"""QE-NES-VMC：量子增强 NES-VMC 的混合拟设模块（H2/6-31G，K=4）。

按《Quantum_Enhanced_NES_VMC方案总结.md》§3 方案 A 实现单态混合 Ansatz：

    ψ_i(x; λ_i, θ_i) = φ_i^NN(x; λ_i) · φ_i^QC(x; θ_i)

数值上以 log 域拼装（实参数、复输出）：

    logψ_i(x) = logA_i^FFNN(x; λ_i)  +  i · φ_i^PQC(x; θ_i)

- φ^NN：FFNN 骨干（tanh×2）+ 振幅头 head_logA（读 full hidden，保证振幅表达力，
  沿用「FFNN振幅_PQC相位混合版-VMC.py」已验证设计，400 步 1.971 mHa）。
- φ^QC：PennyLane PQC 相位头——4 维压缩特征 z = tanh(phase_proj(h)) 经 RX(πz)
  编码，2 层 CNOT 链 + RY/RZ 变分层，⟨Z_i⟩ 读出后线性组合成标量相位。
  QNode 置于模块级（避免 nnx.split 携带设备对象，项目已验证教训）。

多态联合行列式（§4）：

    D(X) = det[ ψ_i(x_j) ]_{i,j=0}^{K-1}

接口与 NES_VMC_tool.create_gauge_reset_total_machines / make_gauge_fn 完全兼容：
暴露 .K、.n_spin、.single_ansatz_list（nnx.List），单态前向支持
(n_spin,) → 标量、(B, n_spin) → (B,) 两种批量；__call__ 返回
(log_Psi_stable, L_stable, shift) 三元组，与 NESTotalAnsatz_stable 一致。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import flax.nnx as nnx
from functools import partial


# ============================================================
# 1. 模块级 PQC QNode（4 qubits；default.qubit 反传，jax.grad 直通）
# ============================================================
N_QUBITS = 4  # PQC 编码 4 维经典压缩特征 z ∈ [-1,1]^4


def pqc_circuit(h, weights, n_qubits=N_QUBITS, n_layers=2):
    """批量 PQC。

    h:       (B, n_qubits) 实数特征（经典网络压缩输出，tanh 限幅）
    weights: (n_layers, 2*n_qubits) 变分参数
    返回：n_qubits 个 <Z_i>，每个形状 (B,)
    """
    for i in range(n_qubits):
        qml.RX(h[:, i] * np.pi, wires=i)
    qml.Barrier(wires=range(n_qubits))

    for layer in range(n_layers):
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.Barrier(wires=range(n_qubits))
        for i in range(n_qubits):
            qml.RY(weights[layer, 2 * i], wires=i)
            qml.RZ(weights[layer, 2 * i + 1], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


_DEV = qml.device("default.qubit", wires=N_QUBITS)
PQC_NODE = qml.QNode(pqc_circuit, _DEV, interface="jax", diff_method="backprop")


# ============================================================
# 2. 单态混合 Ansatz：φ^NN（FFNN 振幅）× φ^QC（PQC 相位）
# ============================================================
class QESingleStateAnsatz(nnx.Module):
    """logψ(x) = logA^FFNN(x) + i·φ^PQC(x)，全部实参数。

    前向批量约定（与 NES_VMC_tool 的调用方式对齐）：
        x: (n_spin,)   → 标量 logψ
        x: (B, n_spin) → (B,) logψ
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        hidden_dim: int = 12,
        qubits: int = N_QUBITS,
        n_layers: int = 2,
        pqc_scale: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.n_so = n_spin_orbitals
        self.qubits = qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # 经典骨干（实参数）
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        # 振幅头（经典，读 full hidden）
        self.head_logA = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        # 相位支路压缩层：hidden -> qubits（PQC 编码输入）
        self.phase_proj = nnx.Linear(hidden_dim, qubits, rngs=rngs, param_dtype=float)

        # PQC 变分参数 + 线性读出（相位头 φ^QC）
        key = rngs.params()
        self.qnn_params = nnx.Param(
            jax.random.uniform(key, (n_layers, 2 * qubits), dtype=jnp.float64)
            * pqc_scale
            * np.pi
        )
        self.readout = nnx.Linear(qubits, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))

        # 振幅：logA（实数）
        logA = jnp.squeeze(self.head_logA(h), axis=-1)

        # 相位：PQC（只支持单一批量维 → reshape (-1, qubits) 后还原）
        z = nnx.tanh(self.phase_proj(h))            # (..., qubits)
        lead_shape = jnp.shape(z)[:-1]
        z2 = jnp.reshape(z, (-1, self.qubits))
        feats = PQC_NODE(z2, self.qnn_params, n_layers=self.n_layers)
        feats = jnp.stack(feats, axis=1)           # (B, qubits)
        phi = jnp.squeeze(self.readout(feats), axis=-1)
        phi = jnp.reshape(phi, lead_shape)

        return logA + 1j * phi


# ============================================================
# 3. 总 Ansatz：K 个混合单态的联合行列式 D(X) = det[ψ_i(x_j)]
# ============================================================
class QETotalAnsatz(nnx.Module):
    """QE-NES 总拟设：K 个「FFNN 振幅 × PQC 相位」单态组成的行列式波函数。

    与 NESTotalAnsatz_stable（K4 RealFFNN notebook）接口一致：
        .K / .n_spin / .single_ansatz_list（nnx.List）
        __call__(x) -> (log_Psi_stable, L_stable, shift)
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        n_states: int = 4,
        hidden_dim: int = 12,
        qubits: int = N_QUBITS,
        n_layers: int = 2,
        pqc_scale: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = QESingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim=hidden_dim,
                qubits=qubits,
                n_layers=n_layers,
                pqc_scale=pqc_scale,
                rngs=sub_rngs,
            )
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)

            # complex128 承接复输出，避免隐式降精度 scatter
            L = jnp.zeros((self.K, self.K), dtype=jnp.complex128)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))

            shift = jax.lax.stop_gradient(jnp.max(jnp.real(L)))
            L_stable = L - shift
            Psi_stable = jnp.exp(L_stable)

            sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi_stable, L_stable, shift

        if x.ndim == 1:
            return _forward_single(x.reshape(self.K, self.n_spin))
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        if x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        if x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        raise ValueError(f"不支持的输入形状: {x.shape}")
