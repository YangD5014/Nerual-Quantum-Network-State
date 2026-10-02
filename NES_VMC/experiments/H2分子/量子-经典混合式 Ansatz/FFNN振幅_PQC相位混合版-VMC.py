# -*- coding: utf-8 -*-
"""
FFNN 振幅头 + PennyLane PQC 相位头 混合式 VMC
================================================
系统：H2 / 6-31G（SINGLE_SIZE=4 自旋轨道，物理空间 16 维）
基准：FCI E0 = -1.05434745 Ha

结构（对应 FFNN振幅相位分离版-VMC-3.ipynb 的改造点）：
    σ(4) → FFNN backbone(tanh ×2) → h(hidden)
    logA = Linear(h → 1)                         # 经典振幅头（读 full hidden）
    z    = tanh(Linear(h → 4))                   # 相位支路压缩特征
    φ    = W · PQC(z) + b                        # 相位头 = PennyLane 参数化量子线路
    logψ = logA + i·φ                            # 实参数、复输出

PQC（4 qubits, L 层）：
    编码层  RX(π·z_i)
    变分层  CNOT 链 (0-1,1-2,2-3) + RY(a)/RZ(b) 每比特
    特征    f_i = <Z_i>（4 维实数）→ 经典线性读出 φ
    梯度    diff_method="backprop"（default.qubit 态矢量反传，jax.grad 直通）

VMC 引擎沿用已验证框架（0925 失败/成功分析文档）：
    - 实参数 Wirtinger 梯度 ∇logψ = ∇Re + i·∇Im
    - machine_pow=1 按 |ψ| 采样 + 重要性加权 w ∝ |ψ|（防零方差冻结）
    - 自然梯度 SR：∇E = 2Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩，度量 Re(S)

用法：
    python FFNN振幅_PQC相位混合版-VMC.py [--n-iter 400] [--seed 21] [--lr 0.1]
        [--layers 2] [--hidden 12] [--pqc-scale 0.1] [--diag-shift 0.01]
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax import flatten_util
from functools import partial
import flax.nnx as nnx
import netket as nk
import numpy as np
import optax
import pennylane as qml

jax.config.update("jax_enable_x64", True)

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from H2_6_31G import (
    SINGLE_SIZE, ha, hi, E_fcis, hf_ground_energy, single_rule,
)

# ============================================================
# 1. PennyLane PQC（模块级 QNode，避免 nnx.split 携带设备对象）
# ============================================================
N_QUBITS = 4  # PQC 编码 4 维经典瓶颈特征 z ∈ [-1,1]^4


def pqc_circuit(h, weights, n_qubits=N_QUBITS, n_layers=2):
    """批量 PQC。

    h:       (B, n_qubits) 实数特征（经典网络输出，已 tanh 到 [-1,1]）
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
PQC_NODE = qml.QNode(
    pqc_circuit, _DEV, interface="jax", diff_method="backprop"
)


# ============================================================
# 2. Ansatz：FFNN 振幅头 + PQC 相位头
# ============================================================
class SingleStateAnsatz_FFNN_PQC(nnx.Module):
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
        self.qubits = qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # 经典骨干（实参数）
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        # 相位支路压缩层：hidden -> qubits（PQC 编码输入）
        self.phase_proj = nnx.Linear(hidden_dim, qubits, rngs=rngs, param_dtype=float)

        # 振幅头（经典，直接读 full hidden，保证振幅表达力）
        self.head_logA = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

        # 相位头 = PQC 变分参数 + 线性读出
        key = rngs.params()
        self.qnn_params = nnx.Param(
            jax.random.uniform(key, (n_layers, 2 * qubits), dtype=jnp.float64)
            * pqc_scale
            * np.pi
        )
        self.readout = nnx.Linear(qubits, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))

        # 振幅头：经典，读 full hidden
        logA = jnp.squeeze(self.head_logA(h), axis=-1)

        # 相位头：PQC（4 qubits），输入是 tanh 限幅的 4 维压缩特征
        z = nnx.tanh(self.phase_proj(h))          # (..., qubits)
        lead_shape = jnp.shape(z)[:-1]
        z2 = jnp.reshape(z, (-1, self.qubits))
        feats = PQC_NODE(z2, self.qnn_params, n_layers=self.n_layers)
        feats = jnp.stack(feats, axis=1)          # (B, qubits)
        phi = jnp.squeeze(self.readout(feats), axis=-1)
        phi = jnp.reshape(phi, lead_shape)

        return logA + 1j * phi


# ============================================================
# 3. VMC 引擎（沿用已验证实现）
# ============================================================
def create_machine(model: nnx.Module):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state


@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    # eta 为 3D (N, n_conn, size)，PQC 只支持单一批维 → 展平后计算再还原
    n_conn = eta.shape[-2]
    logpsi_eta = machine(params, eta.reshape(-1, sigma.shape[-1]))
    logpsi_eta = logpsi_eta.reshape(logpsi_sigma.shape + (n_conn,))
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma[..., None]), axis=-1)


def _grad_logpsi_real_params(machine, params, sigma):
    """实参数、复输出：∇logψ = ∇Re + i·∇Im（手写 Wirtinger，绕开 holomorphic 限制）。"""
    re_grad = jax.vmap(lambda s: jax.grad(lambda p: machine(p, s).real)(params))(sigma)
    im_grad = jax.vmap(lambda s: jax.grad(lambda p: machine(p, s).imag)(params))(sigma)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


@partial(jax.jit, static_argnames=("machine",))
def _sample_weights(machine, params, sigma):
    """w_i = |ψ(σ_i)|，配合 p ∝ |ψ| 采样（machine_pow=1）的重要性权重。"""
    logpsi = machine(params, sigma)
    return jnp.exp(jnp.real(logpsi))


def _weighted_mean_std(x, w):
    w_sum = jnp.sum(w)
    mean = jnp.sum(w * x) / w_sum
    var = jnp.sum(w * jnp.abs(x - mean) ** 2) / w_sum
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """重要性加权 force 梯度（实参数，取实部）：∇⟨E⟩ = 2·Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩_w。"""
    w = _sample_weights(machine, params, sigma)
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = _weighted_mean_std(O_loc, w)
    O_centered = O_loc - O_mean

    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)

    def weight_and_mean(grad_component):
        wc = (w * O_centered).reshape((w.shape[0],) + (1,) * (grad_component.ndim - 1))
        return (jnp.sum(wc * jnp.conj(grad_component), axis=0) / jnp.sum(w)).real

    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    return O_mean, O_std, grad


def compute_qgt(machine, params, sigma, diag_shift=0.01):
    """加权 QGT（实参数，度量取 Re(S)）。"""
    n_samples = sigma.shape[0]
    w = _sample_weights(machine, params, sigma)
    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    w_mean = jnp.sum(w[:, None] * grad_flat, axis=0, keepdims=True) / jnp.sum(w)
    grad_centered = grad_flat - w_mean
    qw = (w[:, None] * jnp.conj(grad_centered)).T @ grad_centered / jnp.sum(w)
    qgt = qw.real
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    return qgt_reg, unravel_fn


_EXACT_STATES_INT = jnp.asarray(np.array(hi.all_states()), dtype=jnp.int32)
_EXACT_STATES = _EXACT_STATES_INT.astype(jnp.float64)
_EXACT_ETA, _EXACT_HE = ha.get_conn_padded(_EXACT_STATES_INT)


@partial(jax.jit, static_argnames=("machine",))
def exact_energy(machine, params):
    """全空间 16 组态精确能量 E = Σ p_σ E_loc(σ)，消除采样噪声的收敛判据。"""
    logpsi = machine(params, _EXACT_STATES)
    n_conn = _EXACT_ETA.shape[-2]
    logpsi_eta = machine(
        params, _EXACT_ETA.reshape(-1, _EXACT_STATES.shape[-1])
    ).reshape(logpsi.shape[0], n_conn)
    e_loc = jnp.sum(_EXACT_HE * jnp.exp(logpsi_eta - logpsi[:, None]), axis=-1)
    psi = jnp.exp(logpsi)
    prob = jnp.abs(psi) ** 2
    prob = prob / jnp.sum(prob)
    return jnp.sum(prob * e_loc).real


# ============================================================
# 4. 训练
# ============================================================
def run(n_iter=400, seed=21, lr=0.1, layers=2, hidden=12, pqc_scale=0.1,
        diag_shift=0.01, n_chains=150, sweep_size=32, chain_length=20,
        log_every=50, tag=""):
    model = SingleStateAnsatz_FFNN_PQC(
        n_spin_orbitals=SINGLE_SIZE, hidden_dim=hidden, n_layers=layers,
        pqc_scale=pqc_scale, rngs=nnx.Rngs(seed),
    )
    sampler = nk.sampler.MetropolisSampler(
        hi, rule=single_rule, n_chains=n_chains, sweep_size=sweep_size, machine_pow=1
    )
    machine, graphdef, params = create_machine(model)
    sampler_state = sampler.init_state(machine, params, seed=1)

    optimizer = optax.sgd(learning_rate=lr)
    opt_state = optimizer.init(params)

    history = {"step": [], "energy": [], "energy_std": [], "error": [], "exact": [], "exact_err": []}

    print("\n" + "=" * 64)
    print(f"开始 FFNN振幅+PQC相位 混合 VMC | layers={layers} hidden={hidden} "
          f"pqc_scale={pqc_scale} lr={lr} diag_shift={diag_shift} seed={seed} {tag}")
    print("=" * 64)

    samples = None
    t0 = time.time()
    for step in range(n_iter):
        sampler_state = sampler.reset(machine, params, sampler_state)
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, chain_length=chain_length
        )
        samples = samples.reshape(-1, hi.size)

        energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)

        qgt_reg, _ = compute_qgt(machine, params, samples, diag_shift=diag_shift)
        grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
        grad = grad_unravel_fn(natural_grad)

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % log_every == 0 or step == n_iter - 1:
            ex = float(exact_energy(machine, params))
            error = abs(float(energy.real) - float(E_fcis[0]))
            ex_err = abs(ex - float(E_fcis[0])) * 1000.0  # mHa
            history["step"].append(step)
            history["energy"].append(float(energy.real))
            history["energy_std"].append(float(energy_std))
            history["error"].append(error)
            history["exact"].append(ex)
            history["exact_err"].append(ex_err)
            print(f"Step {step:3d} | E_sample: {energy.real:.8f} ± {energy_std:.6f} "
                  f"| E_exact: {ex:.8f} | 精确误差: {ex_err:.3f} mHa")

    final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
    final_exact = float(exact_energy(machine, params))
    final_err_mHa = abs(final_exact - float(E_fcis[0])) * 1000.0
    print("\n" + "=" * 64)
    print(f"训练完成! 用时 {time.time() - t0:.1f}s")
    print(f"采样能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
    print(f"全空间精确能量：{final_exact:.8f} Ha")
    print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
    print(f"绝对误差：{abs(final_err_mHa):.4f} mHa  |  相对误差：{abs(final_err_mHa)/abs(float(E_fcis[0]))/1000*100:.4f}%")
    print("=" * 64)

    return model, machine, params, history, final_exact, final_err_mHa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--pqc-scale", type=float, default=0.1)
    parser.add_argument("--diag-shift", type=float, default=0.01)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    model, machine, params, history, final_exact, final_err_mHa = run(
        n_iter=args.n_iter, seed=args.seed, lr=args.lr, layers=args.layers,
        hidden=args.hidden, pqc_scale=args.pqc_scale, diag_shift=args.diag_shift,
        tag=args.tag,
    )

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(history["step"], history["energy"], yerr=history["energy_std"],
                    fmt="o-", capsize=3, label="VMC energy (sampled)")
        ax.plot(history["step"], history["exact"], "s--", ms=4,
                label="VMC energy (exact sum)")
        ax.axhline(float(E_fcis[0]), color="r", ls="--",
                   label=f"FCI E0 = {float(E_fcis[0]):.6f} Ha")
        ax.axhline(float(hf_ground_energy), color="g", ls=":",
                   label=f"HF = {float(hf_ground_energy):.6f} Ha")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Energy (Ha)")
        ax.set_ylim(-1.06, -0.94)
        ax.set_title("FFNN amplitude + PQC phase hybrid VMC (H2/6-31G)")
        ax.legend()
        fig.tight_layout()
        out_png = "FFNN振幅_PQC相位混合版-VMC.png"
        fig.savefig(out_png, dpi=150)
        print(f"训练曲线已保存: {out_png}")
