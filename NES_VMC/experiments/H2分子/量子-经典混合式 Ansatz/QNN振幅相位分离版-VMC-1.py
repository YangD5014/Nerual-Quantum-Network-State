"""
QNN (PQC) 振幅-相位分离版 VMC —— 4-qubit 压缩输入

核心思想：
1. 组态压缩编码：8 位自旋轨道占据组态 σ = [s0..s7] 按相邻对压成 4 个 qubit 的实数输入
       x_j = 2*σ[2j] + σ[2j+1] ∈ {0,1,2,3},  j = 0..3
   例如 HF 组态 [0 0 0 1 0 0 0 1] → x = [0,1,0,1]，即 "0101"。
   量子比特数从 8 → 4（每个 qubit 承载 2 bit 信息）。

2. PQC（纯 JAX 态矢量模拟，4 qubits）：
   - 编码层：每个 qubit 上 RY(θ_j)·RZ(θ_j)，θ_j = π/2 · x_j ∈ {0, π/2, π, 3π/2}，
     四个取值对应布洛赫球上四个不同点，可无损区分 {0,1,2,3}。
   - 变分层 × L：RY(a)·RZ(b) 单比特旋转 + CNOT 环（0-1-2-3-0）纠缠。
   - 输出：测量特征 f = [⟨Z_0..3⟩, ⟨Z0Z1⟩, ⟨Z1Z2⟩, ⟨Z2Z3⟩, ⟨Z3Z0⟩]（实数，8 维）。

3. 复数输出头（量子-经典混合）：PQC 实特征 → tanh 隐层 → logA（振幅）与 φ（相位），
   logψ = logA + i·φ。全部参数为实数、输出为复数。

4. VMC 引擎沿用已验证框架：
   - 实参数 Wirtinger 梯度 ∇logψ = ∇Re + i∇Im；
   - machine_pow=1 按 |ψ| 采样 + 重要性加权 w ∝ |ψ|（防止 |ψ|² 采样零方差冻结在 HF）；
   - 自然梯度 SR：∇E = 2Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩，度量 Re(S)。
"""
import sys
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax import nnx
from functools import partial
from jax import flatten_util

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from H2_631G import SINGLE_SIZE, ha, hi, E_fcis, single_rule, g, hf_ground_energy

jax.config.update("jax_enable_x64", True)


# ===================== 组态压缩编码：8 bit → 4 个 qubit 实数输入 =====================
def encode_configs(sigma):
    """
    sigma: (..., 8) 占据数 0/1
    返回 x: (..., 4) 实数输入，x_j = 2σ[2j] + σ[2j+1] ∈ {0,1,2,3}
    HF [0 0 0 1 0 0 0 1] → x = [0,1,0,1]（即 "0101"）

    编码思想（用户提出）：每个 qubit 承载一对相邻轨道，占据处放门、空处留空，
    HF 编码层即 [I, G, I, G]（"IXIX"）。由于对内两个位置都可能被占（(0,1) 与 (1,0)
    必须区分），纯 X 门会碰撞，故用旋转角 θ = π/2·v 区分：
        v=0 → I          （|0⟩，北极，留空）
        v=2 → X 门       （RY(π)RZ(π) = -iX，精确 X 差全局相位）
        v=1/v=3 → 广义 X（Bloch 赤道上两个不同点）
    四个取值 → 四个不同量子态，无信息损失。
    """
    sigma = jnp.asarray(sigma)
    return (2.0 * sigma[..., 0::2] + sigma[..., 1::2]).astype(jnp.float64)


# ===================== 单比特 / 两比特门（态矢量，支持批量角度） =====================
def ry(theta):
    """theta: (...) → U: (..., 2, 2)"""
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.stack([
        jnp.stack([c, -s], axis=-1),
        jnp.stack([s, c], axis=-1),
    ], axis=-2)


def rz(theta):
    """theta: (...) → U: (..., 2, 2) 对角阵"""
    e = jnp.exp(0.5j * theta)
    return jnp.stack([
        jnp.stack([jnp.conj(e), 0 * e], axis=-1),
        jnp.stack([0 * e, e], axis=-1),
    ], axis=-2)


CNOT = jnp.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=jnp.complex128)


def _bcast(u, p0_ndim):
    """把门元素 u（标量或任意批维）广播到单 qubit 切片 p0 的形状"""
    return u.reshape(u.shape + (1,) * (p0_ndim - u.ndim))


def apply_1q(psi, U, q):
    """
    psi: (..., 2,2,2,2)（qubit 0 = 倒数第 4 轴），U: (..., 2, 2) 或 (2,2)
    """
    ax = psi.ndim - 4 + q
    psi = jnp.moveaxis(psi, ax, -4)
    p0, p1 = psi[..., 0, :, :, :], psi[..., 1, :, :, :]
    u00, u01 = _bcast(U[..., 0, 0], p0.ndim), _bcast(U[..., 0, 1], p0.ndim)
    u10, u11 = _bcast(U[..., 1, 0], p0.ndim), _bcast(U[..., 1, 1], p0.ndim)
    out = jnp.stack([u00 * p0 + u01 * p1, u10 * p0 + u11 * p1], axis=-4)
    return jnp.moveaxis(out, -4, ax)


def apply_cnot_ring(psi):
    """对 qubit (0,1),(1,2),(2,3),(3,0) 依次作用 CNOT（控制在前）"""
    for (c, t) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        psi = _apply_cnot(psi, c, t)
    return psi


def _apply_cnot(psi, c, t):
    axc, axt = psi.ndim - 4 + c, psi.ndim - 4 + t
    psi = jnp.moveaxis(psi, axc, -4)
    if axt < axc:
        axt += 1  # axc 被移到 -4 后，位于其前的轴整体右移一位
    psi = jnp.moveaxis(psi, axt, -3)
    lead = psi.shape[:-4]
    p = psi.reshape(lead + (4, 2, 2))  # 指数 = 2*b_c + b_t
    p = p.at[..., 2, :, :].set(p[..., 3, :, :])
    p = p.at[..., 3, :, :].set(psi.reshape(lead + (4, 2, 2))[..., 2, :, :])
    psi = p.reshape(lead + (2, 2, 2, 2))
    psi = jnp.moveaxis(psi, -3, axt)
    return jnp.moveaxis(psi, -4, axc)


# ===================== PQC 前向：输入实数 → 输出实特征 =====================
N_QUBITS = 4
# ZZ 环特征配对
ZZ_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]
N_FEATURES = N_QUBITS + len(ZZ_PAIRS)  # 8


def pqc_features(x, pqc_params):
    """
    x: (..., 4) 实数输入（取值 0..3）
    pqc_params: (L, 4, 2) 实参数 [RY, RZ]
    返回 features: (..., 8) 实数
    """
    theta = x * (jnp.pi / 2.0)  # (..., 4)
    # 初始 |0000>，qubit 轴在前端便于门操作
    psi = jnp.zeros(theta.shape[:-1] + (2,) * N_QUBITS, dtype=jnp.complex128)
    psi = psi.at[..., 0, 0, 0, 0].set(1.0)

    # 编码层：RY(θ)·RZ(θ)，{0,1,2,3} → 布洛赫球 4 个不同点
    for q in range(N_QUBITS):
        psi = apply_1q(psi, ry(theta[..., q]), q)
        psi = apply_1q(psi, rz(theta[..., q]), q)

    # 变分层
    n_layers = pqc_params.shape[0]
    for l in range(n_layers):
        for q in range(N_QUBITS):
            psi = apply_1q(psi, ry(pqc_params[l, q, 0]), q)
            psi = apply_1q(psi, rz(pqc_params[l, q, 1]), q)
        psi = apply_cnot_ring(psi)

    p = (jnp.abs(psi) ** 2).reshape(theta.shape[:-1] + (2 ** N_QUBITS,))

    # ⟨Z_q⟩：qubit q = 第 q 根轴 = 指数第 q 位（MSB 起）
    z_signs = 1 - 2 * ((jnp.arange(2 ** N_QUBITS)[:, None] >> jnp.arange(3, -1, -1)) & 1)  # (16, 4)
    feats_Z = jnp.einsum('...z,zq->...q', p, z_signs)
    feats_ZZ = jnp.stack(
        [feats_Z[..., a] * feats_Z[..., b] for a, b in ZZ_PAIRS], axis=-1
    )
    return jnp.concatenate([feats_Z, feats_ZZ], axis=-1)


# ===================== Ansatz：PQC 特征 + 经典复数头 =====================
class QNNAnsatz(nnx.Module):
    """实参数、复数输出：logψ = logA + i·φ"""

    def __init__(self, n_layers: int = 3, hidden_dim: int = 16, *,
                 pqc_scale: float = 0.5, rngs: nnx.Rngs):
        self.pqc = nnx.Param(
            jax.random.normal(rngs.params(), (n_layers, N_QUBITS, 2)) * pqc_scale
        )
        self.hid = nnx.Linear(N_FEATURES, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_logA = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_phi = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, sigma: jax.Array) -> jax.Array:
        x = encode_configs(sigma)                      # (..., 4) 实数
        f = pqc_features(x, self.pqc.value)            # (..., 8) 实数
        h = nnx.tanh(self.hid(f))
        logA = jnp.squeeze(self.head_logA(h), axis=-1)
        phi = jnp.squeeze(self.head_phi(h), axis=-1)
        return logA + 1j * phi


def create_machine(model: nnx.Module):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state


# ===================== 局部能量 =====================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


# ===================== 实参数 Wirtinger 梯度 =====================
def _grad_logpsi_real_params(machine, params, sigma):
    """∇ logψ(σ) = ∇ Re logψ(σ) + i · ∇ Im logψ(σ)"""
    re_grad = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s).real)(params)
    )(sigma)
    im_grad = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s).imag)(params)
    )(sigma)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


# ===================== 重要性加权（配合 machine_pow=1 采样 p ∝ |ψ|） =====================
@partial(jax.jit, static_argnames=("machine",))
def _sample_weights(machine, params, sigma):
    """w_i = |ψ(σ_i)|，配合 p ∝ |ψ| 的采样分布"""
    logpsi = machine(params, sigma)
    return jnp.exp(jnp.real(logpsi))


def _weighted_mean_std(x, w):
    w_sum = jnp.sum(w)
    mean = jnp.sum(w * x) / w_sum
    var = jnp.sum(w * jnp.abs(x - mean) ** 2) / w_sum
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """∇⟨E⟩ = 2·Re⟨(E_loc − ⟨E⟩) (∇log ψ)*⟩_w，返回已含因子 2 的实梯度"""
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
    """加权 QGT（实参数度量取 Re(S)）+ 对角正则"""
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


# ===================== 精确校验：16 维全空间波函数 =====================
def exact_energy(machine, params):
    """小系统极限：全空间 16 组态精确求和（|ψ|² 加权），对照采样结果"""
    all_states = hi.all_states()
    logpsi = machine(params, all_states)                       # (16,)
    psi2 = jnp.exp(2 * jnp.real(logpsi))
    eta, H_eta = ha.get_conn_padded(all_states)                # (16,C,8), (16,C)
    logpsi_eta = machine(params, eta)                          # (16,C)
    e_loc = jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi[:, None]), axis=-1)
    return jnp.sum(psi2 * e_loc) / jnp.sum(psi2)


# ===================== 训练 =====================
def main(seed=21, n_iter=600, learning_rate=0.1, diag_shift=0.01,
         n_layers=3, hidden_dim=16, pqc_scale=0.5, tag=""):
    print("=" * 60)
    print("H2 分子基本信息")
    print("=" * 60)
    print(f"HF energy = {hf_ground_energy:.8f} Ha")
    print(f"FCI E0    = {E_fcis[0]:.8f} Ha")

    # 编码演示
    hf_cfg = np.array(hi.all_states()[0])
    print(f"HF 组态 {hf_cfg} → 4-qubit 输入 {np.array(encode_configs(hf_cfg))}（即 0101 编码）")

    model = QNNAnsatz(n_layers=n_layers, hidden_dim=hidden_dim, pqc_scale=pqc_scale, rngs=nnx.Rngs(seed))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"Ansatz: PQC({N_QUBITS} qubits, {n_layers} layers) + 经典复数头, 参数量 {n_params}")

    sampler = nk.sampler.MetropolisSampler(
        hi, rule=single_rule, n_chains=150, sweep_size=32, machine_pow=1
    )
    machine, graphdef, params = create_machine(model)
    sampler_state = sampler.init_state(machine, params, seed=1)

    optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    history = {'step': [], 'energy': [], 'energy_std': [], 'error': []}

    print("\n" + "=" * 60)
    print(f"开始 QNN-VMC 训练 (自然梯度 + 重要性加权, seed={seed})")
    print("=" * 60)

    samples = None
    t0 = time.time()
    for step in range(n_iter):
        sampler_state = sampler.reset(machine, params, sampler_state)
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, chain_length=20
        )
        samples = samples.reshape(-1, hi.size)

        energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)

        qgt_reg, _ = compute_qgt(machine, params, samples, diag_shift=diag_shift)
        grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
        grad = grad_unravel_fn(natural_grad)

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 50 == 0 or step == n_iter - 1:
            error = jnp.abs(energy.real - E_fcis[0])
            e_ex = exact_energy(machine, params)
            history['step'].append(step)
            history['energy'].append(float(energy.real))
            history['energy_std'].append(float(energy_std))
            history['error'].append(float(error))
            print(
                f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} "
                f"| 精确: {e_ex.real:.8f} | Error: {error:.6f}"
            )

    # 精确全空间校验
    e_exact = exact_energy(machine, params)
    final_error = jnp.abs(e_exact.real - E_fcis[0])
    print("\n" + "=" * 60)
    print(f"训练完成! 用时 {time.time() - t0:.1f}s")
    print(f"全空间精确能量：{e_exact.real:.8f} Ha")
    print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
    print(f"绝对误差：{final_error:.6f} Ha")
    print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
    print("=" * 60)

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(history['step'], history['energy'], yerr=history['energy_std'],
                fmt='o-', capsize=3, label='VMC energy')
    ax.axhline(E_fcis[0], color='r', ls='--', label=f'FCI E0 = {E_fcis[0]:.6f} Ha')
    ax.axhline(hf_ground_energy, color='g', ls='--', label=f'HF energy = {hf_ground_energy:.6f} Ha')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title(f'QNN (4-qubit compressed input) VMC, seed={seed}')
    ax.legend()
    fig.tight_layout()
    out_png = f'QNN振幅相位分离版-VMC-1{tag}.png'
    fig.savefig(out_png, dpi=150)
    print(f"训练曲线已保存: {out_png}")

    return float(e_exact.real), float(final_error)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--n-iter', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--pqc-scale', type=float, default=0.5)
    parser.add_argument('--tag', type=str, default="")
    args = parser.parse_args()
    main(seed=args.seed, n_iter=args.n_iter, learning_rate=args.lr,
         n_layers=args.layers, hidden_dim=args.hidden,
         pqc_scale=args.pqc_scale, tag=args.tag)
