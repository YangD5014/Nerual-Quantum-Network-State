"""
FFNN 振幅-相位分离版 VMC（py 版本）

相对 notebook 的修复：
1. jax.grad 报错：实参数、复输出 logψ = logA + 1j·φ 不能直接 jax.grad(holomorphic=False)，
   改为分别对实部/虚部求导后组合 ∇logψ = ∇Re + i·∇Im（Wirtinger）。
2. 实参数 SR 取实部：∇E = 2·Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩；度量用 Re(S)。
3. 重要性加权采样：按 |ψ|（machine_pow=1）采样、估计器加权 w ∝ |ψ|，
   否则 |ψ|² 采样在轨迹接近 HF 本征态时方差→0、梯度恒 0，冻结在 HF（误差 ~57 mHa）。
   修复后能平滑越过 HF 收敛到 FCI（400 步误差 ~3.5 mHa）。
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


# ===================== Ansatz：振幅-相位分离（共享骨干） =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_logA = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_phi = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        logA = jnp.squeeze(self.head_logA(h))
        phi = jnp.squeeze(self.head_phi(h))
        return logA + 1j * phi


def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 machine 函数"""
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


# ===================== 每样本梯度：实参数网络，拆实部/虚部求导 =====================
def _grad_logpsi_real_params(machine, params, sigma):
    """
    实参数、复输出的 logψ 梯度：
        ∇ logψ(σ) = ∇ Re logψ(σ) + i · ∇ Im logψ(σ)
    （这就是 holomorphic=False 时手写的正确版本，绕开 jax.grad 的实输出限制）
    """
    re_grad = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s).real)(params)
    )(sigma)
    im_grad = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s).imag)(params)
    )(sigma)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


# ===================== 重要性加权（配合 machine_pow=1 采样 p ∝ |ψ|） =====================
# 动机：按 |ψ|² 采样时，一旦 |ψ|² 塌缩到单个组态（如 HF），梯度信号 ∝ ψη 的激发组态
# 永远采不到 → force 梯度恒 0 → 冻结在本征态。改按 |ψ| 采样并对估计器加权
# w(σ) = |ψσ|²/p(σ) ∝ |ψσ|，稀有组态访问率提升数个量级，其 w·E_loc·∇logψ 贡献
# 恰好恢复精确公式（w·E_loc ~ ψη·H 为有限量），这是通用的重要性加权 VMC。

@partial(jax.jit, static_argnames=("machine",))
def _sample_weights(machine, params, sigma):
    """w_i = |ψ(σ_i)|（未归一化），配合 p ∝ |ψ| 的采样分布"""
    logpsi = machine(params, sigma)
    return jnp.exp(jnp.real(logpsi))


def _weighted_mean_std(x, w):
    w_sum = jnp.sum(w)
    mean = jnp.sum(w * x) / w_sum
    var = jnp.sum(w * jnp.abs(x - mean) ** 2) / w_sum
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    重要性加权 force 梯度（实参数网络，取实部）：
        ∇⟨E⟩ = 2·Re⟨(E_loc − ⟨E⟩) (∇log ψ)*⟩_w
    E 是实参数的实值函数，梯度必须为实数；复数中间量只保留实部。
    返回的 grad 已含因子 2（即完整 ∇⟨E⟩）。
    """
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


def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    加权 QGT / F 矩阵（实参数网络，取实部）。
    实参数位移 dθ 为实向量时 ⟨|dlogψ|²⟩ = dθᵀ Re(S) dθ，
    虚部是反对称部分、对实二次型无贡献，度量取 Re(S)。
    """
    n_samples = sigma.shape[0]
    w = _sample_weights(machine, params, sigma)
    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    w_mean = jnp.sum(w[:, None] * grad_flat, axis=0, keepdims=True) / jnp.sum(w)
    grad_centered = grad_flat - w_mean
    qw = (w[:, None] * jnp.conj(grad_centered)).T @ grad_centered / jnp.sum(w)
    qgt = qw.real  # 实参数度量
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    return qgt_reg, unravel_fn


# ===================== 训练 =====================
def main():
    print("=" * 60)
    print("H2 分子基本信息")
    print("=" * 60)
    print(f"HF energy = {hf_ground_energy:.8f} Ha")
    print(f"FCI E0 = {E_fcis[0]:.8f} Ha")

    learning_rate = 0.1
    diag_shift = 0.01
    seed = 21

    model = SingleStateAnsatz_SharedBackbone(
        n_spin_orbitals=SINGLE_SIZE, hidden_dim=SINGLE_SIZE * 2 + 4, rngs=nnx.Rngs(seed)
    )
    # machine_pow=1: 按 |ψ|（而非 |ψ|²）采样，配合重要性加权估计器，
    # 避免波函数塌缩后梯度信号丢失（见 _sample_weights 注释）
    sampler = nk.sampler.MetropolisSampler(
        hi, rule=single_rule, n_chains=150, sweep_size=32, machine_pow=1
    )
    machine, graphdef, params = create_machine(model)
    sampler_state = sampler.init_state(machine, params, seed=1)

    optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    n_iter = 400

    history = {'step': [], 'energy': [], 'energy_std': [], 'error': []}

    print("\n" + "=" * 60)
    print("开始纯 JAX VMC 训练 (自然梯度下降法, 重要性加权)")
    print("=" * 60)

    samples = None
    t0 = time.time()
    for step in range(n_iter):
        sampler_state = sampler.reset(machine, params, sampler_state)
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, chain_length=20
        )
        samples = samples.reshape(-1, hi.size)

        # 1. force-based 能量和梯度（grad 已是完整 ∇⟨E⟩，实数）
        energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)

        # 2. 自然梯度 = S^{-1} * grad
        qgt_reg, _ = compute_qgt(machine, params, samples, diag_shift=diag_shift)
        grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
        grad = grad_unravel_fn(natural_grad)

        # 3. 更新参数
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # 4. 记录历史
        if step % 50 == 0 or step == n_iter - 1:
            error = jnp.abs(energy.real - E_fcis[0])
            history['step'].append(step)
            history['energy'].append(float(energy.real))
            history['energy_std'].append(float(energy_std))
            history['error'].append(float(error))
            print(
                f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} "
                f"| FCI: {E_fcis[0]:.8f} | Error: {error:.6f}"
            )

    # 最终结果
    final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
    final_error = jnp.abs(final_energy.real - E_fcis[0])
    print("\n" + "=" * 60)
    print(f"训练完成! 用时 {time.time() - t0:.1f}s")
    print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
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
    ax.set_title('FFNN amplitude-phase VMC (H2/6-31G)')
    ax.legend()
    fig.tight_layout()
    out_png = 'FFNN振幅相位分离版-VMC-1.png'
    fig.savefig(out_png, dpi=150)
    print(f"训练曲线已保存: {out_png}")


if __name__ == '__main__':
    main()
