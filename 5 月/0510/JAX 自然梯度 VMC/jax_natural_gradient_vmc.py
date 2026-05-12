#!/usr/bin/env python3
"""
JAX 自然梯度方法求解 VMC 基态能量

本文档展示如何使用纯 JAX 实现自然梯度下降法来求解变分量子蒙特卡洛（VMC）的基态能量。

核心算法：
1. 神经网络变分波函数 ansatz
2. Metropolis-Hastings 采样
3. Force-based 梯度计算
4. 量子几何张量（QGT）计算
5. 自然梯度更新：θ ← θ - η · S⁻¹∇E
"""

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx
import optax
from functools import partial
from jax import flatten_util
from tqdm import tqdm
import time


print("=" * 70)
print("JAX 自然梯度 VMC 基态能量计算")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"NetKet version: {nk.__version__}")

# ==============================================================================
# 1. H₂ 分子定义与 FCI 基准
# ==============================================================================
print("\n" + "=" * 70)
print("1. 系统设置：H₂ 分子")
print("=" * 70)

bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print(f"键长: {bond_length} Å")
print(f"基组: STO-3G")
print(f"\nFCI 精确基准能量:")
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"  E{i} = {e:.8f} Ha  (激发能: {exc:.4f} eV)")

# ==============================================================================
# 2. 哈密顿量与 Hilbert 空间
# ==============================================================================
print("\n" + "=" * 70)
print("2. 哈密顿量与采样器设置")
print("=" * 70)

ha = nkx.operator.from_pyscf_molecule(mol)

hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

print(f"Hilbert 空间维度: {hi.size}")

edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(
    hi, 
    rule=single_rule, 
    n_chains=16, 
    sweep_size=32
)

print(f"采样器: Metropolis-Hastings")
print(f"  - 链数: {sampler.n_chains}")
print(f"  - 每链步数: {sampler.sweep_size}")

# ==============================================================================
# 3. 神经网络 Ansatz
# ==============================================================================
print("\n" + "=" * 70)
print("3. 神经网络变分波函数")
print("=" * 70)


class SingleStateAnsatz(nnx.Module):
    """
    神经网络变分波函数 ansatz
    
    结构: 4 → 16 → 16 → 1
    激活函数: tanh
    参数类型: complex
    """
    
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(
            n_spin_orbitals, hidden_dim, 
            rngs=rngs, param_dtype=complex
        )
        self.linear2 = nnx.Linear(
            hidden_dim, hidden_dim, 
            rngs=rngs, param_dtype=complex
        )
        self.output = nnx.Linear(
            hidden_dim, 1, 
            rngs=rngs, param_dtype=complex
        )
    
    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


# ==============================================================================
# 4. Machine 函数包装
# ==============================================================================
def create_machine(model: nnx.Module):
    """
    将 Flax NNX 模型包装为 NetKet 风格的 machine 函数
    
    关键操作：
    - nnx.split(): 分离静态结构（graphdef）和动态参数（state）
    - machine(): 接受 (params, sigma)，返回波函数值
    """
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    
    return machine, graphdef, state


# ==============================================================================
# 5. 局部能量计算
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ)
    
    公式: E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)
    
    参数:
        machine: 波函数机器
        params: 网络参数
        sigma: 样本 (n_samples, n_orbitals)
    
    返回:
        E_loc: 局部能量 (n_samples,)
    """
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


# ==============================================================================
# 6. 统计量计算
# ==============================================================================
def statistics(x):
    """
    计算样本统计量
    
    返回:
        mean: 均值
        std: 标准差（考虑样本数）
    """
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


# ==============================================================================
# 7. Force-based 梯度计算
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    Force-based 梯度计算
    
    公式: ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩
    
    关键点:
    - holomorphic=True: 处理复数梯度
    - jax.vmap: 批量计算每个样本的梯度
    - jax.tree_util.tree_map: 处理 PyTree 结构
    
    参数:
        machine: 波函数机器
        params: 网络参数
        sigma: 样本
    
    返回:
        energy: 能量均值
        energy_std: 能量标准差
        grad: 梯度 (PyTree)
    """
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = statistics(O_loc)
    O_centered = O_loc - O_mean
    
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(
            lambda p: log_psi_single(p, s), 
            holomorphic=True
        )(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    def weight_and_mean(grad_component):
        weights = O_centered.reshape(
            (O_centered.shape[0],) + (1,) * (grad_component.ndim - 1)
        )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    
    return O_mean, O_std, grad


# ==============================================================================
# 8. 量子几何张量 (QGT) 计算
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def compute_qgt_jit(machine, params, sigma, diag_shift=0.1):
    """
    完全 JIT 化的 QGT 计算
    
    QGT 定义:
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    
    矩阵形式:
    S = (1/N) · (∇log ψ*)ᵀ · ∇log ψ - ⟨∇log ψ*⟩ᵀ · ⟨∇log ψ⟩
    
    参数:
        machine: 波函数机器
        params: 网络参数
        sigma: 样本
        diag_shift: 对角正则化参数 λ
    
    返回:
        qgt_reg: 正则化后的 QGT 矩阵 (n_params, n_params)
    """
    n_samples = sigma.shape[0]
    
    def log_psi(p, s):
        return machine(p, s)
    
    def grad_fn(s):
        return jax.grad(log_psi, holomorphic=True)(params, s)
    
    grad_matrix = jax.vmap(grad_fn)(sigma)
    
    grad_flat, _ = jax.flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_cent = grad_flat - grad_mean
    
    qgt = (grad_cent.conj().T @ grad_cent) / n_samples
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    
    return qgt_reg


def get_unravel_fn(params, machine, sigma_sample):
    """
    获取 PyTree 结构信息（只需运行一次）
    
    用于将展平的梯度向量恢复为 PyTree 结构
    """
    def log_psi(p, s):
        return machine(p, s)
    g = jax.grad(log_psi, holomorphic=True)(params, sigma_sample[0])
    _, unravel_fn = jax.flatten_util.ravel_pytree(g)
    return unravel_fn


# ==============================================================================
# 9. 自然梯度计算
# ==============================================================================
def compute_natural_gradient(machine, params, sigma, grad, diag_shift=0.1):
    """
    计算自然梯度: S⁻¹ ∇E
    
    步骤:
    1. 计算 QGT S
    2. 展平梯度
    3. 求解线性系统 S · x = grad
    4. 恢复 PyTree 结构
    
    参数:
        machine: 波函数机器
        params: 网络参数
        sigma: 样本
        grad: 普通梯度
        diag_shift: 正则化参数
    
    返回:
        nat_grad: 自然梯度 (PyTree)
    """
    qgt_reg = compute_qgt_jit(machine, params, sigma, diag_shift)
    grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
    nat_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    nat_grad = grad_unravel_fn(nat_grad_flat)
    
    return nat_grad


# ==============================================================================
# 10. 初始化
# ==============================================================================
print("\n" + "=" * 70)
print("4. 模型初始化")
print("=" * 70)

rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, hidden_dim=16, rngs=rngs)
machine, graphdef, params = create_machine(model)

sampler_state = sampler.init_state(machine, params, seed=1)
optimizer = optax.sgd(learning_rate=0.02)
opt_state = optimizer.init(params)

print(f"模型: SingleStateAnsatz")
print(f"  - 输入维度: 4")
print(f"  - 隐藏层维度: 16")
print(f"  - 输出维度: 1")
print(f"  - 激活函数: tanh")
print(f"  - 参数类型: complex")

param_count = sum(
    p.size for p in jax.tree_util.tree_leaves(params)
)
print(f"总参数数: {param_count}")

# ==============================================================================
# 11. 训练参数
# ==============================================================================
N_ITER = 300
N_SAMPLES = 1008
DIAG_SHIFT = 0.001

print(f"\n训练参数:")
print(f"  - 迭代次数: {N_ITER}")
print(f"  - 样本数: {N_SAMPLES}")
print(f"  - 学习率: 0.01")
print(f"  - QGT 正则化: {DIAG_SHIFT}")

# ==============================================================================
# 12. 训练主循环
# ==============================================================================
print("\n" + "=" * 70)
print("5. 开始自然梯度训练")
print("=" * 70)

history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}

start_time = time.time()

for step in range(N_ITER):
    # 1. 采样
    sampler_state = sampler.reset(machine, params, sampler_state)
    samples, sampler_state = sampler.sample(
        machine, params, 
        state=sampler_state, 
        chain_length=N_SAMPLES // sampler.n_chains
    )
    samples = samples.reshape(-1, hi.size)
    
    # 2. 计算 force-based 能量和梯度
    energy, energy_std, grad = forces_expect_hermitian(
        machine, params, samples
    )
    
    # 3. 计算自然梯度
    nat_grad = compute_natural_gradient(
        machine, params, samples, grad, 
        diag_shift=DIAG_SHIFT
    )
    
    # 4. 参数更新
    updates, opt_state = optimizer.update(nat_grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # 5. 记录历史
    if step % 50 == 0 or step == N_ITER - 1:
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        history['energy'].append(float(energy.real))
        history['energy_std'].append(float(energy_std))
        history['error'].append(float(error))
        
        elapsed = time.time() - start_time
        print(
            f"Step {step:4d} | "
            f"E: {energy.real:10.6f} ± {energy_std:.6f} | "
            f"FCI: {E_fcis[0]:10.6f} | "
            f"Error: {error:.6f} | "
            f"Time: {elapsed:6.1f}s"
        )

# ==============================================================================
# 13. 最终结果
# ==============================================================================
print("\n" + "=" * 70)
print("6. 训练结果")
print("=" * 70)

final_energy, final_std, _ = forces_expect_hermitian(
    machine, params, samples
)
final_error = jnp.abs(final_energy.real - E_fcis[0])
total_time = time.time() - start_time

print(f"最终能量:     {final_energy.real:.8f} ± {final_std:.6f} Ha")
print(f"FCI 基准:     {E_fcis[0]:.8f} Ha")
print(f"绝对误差:     {final_error:.6f} Ha")
print(f"相对误差:     {final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print(f"总训练时间:   {total_time:.1f} 秒")
print(f"平均每步:     {total_time / N_ITER * 1000:.1f} 毫秒")
print("=" * 70)


# ==============================================================================
# 14. 可视化历史（可选）
# ==============================================================================
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = history['step']
    energies = history['energy']
    errors = history['error']
    
    axes[0].errorbar(
        steps, energies, 
        yerr=history['energy_std'],
        fmt='o-', capsize=3, capthick=1,
        color='blue', markersize=6,
        label='VMC Energy'
    )
    axes[0].axhline(
        y=E_fcis[0], color='red', linestyle='--',
        linewidth=2, label=f'FCI: {E_fcis[0]:.6f} Ha'
    )
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Energy (Ha)', fontsize=12)
    axes[0].set_title('Energy vs Iteration', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(steps, errors, 'o-', color='green', markersize=6)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Absolute Error (Ha)', fontsize=12)
    axes[1].set_title('Error vs Iteration (Log Scale)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n训练曲线已保存: training_history.png")
    plt.show()
    
except ImportError:
    print("\n提示: 安装 matplotlib 可查看训练曲线")
    print("  pip install matplotlib")


# ==============================================================================
# 15. 主函数入口
# ==============================================================================
if __name__ == "__main__":
    print("\n训练完成!")
    print("\n核心算法总结:")
    print("  1. 神经网络 ansatz: 4 → 16 → 16 → 1")
    print("  2. Force-based 梯度: ∇E = ⟨(E_loc - ⟨E⟩)∇log ψ⟩")
    print("  3. QGT 计算: S = ⟨∇log ψ* ∇log ψᵀ⟩ - ⟨∇log ψ*⟩⟨∇log ψᵀ⟩")
    print("  4. 自然梯度: θ ← θ - η · S⁻¹∇E")
