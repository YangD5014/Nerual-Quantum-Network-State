#!/usr/bin/env python3
"""
纯 JAX 实现 VMC 梯度验证与对比分析

本文档展示如何使用纯 JAX 函数实现 VMC 的梯度计算，并与 NetKet 官方实现进行对比验证。
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


print(f"JAX version: {jax.__version__}")
print(f"NetKet version: {nk.__version__}")

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

ha = nkx.operator.from_pyscf_molecule(mol)
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

# ==============================================================================
# 2. 神经网络 Ansatz
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


# ==============================================================================
# 3. 包装模型为 machine 函数
# ==============================================================================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    
    return machine, graphdef, state


# ==============================================================================
# 4. 纯 JAX 实现的 force-based 梯度计算
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)
    
    这对应 NetKet 的 local_value_kernel
    """
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    核心：复刻 NetKet 的 forces_expect_hermitian 函数
    
    使用 force-based 梯度计算：
    ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩
    
    关键：对于复数值网络，使用 holomorphic=True
    """
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = statistics(O_loc)
    O_centered = O_loc - O_mean
    
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    def weight_and_mean(grad_component):
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    
    return O_mean, O_std, grad


# ==============================================================================
# 5. QGT (量子几何张量) 计算
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）/ F 矩阵
    
    QGT 定义：
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    """
    n_samples = sigma.shape[0]
    
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean
    
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    
    return qgt_reg, unravel_fn


# ==============================================================================
# 6. 完全 JIT 化的 QGT 计算（性能优化版本）
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def compute_qgt_jit(machine, params, sigma, diag_shift=0.1):
    """
    完全 JIT 化的 QGT 计算
    
    关键优化：
    - 在 JIT 内部完成所有计算
    - 不返回函数对象，只返回张量
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
    """获取 PyTree 结构信息（只需运行一次）"""
    def log_psi(p, s):
        return machine(p, s)
    g = jax.grad(log_psi, holomorphic=True)(params, sigma_sample[0])
    _, unravel_fn = jax.flatten_util.ravel_pytree(g)
    return unravel_fn


# ==============================================================================
# 7. 与 NetKet 官方实现对比
# ==============================================================================
def compare_with_netket():
    """比较纯 JAX 实现与 NetKet 官方实现的梯度"""
    print("\n" + "="*60)
    print("对比纯 JAX 实现与 NetKet 官方实现")
    print("="*60)
    
    # 采样器
    edges = [(0, 1), (2, 3)]
    g = nk.graph.Graph(edges=edges)
    single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)
    
    # 初始化模型
    model = SingleStateAnsatz(4, 12, rngs=nnx.Rngs(21))
    machine, graphdef, params = create_machine(model)
    
    # NetKet MCState
    vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
    
    # NetKet 官方梯度
    value_nk, grad_nk = vstate.expect_and_grad(ha)
    print(f"\nNetKet 能量: {value_nk:.8f}")
    
    # 纯 JAX 梯度
    value_jax, _, grad_jax = forces_expect_hermitian(machine, params, vstate.samples.reshape(-1, 4))
    print(f"纯 JAX 能量: {value_jax.real:.8f}")
    
    # 对比梯度
    print("\n梯度对比（部分参数）：")
    print("参数\t\t\tNetKet\t\t\t纯 JAX\t\t\t比例")
    print("-" * 80)
    
    for name, p_nk in grad_nk.items():
        if hasattr(p_nk, 'value'):
            p_nk_val = p_nk.value
        else:
            p_nk_val = np.array(p_nk)
        
        p_jax = grad_jax[name]
        
        # 显示第一个元素
        ratio = np.abs(p_nk_val.flat[0] / (p_jax.flat[0] + 1e-10))
        print(f"{name:15s}\t{p_nk_val.flat[0]:12.6e}\t{p_jax.flat[0]:12.6e}\t{ratio:8.4f}")
    
    return vstate, model, machine, params, sampler


# ==============================================================================
# 8. 自然梯度训练
# ==============================================================================
def train_with_natural_gradient():
    """使用自然梯度进行 VMC 训练"""
    print("\n" + "="*60)
    print("开始纯 JAX VMC 训练 (自然梯度下降法)")
    print("="*60)
    
    # 采样器
    edges = [(0, 1), (2, 3)]
    g = nk.graph.Graph(edges=edges)
    single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)
    
    # 初始化模型
    rngs = nnx.Rngs(21)
    model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
    machine, graphdef, params = create_machine(model)
    
    sampler_state = sampler.init_state(machine, params, seed=1)
    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    # 训练参数
    N_ITER = 300
    N_SAMPLES = 1008
    
    history = {
        'step': [],
        'energy': [],
        'energy_std': [],
        'error': []
    }
    
    sampler_state = sampler.init_state(machine, params, seed=21)
    
    for step in range(N_ITER):
        # 1. 采样
        sampler_state = sampler.reset(machine, params, sampler_state)
        
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, 
            chain_length=N_SAMPLES // sampler.n_chains
        )
        samples = samples.reshape(-1, hi.size)
        
        # 2. 计算 force-based 能量和梯度
        energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
        
        # 3. 梯度缩放（根据实际情况调整）
        grad = jax.tree_map(lambda x: x * 2, grad)
        
        # 4. 计算 QGT
        qgt_reg = compute_qgt_jit(machine, params, samples, diag_shift=0.001)
        
        # 5. 计算自然梯度：S^{-1} * grad
        grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        nat_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
        nat_grad = grad_unravel_fn(nat_grad_flat)
        
        # 6. 使用自然梯度
        grad = nat_grad
        
        # 7. 参数更新
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # 8. 记录历史
        if step % 100 == 0 or step == N_ITER - 1:
            error = jnp.abs(energy.real - E_fcis[0])
            history['step'].append(step)
            history['energy'].append(float(energy.real))
            history['energy_std'].append(float(energy_std))
            history['error'].append(float(error))
            print(f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} | FCI: {E_fcis[0]:.8f} | Error: {error:.6f}")
    
    # 最终结果
    final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
    final_error = jnp.abs(final_energy.real - E_fcis[0])
    print("\n" + "="*60)
    print(f"训练完成!")
    print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
    print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
    print(f"绝对误差：{final_error:.6f} Ha")
    print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
    print("="*60)
    
    return history


# ==============================================================================
# 9. 主函数
# ==============================================================================
if __name__ == "__main__":
    # 对比 NetKet 实现
    vstate, model, machine, params, sampler = compare_with_netket()
    
    # 使用自然梯度训练
    history = train_with_natural_gradient()
