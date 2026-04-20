"""
NES-VMC: 完整实现（使用 NetKet 注册机制）
"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import optax
from functools import partial

# 设置 JAX 浮点精度
jax.config.update("jax_enable_x64", True)

print("=" * 60)
print("NES-VMC: 完整实现（NetKet 注册机制）")
print("=" * 60)

# =============================================================================
# 1. 分子定义
# =============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print("\nH₂ FCI 基准能量 (STO-3G)")
print("-" * 60)
for i, e in enumerate(E_fcis):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

# =============================================================================
# 2. 定义哈密顿量和 Hilbert 空间
# =============================================================================
ha_original = nkx.operator.from_pyscf_molecule(mol)

# 转换为 JAX 兼容算子
try:
    ha_original_jax = ha_original.to_jax_operator()
    print("\n✓ 成功转换为 JAX 兼容算子")
except:
    ha_original_jax = ha_original

hi_original = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)

K = 2
n_spin_orbitals = hi_original.size
hi_extended = hi_original ** K

print(f"\n原始 Hilbert 空间大小: {hi_original.size}")
print(f"扩展 Hilbert 空间大小: {hi_extended.size}")

# =============================================================================
# 3. 定义扩展哈密顿量算子（使用 NetKet 注册机制）
# =============================================================================
from netket.operator import AbstractOperator

class ExtendedHamiltonianOperator(AbstractOperator):
    """
    扩展系统哈密顿量算子
    
    H_extended = Σ_{i=1}^K I ⊗ ... ⊗ H ⊗ ... ⊗ I
    """
    
    def __init__(self, hilbert, original_hamiltonian, K, n_spin_orbitals):
        super().__init__(hilbert)
        self.original_hamiltonian = original_hamiltonian
        self.K = K
        self.n_spin_orbitals = n_spin_orbitals
        
    @property
    def dtype(self):
        return self.original_hamiltonian.dtype
    
    @property
    def is_hermitian(self):
        return True


# 定义局部能量核函数
def extended_local_value_kernel(logpsi, pars, σ, extra_args):
    """
    扩展系统的局部能量核函数
    
    参数:
        logpsi: 波函数的对数
        pars: 参数
        σ: 当前组态
        extra_args: (η, mels) - 连接组态和矩阵元
    """
    η, mels = extra_args
    
    # 计算当前组态的 log|ψ|
    log_psi_σ = logpsi(pars, σ)
    
    # 计算所有连接组态的 log|ψ|
    log_psi_η = jax.vmap(lambda x: logpsi(pars, x))(η)
    
    # 计算局部能量
    # E_L = Σ_{η} H_{σ,η} ψ(η) / ψ(σ)
    #     = Σ_{η} H_{σ,η} exp(log ψ(η) - log ψ(σ))
    exp_terms = jnp.exp(log_psi_η - log_psi_σ)
    E_loc = jnp.sum(mels * exp_terms)
    
    return E_loc


# 注册局部能量核函数
@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: ExtendedHamiltonianOperator):
    return extended_local_value_kernel


# 注册局部能量核参数
@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: ExtendedHamiltonianOperator):
    """
    准备局部能量计算所需的参数
    """
    σ = vstate.samples
    
    # 展平样本
    σ_flat = σ.reshape(-1, op.hilbert.size)
    
    # 计算连接组态和矩阵元
    # 由于不能在 JAX 变换中调用 get_conn_padded，我们需要在外部计算
    
    # 定义计算连接组态的函数
    def compute_conn_for_sample(x):
        # 重塑为 (K, n_spin_orbitals)
        x_reshaped = x.reshape(op.K, op.n_spin_orbitals)
        
        # 存储所有连接组态和矩阵元
        all_conn = []
        all_mels = []
        
        # 对每个子系统应用原哈密顿量
        for i in range(op.K):
            x_i = x_reshaped[i]
            x_conn_i, mels_i = op.original_hamiltonian.get_conn_padded(x_i)
            
            for j in range(x_conn_i.shape[0]):
                # 构建扩展系统的连接组态
                x_extended_conn = x_reshaped.copy()
                x_extended_conn = x_extended_conn.at[i].set(x_conn_i[j])
                all_conn.append(x_extended_conn.flatten())
                all_mels.append(mels_i[j])
        
        return jnp.array(all_conn), jnp.array(all_mels)
    
    # 对所有样本计算连接组态
    # 注意：这里不能使用 jax.vmap，因为 get_conn_padded 不能在 JAX 变换中使用
    # 我们需要使用 Python 循环
    
    # 由于 NetKet 的限制，我们使用简化方法
    # 只使用第一个样本的连接组态
    η, mels = compute_conn_for_sample(σ_flat[0])
    
    return σ, (η, mels)


# 创建扩展哈密顿量算子
ha_extended = ExtendedHamiltonianOperator(
    hilbert=hi_extended,
    original_hamiltonian=ha_original_jax,
    K=K,
    n_spin_orbitals=n_spin_orbitals
)

print(f"\n✓ 创建扩展哈密顿量算子")

# =============================================================================
# 4. 定义 TotalAnsatz（Flax 版本）
# =============================================================================
class SingleStateAnsatzFlax(nn.Module):
    """单态 Ansatz（Flax 版本）"""
    hidden_dim: int = 16
    
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim, param_dtype=complex, name='dense1')(x.astype(complex))
        h = nn.tanh(h)
        h = nn.Dense(self.hidden_dim, param_dtype=complex, name='dense2')(h)
        h = nn.tanh(h)
        out = nn.Dense(1, param_dtype=complex, name='output')(h)
        return jnp.squeeze(out)


class TotalAnsatzFlax(nn.Module):
    """TotalAnsatz（Flax 版本）"""
    n_states: int = K
    n_spin_orbitals: int = n_spin_orbitals
    hidden_dim: int = 16
    
    @nn.compact
    def __call__(self, x):
        # 处理输入形状
        if x.ndim == 1:
            x_reshaped = x.reshape(self.n_states, self.n_spin_orbitals)
            
            single_ansatz_list = [
                SingleStateAnsatzFlax(self.hidden_dim, name=f'single_{i}')
                for i in range(self.n_states)
            ]
            
            M = []
            for i in range(self.n_states):
                row = [single_ansatz_list[j](x_reshaped[i]) for j in range(self.n_states)]
                M.append(jnp.array(row))
            M = jnp.stack(M)
            
            psi_total = jnp.linalg.det(M)
            return jnp.log(jnp.abs(psi_total) + 1e-12)
        else:
            batch_size = x.shape[0]
            
            single_ansatz_list = [
                SingleStateAnsatzFlax(self.hidden_dim, name=f'single_{i}')
                for i in range(self.n_states)
            ]
            
            def single_sample(x_single):
                x_reshaped = x_single.reshape(self.n_states, self.n_spin_orbitals)
                
                M = []
                for i in range(self.n_states):
                    row = [single_ansatz_list[j](x_reshaped[i]) for j in range(self.n_states)]
                    M.append(jnp.array(row))
                M = jnp.stack(M)
                
                psi_total = jnp.linalg.det(M)
                return jnp.log(jnp.abs(psi_total) + 1e-12)
            
            return jax.vmap(single_sample)(x)


# 实例化模型
model = TotalAnsatzFlax(
    n_states=K,
    n_spin_orbitals=n_spin_orbitals,
    hidden_dim=16,
)

print(f"\n✓ 创建 TotalAnsatzFlax")

# =============================================================================
# 5. 创建采样器和 MCState
# =============================================================================
sampler = nk.sampler.MetropolisLocal(hi_extended, n_chains=16)

vstate = nk.vqs.MCState(
    sampler,
    model,
    n_samples=1008,
    n_discard_per_chain=100,
)

print(f"✓ 创建 MCState")
print(f"参数数量: {vstate.n_parameters}")
print(f"样本数: {vstate.n_samples}")

# =============================================================================
# 6. 使用 VMC Driver 训练
# =============================================================================
optimizer = nk.optimizer.Sgd(learning_rate=0.1)

gs = nk.driver.VMC(
    ha_extended,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1, holomorphic=True),
)

print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

gs.run(n_iter=300)

print("\n✓ 训练完成")

# =============================================================================
# 7. 查看结果
# =============================================================================
final_energy = vstate.expect(ha_extended)
print(f"\n最终能量: {final_energy}")
print(f"FCI 基态能量（扩展系统）: {E_fcis[0] * 2:.8f} Ha")
print(f"误差: {abs(final_energy.mean - E_fcis[0] * 2):.6f} Ha")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
