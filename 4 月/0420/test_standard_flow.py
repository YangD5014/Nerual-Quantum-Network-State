"""
测试使用 NetKet 标准流程求解扩展系统
"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax

# 设置 JAX 浮点精度
jax.config.update("jax_enable_x64", True)

print("=" * 60)
print("测试 NetKet 标准流程求解扩展系统")
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

# 转换为 JAX 兼容的算子
try:
    ha_original = ha_original.to_jax_operator()
    print("\n✓ 成功转换为 JAX 兼容算子")
except Exception as e:
    print(f"\n✗ 无法转换为 JAX 算子: {e}")

# 原始 Hilbert 空间
hi_original = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)

# NES-VMC 参数
K = 2

# 扩展 Hilbert 空间
hi_extended = hi_original ** K

print(f"\n原始 Hilbert 空间大小: {hi_original.size}")
print(f"扩展 Hilbert 空间大小: {hi_extended.size}")

# =============================================================================
# 3. 实现扩展哈密顿量算子
# =============================================================================
from netket.operator import DiscreteJaxOperator

class ExtendedHamiltonianOperator(DiscreteJaxOperator):
    """扩展系统哈密顿量算子"""
    
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
    
    def get_conn_padded(self, x):
        """获取连接组态和矩阵元"""
        original_shape = x.shape
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
        
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, self.K, self.n_spin_orbitals)
        
        all_conn = []
        all_mels = []
        
        for i in range(self.K):
            x_i = x_reshaped[:, i, :]
            x_conn_i, mels_i = self.original_hamiltonian.get_conn_padded(x_i)
            
            n_conn_i = x_conn_i.shape[1]
            
            for j in range(n_conn_i):
                x_extended_conn = x_reshaped.copy()
                x_extended_conn = x_extended_conn.at[:, i, :].set(x_conn_i[:, j, :])
                all_conn.append(x_extended_conn.reshape(batch_size, -1))
                all_mels.append(mels_i[:, j])
        
        x_conn = jnp.stack(all_conn, axis=1)
        mels = jnp.stack(all_mels, axis=1)
        
        if len(original_shape) == 1:
            x_conn = x_conn[0]
            mels = mels[0]
        
        return x_conn, mels


# 创建扩展哈密顿量算子
ha_extended = ExtendedHamiltonianOperator(
    hilbert=hi_extended,
    original_hamiltonian=ha_original,
    K=K,
    n_spin_orbitals=hi_original.size
)

print(f"\n✓ 创建扩展哈密顿量算子")

# =============================================================================
# 4. 测试扩展哈密顿量
# =============================================================================
print("\n测试扩展哈密顿量算子...")
test_state = hi_extended.all_states()[0]
conn_states, mels = ha_extended.get_conn_padded(test_state)

print(f"测试状态: {test_state}")
print(f"连接组态数量: {conn_states.shape[0]}")
print(f"前几个连接组态和矩阵元:")
for i in range(min(3, conn_states.shape[0])):
    print(f"  {i}: {conn_states[i]}, mel={mels[i]:.6f}")

# =============================================================================
# 5. 创建 MCState
# =============================================================================
model = nk.models.RBM(
    alpha=2,
    param_dtype=complex,
)

sampler = nk.sampler.MetropolisLocal(
    hi_extended,
    n_chains=16,
)

vstate = nk.vqs.MCState(
    sampler,
    model,
    n_samples=1008,
    n_discard_per_chain=100,
)

print(f"\n✓ 创建 MCState")
print(f"参数数量: {vstate.n_parameters}")

# =============================================================================
# 6. 使用 VMC 训练
# =============================================================================
optimizer = optax.adam(learning_rate=0.01)

vmc = nk.driver.VMC(
    ha_extended,
    optimizer,
    variational_state=vstate,
)

print("\n" + "=" * 60)
print("开始训练扩展系统...")
print("=" * 60)

logger = nk.logging.RuntimeLog()

vmc.run(
    n_iter=100,
    out=logger,
    show_progress=True,
)

print("\n✓ 训练完成")

# =============================================================================
# 7. 查看训练结果
# =============================================================================
energies = logger.data['Energy']['Mean']

print("\n训练结果:")
print(f"初始能量: {energies[0]:.6f} Ha")
print(f"最终能量: {energies[-1]:.6f} Ha")

# =============================================================================
# 8. 精确对角化对比
# =============================================================================
def build_extended_hamiltonian_matrix_direct(hi_original, hi_extended, original_hamiltonian, K):
    """直接构建扩展哈密顿量的矩阵表示"""
    states_original = hi_original.all_states()
    n_states_original = states_original.shape[0]
    
    H_original = jnp.zeros((n_states_original, n_states_original), dtype=complex)
    
    for i, state in enumerate(states_original):
        conn_states, mels = original_hamiltonian.get_conn(state)
        for conn_state, mel in zip(conn_states, mels):
            j = hi_original.states_to_numbers(conn_state)
            H_original = H_original.at[i, j].set(mel)
    
    I = jnp.eye(n_states_original, dtype=complex)
    H_extended = jnp.zeros((hi_extended.n_states, hi_extended.n_states), dtype=complex)
    
    for i in range(K):
        term = jnp.array([[1.0]], dtype=complex)
        for j in range(K):
            if j == i:
                term = jnp.kron(term, H_original)
            else:
                term = jnp.kron(term, I)
        H_extended = H_extended + term
    
    return H_extended

print("\n构建扩展哈密顿量矩阵（精确对角化）...")
H_extended_matrix = build_extended_hamiltonian_matrix_direct(
    hi_original, hi_extended, ha_original, K
)

eigenvalues_extended = jnp.linalg.eigvalsh(H_extended_matrix)
eigenvalues_extended = jnp.sort(eigenvalues_extended)

print("\n扩展哈密顿量的本征值（精确对角化）:")
for i, ev in enumerate(eigenvalues_extended[:4]):
    print(f"  λ_{i} = {ev:.8f} Ha")

# =============================================================================
# 9. 提取原系统的激发态能量
# =============================================================================
print("\n" + "=" * 60)
print("从扩展系统本征值提取原系统本征值")
print("=" * 60)

E0_extracted = eigenvalues_extended[0] / K
delta_E = eigenvalues_extended[1] - eigenvalues_extended[0]
E1_extracted = E0_extracted + delta_E

print(f"\n提取的激发态能量:")
print(f"E0 = {E0_extracted:.8f} Ha  |  误差: {abs(E0_extracted - E_fcis[0]):.6f} Ha")
print(f"E1 = {E1_extracted:.8f} Ha  |  误差: {abs(E1_extracted - E_fcis[1]):.6f} Ha")

print("\nFCI 基准:")
for i, e in enumerate(E_fcis[:K]):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
