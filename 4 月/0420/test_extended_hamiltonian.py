"""
测试扩展系统哈密顿量的构建
"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci

# 设置 JAX 浮点精度
jax.config.update("jax_enable_x64", True)

print("=" * 60)
print("测试扩展系统哈密顿量构建")
print("=" * 60)

# H₂ 分子定义
bond_length = 1.4  # Bohr
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

# 将 PySCF 分子转为 NetKet 离散算符
ha_original = nkx.operator.from_pyscf_molecule(mol)

# 原始 Hilbert 空间
hi_original = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)

print(f"\n原始 Hilbert 空间: {hi_original}")
print(f"原始 Hilbert 空间大小: {hi_original.size}")
print(f"原始 Hilbert 空间状态数: {hi_original.n_states}")

# NES-VMC 参数
K = 2  # 需要计算的态数（基态 + 1 个激发态）

# 扩展 Hilbert 空间：K 个副本的直积
hi_extended = hi_original ** K

print(f"\n扩展 Hilbert 空间: {hi_extended}")
print(f"扩展 Hilbert 空间大小: {hi_extended.size}")
print(f"扩展 Hilbert 空间状态数: {hi_extended.n_states}")

# 构建扩展哈密顿量的矩阵表示
def build_extended_hamiltonian_matrix(hi_original, hi_extended, original_hamiltonian, K):
    """使用矩阵表示构建扩展系统哈密顿量"""
    # 获取原始 Hilbert 空间的所有状态
    states_original = hi_original.all_states()
    n_states_original = states_original.shape[0]
    
    print(f"\n构建扩展哈密顿量矩阵...")
    print(f"原始系统状态数: {n_states_original}")
    print(f"扩展系统状态数: {hi_extended.n_states}")
    
    # 构建原始哈密顿量的矩阵表示
    H_original = jnp.zeros((n_states_original, n_states_original), dtype=complex)
    
    for i, state in enumerate(states_original):
        # 获取连接组态和矩阵元
        conn_states, mels = original_hamiltonian.get_conn(state)
        
        # 填充矩阵
        for conn_state, mel in zip(conn_states, mels):
            # 找到连接组态的索引
            j = hi_original.states_to_numbers(conn_state)
            H_original = H_original.at[i, j].set(mel)
    
    print(f"原始哈密顿量矩阵形状: {H_original.shape}")
    
    # 构建扩展系统的哈密顿量
    # H_extended = H ⊗ I ⊗ ... ⊗ I + I ⊗ H ⊗ ... ⊗ I + ... 
    
    # 单位矩阵
    I = jnp.eye(n_states_original, dtype=complex)
    
    # 初始化扩展哈密顿量
    H_extended = jnp.zeros((hi_extended.n_states, hi_extended.n_states), dtype=complex)
    
    # 对每个子系统构建项
    for i in range(K):
        # 构建第 i 项：I ⊗ I ⊗ ... ⊗ H ⊗ ... ⊗ I
        # 其中 H 在第 i 个位置
        
        term = jnp.array([[1.0]], dtype=complex)
        
        for j in range(K):
            if j == i:
                term = jnp.kron(term, H_original)
            else:
                term = jnp.kron(term, I)
        
        H_extended = H_extended + term
    
    print(f"扩展哈密顿量矩阵形状: {H_extended.shape}")
    
    return H_extended

# 构建扩展哈密顿量
H_extended_matrix = build_extended_hamiltonian_matrix(
    hi_original, hi_extended, ha_original, K
)

# 验证扩展哈密顿量的性质
print("\n" + "=" * 60)
print("扩展哈密顿量验证")
print("=" * 60)

# 检查厄米性
is_hermitian = jnp.allclose(H_extended_matrix, H_extended_matrix.conj().T)
print(f"是否厄米: {is_hermitian}")

# 计算本征值
eigenvalues = jnp.linalg.eigvalsh(H_extended_matrix)
print(f"\n扩展哈密顿量的本征值:")
for i, ev in enumerate(eigenvalues[:8]):
    print(f"  λ_{i} = {ev:.8f}")

# 测试扩展系统的局部能量计算
def compute_extended_local_energy(original_hamiltonian, K, n_spin_orbitals, x_extended):
    """计算扩展系统的局部能量"""
    # 重塑为 K 个子系统
    x_reshaped = x_extended.reshape(K, n_spin_orbitals)
    
    # 存储所有连接组态和矩阵元
    all_conn = []
    all_mels = []
    
    # 对每个子系统应用原哈密顿量
    for i in range(K):
        x_i = x_reshaped[i]
        x_conn_i, mels_i = original_hamiltonian.get_conn_padded(x_i)
        
        for x_c, mel in zip(x_conn_i, mels_i):
            x_extended_conn = x_reshaped.at[i].set(x_c)
            all_conn.append(x_extended_conn.flatten())
            all_mels.append(mel)
    
    return jnp.array(all_conn), jnp.array(all_mels)

# 测试扩展哈密顿量
print("\n" + "=" * 60)
print("测试扩展哈密顿量构建")
print("=" * 60)

# 选择一个测试状态
test_state = hi_extended.all_states()[0]
print(f"测试状态: {test_state}")
print(f"测试状态形状: {test_state.shape}")

# 计算连接组态
conn_states, mels = compute_extended_local_energy(
    ha_original, K, hi_original.size, test_state
)

print(f"\n连接组态数量: {conn_states.shape[0]}")
print(f"连接组态形状: {conn_states.shape}")
print(f"矩阵元形状: {mels.shape}")

print(f"\n前几个连接组态和矩阵元:")
for i in range(min(5, conn_states.shape[0])):
    print(f"  {i}: {conn_states[i]}, mel={mels[i]}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
