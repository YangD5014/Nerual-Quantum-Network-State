"""
测试使用 NetKet 原生 API（MCState + VMC）训练扩展系统
"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from functools import partial

# 设置 JAX 浮点精度
jax.config.update("jax_enable_x64", True)

print("=" * 60)
print("测试 NetKet 原生 API（MCState + VMC）")
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
# 3. 定义 TotalAnsatz（输出 log|ψ|）
# =============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


class TotalAnsatzForNetKet(nnx.Module):
    """
    为 NetKet 设计的 TotalAnsatz
    输出 log|ψ| 而不是 ψ
    """
    def __init__(self, n_spin_orbitals: int, n_states: int = K, hidden_dim: int = 16):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals

        key = jax.random.key(42)
        keys = jax.random.split(key, n_states)

        self.single_ansatz_list = [
            SingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim,
                rngs=nnx.Rngs(keys[i])
            )
            for i in range(n_states)
        ]

    def __call__(self, x):
        """
        x: 扩展系统的组态，形状 (K * n_spin_orbitals,)
        返回: log|ψ|
        """
        # 重塑为 (K, n_spin_orbitals)
        x_reshaped = x.reshape(self.n_states, self.n_spin_orbitals)
        
        # 计算矩阵 M
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_reshaped[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        
        # 计算行列式
        psi_total = jnp.linalg.det(M)
        
        # 返回 log|ψ|
        return jnp.log(jnp.abs(psi_total) + 1e-12)


# 实例化模型
model = TotalAnsatzForNetKet(
    n_spin_orbitals=n_spin_orbitals,
    n_states=K,
    hidden_dim=16,
)

print(f"\n✓ 创建 TotalAnsatzForNetKet")

# =============================================================================
# 4. 创建采样器和 MCState
# =============================================================================
# 创建采样器
sampler = nk.sampler.MetropolisLocal(
    hi_extended,
    n_chains=16,
)

# 创建 MCState
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
# 5. 使用 VMC Driver 训练
# =============================================================================
# 转换为 JAX 兼容的算子
try:
    ha_original_jax = ha_original.to_jax_operator()
    print("\n✓ 成功转换为 JAX 兼容算子")
except Exception as e:
    print(f"\n✗ 无法转换为 JAX 算子: {e}")
    ha_original_jax = ha_original

# 创建优化器
optimizer = nk.optimizer.Sgd(learning_rate=0.1)

# 创建 VMC driver
gs = nk.driver.VMC(
    ha_original_jax,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1, holomorphic=True),
)

print("\n" + "=" * 60)
print("开始训练（使用 NetKet 原生 API）...")
print("=" * 60)

# 运行训练
gs.run(n_iter=300)

print("\n✓ 训练完成")

# =============================================================================
# 6. 查看训练结果
# =============================================================================
final_energy = vstate.expect(ha_original_jax)
print(f"\n最终能量: {final_energy}")
print(f"FCI 基态能量: {E_fcis[0]:.8f} Ha")
print(f"误差: {abs(final_energy.mean - E_fcis[0]):.6f} Ha")

# =============================================================================
# 7. 计算局域能量矩阵
# =============================================================================
print("\n" + "=" * 60)
print("计算局域能量矩阵...")
print("=" * 60)

# 获取样本
samples = vstate.samples
samples_flat = samples.reshape(-1, K, n_spin_orbitals)

print(f"样本数: {samples_flat.shape[0]}")

# 定义局部能量计算函数
def compute_H_psi_for_single_state(ha, single_ansatz, x):
    """计算单个态的 H|ψ>"""
    x_conn, mels = ha.get_conn_padded(x)
    psi_conn = jax.vmap(single_ansatz)(x_conn)
    return jnp.sum(mels * psi_conn)


def compute_H_Psi_matrix(ha, total_ansatz, x_batch):
    """计算 HΨ 矩阵"""
    K = total_ansatz.n_states
    H_mat = []
    
    for i in range(K):
        row = []
        for j in range(K):
            v = compute_H_psi_for_single_state(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(v)
        H_mat.append(row)
    
    return jnp.array(H_mat, dtype=complex)


def compute_local_energy_matrix(ha, total_ansatz, x_batch, eps=1e-6):
    """计算局部能量矩阵"""
    # 重塑 x_batch
    x_reshaped = x_batch.reshape(total_ansatz.n_states, total_ansatz.n_spin_orbitals)
    
    # 计算矩阵 M
    K = total_ansatz.n_states
    M = []
    for i in range(K):
        row = [total_ansatz.single_ansatz_list[j](x_reshaped[i]) for j in range(K)]
        M.append(jnp.array(row))
    M = jnp.stack(M)
    
    M_reg = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    H_Psi = compute_H_Psi_matrix(ha, total_ansatz, x_reshaped)
    E_matrix = jnp.linalg.solve(M_reg, H_Psi)
    E_loc = jnp.trace(E_matrix)
    return E_loc, E_matrix


# 批量计算局域能量矩阵
def compute_average_energy_matrix(total_ansatz, samples, ha):
    def single_E_matrix(x):
        _, E_matrix = compute_local_energy_matrix(ha, total_ansatz, x)
        return E_matrix
    
    E_matrices = jax.vmap(single_E_matrix)(samples)
    E_matrix_avg = E_matrices.mean(axis=0)
    E_avg = jnp.trace(E_matrix_avg)
    
    return E_avg, E_matrix_avg


E_avg, E_matrix_avg = compute_average_energy_matrix(
    model, samples_flat, ha_original_jax
)

print(f"\n平均局部能量: {E_avg:.6f} Ha")
print(f"\n平均局域能量矩阵:\n{E_matrix_avg}")

# =============================================================================
# 8. 对角化得到激发态能量
# =============================================================================
E_matrix_sym = (E_matrix_avg + E_matrix_avg.conj().T) / 2
eigenvalues_vmc = jnp.linalg.eigvalsh(E_matrix_sym)
eigenvalues_vmc = jnp.sort(eigenvalues_vmc)

print("\n" + "=" * 60)
print("NES-VMC 计算得到的激发态能量")
print("=" * 60)
for i, e in enumerate(eigenvalues_vmc):
    exc = (e - eigenvalues_vmc[0]) * 27.2114
    error = abs(e - E_fcis[i])
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV  |  误差: {error:.6f} Ha")

print("\nFCI 基准:")
for i, e in enumerate(E_fcis[:K]):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
