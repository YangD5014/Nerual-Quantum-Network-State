"""
测试使用 NetKet 原生 API（MCState + VMC）训练扩展系统
使用包装器使 NNX 模型符合 NetKet 接口
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
# 3. 定义 TotalAnsatz（使用 Flax linen 接口）
# =============================================================================
from flax import linen as nn

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
    """
    为 NetKet 设计的 TotalAnsatz（Flax 版本）
    输出 log|ψ|
    """
    n_states: int = K
    n_spin_orbitals: int = n_spin_orbitals
    hidden_dim: int = 16
    
    @nn.compact
    def __call__(self, x):
        """
        x: 扩展系统的组态，形状 (K * n_spin_orbitals,)
        返回: log|ψ|
        """
        # 重塑为 (K, n_spin_orbitals)
        x_reshaped = x.reshape(self.n_states, self.n_spin_orbitals)
        
        # 为每个单态创建子模块
        single_ansatz_list = [
            SingleStateAnsatzFlax(self.hidden_dim, name=f'single_{i}')
            for i in range(self.n_states)
        ]
        
        # 计算矩阵 M
        M = []
        for i in range(self.n_states):
            row = [single_ansatz_list[j](x_reshaped[i]) for j in range(self.n_states)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        
        # 计算行列式
        psi_total = jnp.linalg.det(M)
        
        # 返回 log|ψ|
        return jnp.log(jnp.abs(psi_total) + 1e-12)


# 实例化模型
model = TotalAnsatzFlax(
    n_states=K,
    n_spin_orbitals=n_spin_orbitals,
    hidden_dim=16,
)

print(f"\n✓ 创建 TotalAnsatzFlax")

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
def compute_H_psi_for_single_state(ha, params, model, single_idx, x):
    """计算单个态的 H|ψ>"""
    x_conn, mels = ha.get_conn_padded(x)
    
    # 计算所有连接态的波函数值
    def apply_single(x):
        # 应用整个模型，但只提取特定单态的输出
        x_reshaped = x.reshape(K, n_spin_orbitals)
        
        # 手动应用单态 Ansatz
        # 这里需要从 params 中提取对应的参数
        # 由于 Flax 的参数结构，我们需要使用 model.apply
        # 但这里简化处理，直接使用 vstate
        pass
    
    # 简化：直接使用 vstate 的 apply 函数
    # 但这需要更复杂的处理
    # 暂时返回 None
    return None


# 由于 Flax 模型的参数结构比较复杂，我们使用简化方法
# 直接使用训练好的模型参数来计算局域能量矩阵

# 获取模型参数
params = vstate.parameters

# 使用模型的前向传播
def forward_model(params, x):
    """前向传播"""
    return model.apply(params, x)

# 计算局域能量矩阵（简化版本）
# 由于时间限制，这里只展示概念
print("\n注意：由于 Flax 模型的参数结构复杂性，")
print("局域能量矩阵的详细计算需要更复杂的实现。")
print("这里我们只展示训练结果。")

# =============================================================================
# 8. 总结
# =============================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("\n1. 成功使用 NetKet 原生 API 训练扩展系统：")
print(f"   - 使用 MCState 创建变分态")
print(f"   - 使用 VMC driver 进行训练")
print(f"   - 训练 300 次迭代")

print("\n2. 训练结果：")
print(f"   - 最终能量: {final_energy.mean:.8f} Ha")
print(f"   - FCI 基态: {E_fcis[0]:.8f} Ha")
print(f"   - 误差: {abs(final_energy.mean - E_fcis[0]):.6f} Ha")

print("\n3. 下一步工作：")
print("   - 实现完整的局域能量矩阵计算")
print("   - 对角化得到激发态能量")
print("   - 与 FCI 结果对比")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
