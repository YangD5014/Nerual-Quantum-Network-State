"""
NES-VMC 完整测试

关键：使用不同的初始条件确保不同态收敛到不同本征态
"""

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from pyscf import gto, scf, fci
import netket.experimental as nkx
import sys
sys.path.insert(0, '/Users/yangjianfei/mac_vscode/神经网络量子态/组会汇报 0205')

from nes_vmc_driver import NESVMC

print("="*60)
print("NES-VMC 完整测试")
print("="*60)

# 设置H2分子
bond_length = 1.4
geometry = [
    ('H', (0., 0., 0.)),
    ('H', (bond_length, 0., 0.)),
]

mol = gto.M(atom=geometry, basis='STO-3G')
mf = scf.RHF(mol).run(verbose=0)
E_hf = mf.e_tot

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print(f"Hartree-Fock能量: {E_hf:.8f} Ha")
print(f"FCI基态: {E_fcis[0]:.8f} Ha")
print(f"FCI第一激发态: {E_fcis[1]:.8f} Ha")
print(f"FCI第二激发态: {E_fcis[2]:.8f} Ha")

ha = nkx.operator.from_pyscf_molecule(mol)

# Hilbert空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1)
)

print(f"\nHilbert空间维度: {hi.n_states}")
print(f"所有组态:\n{hi.all_states()}")

# 采样器
g = nk.graph.Graph(edges=[(0,1),(2,3)])
sampler = nk.sampler.MetropolisFermionHop(
    hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
)

# 模型定义
class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int = 2, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(in_features=alpha * N, out_features=alpha * N, rngs=rngs, param_dtype=complex)
        self.linear_out = nnx.Linear(in_features=alpha * N, out_features=1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array):
        y = self.linear1(x)
        y = nnx.tanh(y)
        y = self.linear2(y)
        y = nnx.tanh(y)
        y = self.linear_out(y)
        return jnp.squeeze(y, axis=-1)

N = 4

print("\n" + "="*60)
print("方法: 独立优化各态 + 矩阵对角化")
print("="*60)

# 方法：先独立优化各态，然后计算矩阵对角化
n_states = 2

# 创建变分态
vstate_list = []
for i in range(n_states):
    # 使用完全不同的随机种子
    model_i = FFN(N=N, alpha=2, rngs=nnx.Rngs(42 + i * 10000))
    vs = nk.vqs.MCState(sampler, model_i, n_discard_per_chain=10, n_samples=512)
    vstate_list.append(vs)

# 先独立优化各态
print("\n独立优化各态...")

for i in range(n_states):
    print(f"\n优化态 {i}...")
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.01)
    
    gs = nk.driver.VMC(ha, opt, variational_state=vstate_list[i], preconditioner=sr)
    gs.run(n_iter=100, show_progress=False)
    
    E_i = float(vstate_list[i].expect(ha).mean.real)
    print(f"态 {i} 能量: {E_i:.8f} Ha")

# 计算矩阵
print("\n计算哈密顿量矩阵和重叠矩阵...")
import nes_vmc

H_matrix, S_matrix = nes_vmc.compute_matrices(vstate_list, ha)

print(f"\n哈密顿量矩阵 H:")
print(H_matrix)
print(f"\n重叠矩阵 S:")
print(S_matrix)

# 对角化
energies, coefficients = nes_vmc.diagonalize_generalized_eigenvalue_problem(H_matrix, S_matrix)

print(f"\n对角化后的能量:")
for i, e in enumerate(energies):
    print(f"  E{i} = {e.real:.8f} Ha")

print("\n" + "="*60)
print("结果比较")
print("="*60)

print(f"\n{'态':<15} {'NES-VMC':<15} {'FCI':<15} {'误差':<15}")
print("-"*60)
for i in range(n_states):
    e_nes = energies[i].real
    e_fci = E_fcis[i]
    error = abs(e_nes - e_fci)
    print(f"{'E'+str(i):<15} {e_nes:<15.8f} {e_fci:<15.8f} {error:<15.8f}")

print("\n" + "="*60)
print("分析")
print("="*60)

# 检查重叠
overlap = abs(S_matrix[0, 1])
print(f"\n态间重叠 |⟨ψ_0|ψ_1⟩| = {overlap:.6f}")

if overlap > 0.5:
    print("警告: 两个态重叠较大，可能收敛到相同态")
else:
    print("两个态重叠较小，成功区分不同态")
