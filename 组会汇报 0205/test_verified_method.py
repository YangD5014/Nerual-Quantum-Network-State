"""
NES-VMC 正确实现 - 基于验证过的惩罚函数方法

直接使用已有的VMC_ex实现，然后计算矩阵对角化
"""

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from pyscf import gto, scf, fci
import netket.experimental as nkx
import sys
sys.path.insert(0, '/Users/yangjianfei/mac_vscode/神经网络量子态/激发态能量/Netket_excited_state')
sys.path.insert(0, '/Users/yangjianfei/mac_vscode/神经网络量子态/组会汇报 0205')

import vmc_ex
import nes_vmc

print("="*60)
print("NES-VMC: 使用验证过的惩罚函数方法")
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
print("步骤1: 优化基态")
print("="*60)

# 基态优化
model_gs = FFN(N=N, alpha=2, rngs=nnx.Rngs(42))
vs_gs = nk.vqs.MCState(sampler, model_gs, n_discard_per_chain=10, n_samples=512)

opt_gs = nk.optimizer.Sgd(learning_rate=0.05)
sr_gs = nk.optimizer.SR(diag_shift=0.01)

gs = nk.driver.VMC(ha, opt_gs, variational_state=vs_gs, preconditioner=sr_gs)
print("优化基态...")
gs.run(n_iter=200, show_progress=False)

E_gs = float(vs_gs.expect(ha).mean.real)
print(f"基态能量: {E_gs:.8f} Ha (FCI: {E_fcis[0]:.8f} Ha, 误差: {abs(E_gs - E_fcis[0]):.6f} Ha)")

print("\n" + "="*60)
print("步骤2: 优化第一激发态 (与基态正交)")
print("="*60)

# 第一激发态优化（使用惩罚函数法）
model_ex1 = FFN(N=N, alpha=2, rngs=nnx.Rngs(142))
vs_ex1 = nk.vqs.MCState(sampler, model_ex1, n_discard_per_chain=10, n_samples=512)

opt_ex1 = nk.optimizer.Sgd(learning_rate=0.03)
sr_ex1 = nk.optimizer.SR(diag_shift=0.01)

# 使用验证过的VMC_ex
gs_ex1 = vmc_ex.VMC_ex(
    hamiltonian=ha,
    optimizer=opt_ex1,
    variational_state=vs_ex1,
    preconditioner=sr_ex1,
    state_list=[vs_gs],  # 与基态正交
    shift_list=[0.3]     # 惩罚参数
)

print("优化第一激发态...")
gs_ex1.run(n_iter=200, show_progress=False)

E_ex1 = float(vs_ex1.expect(ha).mean.real)
print(f"第一激发态能量: {E_ex1:.8f} Ha (FCI: {E_fcis[1]:.8f} Ha, 误差: {abs(E_ex1 - E_fcis[1]):.6f} Ha)")

print("\n" + "="*60)
print("步骤3: 计算哈密顿量矩阵并对角化")
print("="*60)

# 计算矩阵
vstate_list = [vs_gs, vs_ex1]
H_matrix, S_matrix = nes_vmc.compute_matrices(vstate_list, ha)

print(f"\n哈密顿量矩阵 H:")
print(H_matrix)
print(f"\n重叠矩阵 S:")
print(S_matrix)

# 对角化
energies, coefficients = nes_vmc.diagonalize_generalized_eigenvalue_problem(H_matrix, S_matrix)

print(f"\n对角化后的能量:")
for i, e in enumerate(energies):
    print(f"  E{i} = {e.real:.8f} Ha (FCI: {E_fcis[i]:.8f} Ha)")

# 检查重叠
overlap = abs(S_matrix[0, 1])
print(f"\n态间重叠 |⟨ψ_0|ψ_1⟩| = {overlap:.6f}")

print("\n" + "="*60)
print("最终结果总结")
print("="*60)

print(f"\n{'态':<15} {'VMC (Ha)':<18} {'FCI (Ha)':<18} {'误差 (Ha)':<15} {'激发能 (eV)'}")
print("-"*85)

for i in range(2):
    if i == 0:
        e_vmc = E_gs
        state_name = "基态"
        exc_eV = 0.0
    else:
        e_vmc = E_ex1
        state_name = "第1激发态"
        exc_eV = (E_ex1 - E_gs) * 27.2114
    
    e_fci = E_fcis[i]
    error = abs(e_vmc - e_fci)
    print(f"{state_name:<15} {e_vmc:<18.8f} {e_fci:<18.8f} {error:<15.8f} {exc_eV:.4f}")

print(f"\n态间重叠: {overlap:.6f}")
if overlap < 0.3:
    print("态区分良好！")
else:
    print("态重叠较大")
