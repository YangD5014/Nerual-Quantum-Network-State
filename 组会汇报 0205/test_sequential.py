"""
NES-VMC 分步优化测试

策略：
1. 先优化基态到收敛
2. 固定基态，优化激发态（与基态正交）
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
import nes_vmc

print("="*60)
print("NES-VMC 分步优化测试")
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
n_states = 2

# 创建变分态
print("\n创建变分态...")
vstate_list = []
for i in range(n_states):
    model_i = FFN(N=N, alpha=2, rngs=nnx.Rngs(42 + i * 10000))
    vs = nk.vqs.MCState(sampler, model_i, n_discard_per_chain=10, n_samples=512)
    vstate_list.append(vs)

# 创建NES-VMC驱动器
opt = nk.optimizer.Sgd(learning_rate=0.05)
sr = nk.optimizer.SR(diag_shift=0.01)

nes_driver = NESVMC(
    hamiltonian=ha,
    optimizer=opt,
    variational_states=vstate_list,
    preconditioner=sr,
    n_states=n_states,
    penalty_strength=0.5  # 惩罚强度
)

# 分步优化
print("\n" + "="*60)
print("分步优化")
print("="*60)

log_data = nes_driver.run_sequential(
    n_iter_ground=200,    # 基态优化迭代
    n_iter_excited=200,   # 激发态优化迭代
    out='nes_vmc_sequential',
    show_progress=True
)

# 最终结果
print("\n" + "="*60)
print("最终结果")
print("="*60)

final_energies = nes_driver.get_state_energies()

# 计算重叠
H_matrix, S_matrix = nes_vmc.compute_matrices(vstate_list, ha)
overlap = abs(S_matrix[0, 1])

print(f"\n{'态':<15} {'NES-VMC (Ha)':<18} {'FCI (Ha)':<18} {'误差 (Ha)':<15}")
print("-"*66)
for i in range(n_states):
    e_nes = final_energies[i]
    e_fci = E_fcis[i]
    error = abs(e_nes - e_fci)
    state_name = "基态" if i == 0 else f"第{i}激发态"
    print(f"{state_name:<15} {e_nes:<18.8f} {e_fci:<18.8f} {error:<15.8f}")

print(f"\n态间重叠 |⟨ψ_0|ψ_1⟩| = {overlap:.6f}")

# 哈密顿量矩阵
print(f"\n哈密顿量矩阵对角元素:")
print(f"  H_00 = {H_matrix[0,0]:.6f}")
print(f"  H_11 = {H_matrix[1,1]:.6f}")

if overlap < 0.2:
    print("\n成功: 态区分良好！")
elif overlap < 0.5:
    print("\n部分成功: 态有一定区分")
else:
    print("\n需要改进: 态重叠较大")
