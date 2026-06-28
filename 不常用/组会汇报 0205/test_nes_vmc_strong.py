"""
NES-VMC 完整测试 - 更强的惩罚
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
print("NES-VMC 完整测试 - 更强惩罚")
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
    print(f"态 {i} 初始能量: {float(vs.expect(ha).mean.real):.6f} Ha")

# 使用更强的惩罚强度
penalty_strength = 1.0  # 增加惩罚强度

# 创建NES-VMC驱动器
opt = nk.optimizer.Sgd(learning_rate=0.03)  # 降低学习率
sr = nk.optimizer.SR(diag_shift=0.01)

nes_driver = NESVMC(
    hamiltonian=ha,
    optimizer=opt,
    variational_states=vstate_list,
    preconditioner=sr,
    n_states=n_states,
    penalty_strength=penalty_strength
)

# 运行优化
print(f"\n运行NES-VMC优化 (惩罚强度={penalty_strength})...")
log_data = nes_driver.run(n_iter=300, out='nes_vmc_test', show_progress=True)

# 最终结果
print("\n" + "="*60)
print("最终结果")
print("="*60)

final_energies = nes_driver.get_state_energies()
print(f"\n对角化后的能量:")
for i, e in enumerate(final_energies):
    print(f"  E{i} = {e:.8f} Ha (FCI: {E_fcis[i]:.8f} Ha, 误差: {abs(e - E_fcis[i]):.8f} Ha)")

# 检查重叠
H_matrix, S_matrix = nes_vmc.compute_matrices(vstate_list, ha)
overlap = abs(S_matrix[0, 1])
print(f"\n态间重叠 |⟨ψ_0|ψ_1⟩| = {overlap:.6f}")

# 打印哈密顿量矩阵
print(f"\n哈密顿量矩阵对角元素:")
print(f"  H_00 = {H_matrix[0,0]:.6f}")
print(f"  H_11 = {H_matrix[1,1]:.6f}")
