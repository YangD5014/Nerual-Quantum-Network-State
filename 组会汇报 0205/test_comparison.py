"""
对比测试: NES-VMC vs 惩罚函数法
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
import expect_grad_ex

print("="*60)
print("对比测试: 惩罚函数法 vs NES-VMC")
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
print(f"FCI基态能量: {E_fcis[0]:.8f} Ha")
print(f"FCI第一激发态: {E_fcis[1]:.8f} Ha")

ha = nkx.operator.from_pyscf_molecule(mol)

# Hilbert空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1)
)

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
print("方法1: 惩罚函数法")
print("="*60)

# 基态计算
model_gs = FFN(N=N, alpha=2, rngs=nnx.Rngs(42))
vs_gs = nk.vqs.MCState(sampler, model_gs, n_discard_per_chain=10, n_samples=512)

opt_gs = nk.optimizer.Sgd(learning_rate=0.05)
sr_gs = nk.optimizer.SR(diag_shift=0.01)

gs = nk.driver.VMC(ha, opt_gs, variational_state=vs_gs, preconditioner=sr_gs)
print("计算基态...")
gs.run(n_iter=200, out='test_gs')

E_gs = float(vs_gs.expect(ha).mean.real)
print(f"基态能量: {E_gs:.8f} Ha")
print(f"FCI基态: {E_fcis[0]:.8f} Ha")
print(f"误差: {abs(E_gs - E_fcis[0]):.8f} Ha")

# 第一激发态计算（惩罚函数法）
model_ex1 = FFN(N=N, alpha=2, rngs=nnx.Rngs(142))
vs_ex1 = nk.vqs.MCState(sampler, model_ex1, n_discard_per_chain=10, n_samples=512)

opt_ex1 = nk.optimizer.Sgd(learning_rate=0.03)
sr_ex1 = nk.optimizer.SR(diag_shift=0.01)

# 使用惩罚函数法
gs_ex1 = vmc_ex.VMC_ex(
    hamiltonian=ha,
    optimizer=opt_ex1,
    variational_state=vs_ex1,
    preconditioner=sr_ex1,
    state_list=[vs_gs],
    shift_list=[0.3]  # 惩罚参数
)

print("\n计算第一激发态...")
gs_ex1.run(n_iter=200, out='test_ex1')

E_ex1 = float(vs_ex1.expect(ha).mean.real)
print(f"第一激发态能量: {E_ex1:.8f} Ha")
print(f"FCI第一激发态: {E_fcis[1]:.8f} Ha")
print(f"误差: {abs(E_ex1 - E_fcis[1]):.8f} Ha")

print("\n" + "="*60)
print("结果总结")
print("="*60)
print(f"\n{'态':<15} {'惩罚函数法':<15} {'FCI':<15} {'误差':<15}")
print("-"*60)
print(f"{'基态':<15} {E_gs:<15.8f} {E_fcis[0]:<15.8f} {abs(E_gs-E_fcis[0]):<15.8f}")
print(f"{'第一激发态':<15} {E_ex1:<15.8f} {E_fcis[1]:<15.8f} {abs(E_ex1-E_fcis[1]):<15.8f}")
