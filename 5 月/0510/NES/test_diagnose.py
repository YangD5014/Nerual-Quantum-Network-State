"""测试脚本：诊断NES-VMC代码问题"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
import flax.nnx as nnx
from flax import linen as nn

# H₂ 分子定义
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")

# NetKet 哈密顿量和采样器
ha = nkx.operator.from_pyscf_molecule(mol)

print(f"\nha (from_pyscf_molecule):")
print(f"  - ha.hilbert: {ha.hilbert}")
print(f"  - ha.hilbert.size: {ha.hilbert.size}")

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2
hi_ext = hi ** K

print(f"\nhi (单链):")
print(f"  - hi.size: {hi.size}")

print(f"\nhi_ext (K={K}链):")
print(f"  - hi_ext.size: {hi_ext.size}")
print(f"  - hi_ext: {hi_ext}")

# 创建哈密顿量在hi_ext上
ha_ext = nkx.operator.from_pyscf_molecule(mol, hilbert=hi_ext)
print(f"\nha_ext (在hi_ext上):")
print(f"  - ha_ext.hilbert: {ha_ext.hilbert}")
print(f"  - ha_ext.hilbert.size: {ha_ext.hilbert.size}")

# 测试Ham_psi
class TestAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(n_spin_orbitals, 8, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sum(self.linear(x))

rngs = nnx.Rngs(0)
ansatz = TestAnsatz(4, rngs=rngs)

# 测试单个样本
x_single = hi.random_state(jax.random.PRNGKey(0))
print(f"\nx_single (hi上的随机态):")
print(f"  - shape: {x_single.shape}")
print(f"  - value: {x_single}")

# 测试Ham_psi用ha
x_primes, mels = ha.get_conn_padded(x_single)
print(f"\nha.get_conn_padded(x_single):")
print(f"  - x_primes shape: {x_primes.shape}")
print(f"  - mels shape: {mels.shape}")

# 测试Ham_psi用ha_ext
x_primes_ext, mels_ext = ha_ext.get_conn_padded(x_single)
print(f"\nha_ext.get_conn_padded(x_single):")
print(f"  - x_primes_ext shape: {x_primes_ext.shape}")
print(f"  - mels_ext shape: {mels_ext.shape}")

# 测试扩展态
x_ext = hi_ext.random_state(jax.random.PRNGKey(0))
print(f"\nx_ext (hi_ext上的随机态):")
print(f"  - shape: {x_ext.shape}")
print(f"  - value: {x_ext}")

# 测试ha_ext对扩展态
x_primes_ext2, mels_ext2 = ha_ext.get_conn_padded(x_ext)
print(f"\nha_ext.get_conn_padded(x_ext):")
print(f"  - x_primes_ext2 shape: {x_primes_ext2.shape}")
print(f"  - mels_ext2 shape: {mels_ext2.shape}")

# 测试ha对扩展态 (应该会报错或返回错误结果)
try:
    x_primes_ha, mels_ha = ha.get_conn_padded(x_ext)
    print(f"\nha.get_conn_padded(x_ext) - 不应该工作:")
    print(f"  - shape: {x_primes_ha.shape}")
except Exception as e:
    print(f"\nha.get_conn_padded(x_ext) 出错: {e}")
