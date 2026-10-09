from pyscf import gto, scf, fci,mp, ci
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np
# ============================================================
# LiH 分子 / cc-pVDZ / Active Space 基准
# ============================================================
# LiH 几何结构 (平衡键长约 1.58 Angstrom)
bond_length = 1.58
geometry = [
    ("Li", (0.0, 0.0, 0.0)),
    ("H",  (0.0, 0.0, bond_length)),
]

mol = gto.M(atom=geometry, basis="STO-3G", verbose=0, spin=0)
mf = scf.RHF(mol).run(verbose=0)
hf_ground_energy = mf.e_tot

print("=" * 60)
print("LiH 分子基本信息")
print("=" * 60)
print(f"HF energy = {hf_ground_energy:.8f} Ha")
print(f"Total electrons = {mol.nelec}")
print(f"Total basis functions = {mol.nao_nr()}")

# FCI 计算获取激发态参考能量
cisolver = fci.FCI(mf)
cisolver.nroots = 10
E_fcis, fcivec = cisolver.kernel()

print("\n" + "=" * 60)
print("LiH / STO-3G 基准能量")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114 if i > 0 else 0.0
    print(f"E{i} = {e:.8f} Ha | excitation = {exc:.4f} eV")

# 空间轨道数目（分子轨道 MO）
n_space_orbitals = mf.mo_coeff.shape[1]
print(f"Number of spatial molecular orbitals(空间轨道数) = {n_space_orbitals}")

# 顺便看占据情况
print(f"MO occupation numbers: {mf.mo_occ}")
n_orbitals = 6
n_alpha_electrons = 2
n_beta_electrons  = 2

# Hilbert 空间设置
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,
    s=1/2,
    n_fermions_per_spin=(n_alpha_electrons, n_beta_electrons),
)

K = 4  # 4 个态 (基态 + 3 个激发态)
hi_ext = hi ** K

ha_raw = nkx.operator.from_pyscf_molecule(
    mol,
    mo_coeff=mf.mo_coeff,
    implementation=nk.operator.FermionOperator2ndJax,
)
# 第二步：转换成 NetKet 推荐的高效 operator
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
    ha_raw
)

SINGLE_SIZE = hi.size

print("\n" + "=" * 60)
print("Hilbert 信息")
print("=" * 60)
print(f"K = {K}")
print(f"hi.size = {hi.size}")
print(f"hi_ext.size = {hi_ext.size}")
print(f"SINGLE_SIZE = {SINGLE_SIZE}")

# target_loss = sum of first K FCI energies
target_loss = float(np.sum(E_fcis[:K]))
print(f"target_loss = sum(E_fcis[:K]) = {target_loss:.8f}")
# HF reference state
Hatree_Fock = hi.all_states()[0]
print(f"\nHF reference state: {Hatree_Fock[:20]}")

# Alpha and beta orbital indices
alpha_orbs = list(range(n_orbitals))
beta_orbs  = list(range(n_orbitals, 2 * n_orbitals))

print(f"Alpha orbitals: {alpha_orbs[:5]}... ({len(alpha_orbs)} total)")
print(f"Beta orbitals: {beta_orbs[:5]}... ({len(beta_orbs)} total)")

# Full edges for single excitations
single_edges_full = (
    list(itertools.combinations(alpha_orbs, 2))
    + list(itertools.combinations(beta_orbs, 2))
)
print(f"\nSingle edges (total): {len(single_edges_full)}")

# 构造扩展 Hilbert 的边
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for i, j in single_edges_full:
        ext_edges.append((i + offset, j + offset))

ext_edges = jnp.asarray(ext_edges)