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

# ============================================================
# Active Space 设置
# ============================================================
# LiH 有 4 个电子，通常冻结 Li 的 1s 轨道
# 使用 CAS(2,2): 2 个电子，2 个轨道 (Li 2s, 2p_z)
# 或者 CAS(4,4): 4 个电子，4 个轨道
# 这里使用 CAS(2,2) 活性空间

N_ACTIVE_ELEC = 4  # 活性电子数
N_ACTIVE_ORBS = 4  # 活性轨道数

# 使用 RAS (Restricted Active Space) 策略
# 冻结内层电子，只优化价层轨道

# 获取所有占据轨道信息
n_orbitals_total = mol.nao_nr()
print(f"\nTotal orbitals in STO-3G: {n_orbitals_total}")

# 对于 LiH，使用自然轨道基组来选择活性空间
# 计算 natural orbitals from MP2 or CISD


# 先做 MP2 得到 natural orbitals
my_mp = mp.MP2(mf)
my_mp.kernel()

# 获取 MP2 density matrix 的 natural orbital occupations
nat_orbs = mf.mo_coeff

# 简化方案：使用 HOMO-1, HOMO, LUMO, LUMO+1 作为活性空间
# (对应于 Li 的 2s, 2p 和 H 的 1s)
# 对于闭壳层 RHF，HOMO 和 HOMO-1 是成对占据的

n_electrons = mol.nelec  # total electrons
n_frozen = 0  # 冻结电子数 (这里不冻结，使用 full space)

# 选择活性空间轨道
# 使用 highest occupied and lowest unoccupied orbitals
n_alpha = 2
n_beta  = 2

# FCI 计算获取激发态参考能量
cisolver = fci.FCI(mf)
cisolver.nroots = 6
E_fcis, fcivec = cisolver.kernel()

print("\n" + "=" * 60)
print("LiH / STO-3G 基准能量")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114 if i > 0 else 0.0
    print(f"E{i} = {e:.8f} Ha | excitation = {exc:.4f} eV")

# ============================================================
# 构造 Active Space Hilbert 空间
# ============================================================
# 使用 CAS(2,2) - 2 electrons in 2 orbitals
# 或者根据需要调整为 CAS(4,4)

# 这里使用完整的 FCI space 但设置合理的 n_fermions_per_spin
# 对于闭壳层 LiH, n_alpha = n_beta = 2

n_orbitals = 4
n_alpha_electrons = 2
n_beta_electrons  = 2

# 如果使用 active space approach，可以减小 n_orbitals
# 这里先用完整空间，然后根据需要在后续选择 active orbitals

print(f"\nActive Space Configuration:")
print(f"  n_orbitals = {n_orbitals}")
print(f"  n_alpha = {n_alpha_electrons}, n_beta = {n_beta_electrons}")
print(f"  Total electrons = {n_electrons}")

# Hilbert 空间设置
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,
    s=1/2,
    n_fermions_per_spin=(n_alpha_electrons, n_beta_electrons),
)

K = 4  # 4 个态 (基态 + 3 个激发态)
hi_ext = hi ** K

# 选择 active MO
# 这里先用前 4 个 RHF canonical orbitals
active_idx = [0, 1, 2, 3]   # Python 里是 0-based

mo_active = mf.mo_coeff[:, active_idx]

ha_raw = nkx.operator.from_pyscf_molecule(
    mol,
    mo_coeff=mo_active,
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

# nes_rule = NESFermionHopRule(
#     edges=ext_edges,
#     K=K,
#     single_size=SINGLE_SIZE,
# )

# print(f"Total sampler edges: {len(ext_edges)}")