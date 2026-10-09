from pyscf import gto, scf, fci, ao2mo
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np

# ============================================================
# LiH 分子 / STO-3G / 4 轨道活性空间 · 8 qubit · K=4 基准
# ============================================================
# 与 H2_6_31G.py 同构的数据模块，两处关键修正（相对旧 LiH.py）：
#   1) Hatree_Fock 用真实 HF 行列式 [1,1,0,0,1,1,0,0]（对角元 == E_RHF）。
#      旧 LiH.py 用 hi.all_states()[0]（枚举从最高轨道开始），并非 HF。
#   2) E_fcis 用 4 轨道 active-space FCI，与 ha（mo_coeff=前4轨道投影）
#      的谱逐位一致；旧 LiH.py 的 6 轨道全空间 FCI 与 ha 差 ~16 mHa，
#      会导致训练目标与参考能量对不上。

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
# 活性空间：前 4 个 RHF canonical 轨道（CAS(4e,4o) 全空间）
# ============================================================
N_ORBITALS = 4
n_alpha_electrons = 2
n_beta_electrons = 2

active_idx = [0, 1, 2, 3]
mo_active = mf.mo_coeff[:, active_idx]

# active-space FCI 参考（与 ha 投影空间严格一致）
E_nuc = mol.energy_nuc()
h1 = mo_active.T @ mf.get_hcore() @ mo_active
g2 = ao2mo.incore.full(mf._eri, mo_active)
cisolver = fci.direct_spin1.FCI()
E_active, _ = cisolver.kernel(h1, g2, N_ORBITALS,
                              (n_alpha_electrons, n_beta_electrons), nroots=10)
E_fcis = np.asarray(E_active) + E_nuc

print("\n" + "=" * 60)
print("LiH / STO-3G · 4 轨道活性空间 FCI 基准")
print("=" * 60)
for i, e in enumerate(E_fcis[:6]):
    exc = (e - E_fcis[0]) * 27.2114 if i > 0 else 0.0
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# ============================================================
# Hilbert 空间与哈密顿量
# ============================================================
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=N_ORBITALS,
    s=1/2,
    n_fermions_per_spin=(n_alpha_electrons, n_beta_electrons),
)

K = 4  # NES 扩展副本数
hi_ext = hi ** K

ha_raw = nkx.operator.from_pyscf_molecule(
    mol,
    mo_coeff=mo_active,
    implementation=nk.operator.FermionOperator2ndJax,
)
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
    ha_raw
)

# 真实 HF 行列式：alpha 占据轨道 0,1；beta 占据轨道 0,1
Hatree_Fock = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
_conn, _mels = ha.get_conn(jnp.asarray(Hatree_Fock))
_conn = np.asarray(_conn); _mels = np.asarray(_mels)
_same = np.where((_conn == Hatree_Fock).all(axis=-1))[0]
_diag = float(np.real(_mels[_same].sum()))
assert abs(_diag - hf_ground_energy) < 1e-6, \
    f"HF 参考态对角元 {_diag} != E_RHF {hf_ground_energy}"

print(f"\nHF reference state: {Hatree_Fock}")
print(f"HF 对角元 = {_diag:.8f} == E_RHF ✓")

# ============================================================
# 采样图边（单粒子 hop：同自旋轨道两两组合）
# ============================================================
alpha_orbs = list(range(N_ORBITALS))
beta_orbs = list(range(N_ORBITALS, 2 * N_ORBITALS))
single_edges = (
    list(itertools.combinations(alpha_orbs, 2))
    + list(itertools.combinations(beta_orbs, 2))
)

print(f'single_edges: {single_edges}')
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)

SINGLE_SIZE = hi.size
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for i, j in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.asarray(ext_edges)

print("\n" + "=" * 60)
print("Hilbert 信息")
print("=" * 60)
print(f"K = {K}")
print(f"hi.size = {hi.size}")
print(f"hi_ext.size = {hi_ext.size}")
print(f"SINGLE_SIZE = {SINGLE_SIZE}")
print(f"target_loss = sum(E_fcis[:K]) = {float(np.sum(E_fcis[:K])):.8f}")
