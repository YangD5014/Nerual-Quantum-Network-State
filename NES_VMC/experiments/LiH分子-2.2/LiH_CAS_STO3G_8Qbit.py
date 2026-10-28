from pyscf import gto, scf, fci, mcscf, ao2mo
import netket as nk
import netket.experimental as nkx
import itertools
import numpy as np
import jax.numpy as jnp

# ============================================================
# LiH 分子 / STO-3G / CAS(2e,4o) 活性空间 · 8 qubit 基准
# ============================================================
# 与 Li_ccpvD.py（Li 原子）同构的数据模块，差别只在活性空间的构造：
#   LiH / STO-3G 满空间为 6 个空间轨道 = 12 自旋轨道 = 12 比特。
#   这里用 CAS 把比特数压到 8：
#       - 冻结 Li 1s（2 个电子 + 1 个轨道），只保留核心能量常数 ecore；
#       - 活性空间 4 个空间轨道 = 8 自旋轨道 = 8 比特，活性电子 2 个（CAS(2e,4o)）；
#       - 活性轨道由 CASSCF(2e,4o) 优化后再做自然轨道旋转（natorb=True），
#         保证 8 比特空间对基态的描述尽量接近满空间 FCI（本体系误差 ~0.24 mHa）。
#
# 与旧 LiH.py / Li_STO3G_8Qbit_K4.py 的关键差别：
#   1) 哈密顿量用**冻结核心的有效积分** (h1eff, h2eff) 构造，而不是直接投影
#      前 4 个 canonical 轨道；否则丢掉的核心-活性相互作用会让谱与参考能量对不上。
#   2) E_fcis 由同一组 (h1eff, h2eff, ecore) 的活性空间 FCI 得到，与 ha 严格同谱。
#   3) Hatree_Fock 取 CASSCF 自然轨道下单占据的主参考行列式（活性首个轨道双占）。

bond_length = 2.2
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
print(f"Total basis functions = {mol.nao_nr()} (满空间 {2 * mol.nao_nr()} 比特)")

# ============================================================
# CAS(2e, 4o)：冻结 Li 1s，4 个活性空间轨道 -> 8 比特
# ============================================================
NFROZEN = 1          # 冻结轨道数（Li 1s）
N_ACTIVE_ORBS = 4    # 活性空间轨道数
N_ACTIVE_ELEC = 2    # 活性电子数

# CASSCF：轨道优化 + 自然轨道旋转（natorb=True 会把活性轨道转到 1-RDM 本征基，
# 主参考行列式因此变成"活性首个自然轨道双占"，后续 NES-VMC 的 gauge 参考态取它最稳）
mc = mcscf.CASSCF(mf, N_ACTIVE_ORBS, N_ACTIVE_ELEC)
mc.frozen = NFROZEN
mc.natorb = True
mc.verbose = 0
mc.max_cycle = 300
mc.kernel()
assert mc.converged, "CASSCF 未收敛，请增大 max_cycle 或检查初始轨道"

# 冻结核心的有效单体积分 + 核心能量常数（含核排斥），可直接喂给 netket
h1eff, ecore = mc.get_h1eff()
h2eff = ao2mo.restore(1, mc.get_h2eff(), N_ACTIVE_ORBS)   # (4,4,4,4) 化学家记号 (pq|rs)

dm1_active = mc.fcisolver.make_rdm1(mc.ci, N_ACTIVE_ORBS, mc.nelecas)
active_occ = np.diag(dm1_active)

print("\n" + "=" * 60)
print(f"CASSCF({N_ACTIVE_ELEC}e,{N_ACTIVE_ORBS}o) | 冻结核心 {NFROZEN} 个轨道")
print("=" * 60)
print(f"CASSCF energy = {mc.e_tot:.8f} Ha | core energy ecore = {ecore:.8f} Ha")
print(f"活性自然轨道占据数 = {np.round(active_occ, 6)}")
print(f"活性空间比特数 = {2 * N_ACTIVE_ORBS}")

# ============================================================
# Hilbert 空间与哈密顿量（活性空间 8 比特）
# ============================================================
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=N_ACTIVE_ORBS,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)
SINGLE_SIZE = hi.size   # 8

cutoff = 1e-10
# 自旋轨道索引约定（与 SpinOrbitalFermions 一致）：前 N_ACTIVE_ORBS 个为一个自旋分块，
# 后 N_ACTIVE_ORBS 个为另一个自旋分块，即 spin_orb(p, σ) = p + σ * N_ACTIVE_ORBS。
#
# H = ecore + Σ_{pqσ} h_pq c†_{pσ} c_{qσ}
#           + 1/2 Σ_{pqrs,στ} (pq|rs) c†_{pσ} c†_{rτ} c_{sτ} c_{qσ}
# 逐项显式写进 FermionOperator2ndJax，避免不同版本的内部积分转换约定差异。
_terms = []
_weights = []
for _sigma in range(2):
    _off = _sigma * N_ACTIVE_ORBS
    for _p in range(N_ACTIVE_ORBS):
        for _q in range(N_ACTIVE_ORBS):
            if abs(h1eff[_p, _q]) > cutoff:
                _terms.append([(_p + _off, 1), (_q + _off, 0)])
                _weights.append(h1eff[_p, _q])

for _tau in range(2):
    for _sigma in range(2):
        _o_s, _o_t = _sigma * N_ACTIVE_ORBS, _tau * N_ACTIVE_ORBS
        for _p in range(N_ACTIVE_ORBS):
            for _q in range(N_ACTIVE_ORBS):
                for _r in range(N_ACTIVE_ORBS):
                    for _s in range(N_ACTIVE_ORBS):
                        _v = 0.5 * h2eff[_p, _q, _r, _s]
                        if abs(_v) > cutoff:
                            _terms.append(
                                [(_p + _o_s, 1), (_r + _o_t, 1), (_s + _o_t, 0), (_q + _o_s, 0)]
                            )
                            _weights.append(_v)

ha_raw = nk.operator.FermionOperator2ndJax(hi, _terms, _weights, ecore)
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(ha_raw)

# 活性空间参考能量：与 ha 同一组积分，谱逐位一致
_cisolver = fci.direct_spin1.FCI()
E_active, _ci = _cisolver.kernel(
    h1eff, h2eff, N_ACTIVE_ORBS, (1, 1), nroots=10,
)
E_fcis = np.asarray(E_active) + ecore

print("\n" + "=" * 60)
print(f"LiH / STO-3G · CAS({N_ACTIVE_ELEC}e,{N_ACTIVE_ORBS}o) · 8 qubit FCI 基准")
print("=" * 60)
for i, e in enumerate(E_fcis[:6]):
    exc = (e - E_fcis[0]) * 27.2114 if i > 0 else 0.0
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# 自检：ha 的精确对角化谱必须与活性空间 FCI 谱逐位一致
_ed = np.asarray(nk.exact.lanczos_ed(ha, k=4))
assert np.allclose(_ed, E_fcis[:4], atol=1e-6), (
    f"ha 谱 {_ed} 与 CASCI 参考 {E_fcis[:4]} 不一致"
)
print(f"自检通过：ha 精确对角化谱与 CASCI 参考一致 (max diff = {np.abs(_ed - E_fcis[:4]).max():.2e})")

# ============================================================
# 参考态（gauge 用）：活性首个自然轨道双占的行列式
# 自旋轨道排布：前 4 个为一个自旋分块，后 4 个为另一个自旋分块
# ============================================================
Hatree_Fock = np.zeros(SINGLE_SIZE, dtype=np.int32)
Hatree_Fock[0] = 1
Hatree_Fock[N_ACTIVE_ORBS] = 1

_conn, _mels = ha.get_conn(jnp.asarray(Hatree_Fock))
_conn = np.asarray(_conn)
_mels = np.asarray(_mels)
_same = np.where((_conn == Hatree_Fock).all(axis=-1))[0]
_ref_diag = float(np.real(_mels[_same].sum()))
print(f"\nHF 参考态: {Hatree_Fock}")
print(f"参考行列式对角元 = {_ref_diag:.8f} Ha（CASSCF 基态 = {E_fcis[0]:.8f} Ha）")

# ============================================================
# 采样图边（单粒子 hop：同自旋轨道两两组合）
# ============================================================
alpha_orbs = list(range(N_ACTIVE_ORBS))
beta_orbs = list(range(N_ACTIVE_ORBS, 2 * N_ACTIVE_ORBS))
single_edges = (
    list(itertools.combinations(alpha_orbs, 2))
    + list(itertools.combinations(beta_orbs, 2))
)

print(f"single_edges: {single_edges}")
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)


class LiH_CAS_STO3G_8Qbit:
    """LiH / STO-3G / CAS(2e,4o) / 8 比特 / NES 扩展空间基准数据。"""

    def __init__(self, K=4):
        self.K = K
        self.hi = hi
        self.hi_ext = hi ** K          # 扩展希尔伯特空间
        self.SINGLE_SIZE = SINGLE_SIZE
        self.ha = ha
        self.Hatree_Fock = Hatree_Fock
        self.single_edges = single_edges
        self.g = g
        self.single_rule = single_rule
        self.tensor_rule = nk.sampler.rules.TensorRule(self.hi_ext, [self.single_rule] * self.K)
        self.E_fcis = E_fcis
        self.hf_ground_energy = hf_ground_energy
        self.casscf_energy = mc.e_tot
        self.ecore = ecore
        self.n_active_orbs = N_ACTIVE_ORBS
        self.n_frozen = NFROZEN
        self.ext_edges = []
        self._set_ext_edges()

    def _set_ext_edges(self):
        ext_edges = []
        for k in range(self.K):
            offset = k * SINGLE_SIZE
            for i, j in self.single_edges:
                ext_edges.append((i + offset, j + offset))
        self.ext_edges = jnp.asarray(ext_edges)
