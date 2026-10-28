from pyscf import gto, scf, fci, ao2mo
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np
import itertools
# ============================================================
# LiH 分子 / STO-3G / Active Space 基准
# ============================================================
# 与同级目录 `H2 分子/H2_6_31G_8Qbit.py` 同构的数据模块：
#   H2  : 6-31G 全空间 4 个空间轨道（8 自旋轨道，2 电子）→ 8 qubit
#   LiH : STO-3G 前 4 个 RHF 轨道做 CAS(4e,4o)（8 自旋轨道，4 电子）→ 8 qubit
# 两者 SINGLE_SIZE 都是 8，方便直接复用同一套 VMC 脚本。
#
# 注意：LiH 的 HF 行列式不是 hi.all_states()[0]（枚举顺序从高轨道开始），
# 必须显式用 [1,1,0,0,1,1,0,0]（alpha 占 0,1；beta 占 0,1）。

bond_length = 1.58  # LiH 平衡键长 ≈ 1.58 Angstrom
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
# 活性空间：前 4 个 RHF canonical 轨道（CAS(4e,4o)）
# 与 ha 的 mo_coeff 投影空间严格一致，E_fcis 才与哈密顿量谱逐位对应
# ============================================================
N_ORBITALS = 4
n_alpha_electrons = 2
n_beta_electrons = 2

active_idx = [0, 1, 2, 3]
mo_active = mf.mo_coeff[:, active_idx]

E_nuc = mol.energy_nuc()
h1_active = mo_active.T @ mf.get_hcore() @ mo_active
g2_active = ao2mo.incore.full(mf._eri, mo_active)
cisolver = fci.direct_spin1.FCI()
E_active, fcivec = cisolver.kernel(
    h1_active, g2_active, N_ORBITALS,
    (n_alpha_electrons, n_beta_electrons), nroots=12,
)
E_fcis = np.asarray(E_active) + E_nuc

print("=" * 60)
print("LiH / STO-3G · 4 轨道活性空间 FCI 基准")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# ============================================================
# 采样图边（单粒子 hop：同自旋轨道两两组合）
# ============================================================
alpha_orbs = list(range(N_ORBITALS))          # 0,1,2,3
beta_orbs = list(range(N_ORBITALS, 2 * N_ORBITALS))  # 4,5,6,7
single_edges = (
    list(itertools.combinations(alpha_orbs, 2))
    + list(itertools.combinations(beta_orbs, 2))
)

hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=N_ORBITALS,
    s=1/2,
    n_fermions_per_spin=(n_alpha_electrons, n_beta_electrons),
)

# 真实 HF 行列式：alpha 占据轨道 0,1；beta 占据轨道 0,1
Hatree_Fock = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)

# 注意：默认的 ParticleNumberAndSpinConserving 实现会用 mol.nao=6（12 自旋轨道）
# 建 Hilbert 空间，无法截断到 4 轨道活性空间；这里先用 FermionOperator2ndJax
# 构造（它按 mo_coeff 列数 4 → 8 自旋轨道），再包装成自旋/粒子数守恒算子。
ha_raw = nkx.operator.from_pyscf_molecule(
    mol, mo_coeff=mo_active, implementation=nk.operator.FermionOperator2ndJax
)
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
    ha_raw
)

_conn, _mels = ha.get_conn(jnp.asarray(Hatree_Fock))
_conn = np.asarray(_conn)
_mels = np.asarray(_mels)
_same = np.where((_conn == Hatree_Fock).all(axis=-1))[0]
_diag = float(np.real(_mels[_same].sum()))
assert abs(_diag - hf_ground_energy) < 1e-6, \
    f"HF 参考态对角元 {_diag} != E_RHF {hf_ground_energy}"

print(f"\nHF reference state: {Hatree_Fock}")
print(f"HF 对角元 = {_diag:.8f} == E_RHF ✓")
print(f'single_edges: {single_edges}')
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
SINGLE_SIZE = hi.size


class LiH_STO3G_8Qbit:
    def __init__(self, K=2):
        self.K = K
        self.hi = hi
        self.hi_ext = hi ** K  # 扩展希尔伯特空间
        self.SINGLE_SIZE = SINGLE_SIZE
        self.ha = ha
        self.Hatree_Fock = Hatree_Fock
        self.single_edges = single_edges
        self.g = g
        self.single_rule = single_rule
        self.tensor_rule = nk.sampler.rules.TensorRule(self.hi_ext, [self.single_rule] * self.K)
        self.ext_edges = []
        self.E_fcis = E_fcis
        self.fcivec = fcivec
        self.hf_ground_energy = hf_ground_energy

        self._set_ext_edges()

    def _set_ext_edges(self):
        ext_edges = []
        for k in range(self.K):
            offset = k * SINGLE_SIZE
            for i, j in self.single_edges:
                ext_edges.append((i + offset, j + offset))
        self.ext_edges = jnp.asarray(ext_edges)
