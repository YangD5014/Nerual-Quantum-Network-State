from pyscf import gto, scf, fci,mp, ci
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np
import itertools
# ============================================================
# He 原子 / cc-pVDZ 基准
# ============================================================
bond_length = 1.8
geometry = [
    ("He", (0.0, 0.0, bond_length)),
]

mol = gto.M(atom=geometry, basis="cc-pVDZ", verbose=0, spin=0)
mf = scf.RHF(mol).run(verbose=0)
mo_coeff = mf.mo_coeff
hf_ground_energy = mf.e_tot

print("=" * 60)
print("He 原子基本信息")
print("=" * 60)
print(f"HF energy = {hf_ground_energy:.8f} Ha")
print(f"Total electrons = {mol.nelec}")
print(f"Total basis functions = {mol.nao_nr()}")
cisolver = fci.FCI(mf)
cisolver.nroots = 12
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("He FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")


nums1 = range(mol.nao)
combs1 = itertools.combinations(nums1, r=2)
nums2 = range(mol.nao, 2*mol.nao)
combs2 = itertools.combinations(nums2, r=2)
# combinations返回迭代器，要转成list才能打印看全部结果
combs_list1 = list(combs1)
combs_list2 = list(combs2)
single_edges = combs_list1 + combs_list2
print(f'single_edges: {single_edges}')
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=mol.nao,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

# 真实的 HF 参考态：mo_occ 中 2 = 双占据
# 自旋轨道排布：前 nao 个为 alpha 分块，后 nao 个为 beta 分块
occ_alpha = np.where(mf.mo_occ >= 1)[0]
occ_beta = np.where(mf.mo_occ == 2)[0]
Hatree_Fock = np.zeros(hi.size, dtype=np.int32)
Hatree_Fock[occ_alpha] = 1
Hatree_Fock[mol.nao + occ_beta] = 1

print(f"\nHF reference state: {Hatree_Fock[:20]}")
ha = nkx.operator.from_pyscf_molecule(mol, mo_coeff=mo_coeff)
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
SINGLE_SIZE = hi.size

class He_ccpVD_10Qbit:
    def __init__(self, K=2):
        self.K = K
        self.hi = hi
        self.hi_ext = hi ** K  # 扩展希尔伯特空间
        self.SINGLE_SIZE = SINGLE_SIZE
        self.ha = nkx.operator.from_pyscf_molecule(mol, mo_coeff=mo_coeff)
        self.Hatree_Fock = Hatree_Fock
        self.single_edges = single_edges
        self.g = g
        self.single_rule = single_rule
        self.tensor_rule = nk.sampler.rules.TensorRule(self.hi_ext, [self.single_rule] * self.K)
        self.ext_edges = []
        self.E_fcis, self.fcivec = cisolver.kernel()
        self.hf_ground_energy = hf_ground_energy

        self._set_ext_edges()

    def _set_ext_edges(self):
        ext_edges=[]
        for k in range(self.K):
            offset = k * SINGLE_SIZE
            for i, j in self.single_edges:
                ext_edges.append((i + offset, j + offset))
        self.ext_edges = jnp.asarray(ext_edges)
