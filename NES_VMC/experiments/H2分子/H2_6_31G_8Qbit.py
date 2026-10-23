from pyscf import gto, scf, fci,mp, ci
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np
import itertools
# ============================================================
# LiH 分子 / cc-pVDZ / Active Space 基准
# ============================================================
bond_length = 1.5
geometry = [
    ("H", (0.0, 0.0, 0.0)),
    ("H",  (0.0, 0.0, bond_length)),
]

mol = gto.M(atom=geometry, basis="6-31G", verbose=0, spin=0)
mf = scf.RHF(mol).run(verbose=0)
hf_ground_energy = mf.e_tot

print("=" * 60)
print("H2 分子基本信息")
print("=" * 60)
print(f"HF energy = {hf_ground_energy:.8f} Ha")
print(f"Total electrons = {mol.nelec}")
print(f"Total basis functions = {mol.nao_nr()}")
cisolver = fci.FCI(mf)
cisolver.nroots = 12
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")
    

nums1 = [0,1,2,3]
combs1 = itertools.combinations(nums1, r=2)
nums2 = [4,5,6,7]
combs2 = itertools.combinations(nums2, r=2)
# combinations返回迭代器，要转成list才能打印看全部结果
combs_list1 = list(combs1)
combs_list2 = list(combs2)
single_edges = combs_list1 + combs_list2

hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=4,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

Hatree_Fock = hi.all_states()[0]
print(f"\nHF reference state: {Hatree_Fock[:20]}")
ha = nkx.operator.from_pyscf_molecule(mol)
Hatree_Fock = hi.all_states()[0]
print(f'single_edges: {single_edges}')
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
SINGLE_SIZE = hi.size

class H2_6_31G_8Qbit:
    def __init__(self, K=2):
        self.K = K
        self.hi = hi
        self.hi_ext = hi ** K  # 扩展希尔伯特空间
        self.SINGLE_SIZE = SINGLE_SIZE
        self.ha = nkx.operator.from_pyscf_molecule(mol)
        self.Hatree_Fock = hi.all_states()[0]
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