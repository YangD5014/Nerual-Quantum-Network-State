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
cisolver.nroots = 4
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


K = 4  # NES 扩展副本数
hi_ext = hi ** K  # 扩展希尔伯特空间
ha = nkx.operator.from_pyscf_molecule(mol)
Hatree_Fock = hi.all_states()[0]
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