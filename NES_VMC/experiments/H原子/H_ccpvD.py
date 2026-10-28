from pyscf import gto, scf, fci,mp, ci
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np
import itertools



bond_length = 1.8
geometry = [
    ("H", (0.0, 0.0, bond_length))
]

mol = gto.M(atom=geometry, basis="cc-pVDZ", verbose=0, spin=1)
mf = scf.RHF(mol).run(verbose=0)
hf_ground_energy = mf.e_tot

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print("=" * 60)
print("H原子 / cc-pVDZ 基准")
print("=" * 60)
print(f"HF energy = {hf_ground_energy:.8f} Ha")
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha | excitation = {exc:.4f} eV")

hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=5,
    s=1/2,
    n_fermions_per_spin=(1, 0),
)

K = 4
hi_ext = hi ** K
ha = nkx.operator.from_pyscf_molecule(mol)

SINGLE_SIZE = hi.size

print("=" * 60)
print("Hilbert 信息")
print("=" * 60)
print(f"K = {K}")
print(f"hi.size = {hi.size}")
print(f"hi_ext.size = {hi_ext.size}")
print(f"SINGLE_SIZE = {SINGLE_SIZE}")

target_loss = float(np.sum(E_fcis[:K]))
print(f"target_loss = sum(E_fcis[:K]) = {target_loss:.8f}")


Hatree_Fock = hi.all_states()[0]
print(f'Hatree_Fock = {Hatree_Fock}')
single_edges_full = list(itertools.combinations(range(int(SINGLE_SIZE/2)), 2))
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for i, j in single_edges_full:
        ext_edges.append((i + offset, j + offset))

ext_edges = jnp.asarray(ext_edges)


class H_ccpVD_8Qbit:
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
        
        