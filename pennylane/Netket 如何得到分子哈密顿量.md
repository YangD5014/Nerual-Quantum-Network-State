Netket 官方的一些案例：

from pyscf import gto, scf, fci
import netket as nk; import netket.experimental as nkx

bond_length = 1.5109
geometry = [('Li', (0., 0., -bond_length/2)), ('H', (0., 0., bond_length/2))]
mol = gto.M(atom=geometry, basis='STO-3G')

mf = scf.RHF(mol).run(verbose=0)
E_hf = sum(mf.scf_summary.values())

E_fci = fci.FCI(mf).kernel()[0]

ha = nkx.operator.from_pyscf_molecule(mol)
E0 = float(nk.exact.lanczos_ed(ha))
print(f"{E0 = :.5f}, {E_fci = :.5f}")
E0 = -7.88253, E_fci = -7.88253