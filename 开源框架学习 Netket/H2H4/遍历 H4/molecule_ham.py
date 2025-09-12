import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx

def build_H4(bond_length: float)->(nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd,np.float64):
    geometry = [
        ('H', (0., 0., 0.)),
        ('H', (bond_length, 0., 0.)),
        ('H', (2*bond_length, 0., 0.)),
        ('H', (3*bond_length, 0., 0.))
    ]

    # 创建分子对象，使用STO-3G基组
    mol = gto.M(atom=geometry, basis='STO-3G')
    mf = scf.RHF(mol).run(verbose=0)
    cisolver = fci.FCI(mf)
    E_fci, fcivec = cisolver.kernel()
    ha = nkx.operator.from_pyscf_molecule(mol)
    return ha,E_fci

def build_H2(bond_length: float)->(nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd,np.float64):
    geometry = [
        ('H', (0., 0., 0.)),
        ('H', (bond_length, 0., 0.))
    ]

    # 创建分子对象，使用STO-3G基组
    mol = gto.M(atom=geometry, basis='STO-3G')
    mf = scf.RHF(mol).run(verbose=0)
    cisolver = fci.FCI(mf)
    E_fci, fcivec = cisolver.kernel()
    ha = nkx.operator.from_pyscf_molecule(mol)
    return ha,E_fci

def build_LiH(bond_length: float)->(nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd,np.float64):
    geometry = [
        ('Li', (0., 0., 0.)), 
        ('H', (bond_length, 0., 0.))
    ]

    # 创建分子对象，使用STO-3G基组
    mol = gto.M(atom=geometry, basis='STO-3G')
    mf = scf.RHF(mol).run(verbose=0)
    cisolver = fci.FCI(mf)
    E_fci, fcivec = cisolver.kernel()
    ha = nkx.operator.from_pyscf_molecule(mol)
    return ha,E_fci