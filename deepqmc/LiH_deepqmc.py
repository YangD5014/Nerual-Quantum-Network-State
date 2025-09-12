from deepqmc.molecule import Molecule
from deepqmc.hamil import MolecularHamiltonian
import os
import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import deepqmc

mol = Molecule.from_name('LiH')
mol = Molecule(  # LiH
    coords=[[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]],
    charges=[3, 1],
    charge=0,
    spin=0,
    unit='bohr',
)
H = MolecularHamiltonian(mol=mol)
