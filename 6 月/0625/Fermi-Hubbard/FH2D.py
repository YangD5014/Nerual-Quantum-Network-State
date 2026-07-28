from pyscf import gto, scf, fci,mp, ci
import netket as nk
import netket.experimental as nkx
import itertools
import jax.numpy as jnp
import numpy as np
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd


L = 4  # Side of the square
graph = nk.graph.Square(L)
N = graph.n_nodes
N_f = 5
hi = nk.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)

t = 1.0
V = 4.0

H = 0.0
for i, j in graph.edges():
    H -= t * (cdag(hi, i) * c(hi, j) + cdag(hi, j) * c(hi, i))
    H += V * nc(hi, i) * nc(hi, j)
    H_pnc = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(H)
    
    