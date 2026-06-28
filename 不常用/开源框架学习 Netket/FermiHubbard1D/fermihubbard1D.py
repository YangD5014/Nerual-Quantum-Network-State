import netket as nk
import netket.experimental as nkx
import numpy as np
import matplotlib.pyplot as plt
import json
from flax import nnx
import jax.numpy as jnp
import jax 

L = 4  # take a 2x2 lattice
D = 1
t = 1.0  # tunneling/hopping
U = 0.1  # coulomb

# create the graph our fermions can hop on
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=False)
n_sites = g.n_nodes
# create a hilbert space with 2 up and 2 down spins
hilber_space = nk.hilbert.SpinOrbitalFermions(n_sites, s=1 / 2, n_fermions_per_spin=(2, 2))
fermi_ham = nkx.operator.FermiHubbardJax(hilber_space, t=t, U=U, graph=g)