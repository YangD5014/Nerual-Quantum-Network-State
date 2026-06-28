Ground-State: Heisenberg model

fter having defined this environment variable, we can load netket and the various libraries that we will be using throughout the tutorial.

# Import netket library
import netket as nk

# Import Json, this will be needed to load log files
import json

# Helper libraries
import matplotlib.pyplot as plt
import time
1. Defining the Hamiltonian
NetKet covers quite a few standard Hamiltonians and lattices, so let’s use this to quickly define the antiferromagnetic Heisenberg chain. For the moment we assume 
 and simply define a chain lattice in this way (using periodic boundary conditions for now).

# Define a 1d chain
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
Next, we need to define the Hilbert space on this graph. We have here spin-half degrees of freedom, and as we know that the ground-state sits in the zero magnetization sector, we can already impose this as a constraint in the Hilbert space. This is not mandatory, but will nicely speeds things up in the following.

# Define the Hilbert space based on this graph
# We impose to have a fixed total magnetization of zero
hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
The final element of the triptych is of course the Hamiltonian acting in this Hilbert space, which in our case in already defined in NetKet. Note that the NetKet Hamiltonian uses Pauli Matrices (if you prefer to work with spin-
 operators, it’s pretty trivial to define your own custom Hamiltonian, as covered in another tutorial)

# calling the Heisenberg Hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)