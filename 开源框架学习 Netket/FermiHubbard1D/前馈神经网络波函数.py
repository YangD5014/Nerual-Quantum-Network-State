import netket as nk
import netket.experimental as nkx
import numpy as np
import matplotlib.pyplot as plt
import json
from flax import nnx
import jax.numpy as jnp
import jax 

L = 4  # take a 1x4 lattice
D = 1
t = 1.0  # tunneling/hopping
U = 0.1  # coulomb

# create the graph our fermions can hop on
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=False)
n_sites = g.n_nodes
# create a hilbert space with 2 up and 2 down spins
hilber_space = nk.hilbert.SpinOrbitalFermions(n_sites, s=1 / 2, n_fermions_per_spin=(2, 2))



# create the graph our fermions can hop on
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=False)
n_sites = g.n_nodes
# create a hilbert space with 2 up and 2 down spins
hilber_space = nk.hilbert.SpinOrbitalFermions(n_sites, s=1 / 2, n_fermions_per_spin=(2, 2))
ham = nkx.operator.FermiHubbardJax(hilber_space, t=t, U=U, graph=g)

sampler = nk.sampler.MetropolisFermionHop(hilber_space, graph=g, n_chains=16, sweep_size=64)
# or let netket copy the graph per spin sector
saampler = nk.sampler.MetropolisFermionHop(
    hilber_space, graph=g, n_chains=16, sweep_size=64, spin_symmetric=True
)
# since the hilbert basis is a set of occupation numbers, we can take a general RBM
# we take complex parameters, since it learns sign structures more easily, and for even fermion number, the wave function might be complex

class FFN(nnx.Module):

    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        Construct a Feed-Forward Neural Network with a single hidden layer.

        Args:
            N: The number of input nodes (number of spins in the chain).
            alpha: The density of the hidden layer. The hidden layer will have
                N*alpha nodes.
            rngs: The random number generator seed.
        """
        self.alpha = alpha

        # We define a linear (or dense) layer with `alpha` times the number of input nodes
        # as output nodes.
        # We must pass forward the rngs object to the dense layer.
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        print(f"x.shape = {x.shape}|x ={x}")

        # we apply the linear layer to the input
        y = self.linear(x)

        # the non-linearity is a simple ReLu
        y = nnx.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)
    
model_FNN = FFN(N=L*2, alpha=1, rngs=nnx.Rngs(2))
vstate_FNN = nk.vqs.MCState(sampler, model_FNN, n_samples=1008)

# vs_FNN = nk.vqs.MCState(sampler, model1, n_discard_per_chain=10, n_samples=512)

# we will use sgd with Stochastic Reconfiguration
opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=False)
gs_FNN = nk.driver.VMC(ham, opt, variational_state=vstate_FNN, preconditioner=sr)
exp_name = "fermions_test0723_FNN"
gs_FNN.run(500, out=exp_name)

