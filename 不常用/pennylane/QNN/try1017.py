from QNN_jax import quantum_neural_network,initialize_parameters
import pennylane as qml
import jax
from jax import numpy as jnp, random
from flax import nnx
import os
import netket as nk
from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmax, sigmaz

N = 3
hi = nk.hilbert.Spin(s=1 / 2, N=N)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
hi.random_state(jax.random.key(0), 3)
Gamma = -1
H = sum([Gamma * sigmax(hi, i) for i in range(N)])
V = -1
H += sum([V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) for i in range(N)])
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
print("eigenvalues with scipy sparse:", eig_vals)
E_gs = eig_vals[0]

# Create the local sampler on the hilbert space
sampler = nk.sampler.MetropolisLocal(hi)


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
        print(f'input X={x}')
        # we apply the linear layer to the input
        y = self.linear(x)

        # the non-linearity is a simple ReLu
        y = nnx.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)


model = FFN(N=N, alpha=1, rngs=nnx.Rngs(2))

vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

optimizer = nk.optimizer.Sgd(learning_rate=0.1)

# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

ffn_energy = vstate.expect(H)
error = abs((ffn_energy.mean - eig_vals[0]) / eig_vals[0])
print("Optimized energy and relative error: ", ffn_energy, error)