class netket.sampler.MetropolisSampler[source]

init_state(machine, parameters, seed=None)[source]
Creates the structure holding the state of the sampler.

If you want reproducible samples, you should specify seed, otherwise the state will be initialised randomly.

If running across several JAX processes, all sampler_state`s are guaranteed to be in a different (but deterministic) state. This is achieved by first reducing (summing) the seed provided to every JAX process, then generating `n_process seeds starting from the reduced one, and every process is initialized with one of those seeds.

The resulting state is guaranteed to be a frozen Python dataclass (in particular, a Flax dataclass), and it can be serialized using Flax serialization methods.

Parameters
:
machine (Union[Module, Callable[[Any, Union[ndarray, Array]], Union[ndarray, Array]]]) – A Flax module or callable with the forward pass of the log-pdf. If it is a callable, it should have the signature f(parameters, σ) -> jax.Array.

parameters (Any) – The PyTree of parameters of the model.

seed (Union[int, Any, None]) – An optional seed or jax PRNGKey. If not specified, a random seed will be used.

Return type
:
SamplerState

Returns
:
The structure holding the state of the sampler. In general you should not expect it to be in a valid state, and should reset it before use.



```python
from flax.linen import nn
# Mean Field Ansatz
class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)
        p = nn.log_sigmoid(lam * x)
        return 0.5 * jnp.sum(p, axis=-1)
# Example with Mean Field model
model = MF()
parameters = model.init(jax.random.key(0), np.ones((hi.size,)))

# Initialize sampler state
sampler_state = sampler.init_state(model, parameters, seed=1)
```

但是我现在的 model 是基于 flax.nnx写的:
请你教我如何提取到符合 sampler.init_state要求的 parameters？

```python

class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b

model = Linear(2, 5, rngs=nnx.Rngs(params=0))
parameters = model.init(jax.random.key(0), np.ones((hi.size,))) #会报错


```