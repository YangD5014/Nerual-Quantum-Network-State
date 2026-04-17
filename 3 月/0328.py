import platform
import netket as nk
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b

model = Linear(2, 5, rngs=nnx.Rngs(params=0))

