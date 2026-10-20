"""诊断：从训练平台期的参数出发，做「全空间精确 SR」下降（仅诊断）。

目的：区分两种平台期成因——
  A) 优化/估计器问题：MC-SR 卡住，但精确 SR 能继续下降到更低能量；
  B) 模型/景观问题：精确 SR 也卡在同一能量 → 需要更大容量/不同参数化。

用法：python diag_exact_sr.py params_npz路径 [lr] [steps]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "..")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import flatten_util

import H2_631G_AR_VMC as M
from ar_nqs import CausalMaskedARNQS, SegmentLayout, create_machine

npz = sys.argv[1] if len(sys.argv) > 1 else "H2_631G_AR_VMC_params.npz"
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60

layout = SegmentLayout(8, [(0, 4, 1), (4, 8, 1)])
model = CausalMaskedARNQS(8, 4, rngs=nnx.Rngs(21))
machine, net_out, graphdef, init_state = create_machine(model, layout)

flat = np.load(npz)["flat"]
_, unravel = flatten_util.ravel_pytree(init_state)
params = unravel(jnp.asarray(flat))

states, H = M.build_H_dense()

def energy_full(p):
    lp = machine(p, states)
    psi = jnp.exp(lp - jnp.max(lp.real))
    return (jnp.vdot(psi, H @ psi) / jnp.vdot(psi, psi)).real

E0, gn0 = M.exact_grad(machine, params, states, H)
dE0, var0, ovl0, _ = M.full_space_diag(machine, params, (states, H))
print(f"起点: E={E0:.8f} |grad|={gn0:.4e} Var={var0:.3e} Ovl={ovl0:.6f}")

# 精确 force matrix（16 样本、|ψ|² 权重，全空间）
p16 = np.exp(2 * np.asarray(machine(params, states)).real)
p16 = p16 / p16.sum()

def force_tree(p):
    re = jax.vmap(lambda s: jax.grad(lambda q: machine(q, s).real)(p))(states)
    im = jax.vmap(lambda s: jax.grad(lambda q: machine(q, s).imag)(p))(states)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re, im)

def exact_sr_step(p, shift=0.001):
    g = jax.grad(energy_full)(p)
    gf, _ = flatten_util.ravel_pytree(g)
    pw = jnp.exp(2 * machine(p, states).real)
    pw = pw / pw.sum()
    O = force_tree(p)
    Of, _ = flatten_util.ravel_pytree(O)
    Of = Of.reshape(16, -1)                       # (16, P) 复
    Of = Of - jnp.sum(pw[:, None] * Of, axis=0, keepdims=True)
    S = jnp.real((pw[:, None] * jnp.conj(Of)).T @ Of)
    nat = jnp.linalg.solve(S + shift * jnp.eye(S.shape[0]), gf)
    return nat

for t in range(n_steps):
    nat = exact_sr_step(params)
    _, unr = flatten_util.ravel_pytree(params)
    params = jax.tree_util.tree_map(lambda p, d: p - lr * d, params, unr(nat))
    if t % 10 == 0 or t == n_steps - 1:
        E, gn = M.exact_grad(machine, params, states, H)
        dE, var, ovl, _ = M.full_space_diag(machine, params, (states, H))
        print(f"exactSR {t:3d} | E={E:.8f} |grad|={gn:.3e} Var={var:.3e} Ovl={ovl:.6f}")
