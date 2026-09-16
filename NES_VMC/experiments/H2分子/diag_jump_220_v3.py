# -*- coding: utf-8 -*-
"""验证 MCMC 链冻结假设：每步独立构型数 + 逐 walker ∇logΨ 范数。"""
import pickle, sys, os
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from NES_VMC_V1 import NESTotalAnsatz_stable, create_single_machine_gauge_fixed, ravel_pytree
from NES_VMC_tool import create_gauge_reset_total_machines
from H2_631G import SINGLE_SIZE, ha, hi_ext, K, Hatree_Fock

HIST = "./data/26-09-15-17-57_history_natural_gradient_H2_molecule_K4.pkl"
with open(HIST, "rb") as f:
    hist = pickle.load(f)
steps = list(hist["steps"])
idx_of = {int(s): i for i, s in enumerate(steps)}

total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K,
    hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(11),
)
(total_machine, total_matrix_machine, total_max_machine,
 total_matrix_machine_raw, graphdef, _s) = create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)
single_machine_list = [
    create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list
]

grad_logpsi = jax.grad(total_machine, argnums=0, holomorphic=True)
vmap_grad = jax.vmap(grad_logpsi, in_axes=(None, 0, None))

print(f"{'step':>4} | {'uniq':>5}/3200 {'uniq/链':>7} | {'rawGrad':>9} | 逐链独构型数(16链) 前8个 | max‖∇logΨ‖Walker "
      f"| argmax链")
for s in [214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 230, 233, 236, 240, 245, 250, 299]:
    i = idx_of[s]
    samples = np.asarray(hist["samples"][s]).reshape(-1, K, SINGLE_SIZE)
    # 独立构型数：把 (K,n_spin) 展平成 tuple
    flat = samples.reshape(len(samples), -1)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    # 每 200 个样本 = 1 条链（reshape 顺序：n_chains × chain_length）
    per_chain = [len(np.unique(flat[c * 200:(c + 1) * 200], axis=0)) for c in range(16)]
    params = hist["params"][s]
    x = jnp.asarray(samples)
    gt = vmap_grad(params, x, jnp.zeros(K, jnp.complex64))
    gf = jax.vmap(lambda t: jnp.linalg.norm(ravel_pytree(t)[0]))(gt)
    gf_np = np.asarray(gf)
    amax = int(np.argmax(gf_np))
    top5 = np.sort(gf_np)[-5:]
    print(f"{s:>4} | {len(uniq):>5} {per_chain[0]:>4}... | {hist['grad_norm_raw'][i]:9.3f} | "
          f"{str(per_chain[:8]):>40} | {gf_np.max():10.3e} | "
          f"walker{amax}(链{amax//200},内{amax%200}) top5={np.array2string(top5, precision=2)}")
