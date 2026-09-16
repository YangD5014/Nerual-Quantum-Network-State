# -*- coding: utf-8 -*-
"""精确定位：回放训练 g 序列，分解 raw 梯度爆跳 = tr_n 离群 + 规范数值通道。"""
import pickle, sys, os
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from NES_VMC_V1 import NESTotalAnsatz_stable, create_single_machine_gauge_fixed, Ham_Psi_scaled, ravel_pytree
from NES_VMC_tool import (
    create_gauge_reset_total_machines, make_gauge_fn, make_grad_fn_gauge,
    NES_loss_energy_stable_gauge,
)
from H2_631G import SINGLE_SIZE, ha, hi_ext, K, Hatree_Fock, E_fcis

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
gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)
single_machine_list = [
    create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list
]
grad_fn = make_grad_fn_gauge(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list)

# ---- 回放训练中的 g 序列（g 每 10 步被替换为当步 batch 的 raw colmean）----
g_seq = {}
g_cur = jnp.zeros(K, dtype=jnp.complex64)
for s in steps:
    g_seq[int(s)] = g_cur
    if (int(s) + 1) % 10 == 0:
        params = hist["params"][int(s)]
        x = jnp.asarray(np.asarray(hist["samples"][int(s)]).reshape(-1, K, SINGLE_SIZE))
        g_cur = col_mean_fn(params, x)
print("回放 |g| 序列:", {k: round(float(jnp.linalg.norm(v)), 4) for k, v in g_seq.items() if k % 10 == 9})

def tr_stats(params, x_batch, g):
    """逐 walker E_L 迹分布 + raw 梯度范数。"""
    loss_batch, E_L_batch, aux = NES_loss_energy_stable_gauge(
        ha=ha, total_matrix_machine=total_matrix_machine, total_max_machine=total_max_machine,
        single_machine_list=single_machine_list, total_params=params, x=x_batch, g=g, return_aux=True)
    tr_n = jnp.real(jnp.trace(E_L_batch, axis1=-2, axis2=-1))
    grad, loss_mean, E_L_mean, gaux = grad_fn(params, x_batch, g)
    gf, _ = ravel_pytree(grad)
    tr = np.asarray(tr_n)
    return dict(
        raw=float(jnp.linalg.norm(gf)), loss=float(jnp.real(loss_mean)),
        tr_med=float(np.median(tr)), tr_p99=float(np.percentile(np.abs(tr), 99)),
        tr_max=float(np.max(np.abs(tr))), tr_min=float(tr.min()), tr_maxr=float(tr.max()),
        n_out=int((np.abs(tr) > 10 * (np.percentile(np.abs(tr), 50) + 1)).sum()),
        shift=float(np.asarray(aux["shift"]).mean()),
        Lmin=float(np.asarray(aux["L_stable"]).min()),
        cond_mean=float(np.asarray(aux["cond_Psi"]).mean()),
    )

print(f"\n{'step':>4} {'g来源':>10} | {'rawGrad':>9} {'loss':>9} | {'tr_med':>8} {'|tr|p99':>9} {'|tr|max':>10} "
      f"{'tr_min':>9} {'tr_max':>9} {'n_out':>5} | {'shift':>6} {'L_min':>7} {'condΨ̄':>7}")
for s in [216, 217, 218, 219, 220, 221, 222, 223, 225, 228, 230, 233, 236, 240, 245]:
    i = idx_of[s]
    params = hist["params"][s]
    x = jnp.asarray(np.asarray(hist["samples"][s]).reshape(-1, K, SINGLE_SIZE))
    g_true = g_seq[s]
    for tag, g in [("训练g", g_true), ("colmean", col_mean_fn(params, x)), ("零规范", jnp.zeros(K, jnp.complex64))]:
        r = tr_stats(params, x, g)
        print(f"{s:>4} {tag:>10} | {r['raw']:9.3f} {r['loss']:9.5f} | {r['tr_med']:8.4f} {r['tr_p99']:9.3f} "
              f"{r['tr_max']:10.3e} {r['tr_min']:9.3f} {r['tr_maxr']:9.3f} {r['n_out']:5d} | "
              f"{r['shift']:6.2f} {r['Lmin']:7.2f} {r['cond_mean']:7.1f}")
    print("-" * 120)
