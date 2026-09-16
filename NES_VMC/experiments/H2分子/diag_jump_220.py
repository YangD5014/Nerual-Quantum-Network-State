# -*- coding: utf-8 -*-
"""诊断 Step 219->220 raw 梯度爆跳：从 pickle 重建 (params, samples)，逐项检查数值通道。"""
import pickle, sys, os
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from NES_VMC_V1 import NESTotalAnsatz_stable, create_single_machine_gauge_fixed
from NES_VMC_tool import (
    create_gauge_reset_total_machines, make_gauge_fn, make_MS_estimator_fn,
    compute_lam_v_from_samples,
)
from H2_631G import SINGLE_SIZE, ha, hi_ext, K, Hatree_Fock, E_fcis

HIST = "./data/26-09-15-17-57_history_natural_gradient_H2_molecule_K4.pkl"
with open(HIST, "rb") as f:
    hist = pickle.load(f)

steps = np.array(hist["steps"])
print("steps:", steps[0], "->", steps[-1], "n =", len(steps))

# --- 重建机器 ---
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K,
    hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(11),
)
(total_machine, total_matrix_machine, total_max_machine,
 total_matrix_machine_raw, graphdef, _state) = create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)
gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)
single_machine_list = [
    create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list
]
estimator = make_MS_estimator_fn(ha=ha, single_machine_list=single_machine_list)

FOCUS = list(range(208, 246))
idx_of = {int(s): i for i, s in enumerate(steps)}

print(f"{'step':>4} | {'rawGrad':>9} {'loss':>9} {'E3':>9} | "
      f"{'gRe(col0..3)':>28} | {'condΨ_w0':>8} {'condΨ_max':>9} {'condΨ_med':>9} | "
      f"{'|tr_n|max':>10} {'tr_n_p99':>9} | {'condŜ':>8} {'valid%':>6}")

rng = np.random.default_rng(0)
for s in FOCUS:
    i = idx_of[s]
    params = hist["params"][i]
    samples = np.asarray(hist["samples"][i])
    x_batch = jnp.asarray(samples.reshape(-1, K, SINGLE_SIZE))
    g = hist["g_history"][i] if "g_history" in hist else None

    # raw L 与列均值（重建 reset 时刻用的 g）
    L_raw = np.asarray(total_matrix_machine_raw(params, x_batch))   # (N,K,K)
    colmean = L_raw.mean(axis=(0, 1))                               # (K,) 复数
    # 有效 L（当前 g 未知则用 colmean 近似 reset 后状态；这里用 logPsi 反推有效性）
    Psi_st = np.asarray(total_matrix_machine(params, x_batch, jnp.asarray(colmean)))
    cond_w = np.linalg.cond(Psi_st)
    tr_n = None
    Mhat, Shat, finite = estimator(params, x_batch)
    Sh = np.asarray(Shat)
    cond_S = np.linalg.cond(Sh)
    # E_L per walker
    HPsi = None
    lam, v, n_valid, order = compute_lam_v_from_samples(
        ha=ha, single_machine_list=single_machine_list, total_params=params,
        x_batch=x_batch, estimator=estimator)
    g_re = np.real(colmean)
    print(f"{s:>4} | {hist['grad_norm_raw'][i]:9.3f} {hist['loss'][i]:9.5f} {hist['Energy_levels'][i][3]:9.4f} | "
          f"{np.array2string(g_re, precision=2, floatmode='fixed'):>28} | "
          f"{cond_w[0]:8.1f} {cond_w.max():9.1f} {np.median(cond_w):9.1f} | "
          f"{'-':>10} {'-':>9} | {cond_S:8.1e} {100*n_valid/len(samples):6.1f}")
