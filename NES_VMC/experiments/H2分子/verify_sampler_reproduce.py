# -*- coding: utf-8 -*-
"""
用训练同款采样器 + 训练好的参数, 复现训练最后一步的 E_L_mean 及本征值,
验证: 训练记录的 Energy_levels 是有限样本估计, 还是精确值。
"""
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk

from NES_VMC_V1 import NESTotalAnsatz_stable, NESFermionHopRule
from NES_VMC_tool import create_gauge_reset_total_machines, make_grad_fn_gauge
from H2_631G import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock, E_fcis

HISTORY_FILE = r'./data/26-09-05-03-48_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)
total_params = history['params'][-1]
trained_levels = np.sort(np.asarray(history['Energy_levels'][-1]).real)
print(f"训练记录 Energy_levels: {trained_levels}")
print(f"FCI:                   {E_fcis}")

total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
g_current = jnp.zeros(K, dtype=jnp.complex64)
(total_machine, total_matrix_machine, total_max_machine,
 total_matrix_machine_raw, total_graphdef, _) = create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, _, _ = __import__('NES_VMC_V1', fromlist=['create_single_machine_gauge_fixed']).create_single_machine_gauge_fixed(ansatz, Hatree_Fock)
    single_machine_list.append(m)

grad_fn = make_grad_fn_gauge(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list)

nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(hilbert=hi_ext, rule=nes_rule, n_chains=16, sweep_size=10)

def sample_machine(params, sigma):
    return total_machine(params, sigma, g_current)

sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

# 多组独立 batch, 观察 E_L_mean 本征值的波动
results = []
for trial in range(5):
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine, parameters=total_params, state=sampler_state,
        chain_length=200,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g_current)
    E_L_mean = np.asarray(E_L_mean)
    eig = np.sort(np.linalg.eigvals(E_L_mean).real)
    offdiag = np.abs(E_L_mean - np.diag(np.diag(E_L_mean))).max()
    results.append(eig)
    print(f"trial {trial}: eigvals = {eig} | offdiag = {offdiag:.6f} | cond={np.linalg.cond(E_L_mean):.2e}")

print("\n所有 trial 的 E_L 本征值 (训练记录应落在这些附近):")
for r in results:
    print("  ", r)
