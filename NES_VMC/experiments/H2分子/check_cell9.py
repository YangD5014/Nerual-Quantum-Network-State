# -*- coding: utf-8 -*-
"""核对: notebook cell 5/9 的值是否与脚本一致"""
import pickle
import numpy as np
import flax.nnx as nnx
from NES_VMC_V1 import NESTotalAnsatz
from H2_631G import SINGLE_SIZE, ha, hi, K

HISTORY_FILE = r'./data/26-09-05-03-48_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)
finnal_params = history['params'][-1]

total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
single_graphdef, _ = nnx.split(total_ansatz.single_ansatz_list[0])
fine_single_ansatz = nnx.merge(single_graphdef, finnal_params['single_ansatz_list'][0])

logpsi = np.asarray(fine_single_ansatz(hi.all_states()))
print("dtype:", logpsi.dtype)
print("notebook cell5 前 5 个值:")
print(logpsi[:5])
print("标量 logψ†Hlogψ =", (logpsi.conj().T @ ha.to_dense() @ logpsi))
