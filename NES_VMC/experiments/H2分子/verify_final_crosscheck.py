# -*- coding: utf-8 -*-
"""最终交叉验证: scipy eigh 广义本征问题 与 |Ψ|²加权 E_L 平均是否一致"""
import pickle
import itertools
import numpy as np
import flax.nnx as nnx
from scipy.linalg import eigh

from NES_VMC_V1 import NESTotalAnsatz
from H2_631G import SINGLE_SIZE, ha, hi, K, E_fcis

HISTORY_FILE = r'./data/26-09-05-03-48_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)
finnal_params = history['params'][-1]

total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
x = hi.all_states()
H = np.asarray(ha.to_dense())

psi_full = np.exp(np.stack([
    np.asarray(nnx.merge(nnx.split(total_ansatz.single_ansatz_list[j])[0],
                          finnal_params['single_ansatz_list'][j])(x))
    for j in range(K)
], axis=1))                              # (16, K)
Hpsi_full = H @ psi_full

M = psi_full.conj().T @ Hpsi_full
S = psi_full.conj().T @ psi_full
lam, v = eigh(M, S)                      # 广义本征问题 M v = λ S v
print("scipy eigh 广义本征值 (Rayleigh-Ritz):", np.sort(lam))

# 对比 |Ψ|² 加权 E_L 平均
all_idx = np.asarray(list(itertools.product(range(16), repeat=K)))
det2 = np.array([abs(np.linalg.det(psi_full[idx])) ** 2 for idx in all_idx])
Z = det2.sum()
E_L_acc = np.zeros((K, K), dtype=np.complex128)
for w, idx in zip(det2, all_idx):
    if w > 0:
        E_L_acc += w * np.linalg.solve(psi_full[idx], Hpsi_full[idx])
E_L_mean = E_L_acc / Z
print("加权 E_L 本征值:", np.sort(np.linalg.eigvals(E_L_mean).real))
print("E_L_mean == S^{-1} M ?", np.allclose(E_L_mean, np.linalg.solve(S, M), atol=1e-8))
print("FCI:", E_fcis)
