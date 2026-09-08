# -*- coding: utf-8 -*-
"""为什么单列期望(-0.8506) 对不上 Energy_levels(-1.0839, ...)?

把 K×K 的 M / S 矩阵显式打出来, 看本征值与对角元(单列期望)的关系。
"""
import pickle
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

M = psi_full.conj().T @ Hpsi_full       # M_ij = <ψ_i|H|ψ_j>
S = psi_full.conj().T @ psi_full        # S_ij = <ψ_i|ψ_j>

np.set_printoptions(precision=4, suppress=True)
print("=== M 矩阵 (实部),  M_ij = <ψ_i|H|ψ_j> ===")
print(np.real(M))
print("对角元 = 各单列的期望值 <ψ_j|H|ψ_j>（未归一化）:")
print("   ", np.real(np.diag(M)))

print("\n=== S 矩阵 (实部),  S_ij = <ψ_i|ψ_j> ===")
print(np.real(S))

print("\n=== 归一化后的单列期望 <ψ_j|H|ψ_j>/<ψ_j|ψ_j> ===")
print("   ", np.real(np.diag(M) / np.diag(S)))

# 广义本征问题 M v = λ S v  (等价于训练同款 E_L 本征值)
lam, v = eigh(M, S)
order = np.argsort(lam)
lam, v = lam[order], v[:, order]
print("\n=== 广义本征值 λ (Rayleigh-Ritz) ===")
print("   ", lam)
print("\n=== 对应本征向量 v (能级 = v 列对应的线性组合 Σ_j v_j ψ_j) ===")
print(np.real(v))
print("\n=== 参考 ===")
print("FCI:            ", E_fcis)
print("训练记录能级:   ", np.real(np.asarray(history['Energy_levels'][-1])))
print("单列期望(state0):", np.real(np.diag(M) / np.diag(S))[0])
