# -*- coding: utf-8 -*-
"""
决定性验证: 用训练好的参数, 穷举全部扩展空间构型, 复现训练的能级定义
E_levels = eig( E_{X ~ |Ψ_total|²}[ Ψ(X)^{-1} H Ψ(X) ] )
并对比 单列期望 与 Rayleigh-Ritz。
"""
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
trained_levels = np.sort(np.asarray(history['Energy_levels'][-1]).real)
print(f"训练记录 Energy_levels (最后一步): {trained_levels}")

total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
x = hi.all_states()          # (16, 8)
H = np.asarray(ha.to_dense())  # (16, 16)

# 各列 logψ 与振幅 (全部 16 构型)
logpsi_full = np.stack([
    np.asarray(nnx.merge(
        nnx.split(total_ansatz.single_ansatz_list[j])[0],
        finnal_params['single_ansatz_list'][j],
    )(x))
    for j in range(K)
], axis=1)                     # (16, K)  第 j 列 = logψ_j
psi_full = np.exp(logpsi_full)  # (16, K)  振幅
Hpsi_full = H @ psi_full        # (16, K)  (Hψ_j)(x_i)

# ---------------- 1) 单列期望 (不做行列式) ----------------
S0 = psi_full[:, 0].conj() @ psi_full[:, 0]
M0 = psi_full[:, 0].conj() @ Hpsi_full[:, 0]
print(f"\n[B] 单列期望 <ψ_0|H|ψ_0>/<ψ_0|ψ_0> = {M0/S0.real:.8f}   (FCI E0 = {E_fcis[0]:.8f})")

# ---------------- 2) Rayleigh-Ritz: 广义本征问题 ----------------
M = psi_full.conj().T @ Hpsi_full      # (K,K)
S = psi_full.conj().T @ psi_full       # (K,K)
lam = np.sort(np.linalg.eigvalsh(np.linalg.solve(S, M)))  # S^{-1}M 本征值
print(f"[C] Rayleigh-Ritz 广义本征值: {lam}")

# ---------------- 3) 训练同款: |Ψ_total|² 加权平均 E_L 的本征值 ----------------
# 穷举全部 16^K 个扩展构型 X=(x_1..x_K), x_i ∈ 16 单态
all_idx = np.asarray(list(itertools.product(range(16), repeat=K)))  # (65536, K)
print(f"扩展空间构型数: {all_idx.shape[0]}")

Z = 0.0
E_L_acc = np.zeros((K, K), dtype=np.complex128)
Psi_cache = {}
det_abs2 = np.empty(all_idx.shape[0])
for n, idx in enumerate(all_idx):
    Psi = psi_full[idx, :]          # (K,K)  Psi[i,j] = ψ_j(x_i)
    det = np.linalg.det(Psi)
    det_abs2[n] = abs(det) ** 2
# 一次加权: E_L = solve(Psi, Hpsi)
Z = det_abs2.sum()
for n, idx in enumerate(all_idx):
    w = det_abs2[n]
    if w <= 0:
        continue
    Psi = psi_full[idx, :]
    HPsi = Hpsi_full[idx, :]
    E_L_acc += w * np.linalg.solve(Psi, HPsi)
E_L_mean = E_L_acc / Z
eig_D = np.sort(np.linalg.eigvals(E_L_mean).real)
print(f"[D] 训练同款 |Ψ|²-加权 E_L 本征值: {eig_D}")
print(f"     (训练记录: {trained_levels})")
print(f"     E_L 最大非对角元: {np.abs(E_L_mean - np.diag(np.diag(E_L_mean))).max():.6f}")

print("\n对照:")
print(f"FCI          : {E_fcis}")
