# -*- coding: utf-8 -*-
"""
基于 验证报告/energy_reconstruction_failure_analysis.md 的正确做法,
重建训练好的 NES-VMC 的 K 个能级解(能量 + 波函数):

    1) 取最终参数, 构造 K 列振幅  ψ_j(x) = exp(logψ_j(x))   (全部 16 构型)
    2) 构造  M_ij = <ψ_i|H|ψ_j>,  S_ij = <ψ_i|ψ_j>
    3) 解广义本征问题  M v = λ S v  → 能级 λ 与混合系数 v
    4) 重建波函数  φ_i = Σ_j v_ij ψ_j, 并直接验证 <φ_i|H|φ_i>/<φ_i|φ_i> = λ_i
    5) 与 FCI 精确本征态对比 (重叠矩阵)
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
x = hi.all_states()          # (16, n_spin)
H = np.asarray(ha.to_dense())  # (16, 16)

# ---------- 1) 各列振幅 ----------
logpsi_full = np.stack([
    np.asarray(nnx.merge(
        nnx.split(total_ansatz.single_ansatz_list[j])[0],
        finnal_params['single_ansatz_list'][j],
    )(x))
    for j in range(K)
], axis=1)                     # (16, K)
psi_full = np.exp(logpsi_full)  # (16, K)

# ---------- 2) M / S ----------
M = psi_full.conj().T @ (H @ psi_full)   # <ψ_i|H|ψ_j>
S = psi_full.conj().T @ psi_full         # <ψ_i|ψ_j>

# ---------- 3) 广义本征问题 ----------
lam, v = eigh(M, S)                      # M v = λ S v
order = np.argsort(lam.real)
lam, v = lam[order], v[:, order]

# ---------- 4) 重建波函数并验证 ----------
phi = psi_full @ v                       # (16, K), 第 i 列 = φ_i
print("=" * 84)
print("重建结果 (最终参数, 精确全空间求和)")
print("=" * 84)
print(f"{'#':<3}{'λ (Rayleigh-Ritz)':>18}{'<φ|H|φ>/<φ|φ> 直接验证':>22}{'FCI':>14}{'误差(eV)':>12}")
for i in range(K):
    E_direct = (phi[:, i].conj().T @ (H @ phi[:, i]) / (phi[:, i].conj() @ phi[:, i])).real
    print(f"{i:<3}{lam[i].real:>18.8f}{E_direct:>22.8f}{E_fcis[i]:>14.8f}{(lam[i].real - E_fcis[i]) * 27.2114:>12.4f}")

# ---------- 5) 与 FCI 本征态的重叠 ----------
_, eigvecs_fci = np.linalg.eigh(H)
print("\n与 FCI 精确本征态的重叠矩阵 |<φ_recon|FCI>|^2 (行=重建态, 列=FCI态):")
overlap = np.abs(phi.conj().T @ eigvecs_fci) ** 2
np.set_printoptions(precision=4, suppress=True)
print(np.round(overlap, 4))

# ---------- 6) 混合系数 (能级对应的波函数怎么由 4 列组成) ----------
print("\n混合系数 v (φ_i = Σ_j v_ij ψ_j, 行 i = 能级):")
print(np.round(np.real(v), 4))

# ---------- 7) 保存重建结果 ----------
out = {
    'levels': lam.real,                 # 重建能级 (Ha)
    'coeffs': v,                        # 混合系数 φ = ψ_full @ v
    'psi_full': psi_full,               # 原始 4 列振幅
    'logpsi_full': logpsi_full,         # 原始 4 列对数振幅
}
np.savez_compressed('reconstructed_solutions.npz', **out)
print("\n已保存: reconstructed_solutions.npz  (levels / coeffs / psi_full / logpsi_full)")
