# -*- coding: utf-8 -*-
"""
基于训练好的 NES-VMC 总波函数重建能级能量。

背景:
    总波函数 Psi(X) = det[ psi_j(x_i) ], 即 K 个"列" ansatz 构成 K×K 矩阵的行列式。
    训练只保证"扩展空间局域能量矩阵 E_L(X) = Psi(X)^{-1} H Psi(X)"的本征值 = 能级,
    并不保证单列 <psi_j|H|psi_j>/<psi_j|psi_j> 等于能级。

    因此要由训练好的参数重建能级, 应求解由 K 列展开的 K 维子空间上的广义本征问题
    (Rayleigh-Ritz):
        M v = λ S v
        M_ij = <ψ_i | H | ψ_j>,   S_ij = <ψ_i | ψ_j>
    本征值 {λ_k} 即重建的 K 个能级, 本征向量 v 给出能级波函数 φ = Σ_j v_j ψ_j。

    说明: 训练中的列规范(gauge)只是给每列乘一个常数, 而广义本征值对列缩放不变,
          因此不受 gauge reset / 偏移影响。

本脚本针对 gauge_reset 训练的历史文件:
    data/26-09-08-18-09_history_natural_gradient_H2_molecule_K4.pkl
"""
import pickle
import numpy as np
import flax.nnx as nnx
from scipy.linalg import eigh

from NES_VMC_V1 import NESTotalAnsatz
from H2_631G import SINGLE_SIZE, ha, hi, K, E_fcis

# ========================= 配置 =========================
HISTORY_FILE = r'./data/26-09-08-18-09_history_natural_gradient_H2_molecule_K4.pkl'
USE_LAST_N = 1            # 用最后 N 组参数各重建一次, 取平均/逐个打印
# ========================================================

with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)

x = np.asarray(hi.all_states(), dtype=np.complex64)   # (16, n_spin)
H = np.asarray(ha.to_dense())                          # (16, 16) 物理哈密顿量(单子系统)
n_steps = len(history['params'])
print(f"历史步数: {n_steps}, 使用最后 {USE_LAST_N} 组参数重建")

# 常量参考: 与训练同构的 plain NESTotalAnsatz 空模型, 用于拿 graphdef
total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
graphdefs = [nnx.split(total_ansatz.single_ansatz_list[j])[0] for j in range(K)]

recon_levels_all = []

print("\n" + "=" * 92)
print(f"{'步':<5}{'λ0':>14}{'λ1':>14}{'λ2':>14}{'λ3':>14}{'|λ0-FCI0|(eV)':>14}")
print("-" * 92)
for step in range(n_steps - USE_LAST_N, n_steps):
    params = history['params'][step]

    # ---- 1) 每列在全部 16 构型上的振幅 ψ_j(x) = exp(logψ_j(x)) ----
    psi_cols = []
    for j in range(K):
        ans_j = nnx.merge(graphdefs[j], params['single_ansatz_list'][j])
        psi_cols.append(np.exp(np.asarray(ans_j(x), dtype=np.complex128)))
    Psi = np.stack(psi_cols, axis=1)          # (16, K)

    # ---- 2) M / S 矩阵 ----
    HPsi = H @ Psi
    M = Psi.conj().T @ HPsi                    # M_ij = <ψ_i|H|ψ_j>
    S = Psi.conj().T @ Psi                     # S_ij = <ψ_i|ψ_j>
    S = 0.5 * (S + S.conj().T)                 # 强制 Hermitian (去浮点误差)
    M = 0.5 * (M + M.conj().T)

    # ---- 3) 广义本征问题 M v = λ S v ----
    lam, v = eigh(M, S)
    order = np.argsort(lam.real)
    lam, v = lam[order], v[:, order]

    recon_levels = lam.real
    recon_levels_all.append(recon_levels)

    err0 = (recon_levels[0] - E_fcis[0]) * 27.2114
    print(f"{step:<5}"
          + "".join(f"{e:>14.6f}" for e in recon_levels)
          + f"{err0:>14.4f}")

# ========================= 汇总 =========================
levels = np.mean(recon_levels_all, axis=0)
print("\n" + "=" * 92)
print("最终结果 (取最后 N 组参数重建的平均)")
print("=" * 92)
print(f"{'能级':<8}{'重建 λ (Ha)':>18}{'FCI (Ha)':>16}{'误差 (meV)':>12}")
print("-" * 92)
for i in range(K):
    err_meV = (levels[i] - E_fcis[i]) * 1000
    print(f"{i:<8}{levels[i]:>18.8f}{E_fcis[i]:>16.8f}{err_meV:>12.3f}")

# ---- 最后一组参数, 重建波函数并验证重叠 ----
params = history['params'][-1]
psi_cols = []
for j in range(K):
    ans_j = nnx.merge(graphdefs[j], params['single_ansatz_list'][j])
    psi_cols.append(np.exp(np.asarray(ans_j(x), dtype=np.complex128)))
Psi = np.stack(psi_cols, axis=1)
HPsi = H @ Psi
M = Psi.conj().T @ HPsi
S = Psi.conj().T @ Psi
M = 0.5 * (M + M.conj().T)
S = 0.5 * (S + S.conj().T)
lam, v = eigh(M, S)
order = np.argsort(lam.real)
lam, v = lam[order], v[:, order]
phi = Psi @ v
_, eigvecs_fci = np.linalg.eigh(H)
overlap = np.abs(phi.conj().T @ eigvecs_fci) ** 2
print("\n重建态与 FCI 本征态的重叠矩阵 |<φ_recon|FCI>|² (行=重建, 列=FCI):")
np.set_printoptions(precision=4, suppress=True)
print(np.round(overlap, 4))

out = {
    'levels': levels,
    'levels_last': lam.real,
    'psi_full': Psi,
    'coeffs_last': v,
}
np.savez_compressed('reconstructed_levels_gauge_reset.npz', **out)
print("\n已保存: reconstructed_levels_gauge_reset.npz")