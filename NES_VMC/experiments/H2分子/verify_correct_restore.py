# -*- coding: utf-8 -*-
"""
复原训练好的 NES-VMC ansatz: 三种能量计算方式对比

关键事实:
1. SingleStateAnsatz.__call__ 返回 log ψ(x) (对数振幅), 不是振幅 ψ(x)。
2. NES 训练的对象是"扩展空间上的行列式波函数" Ψ(X) = det[ψ_j(x_i)],
   能量级别是 K×K 局域能量矩阵 E_L(X) = Ψ(X)^{-1} H Ψ(X) 的**本征值**,
   不是某个单列 <ψ_j|H|ψ_j>/<ψ_j|ψ_j>。
3. 因此单列复原(哪怕正确取 exp)也不会等于训练得到的能级。
"""
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from scipy.linalg import eigh

from NES_VMC_V1 import NESTotalAnsatz, create_single_machine_gauge_fixed
from NES_VMC_tool import create_gauge_reset_total_machines, NES_loss_energy_stable_gauge
from H2_631G import SINGLE_SIZE, ha, hi, K, E_fcis

HISTORY_FILE = r'./data/26-09-05-03-48_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)

finnal_params = history['params'][-1]
trained_levels = np.asarray(history['Energy_levels'][-1])  # 训练记录的能级(本征值)
print(f"训练记录 Energy_levels(最后一步): {trained_levels.real}")

total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
x = hi.all_states()          # (16, 8)
H = ha.to_dense()            # (16, 16)

# 构造所有单态机器 (gauge-fixed: logψ_j(x) - logψ_j(ref), 每列常数, 不影响能级)
ref_state = hi.all_states()[0]
single_machine_list = []
for j in range(K):
    graphdef_j, _ = nnx.split(total_ansatz.single_ansatz_list[j])
    params_j = finnal_params['single_ansatz_list'][j]
    ans_j = nnx.merge(graphdef_j, params_j)
    machine_j, _, _ = create_single_machine_gauge_fixed(ans_j, ref_state)
    single_machine_list.append(machine_j)

# 各列的振幅 (在全部 16 个构型上)
psi_cols = []
for j in range(K):
    graphdef_j, _ = nnx.split(total_ansatz.single_ansatz_list[j])
    ans_j = nnx.merge(graphdef_j, finnal_params['single_ansatz_list'][j])
    psi_cols.append(np.exp(np.asarray(ans_j(x))))
Psi = np.stack(psi_cols, axis=1)   # (16, K)  第 j 列 = ψ_j

# ---------------- 做法 A: notebook 的写法 (错) ----------------
logpsi = np.log(Psi)
num_raw = logpsi.conj().T @ H @ logpsi
norm_psi = Psi.conj().T @ Psi
E_A1 = num_raw[0, 0]                      # cell 9: logψ^† H logψ
E_A2 = num_raw[0, 0] / norm_psi[0, 0]     # cell 12: 混用 logψ 与 exp 的归一化

# ---------------- 做法 B: 正确取 exp 的单列期望 (仍不对) ----------------
E_B = np.diag(Psi.conj().T @ H @ Psi) / np.diag(norm_psi)

# ---------------- 做法 C: 正确做法——K×K 广义本征问题 (Rayleigh-Ritz) ----------------
M = Psi.conj().T @ H @ Psi          # (K,K)  <ψ_i|H|ψ_j>
S = norm_psi                        # (K,K)  <ψ_i|ψ_j>
lam, _ = eigh(M, S)                 # M v = λ S v
E_C = np.sort(lam)

# ---------------- 做法 D: 训练同款 —— 扩展空间局域能量矩阵 E_L 的本征值 ----------------
# 用训练同款 machinery (gauge=0 不影响本征值, 只是列缩放)
total_machine, total_matrix_machine, total_max_machine, total_matrix_machine_raw, gdef, st = \
    create_gauge_reset_total_machines(total_ansatz, ref_state)
g = jnp.zeros(K, dtype=jnp.complex64)

# 取一个"好样本": 每副本用 HF 态附近
x_ext = jnp.stack([hi.all_states()[i] for i in range(K)])  # (K, n_spin) 一个 walker
L = np.asarray(total_matrix_machine(finnal_params, x_ext, g))
Psi_est = np.exp(L)                    # (K,K)
E_L = np.asarray(NES_loss_energy_stable_gauge(
    ha, total_matrix_machine, total_max_machine, single_machine_list,
    finnal_params, x_ext, g, return_aux=False,
)[1])
E_D = np.sort(np.linalg.eigvals(E_L))

# 检查 E_L 的对角化程度 (说明单列不是本征函数)
offdiag = np.abs(E_L - np.diag(np.diag(E_L))).max()

print("\n" + "=" * 78)
print(f"{'方法':<52}{'E0':>12}{'E1':>12}{'E2':>12}{'E3':>12}")
print("-" * 78)
print(f"{'FCI 基准':<52}{E_fcis[0]:>12.6f}{E_fcis[1]:>12.6f}{E_fcis[2]:>12.6f}{E_fcis[3]:>12.6f}")
print(f"{'A1 notebook cell9: logψ†H logψ (scalar)':<52}{E_A1.real:>12.6f}")
print(f"{'A2 notebook cell12: 混用归一化':<52}{E_A2.real:>12.6f}")
print(f"{'B  单列期望 exp(logψ)  (state0):':<52}{E_B[0]:>12.6f}{E_B[1]:>12.6f}{E_B[2]:>12.6f}{E_B[3]:>12.6f}")
print(f"{'C  广义本征问题 Mv=λSv (Rayleigh-Ritz)':<52}{E_C[0]:>12.6f}{E_C[1]:>12.6f}{E_C[2]:>12.6f}{E_C[3]:>12.6f}")
print(f"{'D  训练同款 E_L 本征值 (单样本)':<52}{E_D[0].real:>12.6f}{E_D[1].real:>12.6f}{E_D[2].real:>12.6f}{E_D[3].real:>12.6f}")
print(f"{'训练记录 Energy_levels (最后一步)':<52}{trained_levels[0].real:>12.6f}{trained_levels[1].real:>12.6f}{trained_levels[2].real:>12.6f}{trained_levels[3].real:>12.6f}")
print("-" * 78)
print(f"E_L 矩阵最大非对角元 = {offdiag:.6f}  (越接近 0, 说明各列越接近本征函数)")

print("\n说明: 做法 C 是在全部 16 个构型上精确求 M/S (无蒙特卡洛噪声),")
print("      是'复原参数后求能级'的正确且最准的方式。做法 D 是训练同款(有采样噪声)。")
