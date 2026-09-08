# -*- coding: utf-8 -*-
"""验证: 训练好的单态 ansatz 参数还原后, 直接计算能量的正确做法

要点:
- SingleStateAnsatz.__call__ 返回的是 log ψ(x) (对数振幅), 不是振幅本身
- 因此用  logψ 直接当 ψ 代入  <ψ|H|ψ>/<ψ|ψ>  是错的
"""
import pickle
import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx

from NES_VMC_V1 import NESTotalAnsatz, SingleStateAnsatz
from H2_631G import SINGLE_SIZE, ha, hi, K, E_fcis

HISTORY_FILE = r'./data/26-09-05-03-48_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)

finnal_params = history['params'][-1]
total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)
single_graphdef, _ = nnx.split(total_ansatz.single_ansatz_list[0])
fine_single_ansatz = nnx.merge(single_graphdef, finnal_params['single_ansatz_list'][0])

x = hi.all_states()          # (16, 8)  全部单态构型
H = ha.to_dense()            # (16, 16) 完整 Fock 空间哈密顿量
logpsi = np.asarray(fine_single_ansatz(x))          # 对数振幅
psi = np.exp(logpsi)                                 # 振幅

# ---- 错误做法 (notebook 中的写法) ----
num_raw = logpsi.conj().T @ H @ logpsi
num_psi = psi.conj().T @ H @ psi
norm_psi = psi.conj().T @ psi

# ---- 正确做法: 振幅参与变分期望 ----
E_correct = num_psi / norm_psi

print("=" * 70)
print("FCI 基准能级:")
for i, e in enumerate(E_fcis):
    print(f"  E{i} = {e:.8f} Ha")
print("=" * 70)
print(f"notebook 写法1: logψ^† H logψ             = {num_raw.real: .8f} Ha")
print(f"notebook 写法2: logψ^† H logψ / ||ψ||²     = {(num_raw / norm_psi).real: .8f} Ha")
print(f"正确做法:      ψ^† H ψ / ||ψ||²           = {E_correct.real: .8f} Ha")
print(f"正确做法误差 vs E0: {abs(E_correct.real - E_fcis[0])*27.2114:.6f} eV")

# 各列 ansatz (激发态) 也用正确做法算一遍
print("=" * 70)
print("正确做法下, 每个单态 ansatz 对应的能量 (K=4):")
for j in range(K):
    graphdef_j, _ = nnx.split(total_ansatz.single_ansatz_list[j])
    ans_j = nnx.merge(graphdef_j, finnal_params['single_ansatz_list'][j])
    psij = np.exp(np.asarray(ans_j(x)))
    Ej = (psij.conj().T @ H @ psij) / (psij.conj().T @ psij)
    print(f"  state {j}: E = {Ej.real: .8f} Ha  (FCI E{j} = {E_fcis[j]:.8f})")
