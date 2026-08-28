"""
LiH / STO-3G / K=4 NES-VMC 最小训练脚本（30 轮）
基于 LiH_molecule_K4_STO-3G.ipynb 精简，运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min30.py
训练日志同时输出到控制台与 日志/ 目录。
"""
import logging
import os
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from scipy.sparse.linalg import eigsh
from jax.flatten_util import ravel_pytree

from LiH import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock
from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_gauge_fixed_total_machines,
    create_single_machine_gauge_fixed,
    make_grad_fn,
    make_qgt_fn,
    NESFermionHopRule,
)

# ====================== 日志配置 ======================
logger = logging.getLogger("NES_VMC_K4_min30")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")

os.makedirs("./日志", exist_ok=True)
log_path = "./日志/nes_vmc_0829_K4_LiH_STO-3G_min30.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# ====================== 超参配置 ======================
N_CHAINS = 16                    # Metropolis 链数
N_SAMPLES_PER_CHAIN = 200        # 每条链每步采样数
SWEEP_SIZE = 30
N_ITER = 30                      # 训练轮数（最小化）
Natural_Grad = True              # 是否使用自然梯度（QGT 预条件）
clip_norm = 20.0                 # 全局梯度裁剪上限
lr = 0.1
qgt_diag_shift = 0.1             # QGT 对角正则

# ====================== 模型与采样器 ======================
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)

(
    total_machine,
    total_matrix_machine,
    total_max_machine,
    total_graphdef,
    total_params,
) = create_gauge_fixed_total_machines(total_ansatz, Hatree_Fock)

single_machine_list = [
    create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
    for ansatz in total_ansatz.single_ansatz_list
]

grad_fn = make_grad_fn(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
)
qgt_fn = make_qgt_fn(total_machine)

nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
)

# FCI 精确参考能量
eigvals, _ = eigsh(ha.to_sparse(), k=K, which="SA", tol=1e-10)

# ====================== 优化器 ======================
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)

sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ====================== 训练循环 ======================
logger.info("\n" + "=" * 60)
logger.info(f"开始多链 NES-VMC 训练 | 使用{'自然' if Natural_Grad else '原始'}梯度")
logger.info("=" * 60)
logger.info(
    f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
    f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
)
logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
logger.info(f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}")

start_time = time.time()
for step in range(N_ITER):
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # 2. 梯度与损失
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch)

    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
    grad_update = grad_raw

    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

    # 3. QGT 自然梯度
    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift)
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_norm_natural = grad_norm_raw

    # 4. 优化器内部完成梯度裁剪 + SGD 更新
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    # 裁剪后梯度范数：clip_by_global_norm 会把超限范数压到 clip_norm
    grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

    # 5. 能量监控：E_L 矩阵本征值（排序后作为各态能量估计）
    eig_vals, _ = jnp.linalg.eig(E_L_mean)
    eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

    # 6. 波函数与Ψ矩阵条件数监控
    log_Psi_batch = total_machine(total_params, x_batch)
    x_single = x_batch[0:1, ...]
    psi_mat = total_matrix_machine(total_params, x_single)[0]  # (K,K)
    psi_cond = jnp.linalg.cond(psi_mat)

    # 7. 日志（格式与 GaugeFixing00_K4_LiH_STO-3G.log 对齐）
    logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
    logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
    logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
    logger.info(
        f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
        f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
    )
    logger.info("#-----------------------------------------#")

end_time = time.time()
logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
logger.info("\n" + "=" * 60)
logger.info("训练完成!")
logger.info("=" * 60)