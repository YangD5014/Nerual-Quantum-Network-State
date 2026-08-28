"""
LiH / STO-3G / K=4 NES-VMC 最小训练脚本（30 轮）——diag_shift 放大版（P3 方案）

核心思路（与基线的唯一区别在自然梯度步）：
    基线中 logΨ mean 漂移 = 行规范坐标 G = mean_batch Σ_i row_mean_i 的噪声驱动随机游走，
    通过 QGT 的 diag_shift=0.1 被放大（S 沿 G 方向近似零模 → (S+0.1I)^{-1} ≈ 1/0.1 放大）。
    由于 loss 对 G 严格不变（规范不变性），沿 G 方向加一个 rank-1 脊：
        S' = S + diag_shift·I + μ · vv^H/(v^H v),   v = ∇_θ G   （holomorphic 梯度）
    把该模式的逆增益从 1/diag_shift 压低到 ~1/μ，而其它方向的优化动力学完全不变
    （v 方向的 loss 是平坦的，加脊不改变能量景观与极值）。

    - 采样、grad_fn、loss 路径：与原版 baseline 完全一致（不动值、不动梯度结构）
    - 只改 QGT 预条件子 → 物理轨迹应与基线几乎一致，但 G 不再漂移

运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min30_gaugepin.py
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


def make_gauge_fn(total_model, ref_state):
    """
    行规范坐标函数（与 create_gauge_fixed_total_machines 内部 L 完全一致）：
        G(params, x_batch) = mean_batch Σ_i row_mean_i,  row_mean_i = (1/K)Σ_j L[i,j]
        L[i,j] = logψ_j(x_i) - logψ_j(ref)   （列规范）
    该坐标的梯度 v = ∇_θ G 即行规范漂移方向（loss 对其不变）。
    """
    graphdef, state = nnx.split(total_model)
    K_ = total_model.K
    n_spin = total_model.n_spin
    ref_state = jnp.asarray(ref_state)

    def _gauge_single(m, x_single):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            cols.append(ansatz_j(x_single) - ansatz_j(ref_state))
        L = jnp.stack(cols, axis=1)          # (K, K) 列规范
        row_mean = jnp.mean(L, axis=1)       # (K,)
        return jnp.sum(row_mean)             # 行规范坐标（复数）

    @jax.jit
    def gauge_fn(params, x_batch):
        m = nnx.merge(graphdef, params)
        gs = jax.vmap(lambda x: _gauge_single(m, x))(x_batch)  # (n,)
        return jnp.mean(gs)

    return gauge_fn


# ====================== 日志配置 ======================
logger = logging.getLogger("NES_VMC_K4_diagshift")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")

os.makedirs("./日志", exist_ok=True)
log_path = "./日志/nes_vmc_0829_K4_LiH_STO-3G_min30_diagshift.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# ====================== 超参配置 ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER = 30
Natural_Grad = True
clip_norm = 20.0
lr = 0.1
qgt_diag_shift = 1.0
mu_ridge = 0.0             # 不使用规范脊，纯测 diag_shift 效果

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

# 行规范坐标函数与梯度
gauge_fn = make_gauge_fn(total_ansatz, Hatree_Fock)
grad_gauge_fn = jax.grad(gauge_fn, holomorphic=True)

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
logger.info("开始多链 NES-VMC 训练 | diag_shift 放大版（P3 方案）")
logger.info("=" * 60)
logger.info(
    f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
    f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
)
logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
logger.info(
    f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}, "
    f"gauge_ridge μ={mu_ridge}"
)

start_time = time.time()
gauge_init = None
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

    # 3. QGT 自然梯度（含规范方向脊）
    if Natural_Grad:
        ridge = 0.0
        if mu_ridge > 0:
            v_tree = grad_gauge_fn(total_params, x_batch)
            v_flat, _ = ravel_pytree(v_tree)
            vHv = jnp.real(jnp.vdot(v_flat, v_flat)) + 1e-12
            ridge = mu_ridge * (jnp.outer(v_flat, jnp.conj(v_flat)) / vHv)

        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift) + ridge
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_norm_natural = grad_norm_raw

    # 4. 优化器内部完成梯度裁剪 + SGD 更新
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

    # 5. 能量监控
    eig_vals, _ = jnp.linalg.eig(E_L_mean)
    eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

    # 6. 波函数 / Ψ 条件数 / 行规范坐标监控
    log_Psi_batch = total_machine(total_params, x_batch)
    x_single = x_batch[0:1, ...]
    psi_mat = total_matrix_machine(total_params, x_single)[0]
    psi_cond = jnp.linalg.cond(psi_mat)

    gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))
    if gauge_init is None:
        gauge_init = gauge_now

    # 7. 日志（格式与基线对齐 + gauge 监控）
    logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
    logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
    logger.info(f"行规范坐标 G(Re Σrow_mean) = {gauge_now:+.4f} | ΔG from init = {gauge_now - gauge_init:+.4f}")
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