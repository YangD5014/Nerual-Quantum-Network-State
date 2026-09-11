# -*- coding: utf-8 -*-
"""
NES-VMC 动态分段冻结（H2 / 6-31G / K=4）
==========================================
相比 NES_VMC_frozen.py 的"硬编码调度"，本版本允许在训练循环中**动态指定冻结哪些列**，
且**一旦冻结就永久冻结，无法恢复**（单向、单调递增的冻结集合）。

核心接口：`OneWayFreezer`
    freeze_ctrl = OneWayFreezer(K=4)          # 开始时无冻结
    freeze_ctrl.freeze_cols(1)                # 任意时刻调用：永久冻结列 1
    freeze_ctrl.freeze_cols(0, 3)             # 再冻结列 0、3（已冻结的自动跳过）
    changed = freeze_ctrl.consume_changed()   # 若本轮有新增冻结 → 触发机器重建

- 冻结集合只增不减（frozen_set 为 set，freeze_cols 用差集去重），**永不恢复**；
- 每次新增冻结返回 changed=True，主循环据此重建 grad_fn / qgt_fn / idx_active；
- 你可以：
    1) 在循环内直接调用 freeze_ctrl.freeze_cols(...)（真正的"动态指定"）；
    2) 或用 FREEZE_EVENTS 列表（step -> cols）按步触发，作为示例驱动方式。

机制（同 NES_VMC_frozen）：列级 stop_gradient 冻结 + 活跃索引 QGT 裁剪（P^3 → P_a^3）。

运行：
    cd experiments/H2分子 && /opt/miniconda3/envs/Netket/bin/python NES_VMC_frozen_dynamic.py
"""

import logging
import os
import time
import pickle

import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax

from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
    ravel_pytree,
)
from NES_VMC_tool import make_grad_fn_gauge, make_gauge_fn
from NES_VMC_frozen import (
    create_gauge_reset_total_machines,
    make_qgt_fn_gauge_frozen,
    build_active_index,
)
from H2_631G import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock, hi, E_fcis


# ====================== 动态、单向冻结控制器 ======================
_LOGGER = logging.getLogger("NES_VMC_K4_H2_frozen_dynamic")


class OneWayFreezer:
    """双向不可恢复的冻结控制器：随时调用 freeze_cols 冻结任意列；冻结集只增不减。

    - 不保存 step：由调用方在想要的时刻调用（真正的"循环中动态指定"）。
    - freeze_cols(*cols)：把尚未冻结的列加入冻结集；已冻结的自动忽略。
    - consume_changed()：返回并通过 flag —— 本轮是否有新增冻结（用于触发机器重建）。
    """

    def __init__(self, K, frozen_set=None):
        self.K = K
        self.frozen_set = set(frozen_set) if frozen_set else set()
        self.events = []
        self._changed = False

    @property
    def frozen_mask(self):
        return tuple(j in self.frozen_set for j in range(self.K))

    def freeze_cols(self, *cols):
        """动态冻结指定列（可同时冻结多个）。单向：已冻结的不再重复、不可恢复。"""
        new_frozen = set(int(c) for c in cols) - self.frozen_set
        if not new_frozen:
            return self._changed
        for j in sorted(new_frozen):
            if not (0 <= j < self.K):
                raise ValueError(f"非法列索引 {j}（K={self.K}）")
            self.frozen_set.add(j)
            self.events.append(j)
            _LOGGER.info(f"[Freeze-dynamic] 列 {j} 冻结（frozen_set={sorted(self.frozen_set)}）")
        self._changed = True
        return self._changed

    def consume_changed(self):
        """取出"本轮是否有新增冻结"标志，并复位。"""
        c = self._changed
        self._changed = False
        return c


# ====================== 主程序 ======================
if __name__ == "__main__":

    # ====================== 日志配置 ======================
    time_str = time.strftime("%y-%m-%d-%H-%M")
    logger = logging.getLogger("NES_VMC_K4_H2_frozen_dynamic")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

    os.makedirs("./日志", exist_ok=True)
    log_path = f"./日志/{time_str}_nes_vmc_K4_H2_6-31G_frozen_dynamic.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(simple_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # ====================== 超参配置 ======================
    N_CHAINS = 16
    N_SAMPLES_PER_CHAIN = 200
    SWEEP_SIZE = 10
    N_ITER = 300
    Natural_Grad = True
    clip_norm = 20.0
    lr = 0.1
    qgt_diag_shift = 0.1
    RESET_PERIOD = 10
    SAVE_INTERVAL = 20
    HISTORY_FILE = f"./data/{time_str}_history_frozen_dynamic_H2_molecule_K4.pkl"
    os.makedirs("./data", exist_ok=True)

    # ---- 示例驱动方式：FREEZE_EVENTS（step -> 列列表）。可改，或改直接在循环里调 freeze_cols ----
    #    置空 {} 则全程不冻结（只体现控制器能力）。
    FREEZE_EVENTS = {
        120: [1],   # step 120 动态冻结列 1
        200: [3],   # step 200 动态冻结列 3（演示多阶段、单向累积）
    }

    # ====================== 模型与采样器 ======================
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(11),
    )

    g_current = jnp.zeros(K, dtype=jnp.complex64)

    total_machine, total_matrix_machine, total_max_machine, _, total_params = \
        create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)

    single_machine_list = [
        create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]

    def build_grad_machines(frozen_mask):
        tm, tmm, tmmax, _, _ = create_gauge_reset_total_machines(
            total_ansatz, Hatree_Fock, frozen_mask=frozen_mask
        )
        grad_fn = make_grad_fn_gauge(ha, tmm, tmmax, tm, single_machine_list)
        qgt_fn = make_qgt_fn_gauge_frozen(tm)
        return grad_fn, qgt_fn

    grad_fn, qgt_fn = build_grad_machines(frozen_mask=())

    gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    nes_sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext,
        rule=nes_rule,
        n_chains=N_CHAINS,
        sweep_size=SWEEP_SIZE,
    )

    exact_eigvals = np.asarray(E_fcis).real

    # ====================== 动态冻结控制器 ======================
    freeze_ctrl = OneWayFreezer(K=K)
    idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
    N_TOTAL = int(ravel_pytree(total_params)[0].size)

    # ====================== 优化器 ======================
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.sgd(learning_rate=lr),
    )
    opt_state = optimizer.init(total_params)

    sampler_rng = jax.random.PRNGKey(21)

    def sample_machine(params, sigma):
        return total_machine(params, sigma, g_current)

    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    # ====================== 训练循环 ======================
    logger.info("\n" + "=" * 60)
    logger.info("开始多链 NES-VMC | P8 双侧列规范 + gauge reset + 动态单向分段冻结")
    logger.info("=" * 60)
    logger.info(f"精确CAS基准：E0={exact_eigvals[0]:.8f} | E1={exact_eigvals[1]:.8f} | "
                f"E2={exact_eigvals[2]:.8f} | E3={exact_eigvals[3]:.8f} Ha")
    logger.info(f"总变分参数 P = {N_TOTAL}（每能级 {N_TOTAL // K}）")
    logger.info(f"示例动态冻结事件（可用 freeze_ctrl.freeze_cols 任意更改）：{FREEZE_EVENTS}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_history = []
    steps_history = []
    logpsi_mean_history = []
    grad_norm_raw_history = []
    grad_norm_nat_history = []
    Energy_levels_history = []
    err_history = {j: [] for j in range(K)}
    n_active_params_history = []
    qgt_size_history = []
    frozen_set_history = []
    t_grad_history = []
    t_qgt_history = []

    start_time = time.time()

    for step in range(N_ITER):
        # 1. 采样
        samples_raw, sampler_state = nes_sampler.sample(
            machine=sample_machine,
            parameters=total_params,
            state=sampler_state,
            chain_length=N_SAMPLES_PER_CHAIN,
        )
        samples = samples_raw.reshape(-1, hi_ext.size)
        x_batch = samples.reshape(-1, K, SINGLE_SIZE)

        # 2. 梯度与损失
        t0_grad = time.perf_counter()
        grad_raw, loss_mean, E_L_mean, _ = grad_fn(total_params, x_batch, g_current)
        t_grad = time.perf_counter() - t0_grad

        grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
        grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
        grad_update = grad_raw

        has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
        if has_nan or grad_norm_raw > 5000.0:
            logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

        # 3. QGT 自然梯度（活跃子矩阵切片）
        n_active = int(idx_active.size)
        qgt_size = n_active * n_active
        t0_qgt = time.perf_counter()
        if Natural_Grad:
            qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift, g_current, idx_active)
            grad_a = grad_raw_flat[idx_active]
            ng_a = jnp.linalg.solve(qgt_reg_mat, grad_a)
            ng_flat = grad_raw_flat.at[idx_active].set(ng_a)
            grad_update = unravel_fn(ng_flat)
            grad_norm_natural = jnp.linalg.norm(ng_flat)
        else:
            grad_norm_natural = grad_norm_raw
        t_qgt = time.perf_counter() - t0_qgt

        # 4. 优化器更新（冻结列梯度为 0 ⇒ 参数逐位不动）
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

        # 5. 能量监控
        eig_vals, _ = jnp.linalg.eig(E_L_mean)
        eig_vals = eig_vals[jnp.argsort(eig_vals.real)]
        eig_vals_re = np.asarray(eig_vals.real)
        errors_j = np.abs(eig_vals_re - exact_eigvals)
        for j in range(K):
            err_history[j].append(float(errors_j[j]))

        # ---- 6. 动态冻结（单向）：先按 FREEZE_EVENTS 驱动，再用 consume_changed 触发重建 ----
        if step in FREEZE_EVENTS:
            freeze_ctrl.freeze_cols(*FREEZE_EVENTS[step])
        if freeze_ctrl.consume_changed():
            t0_fz = time.perf_counter()
            grad_fn, qgt_fn = build_grad_machines(freeze_ctrl.frozen_mask)
            idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
            logger.info(
                f"[重建@{step}] frozen_set={sorted(freeze_ctrl.frozen_set)} "
                f"| P_a/P = {int(idx_active.size)}/{N_TOTAL} "
                f"| 重建耗时 {time.perf_counter()-t0_fz:.2f}s（含编译，一次性）"
            )

        # 7. 监控
        log_Psi_batch = total_machine(total_params, x_batch, g_current)
        x_single = x_batch[0:1, ...]
        psi_mat = total_matrix_machine(total_params, x_single, g_current)[0]
        psi_cond = jnp.linalg.cond(psi_mat)
        gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))
        g_abs = float(jnp.linalg.norm(g_current))

        # 8. 日志
        logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f}")
        logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm}) | t_grad={t_grad*1e3:.0f}ms t_qgt={t_qgt*1e3:.0f}ms")
        logger.info(f"冻结状态 | frozen_set={sorted(freeze_ctrl.frozen_set)} | P_a/P={n_active}/{N_TOTAL} | QGT={qgt_size} | 本轮实际训练参数={n_active}")
        logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e} | 累计|g| = {g_abs:.4f}")
        logger.info(
            f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
            f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
        )
        logger.info(
            f"误差(Ha): " + " | ".join(f"e{j}={errors_j[j]:.2e}" for j in range(K))
        )
        logger.info("#-----------------------------------------#")

        loss_history.append(float(jnp.real(loss_mean)))
        logpsi_mean_history.append(float(jnp.real(log_Psi_batch.mean())))
        steps_history.append(step)
        grad_norm_raw_history.append(float(grad_norm_raw))
        grad_norm_nat_history.append(float(grad_norm_natural))
        Energy_levels_history.append(eig_vals[:K])
        n_active_params_history.append(n_active)
        qgt_size_history.append(qgt_size)
        frozen_set_history.append(sorted(freeze_ctrl.frozen_set))
        t_grad_history.append(t_grad)
        t_qgt_history.append(t_qgt)

        if (step + 1) % SAVE_INTERVAL == 0:
            history = {
                "steps": [int(s) for s in steps_history],
                "logpsi_mean": logpsi_mean_history,
                "grad_norm_raw": grad_norm_raw_history,
                "grad_norm_natural": grad_norm_nat_history,
                "loss": loss_history,
                "save_interval": SAVE_INTERVAL,
                "Energy_levels": Energy_levels_history,
                "err": err_history,
                "n_active_params": n_active_params_history,
                "qgt_size": qgt_size_history,
                "frozen_set_history": frozen_set_history,
                "frozen_set": sorted(freeze_ctrl.frozen_set),
                "params": [total_params],
            }
            with open(HISTORY_FILE, "wb") as f:
                pickle.dump(history, f)
            logger.info(f"[保存] Step {step} → pickle 已更新 ({len(steps_history)} 条记录)")

        # 9. 周期性 gauge reset
        if (step + 1) % RESET_PERIOD == 0:
            new_col_mean = col_mean_fn(total_params, x_batch)
            g_current = new_col_mean
            absorbed = float(jnp.sum(jnp.real(new_col_mean)))
            logger.info(
                f"[GaugeReset@{step+1}] 吸收 Δg = Σ Re(col_mean) = {absorbed:+.4f} "
                f"| 新|g| = {float(jnp.linalg.norm(g_current)):.4f}"
            )

    end_time = time.time()
    logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")

    history = {
        "steps": [int(s) for s in steps_history],
        "logpsi_mean": logpsi_mean_history,
        "grad_norm_raw": grad_norm_raw_history,
        "grad_norm_natural": grad_norm_nat_history,
        "loss": loss_history,
        "save_interval": SAVE_INTERVAL,
        "Energy_levels": Energy_levels_history,
        "err": err_history,
        "n_active_params": n_active_params_history,
        "qgt_size": qgt_size_history,
        "frozen_set_history": frozen_set_history,
        "frozen_set": sorted(freeze_ctrl.frozen_set),
        "params": [total_params],
    }
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"[保存] 训练结束，pickle 最终写入 → {HISTORY_FILE}")

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)