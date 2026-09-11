# -*- coding: utf-8 -*-
"""
NES-VMC 按化学精度自动冻结（H2 / 6-31G / K=4）
=================================================
在训练循环中**实时监控每个能级的能量**，当某一能级 `i` 达到化学精度（相对精确解
`E_fcis[i]` 的绝对误差 < 阈值）并**连续稳定 N 轮**后，自动冻结该能级对应的那一列
子 ansatz，且一旦冻结即永久冻结（单向、单调递增）。

相比 `NES_VMC_frozen_dynamic.py`（人为用 FREEZE_EVENTS 指定 step->cols），本文件把
「何时冻结」交由化学精度判据自动触发，并把核心难点说清楚：

如何找到「达到化学精度的能级」对应的列？
------------------------------------------------
    能级 i 的判据直接用训练估计的本征值：
        err[i] = |eig_vals[i] - E_fcis[i]| < CHEM_ACC
    其中 eig_vals 已在循环内按能量升序排序，E_fcis 也是升序，故下标一一对应。

    但这里的"列"不是天然指原始列：NES-VMC 的 K 个列 ψ_j 只是 K 维本征子空间的
    任意基（GL(K) 规范自由），单列并不是 H 的本征函数。因此「能级 i ↔ 列 j」的
    对应必须通过在 K 列张成的子空间上解**广义本征问题**确定：

        M v = λ S v,   M_ij = <ψ_i|H|ψ_j>,   S_ij = <ψ_i|ψ_j>
        φ_k  = Σ_j ψ_j v[k,j]     （旋转后的第 k 列即第 k 个本征态）

    旋转系数矩阵 v 的第 i 行给出能级 i 在 K 个原始列上的投影系数 |v[i,j]|。
    我们取系数最大的那一列作为该能级的"主动列"并冻结之：
        col_i = argmax_j |v[i,j]|

    这是本项目 `reconstruct_rotate_eigenstates.ipynb` 确立的"列↔能级"对应方法。
    当前冻结机制按原始列 stop_gradient（无法冻结旋转后的组合列），故冻结
    argmax|v[i,j]| 的原始列是该机制下对能级 i 最优的近似。

本文件依赖的冻结机制（源自 NES_VMC_frozen.py / _dynamic.py）：
    列级 stop_gradient 冻结 + 活跃索引 QGT 裁剪（P^3 → P_a^3）。

运行：
    cd experiments/H2分子 && /opt/miniconda3/envs/Netket/bin/python NES_VMC_auto_freeze.py
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

_LOGGER = logging.getLogger("NES_VMC_K4_H2_auto_freeze")


# ====================== 能级→列 对应：广义本征问题 ======================
def make_column_mapper(total_ansatz, ha_dense, x_all):
    """构造把「能级 index」映射到「原始列 index」的判定器。

    方法（详见 reconstruct_rotate_eigenstates.ipynb）：
        1) 在全部构型上对 K 列求振幅 ψ_j(x)，拼成 (n_states, K) 的 Psi；
        2) 求 M = Psi^H H Psi、S = Psi^H Psi；
        3) 解广义本征问题 M v = λ S v（eigh），v 的列按 λ 升序排；
        4) 返回 callable column_for_level(level, params)：
               列 = argmax_j |v[level, j]|，并附 (λ升序, v, Psi) 供诊断。

    参数：
        total_ansatz : NESTotalAnsatz_stable 实例（取其 graphdef，惰性 merge 参数）
        ha_dense     : `np.asarray(ha.to_dense())`（张成空间上的哈密顿量矩阵）
        x_all        : `np.asarray(hi.all_states(), np.complex64)` 全部单态构型
    """
    graphdefs = [nnx.split(total_ansatz.single_ansatz_list[j])[0]
                 for j in range(K)]
    x_all = jnp.asarray(x_all)
    H = jnp.asarray(ha_dense, dtype=jnp.complex128)

    def column_for_level(level, params, keep=None):
        # 各列在全部构型上的对数振幅 -> 按列最大居中 -> 振幅
        cols = []
        for j in range(K):
            m = nnx.merge(graphdefs[j], params["single_ansatz_list"][j])
            cols.append(m(x_all))
        big_psi = jnp.stack(cols, axis=1)                       # (n_states, K)
        big_psi = big_psi - jnp.max(big_psi, axis=0, keepdims=True)  # 防溢出（列缩放不改变能级）
        Psi = jnp.exp(big_psi)

        HPsi = H @ Psi
        M = Psi.conj().T @ HPsi
        S = Psi.conj().T @ Psi
        M = 0.5 * (M + M.conj().T)
        S = 0.5 * (S + S.conj().T)

        lam, v = jnp.linalg.eigh(M, S)                          # M v = λ S v
        order = jnp.argsort(lam.real)
        lam, v = lam[order], v[:, order]                        # 列按能级升序

        coeff = jnp.abs(v[level, :])                            # 能级 level 在各列上的投影
        col = int(jnp.argmax(coeff))
        if keep is not None:
            keep[0] = (float(lam[level].real), jnp.asarray(coeff))
        return col

    return column_for_level


# ====================== 化学精度判据的自动冻结控制器 ======================
class ChemAccFreezer:
    """当能级能量达到化学精度并稳定 N 轮，自动冻结其对应列。

    - 阈值 CHEM_ACC（Ha）：相对 E_fcis 的绝对误差；化学精度通常 1.6 mHa = 1.6e-3 Ha。
    - strategy（列确定方式）：
        "gauge"    -> 用 column_mapper（广义本征问题）求该能级对应的列（推荐，正确）；
        "assume"   -> 直接用假 "列 j == 能级 j"（按能量升序对应，需列已基本本征化）。
    - _changed 标记用于触发 grad_fn / qgt_fn / idx_active 机器重建。
    """

    def __init__(self, K, E_ref, chem_acc=1.6e-3, stable_rounds=10,
                 strategy="gauge", column_mapper=None):
        self.K = K
        self.E_ref = np.asarray(E_ref).real
        self.chem_acc = chem_acc                      # Ha
        self.stable_rounds = int(stable_rounds)       # 需连续达标轮数
        self.strategy = strategy
        self.column_mapper = column_mapper
        self.frozen_levels = set()                    # 已达化学精度并按此能级冻结
        self.frozen_set = set()                       # 已冻结的列 j
        self.events = []                              # (step, level, col, err, mode)
        self.streak = {i: 0 for i in range(K)}        # 连续达标轮数
        self._changed = False

    @property
    def frozen_mask(self):
        return tuple(j in self.frozen_set for j in range(self.K))

    @property
    def pending_levels(self):
        return [i for i in range(self.K) if i not in self.frozen_levels]

    def decide(self, eig_vals, step, total_params):
        """在循环内每轮调用：返回本轮新增冻结的列列表（可能为空）。

        eig_vals : length K，训练估计的本征值，**已按能量升序**。
        """
        eig = np.asarray(eig_vals).real
        to_freeze = []
        diagnose = {}
        for i in sorted(self.pending_levels):
            err = abs(float(eig[i]) - self.E_ref[i])
            if err < self.chem_acc:
                self.streak[i] += 1
            else:
                self.streak[i] = 0

            if self.streak[i] >= self.stable_rounds:
                # —— 稳定达标：确定该能级对应的列 ——
                if self.strategy == "gauge":
                    col = self.column_mapper(i, total_params)
                    mode = "gauge"
                else:
                    col = int(i)
                    mode = "assume"
                self.frozen_levels.add(i)
                if col not in self.frozen_set:
                    self.frozen_set.add(col)
                    self.events.append((step, i, col, err, mode))
                    self._changed = True
                    to_freeze.append(col)
                    _LOGGER.info(
                        f"[AutoFreeze@{step}] 能级 {i} 达标 err={err:.3e}<{self.chem_acc:.1e} "
                        f"连续 {self.streak[i]} 轮 → 冻结列 {col}（mode={mode}）"
                    )
                diagnose[i] = col
        return to_freeze

    def consume_changed(self):
        c = self._changed
        self._changed = False
        return c


# ====================== 主程序 ======================
if __name__ == "__main__":

    # ====================== 日志配置 ======================
    time_str = time.strftime("%y-%m-%d-%H-%M")
    logger = logging.getLogger("NES_VMC_K4_H2_auto_freeze")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    simple_formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

    os.makedirs("./日志", exist_ok=True)
    log_path = f"./日志/{time_str}_nes_vmc_K4_H2_6-31G_auto_freeze.log"
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
    HISTORY_FILE = f"./data/{time_str}_history_auto_freeze_H2_molecule_K4.pkl"
    os.makedirs("./data", exist_ok=True)

    # ---- 自动冻结判据参数 ----
    CHEM_ACC = 1.6e-3          # 化学精度 1.6 mHa（绝对误差阈值，Ha）
    STABLE_ROUNDS = 10         # 需连续达标多少轮才冻结（"稳定 N 轮"）
    COLUMN_STRATEGY = "gauge"  # "gauge"=广义本征判定列；"assume"=按下标列j==能级j

    exact_eigvals = np.asarray(E_fcis).real

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
            total_ansatz, Hatree_Fock, frozen_mask=frozen_mask)
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

    # ====================== 列判定器（能级→列） ======================
    ha_dense = np.asarray(ha.to_dense(), dtype=np.complex128)   # 张成空间 H 矩阵
    x_all = np.asarray(hi.all_states(), dtype=np.complex64)     # 全部单态构型
    column_mapper = make_column_mapper(total_ansatz, ha_dense, x_all)

    # ====================== 手动冻结基线（可选） ======================
    # 置空 {} 则完全交给化学精度判据自动触发
    MANUAL_FREEZE = {}

    # ====================== 自动冻结控制器 ======================
    auto_freezer = ChemAccFreezer(
        K=K,
        E_ref=exact_eigvals,
        chem_acc=CHEM_ACC,
        stable_rounds=STABLE_ROUNDS,
        strategy=COLUMN_STRATEGY,
        column_mapper=column_mapper,
    )
    optax_v_mapper = None   # 备用

    idx_active = build_active_index(total_params, auto_freezer.frozen_set)
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
    logger.info("开始多链 NES-VMC | 化学精度自动冻结")
    logger.info("=" * 60)
    logger.info(f"精确CAS基准：E_fcis = {np.round(exact_eigvals, 6)} Ha")
    logger.info(f"化学精度阈值 = {CHEM_ACC:.1e} Ha | 连续达标 {STABLE_ROUNDS} 轮冻结 | "
                f"列判定策略 = {COLUMN_STRATEGY}")
    logger.info(f"总变分参数 P = {N_TOTAL}（每能级 {N_TOTAL // K}）")
    logger.info("能级→列判定器已就绪（广义本征 M v = λ S v）")

    import matplotlib
    matplotlib.use("Agg")

    loss_history = []
    steps_history = []
    err_history = {j: [] for j in range(K)}
    n_active_params_history = []
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

        # 5. 能量监控（升序本征值）
        eig_vals, _ = jnp.linalg.eig(E_L_mean)
        eig_vals = eig_vals[jnp.argsort(eig_vals.real)]
        eig_vals_re = np.asarray(eig_vals.real)
        errors_j = np.abs(eig_vals_re - exact_eigvals)

        # ---- 6. 化学精度自动冻结判定 ----
        if COLUMN_STRATEGY == "assume" or auto_freezer.pending_levels:
            # 只在仍有未冻结能级时才判定（否则纯监控）
            auto_freezer.decide(eig_vals, step, total_params)
        for step_ev, cols in MANUAL_FREEZE.items():
            if step == step_ev:
                for c in cols:
                    auto_freezer.frozen_set.add(int(c))
                auto_freezer._changed = True
                logger.info(f"[Manual@{step}] 手动冻结列 {cols}")
        if auto_freezer.consume_changed():
            t0_fz = time.perf_counter()
            grad_fn, qgt_fn = build_grad_machines(auto_freezer.frozen_mask)
            idx_active = build_active_index(total_params, auto_freezer.frozen_set)
            logger.info(
                f"[重建@{step}] frozen_levels={sorted(auto_freezer.frozen_levels)} "
                f"frozen_set={sorted(auto_freezer.frozen_set)} "
                f"| P_a/P = {int(idx_active.size)}/{N_TOTAL} | 重建耗时 "
                f"{time.perf_counter()-t0_fz:.2f}s（含编译，一次性）"
            )

        # 7. 监控
        log_Psi_batch = total_machine(total_params, x_batch, g_current)

        # 8. 日志
        logger.info(f"[Step {step:3d}] logΨmean={float(jnp.real(log_Psi_batch.mean())):.3f} "
                    f"| grad raw={float(grad_norm_raw):.4f} "
                    f"nat={float(grad_norm_natural):.4f}")
        logger.info(
            f"Energy: " + " | ".join(f"E{i}={eig_vals_re[i]:.6f}" for i in range(K))
        )
        logger.info(
            f"err(Ha): " + " | ".join(f"e{i}={errors_j[i]:.2e}"
                                       f"{'●' if auto_freezer.streak[i]>=STABLE_ROUNDS else ''}"
                                       for i in range(K))
        )
        logger.info(
            f"冻结状态 | levels={sorted(auto_freezer.frozen_levels)} "
            f"cols={sorted(auto_freezer.frozen_set)} P_a/P={n_active}/{N_TOTAL}"
        )
        logger.info("#-----------------------------------------#")

        loss_history.append(float(jnp.real(loss_mean)))
        for j in range(K):
            err_history[j].append(float(errors_j[j]))
        steps_history.append(step)
        n_active_params_history.append(n_active)
        frozen_set_history.append(sorted(auto_freezer.frozen_set))
        t_grad_history.append(t_grad)
        t_qgt_history.append(t_qgt)

        if (step + 1) % SAVE_INTERVAL == 0:
            history = {
                "steps": [int(s) for s in steps_history],
                "loss": loss_history,
                "err": err_history,
                "exact": exact_eigvals.tolist(),
                "chem_acc": CHEM_ACC,
                "stable_rounds": STABLE_ROUNDS,
                "frozen_set_history": frozen_set_history,
                "frozen_events": auto_freezer.events,
                "n_active_params": n_active_params_history,
                "frozen_set": sorted(auto_freezer.frozen_set),
                "params": [total_params],
            }
            with open(HISTORY_FILE, "wb") as f:
                pickle.dump(history, f)
            logger.info(f"[保存] Step {step} → pickle 已更新")

        # 9. 周期性 gauge reset
        if (step + 1) % RESET_PERIOD == 0:
            new_col_mean = col_mean_fn(total_params, x_batch)
            g_current = new_col_mean
            logger.info(f"[GaugeReset@{step+1}] 新|g| = "
                        f"{float(jnp.linalg.norm(g_current)):.4f}")

    end_time = time.time()
    logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")

    history = {
        "steps": [int(s) for s in steps_history],
        "loss": loss_history,
        "err": err_history,
        "exact": exact_eigvals.tolist(),
        "chem_acc": CHEM_ACC,
        "stable_rounds": STABLE_ROUNDS,
        "frozen_set_history": frozen_set_history,
        "frozen_events": auto_freezer.events,
        "n_active_params": n_active_params_history,
        "frozen_set": sorted(auto_freezer.frozen_set),
        "params": [total_params],
    }
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"[保存] 训练结束，pickle 最终写入 → {HISTORY_FILE}")

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)