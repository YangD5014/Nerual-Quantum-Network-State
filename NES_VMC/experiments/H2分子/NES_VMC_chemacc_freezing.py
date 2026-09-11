# -*- coding: utf-8 -*-
"""
NES-VMC 化学精度阈值动态冻结（H2 / 6-31G / K=4）
==================================================
当某能级**连续 NEED_STABLE_ROUNDS 轮**达到化学精度（|E_i - E_fcis_i| < CHEM_ACC）时，
自动确定该能级对应的"主动列"并永久冻结。

设计要点
--------
1) 动态冻结控制：`OneWayFreezer`（单向、只增不减），任意时刻调用 freeze_cols，配合
   consume_changed() 触发机器/QGT 重建。本脚本用"化学精度达标"作为自动触发源。
2) 判别标准（CRITERION 可切换）：
     "chemacc"         : |E_i - E_fcis_i| < CHEM_ACC 连续 NEED_STABLE 轮（默认；需 FCI 精确解）
     "chemacc_grad"    : 上述 且 该列梯度范数 < GRAD_TOL（防能量虽低但仍在剧烈变化）
     "energy_plateau"  : 最近 PLATEAU_W 轮能量极差 < PLATEAU_EPS —— 纯自监督、无需 E_fcis
   注：化学精度本身是"额外信息"（依赖精确解 E_fcis）；该默认选项之外，其余标准要么自监督、
       要么只用训练内部量，作为不依赖精确解的探索选项。
3) 能级 -> 列 的对应：解 K×K 广义本征问题 Mv=λSv（基于 MCMC 样本重建 M,S），
   取第 i 行 argmax_j |v[i,j]| 即"能级 i 对应的主动列"。M,S,v 全部 K×K。
4) 冻结机制：列级 stop_gradient 剪枝 + QGT 活跃索引切片（P³→P_a³）。冻结列参数永远不动，
   其余列继续用自然梯度更新（冻结列不影响它们达成正交性的耦合——见文档）。

运行：
    cd experiments/H2分子 && /opt/miniconda3/envs/Netket/bin/python NES_VMC_chemacc_freezing.py
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

from scipy.linalg import eigh

from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_single_machine_gauge_fixed,
    Ham_Psi_scaled,
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


_LOGGER = logging.getLogger("NES_VMC_K4_H2_chemacc_freeze")


# ====================== 1. 单向冻结控制器 ======================
class OneWayFreezer:
    """动态、单向、不可恢复的冻结控制器。

    - freeze_cols(*cols)：把尚未冻结的列加入冻结集（已冻结自动忽略，永不恢复）。
    - frozen_mask：长度 K 的布尔元组，供重建工厂（静态 trace 期常量）使用。
    - consume_changed()：取出"本轮是否有新增冻结"标志并复位（触发机器/QGT 重建）。
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
        new_frozen = set(int(c) for c in cols) - self.frozen_set
        if not new_frozen:
            return self._changed
        for j in sorted(new_frozen):
            if not (0 <= j < self.K):
                raise ValueError(f"非法列索引 {j}（K={self.K}）")
            self.frozen_set.add(j)
            self.events.append(j)
            _LOGGER.info(f"[Freeze] 列 {j} 冻结（frozen_set={sorted(self.frozen_set)}）")
        self._changed = True
        return self._changed

    def consume_changed(self):
        c = self._changed
        self._changed = False
        return c


# ====================== 2. 能级 -> 主动列 的映射 ======================
def compute_level_to_col(single_machine_list, total_params, x_batch):
    """用 MCMC 样本解 K×K 广义本征问题 Mv=λSv，得到"能级->列"旋转系数。

    返回 (v, n_used):
        v       : K×K 复数矩阵，行=能级（能量升序），列=原始列。
                  能级 i 对应的主动列 = argmax_j |v[i,j]|。
        n_used  : 参与统计的有效 walker 数。
    全程只出现 K×K 矩阵，不触碰完整 16×16 的 H，也不做精确对角化。
    """
    N = x_batch.shape[0]
    # P_{n α β} = ψ_β(x_{n α})：逐列拟设算到每个 walker 的 K 个副本构型
    logP_cols = [
        single_machine_list[j](total_params["single_ansatz_list"][j], x_batch)
        for j in range(K)
    ]
    logP = jnp.stack(logP_cols, axis=-1)                      # (N, α, β)
    shift_n = jnp.max(logP.real, axis=(1, 2)).reshape(-1)     # 每 walker 公共安全标度
    P = jnp.exp(logP - shift_n.reshape(-1, 1, 1))             # (N, K, K)，有限

    # H·ψ：局部能量算子作用到被采样构型（不构造 16×16 H）
    HP = Ham_Psi_scaled(
        ha=ha, single_machine_list=single_machine_list,
        total_params=total_params, x=x_batch, shift=shift_n,
    )                                                         # (N, K, K)

    finite = (
        jnp.all(jnp.isfinite(P), axis=(-2, -1))
        & jnp.all(jnp.isfinite(HP), axis=(-2, -1))
    )
    Pn = jnp.where(finite.reshape(-1, 1, 1), P, 0.0)
    HPn = jnp.where(finite.reshape(-1, 1, 1), HP, 0.0)
    norm = jnp.sqrt(jnp.sum(jnp.abs(Pn) ** 2, axis=(1, 2)) + 1e-30).reshape(-1, 1, 1)
    Pn = Pn / norm
    HPn = HPn / norm

    n_used = int(finite.sum())
    N_ = max(n_used, 1)
    Shat = (jnp.einsum("nαi,nαj->ij", jnp.conj(Pn), Pn) / N_).astype(jnp.complex128)
    Mhat = (jnp.einsum("nαi,nαj->ij", jnp.conj(Pn), HPn) / N_).astype(jnp.complex128)
    Shat = 0.5 * (Shat + Shat.conj().T)
    Mhat = 0.5 * (Mhat + Mhat.conj().T)

    lam, v = eigh(np.asarray(Mhat), np.asarray(Shat))
    order = np.argsort(lam.real)
    lam, v = lam[order], v[:, order]
    return np.asarray(v), n_used


def level_to_active_col(v, level):
    """能级 level 对应的主动列：旋转矩阵第 level 行绝对值最大的列。"""
    return int(np.argmax(np.abs(v[level])))


# ====================== 3. 冻结判据 ======================
def col_grad_norms(grad_raw):
    """每列变分梯度的 Ravel 范数（用于"chemacc_grad"判据）。"""
    norms = []
    for j in range(K):
        flat, _ = ravel_pytree(grad_raw["single_ansatz_list"][j])
        norms.append(float(jnp.linalg.norm(flat)))
    return norms


def energy_plateau_reached(series, window, eps):
    """最近 window 轮能量极差 < eps（纯自监督判据）。"""
    tail = series[-window:]
    if len(tail) < window:
        return False
    return bool(float(np.max(tail) - np.min(tail)) < eps)


# ====================== 主程序 ======================
if __name__ == "__main__":
    _LOGGER = logging.getLogger("NES_VMC_K4_H2_chemacc_freeze")

    # ----- 日志 -----
    time_str = time.strftime("%y-%m-%d-%H-%M")
    logger = logging.getLogger("NES_VMC_K4_H2_chemacc_freeze")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
    os.makedirs("./日志", exist_ok=True)
    log_path = f"./日志/{time_str}_nes_vmc_K4_H2_chemacc_freeze.log"
    logger.addHandler(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(simple_formatter)

    # ----- 超参 -----
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
    HISTORY_FILE = f"./data/{time_str}_history_chemacc_freeze_H2_K4.pkl"
    os.makedirs("./data", exist_ok=True)

    # ----- 冻结判据配置 -----
    CHEM_ACC = 1.6e-3          # 化学精度 1 kcal/mol ≈ 1.6 mHa（需要的"额外信息"）
    NEED_STABLE = 5            # 连续达标轮数（"稳定 N 轮"）
    CRITERION = "chemacc"      # chemacc | chemacc_grad | energy_plateau
    GRAD_TOL = 5.0             # chemacc_grad 用：该列梯度范数阈值
    PLATEAU_W = 10             # energy_plateau 用：窗口
    PLATEAU_EPS = 1e-3         # energy_plateau 用：能量极差阈值

    exact = np.asarray(E_fcis).real

    # ----- 模型 / 采样器 -----
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE + K,
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
        hilbert=hi_ext, rule=nes_rule, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE,
    )

    # ----- 控制器 / 索引 -----
    freeze_ctrl = OneWayFreezer(K=K)
    idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
    N_TOTAL = int(ravel_pytree(total_params)[0].size)
    frozen_levels = set()          # 已触发冻结的"能级"（与列可能不同）
    stable_rounds = {i: 0 for i in range(K)}
    level_to_col = {i: None for i in range(K)}

    # ----- 优化器 -----
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.sgd(learning_rate=lr),
    )
    opt_state = optimizer.init(total_params)
    sampler_rng = jax.random.PRNGKey(21)

    def sample_machine(params, sigma):
        return total_machine(params, sigma, g_current)

    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    # ----- 训练循环 -----
    logger.info("=" * 60)
    logger.info("| P8 双侧列规范 + gauge reset | 化学精度阈值动态冻结 (单向)")
    logger.info(f"CRITERION={CRITERION} | CHEM_ACC={CHEM_ACC:.4f} | NEED_STABLE={NEED_STABLE}")
    logger.info(f"精确基准 E0..E3 = {np.round(exact, 6)}")
    logger.info(f"总参数 P = {N_TOTAL}（每列 {N_TOTAL//K}）")

    logger.info("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_history, Energy_levels_history = [], []
    err_history = {i: [] for i in range(K)}
    frozen_set_history = []
    level_to_col_history = []

    start_time = time.time()
    for step in range(N_ITER):
        # 1) 采样
        samples_raw, sampler_state = nes_sampler.sample(
            machine=sample_machine, parameters=total_params, state=sampler_state,
            chain_length=N_SAMPLES_PER_CHAIN,
        )
        samples = samples_raw.reshape(-1, hi_ext.size)
        x_batch = samples.reshape(-1, K, SINGLE_SIZE)     # 整数构型 (N,K,8)

        # 2) 梯度与损失（E_L_mean 本征值 = 训练能级估计）
        grad_raw, loss_mean, E_L_mean, _ = grad_fn(total_params, x_batch, g_current)
        grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)

        # 3) QGT 自然梯度（活跃子矩阵 P_a×P_a）
        n_active = int(idx_active.size)
        if Natural_Grad:
            qgt_reg = qgt_fn(total_params, x_batch, qgt_diag_shift, g_current, idx_active)
            grad_a = grad_raw_flat[idx_active]
            ng_a = jnp.linalg.solve(qgt_reg, grad_a)
            ng_flat = grad_raw_flat.at[idx_active].set(ng_a)
            grad_update = unravel_fn(ng_flat)
        else:
            grad_update = grad_raw
        # 4) 优化更新（冻结列梯度=0 ⇒ 参数不动）
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        # 5) 能级与误差
        eig_vals, _ = jnp.linalg.eig(E_L_mean)
        eig_vals = eig_vals[jnp.argsort(eig_vals.real)]
        eig_vals_re = np.asarray(eig_vals.real)
        errors_i = np.abs(eig_vals_re - exact)

        # 6) 判据 & 稳定计数 & 触发冻结
        need_col_norms = CRITERION == "chemacc_grad"
        col_norms = col_grad_norms(grad_raw) if need_col_norms else None
        for i in range(K):
            if i in frozen_levels:
                continue
            if CRITERION == "chemacc":
                met = errors_i[i] < CHEM_ACC
            elif CRITERION == "chemacc_grad":
                met = (errors_i[i] < CHEM_ACC) and (col_norms[i] < GRAD_TOL)
            elif CRITERION == "energy_plateau":
                met = energy_plateau_reached(
                    [e.real for e in Energy_levels_history] + [eig_vals_re[i]], PLATEAU_W, PLATEAU_EPS
                )
            else:
                raise ValueError(f"未知 CRITERION={CRITERION}")
            stable_rounds[i] = stable_rounds[i] + 1 if met else 0
            if stable_rounds[i] >= NEED_STABLE:
                # 能级 -> 列：解 K×K 广义本征问题（按需，仅当有冻结需求时）
                try:
                    v_map, n_used = compute_level_to_col(single_machine_list, total_params, x_batch)
                    col = level_to_active_col(v_map, i)
                    logger.info(f"[映射@{step}] 能级{i}达标 → 主动列 col={col} (v 行范数峰值; n_used={n_used})")
                except Exception as _e:
                    col = i  # 回退：假设列下标=能级下标（仅当重建失败时）
                    logger.warning(f"[映射@{step}] 能级{i} 广义本征失败，回退 col={i}: {_e}")
                if col not in freeze_ctrl.frozen_set:
                    freeze_ctrl.freeze_cols(col)
                    frozen_levels.add(i)
                level_to_col[i] = col

        # 7) 冻结变更 → 重建机器/QGT/索引
        if freeze_ctrl.consume_changed():
            grad_fn, qgt_fn = build_grad_machines(freeze_ctrl.frozen_mask)
            idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
            logger.info(
                f"[重建@{step}] frozen_set={sorted(freeze_ctrl.frozen_set)} "
                f"| P_a/P = {int(idx_active.size)}/{N_TOTAL}"
            )

        # 8) 周期性 gauge reset
        if (step + 1) % RESET_PERIOD == 0:
            new_col_mean = col_mean_fn(total_params, x_batch)
            g_current = new_col_mean

        # 9) 日志 / 记录
        logger.info(
            f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.6f} | "
            + " | ".join(f"E{i}={eig_vals_re[i]:.6f}(e{errors_i[i]:.1e})" for i in range(K))
            + f" | frozen={sorted(freeze_ctrl.frozen_set)} | Pa/P={n_active}/{N_TOTAL}"
        )
        steps_history.append(step)
        Energy_levels_history.append(eig_vals_re)
        frozen_set_history.append(sorted(freeze_ctrl.frozen_set))
        level_to_col_history.append(dict(level_to_col))
        for i in range(K):
            err_history[i].append(float(errors_i[i]))

        if (step + 1) % SAVE_INTERVAL == 0:
            history = {
                "steps": steps_history, "Energy_levels": Energy_levels_history,
                "err": err_history, "frozen_set_history": frozen_set_history,
                "level_to_col": dict(level_to_col), "level_to_col_history": level_to_col_history,
                "frozen_set": sorted(freeze_ctrl.frozen_set), "params": [total_params],
            }
            with open(HISTORY_FILE, "wb") as f:
                pickle.dump(history, f)
            logger.info(f"[保存] Step {step} → {HISTORY_FILE}")

    # ----- 结尾保存 -----
    history = {
        "steps": steps_history, "Energy_levels": Energy_levels_history,
        "err": err_history, "frozen_set_history": frozen_set_history,
        "level_to_col": dict(level_to_col), "level_to_col_history": level_to_col_history,
        "frozen_set": sorted(freeze_ctrl.frozen_set), "params": [total_params],
    }
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"训练完成，耗时 {time.time()-start_time:.1f}s → {HISTORY_FILE}")