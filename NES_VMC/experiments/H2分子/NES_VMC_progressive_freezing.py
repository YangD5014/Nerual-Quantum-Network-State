# -*- coding: utf-8 -*-
"""
NES-VMC Progressive Freezing（分段冻结）Phase I —— H2 / 6-31G / K=4
====================================================================
严格对应《NES_VMC/文档/NES_VMC_Progressive_Freezing方案.md》。

核心机制（方案 §3/§5-§9/§16/§17/§19-PhaseI）
----------
1. 每轮用 MCMC 样本估计 K×K 的 M̂、Ŝ（方案 §3，复用 NES_VMC_tool.make_MS_estimator_fn），
   解广义本征问题  M v = λ S v（能量升序）。
2. 收敛判据（方案 §5「明确的、多轮稳定的 stopping criterion」）：
   能级 i 连续 NEED_STABLE 轮满足 |λ_i − E_fcis[i]| < CHEM_ACC（化学精度 1.6 mHa）。
3. 冻结动作（方案 §6/§16.1-3）：
   snapshot v_f = V[:, i]（Ritz 系数）+ 构成该 Ritz state 的**全部**子网络参数
   θ_0..θ_{K-1}（方案 §4：φ_i = Σ_j V_ji ψ_j，冻结 ψ_j ≠ 冻结 φ_i，必须整体 snapshot）
   + 冻结时刻 energy / overlap / 收敛信息，持久化到 data/snapshots_*/。
4. 后续训练（方案 §7/§9/§16.4-7）：
   - V 的 frozen 列不再被新本征向量覆盖（新本征向量中与 v_f 的 S-重叠最大者被
     frozen 列"消费"，其余按能量升序填入 active 列）；
   - frozen state 始终用 snapshot 参数 + snapshot 系数重构（§18 Frozen representation），
     并在 16 维全空间上**精确**评估 E_f（无 MCMC 噪声，漂移只可能来自机制本身）；
   - M、S 继续按原算法 K×K 更新；active states 继续训练
     （Phase I 训练回路与 baseline 逐位一致 —— 不改梯度、不加 penalty、不删 M/S 行列）。
5. 监控（方案 §16）：能量漂移 / 突然劣化 / generalized eigenvalue fluctuation（RQ 监控）/
   state overlap fluctuation / MCMC variance / QGT condition number /
   natural-gradient update stability / frozen state 化学精度保持。

运行
----
    cd NES_VMC/experiments/H2分子
    /opt/miniconda3/envs/Netket/bin/python NES_VMC_progressive_freezing.py            # frozen 模式
    /opt/miniconda3/envs/Netket/bin/python NES_VMC_progressive_freezing.py baseline   # baseline 模式
"""

import logging
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from scipy.linalg import eigh as scipy_eigh

from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
    ravel_pytree,
)
from NES_VMC_tool import (
    create_gauge_reset_total_machines,
    make_gauge_fn,
    make_MS_estimator_fn,
    make_qgt_fn_gauge,
    nes_vmc_gradient_stable_gauge,
)
from H2_631G import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock, hi, E_fcis

EXACT = np.asarray(E_fcis)[:K].real  # FCI 精确参考（前 K 个能级）

# ====================== 默认配置（训练超参与 h2-6-31G-K4.ipynb 完全一致） ======================
DEFAULT_CONFIG = dict(
    # —— 训练超参 ——
    N_CHAINS=16,
    N_SAMPLES_PER_CHAIN=400,
    SWEEP_SIZE=20,
    N_ITER=300,
    Natural_Grad=True,
    clip_norm=20.0,
    lr=0.1,
    qgt_diag_shift=0.1,
    RESET_PERIOD=10,          # 周期性 gauge reset（P8，双侧一致列规范）
    SAVE_INTERVAL=20,         # pickle 追加保存周期
    QGT_COND_INTERVAL=10,     # QGT 条件数监控周期（SVD 较贵）
    model_seed=11,
    sampler_seed=21,
    # —— 渐进冻结（方案 §5/§16）——
    CHEM_ACC=1.6e-3,          # 化学精度 1 kcal/mol ≈ 1.6 mHa
    NEED_STABLE=5,            # 连续达标轮数（多轮稳定判据）
    EXACT_CHECK=True,         # 触发时用 16 维精确能量交叉验证（防样本估计器噪声导致过早冻结）
    FROZEN_OVERLAP_GUARD=0.95,  # 新 frozen 向量与已有 frozen 向量的 S-重叠上限（防重复冻结同一态）
    # —— 测试钩子（仅冒烟测试用）——
    FORCE_FREEZE_AT=None,     # e.g. {0: 3} → step>=3 时强制冻结能级 0
)


# ====================== 1. 单向冻结控制器（方案 §9：frozen 列只增不减） ======================
class ProgressiveFreezer:
    """Progressive Freezing 控制器。

    - freeze()：记录能级 → snapshot 系数 v_f（K 维 Ritz 系数向量）+ 全部子网络参数
      θ_0..θ_{K-1}（numpy 深拷贝，与继续训练的 live 参数彻底解耦）+ 元信息。
    - frozen_vectors / frozen_params：后续重构 frozen state 的唯一数据来源
      （方案 §18：Frozen representation ≡ Σ_j v*_jf ψ_j(x; θ*_j)，固定物理态）。
    - 单向、只增不减：已冻结能级永不恢复、其 V 列永不被新本征向量覆盖。
    """

    def __init__(self, K: int):
        self.K = K
        self.frozen_levels = []      # 按冻结顺序记录
        self.frozen_vectors = {}     # lvl -> (K,) complex128（snapshot v_f）
        self.frozen_params = {}      # lvl -> params pytree（numpy 叶子快照）
        self.frozen_meta = {}        # lvl -> 冻结时刻元信息

    def is_frozen(self, lvl) -> bool:
        return lvl in self.frozen_vectors

    def freeze(self, lvl, v_f, params_snapshot, meta) -> bool:
        if self.is_frozen(lvl):
            return False
        self.frozen_levels.append(int(lvl))
        self.frozen_vectors[int(lvl)] = np.asarray(v_f, dtype=np.complex128)
        self.frozen_params[int(lvl)] = params_snapshot
        self.frozen_meta[int(lvl)] = meta
        return True

    @property
    def frozen_mask(self):
        return tuple(self.is_frozen(l) for l in range(self.K))


def save_freeze_snapshot(snap_dir, freezer: ProgressiveFreezer, lvl, step, cfg):
    """方案 §16.1-3：持久化 snapshot（v_f + 全部子网络参数 + 能量/重叠/收敛信息）。"""
    os.makedirs(snap_dir, exist_ok=True)
    payload = {
        "level": int(lvl),
        "freeze_step": int(step),
        "v_f": freezer.frozen_vectors[lvl],
        "params_snapshot": freezer.frozen_params[lvl],   # 全部子网络参数 θ_0..θ_{K-1}
        "meta": freezer.frozen_meta[lvl],
        "config": {k: v for k, v in cfg.items() if k != "FORCE_FREEZE_AT"},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(snap_dir, f"level{lvl}_step{step:04d}_snapshot.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    # registry：本目录所有 snapshot 的索引
    reg_path = os.path.join(snap_dir, "registry.pkl")
    registry = []
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "rb") as f:
                registry = pickle.load(f)
        except Exception:
            registry = []
    registry.append({"level": int(lvl), "step": int(step), "path": path})
    with open(reg_path, "wb") as f:
        pickle.dump(registry, f)
    return path


def load_and_verify_snapshots(snap_dir, exact_eval):
    """重载全部 snapshot 并独立复评 E_f（notebook 验证单元也复用本函数）。

    返回 list of dict：{level, step, E_f_reloaded, E_f_recorded, abs_diff, v_f, params}
    """
    reg_path = os.path.join(snap_dir, "registry.pkl")
    with open(reg_path, "rb") as f:
        registry = pickle.load(f)
    eval_phi, energy_of, _ = exact_eval
    results = []
    for item in registry:
        with open(item["path"], "rb") as f:
            payload = pickle.load(f)
        phi = eval_phi(jnp.asarray(payload["v_f"]), payload["params_snapshot"])
        e_re = energy_of(phi)
        results.append({
            "level": payload["level"],
            "step": payload["freeze_step"],
            "E_f_reloaded": e_re,
            "E_f_recorded": payload["meta"]["E_f_exact_at_freeze"],
            "abs_diff": abs(e_re - payload["meta"]["E_f_exact_at_freeze"]),
            "v_f": payload["v_f"],
            "params_snapshot": payload["params_snapshot"],
        })
    return results


# ====================== 2. 能级→V 列组装（方案 §9/§10：V^(t) = [v_f | V_active]） ======================
def assemble_levels(lam_new, V_new, Shat, frozen_vectors):
    """组装当前步的 V 矩阵与能级指派（方案 §9）。

    frozen 列固定为 snapshot v_f（不被覆盖，§16.4）；每条 frozen 列"消费"与其
    S-重叠 |v_f† Ŝ w_k| 最大的新本征向量；其余新本征向量按能量升序填入 active 列。

    返回:
        frozen_consumed: dict lvl -> k（被消费的新本征向量下标，监控用）
        active_assign  : dict lvl -> (k, lam_k, vec_k)  active 能级的本征对
    """
    K_ = lam_new.size
    frozen_consumed = {}
    consumed = set()
    for lvl in sorted(frozen_vectors):
        v_f = np.asarray(frozen_vectors[lvl])
        ovs = np.full(K_, -1.0)
        for k in range(K_):
            if k in consumed:
                continue
            ovs[k] = abs(np.vdot(v_f, np.asarray(Shat) @ V_new[:, k]))
        k_star = int(np.argmax(ovs))
        frozen_consumed[int(lvl)] = k_star
        consumed.add(k_star)

    remaining = [k for k in range(K_) if k not in consumed]  # 保持能量升序
    active_assign = {}
    for lvl in range(K_):
        if int(lvl) in frozen_vectors:
            continue
        k = remaining.pop(0)
        active_assign[int(lvl)] = (k, float(lam_new[k]), V_new[:, k])
    return frozen_consumed, active_assign


# ====================== 3. frozen state 精确评估（16 维全空间，无 MCMC 噪声） ======================
def make_exact_evaluator(hi_single, ha_op, single_machine_list):
    """在 16 维全空间精确评估 Ritz state（方案 §18 Frozen representation）。

    φ(x) = Σ_j v[j] ψ̃_j(x; θ_j)   —— gauge-fixed 基（与 M̂、Ŝ、v 完全一致）
    E   = ⟨φ|H|φ⟩ / ⟨φ|φ⟩        —— ha 在 hi 上的 16×16 稠密矩阵，精确算
    overlap²(a,b) = |⟨a|b⟩|² / (⟨a|a⟩⟨b|b⟩)

    返回 (eval_phi, energy_of, overlap2)。
    """
    X_all = jnp.asarray(hi_single.all_states())          # (16, n_spin)
    H_dense = np.asarray(ha_op.to_dense())               # (16, 16) complex
    K_ = len(single_machine_list)

    @jax.jit
    def eval_phi(v, params):
        logs = [
            single_machine_list[j](params["single_ansatz_list"][j], X_all)
            for j in range(K_)
        ]
        Lmat = jnp.stack(logs, axis=0)                   # (K, 16)
        return jnp.sum(jnp.asarray(v)[:, None] * jnp.exp(Lmat), axis=0)  # (16,)

    def energy_of(phi):
        p = np.asarray(phi)
        return float(np.real(np.vdot(p, H_dense @ p) / np.vdot(p, p)))

    def overlap2(phi_a, phi_b):
        a, b = np.asarray(phi_a), np.asarray(phi_b)
        return float(np.abs(np.vdot(a, b)) ** 2 / (np.real(np.vdot(a, a)) * np.real(np.vdot(b, b))))

    return eval_phi, energy_of, overlap2


# ====================== 4. 梯度函数（与 baseline 完全一致 + 免费统计监控） ======================
def make_grad_fn_stats(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list):
    """包装 nes_vmc_gradient_stable_gauge：训练路径逐位等同 baseline，
    额外从已有 Psi/HPsi 免费重算逐 walker tr(E_L) 的均值/标准差（MCMC variance 监控）。"""

    @jax.jit
    def grad_fn(total_params, x_batch, g):
        grad, loss_mean, E_L_mean, aux = nes_vmc_gradient_stable_gauge(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_max_machine=total_max_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
            g=g,
            return_aux=True,
        )
        la = aux["loss_aux"]
        E_L_batch = jnp.linalg.solve(la["Psi_Matrix_stable"], la["HPsi_stable"])
        tr_batch = jnp.real(jnp.trace(E_L_batch, axis1=-2, axis2=-1))
        valid = aux["valid"]
        n_valid = jnp.maximum(aux["n_valid"], 1.0)
        tr_mean = jnp.sum(jnp.where(valid, tr_batch, 0.0)) / n_valid
        tr_var = jnp.sum(jnp.where(valid, (tr_batch - tr_mean) ** 2, 0.0)) / n_valid
        stats = {
            "tr_mean": tr_mean,
            "tr_std": jnp.sqrt(tr_var),           # MCMC variance（方案 §16）
            "valid_ratio": aux["valid_ratio"],
            "cond_psi_mean": jnp.mean(la["cond_Psi"]),
        }
        return grad, loss_mean, E_L_mean, stats

    return grad_fn


# ====================== 5. 主训练循环 ======================
def _build_logger(time_str, mode):
    logger = logging.getLogger(f"NES_VMC_K4_H2_progressive_freeze_{mode}_{time_str}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
    os.makedirs("./日志", exist_ok=True)
    fh = logging.FileHandler(f"./日志/{time_str}_nes_vmc_K4_H2_progressive_freeze_{mode}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_training(mode="frozen", overrides=None, logger=None):
    """运行一次实验。

    mode="frozen"   : NES-VMC + Progressive Freezing（方案 §16 Frozen）
    mode="baseline" : NES-VMC 持续训练（方案 §16 Baseline；监控路径相同、无冻结动作，
                      训练回路与 frozen 模式逐位一致 → 可验证 Phase I 的非侵入性）

    返回 (history, freezer)。
    """
    assert mode in ("frozen", "baseline")
    cfg = {**DEFAULT_CONFIG, **(overrides or {})}
    time_str = time.strftime("%y-%m-%d-%H-%M")
    if logger is None:
        logger = _build_logger(time_str, mode)

    # ----- 模型 / 机器 / 采样器（与 h2-6-31G-K4.ipynb 一致）-----
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(cfg["model_seed"]),
    )
    g_current = jnp.zeros(K, dtype=jnp.complex64)
    (
        total_machine,
        total_matrix_machine,
        total_max_machine,
        total_matrix_machine_raw,
        total_graphdef,
        total_params,
    ) = create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)

    single_machine_list = [
        create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]

    grad_fn = make_grad_fn_stats(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list)
    qgt_fn = make_qgt_fn_gauge(total_machine)
    gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

    # M̂、Ŝ 估计器（方案 §3；K×K，基于 MCMC 样本）与 16 维精确评估器
    ms_estimator = make_MS_estimator_fn(ha, single_machine_list)
    exact_eval = make_exact_evaluator(hi, ha, single_machine_list)
    eval_phi, energy_of, overlap2 = exact_eval

    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    nes_sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext, rule=nes_rule, n_chains=cfg["N_CHAINS"], sweep_size=cfg["SWEEP_SIZE"],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["clip_norm"]),
        optax.sgd(learning_rate=cfg["lr"]),
    )
    opt_state = optimizer.init(total_params)

    def sample_machine(params, sigma):
        return total_machine(params, sigma, g_current)

    sampler_rng = jax.random.PRNGKey(cfg["sampler_seed"])
    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    N_TOTAL = int(ravel_pytree(total_params)[0].size)

    # ----- 冻结基础设施 -----
    freezer = ProgressiveFreezer(K)
    snap_dir = f"./data/snapshots_{time_str}_{mode}"
    stable_rounds = {i: 0 for i in range(K)}

    # ----- 历史 -----
    hist = {
        "mode": mode, "config": cfg, "exact": EXACT, "N_TOTAL": N_TOTAL,
        "steps": [], "loss": [], "loss_std": [], "tr_mean": [],
        "grad_norm_raw": [], "grad_norm_natural": [], "grad_norm_per_param": [],
        "logpsi_mean": [], "logpsi_min": [], "logpsi_max": [],
        "cond_psi_mean": [], "qgt_cond": [], "qgt_cond_steps": [],
        "gauge_abs": [],
        "lam_levels": [],        # (K,) 汇报能级：active=新本征值；frozen=np.nan
        "lam_MS_raw": [],        # (K,) 新广义本征解原始 λ（能量升序）
        "E_L_eigvals": [],       # (K,) E_L 矩阵本征值（与旧 notebook 口径一致，辅助监控）
        "V_history": [], "M_history": [], "S_history": [],
        "E_frozen_exact": {},    # lvl -> [(step, E), ...]
        "E_frozen_RQ": {},       # lvl -> [(step, RQ), ...]   v_f†M̂v_f / v_f†Ŝv_f（当前统计）
        "overlap_frozen": {},    # lvl -> [((K,) overlaps², ...)]
        "freeze_events": [],     # 冻结事件元信息
        "params_every": [],      # (step, params) 每 SAVE_INTERVAL + 最后一步
    }

    logger.info("=" * 70)
    logger.info(f"NES-VMC Progressive Freezing Phase I | mode={mode}")
    logger.info(f"""超参数配置:
N_CHAINS={cfg['N_CHAINS']} | N_SAMPLES_PER_CHAIN={cfg['N_SAMPLES_PER_CHAIN']} | SWEEP_SIZE={cfg['SWEEP_SIZE']}
N_ITER={cfg['N_ITER']} | Natural_Grad={cfg['Natural_Grad']} | clip_norm={cfg['clip_norm']} | lr={cfg['lr']}
qgt_diag_shift={cfg['qgt_diag_shift']} | RESET_PERIOD={cfg['RESET_PERIOD']} | SAVE_INTERVAL={cfg['SAVE_INTERVAL']}
CHEM_ACC={cfg['CHEM_ACC']:.4f} | NEED_STABLE={cfg['NEED_STABLE']} | model_seed={cfg['model_seed']} | sampler_seed={cfg['sampler_seed']}""")
    logger.info("=" * 70)
    logger.info("精确FCI基准：" + " | ".join(f"E{i}={EXACT[i]:.8f}" for i in range(K)))
    logger.info(f"理论 Loss 上限：{EXACT.sum():.8f} | 总参数 P={N_TOTAL}（每列 {N_TOTAL // K}）")

    start_time = time.time()
    for step in range(cfg["N_ITER"]):
        # ---------- 1. 采样（与 baseline 一致）----------
        samples_raw, sampler_state = nes_sampler.sample(
            machine=sample_machine, parameters=total_params, state=sampler_state,
            chain_length=cfg["N_SAMPLES_PER_CHAIN"],
        )
        samples = samples_raw.reshape(-1, hi_ext.size)
        x_batch = samples.reshape(-1, K, SINGLE_SIZE)

        # ---------- 2. 梯度 / 自然梯度 / 更新（Phase I 与 baseline 逐位一致）----------
        grad_raw, loss_mean, E_L_mean, stats = grad_fn(total_params, x_batch, g_current)
        grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
        grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
        grad_norm_per_param = grad_norm_raw / jnp.sqrt(grad_raw_flat.size)

        if cfg["Natural_Grad"]:
            qgt_reg_mat = qgt_fn(total_params, x_batch, cfg["qgt_diag_shift"], g_current)
            ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
            grad_update = unravel_fn(ng_flat)
            grad_norm_natural = jnp.linalg.norm(ng_flat)
        else:
            qgt_reg_mat = None
            grad_update = grad_raw
            grad_norm_natural = grad_norm_raw

        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        # ---------- 3. M̂、Ŝ 统计 + 广义本征解（方案 §3；K×K 不缩小，§8/§11）----------
        Mhat, Shat, finite = ms_estimator(total_params, x_batch)
        M_np, S_np = np.asarray(Mhat), np.asarray(Shat)
        lam_new, V_new = scipy_eigh(M_np, S_np)
        order = np.argsort(lam_new.real)
        lam_new, V_new = lam_new[order].real, V_new[:, order]

        # ---------- 4. V 组装（方案 §9：frozen 列固定，新本征向量按 S-重叠消费）----------
        frozen_consumed, active_assign = assemble_levels(lam_new, V_new, S_np, freezer.frozen_vectors)

        lam_levels = np.full(K, np.nan)
        for lvl, (k, lam_k, _vec) in active_assign.items():
            lam_levels[lvl] = lam_k

        # ---------- 5. frozen 监控（方案 §16/§18）----------
        if freezer.frozen_vectors:
            # 当前 16 维全空间波函数：frozen 用 snapshot 参数；live 用当前参数
            phis_live = {lvl: eval_phi(vec, total_params) for lvl, (_k, _l, vec) in active_assign.items()}
            for lvl, v_f in freezer.frozen_vectors.items():
                vj = jnp.asarray(v_f)
                rq = float(jnp.real(jnp.conj(vj) @ Mhat @ vj / (jnp.conj(vj) @ Shat @ vj)))
                phi_f = eval_phi(vj, freezer.frozen_params[lvl])   # snapshot 参数 + snapshot 系数（§18）
                e_f = energy_of(phi_f)
                ovs = np.array(
                    [overlap2(phi_f, phis_live[l]) if l in phis_live else np.nan for l in range(K)]
                )
                hist["E_frozen_exact"].setdefault(lvl, []).append((step, e_f))
                hist["E_frozen_RQ"].setdefault(lvl, []).append((step, rq))
                hist["overlap_frozen"].setdefault(lvl, []).append(ovs)

        # ---------- 6. 收敛判据 + 触发冻结（方案 §5/§6/§16.1-3）----------
        if mode == "frozen":
            for lvl in range(K):
                if freezer.is_frozen(lvl):
                    continue
                lam_i = lam_levels[lvl]
                if not np.isfinite(lam_i):
                    continue
                err_i = abs(lam_i - EXACT[lvl])
                met = err_i < cfg["CHEM_ACC"]
                stable_rounds[lvl] = stable_rounds[lvl] + 1 if met else 0
                forced = (cfg.get("FORCE_FREEZE_AT") or {}).get(lvl)
                if stable_rounds[lvl] >= cfg["NEED_STABLE"] or (forced is not None and step >= forced):
                    v_f = active_assign[lvl][2]
                    # 防重复冻结同一物理态：与已有 frozen 向量的 S-重叠过高则跳过
                    guard_ok = all(
                        abs(np.vdot(v_f, np.asarray(S_np) @ v_existing)) < cfg["FROZEN_OVERLAP_GUARD"]
                        for v_existing in freezer.frozen_vectors.values()
                    )
                    if not guard_ok:
                        logger.warning(f"[Guard@{step}] 能级{lvl} 候选 v_f 与已有 frozen 向量 S-重叠过高，跳过")
                        stable_rounds[lvl] = 0
                        continue
                    # ---- 精确能量交叉验证：M̂,Ŝ 是样本估计器（近简并/病态时 λ̂ 可能偏乐观），
                    #      在 16 维全空间精确评估候选态能量，只有同样达到化学精度才允许冻结 ----
                    e_f_candidate = energy_of(eval_phi(jnp.asarray(v_f), total_params))
                    exact_ok = (
                        abs(e_f_candidate - EXACT[lvl]) < cfg["CHEM_ACC"]
                        or not cfg.get("EXACT_CHECK", True)
                        or forced is not None   # 测试钩子：强制冻结不受交叉验证约束
                    )
                    if not exact_ok:
                        if stable_rounds[lvl] % 10 == 0:
                            logger.info(
                                f"[ExactCheck@{step}] 能级{lvl} λ̂已连续达标{stable_rounds[lvl]}轮，"
                                f"但精确能量 E_f={e_f_candidate:.8f}（err={abs(e_f_candidate - EXACT[lvl]):.2e}）"
                                f"未达化学精度 → 暂缓冻结，继续观察"
                            )
                        continue
                    # ---- snapshot：v_f + 全部子网络参数 + 元信息（方案 §6）----
                    params_snapshot = jax.tree_util.tree_map(np.asarray, total_params)
                    phi_f = eval_phi(jnp.asarray(v_f), params_snapshot)
                    meta = {
                        "freeze_step": int(step),
                        "lambda_at_freeze": float(lam_i),
                        "error_at_freeze": float(err_i),
                        "target": float(EXACT[lvl]),
                        "E_f_exact_at_freeze": energy_of(phi_f),
                        "stable_rounds": int(stable_rounds[lvl]),
                        "forced": bool(forced is not None and step >= forced),
                        "overlap_to_levels_at_freeze": {
                            l: overlap2(phi_f, eval_phi(vec, total_params))
                            for l, (_k, _lm, vec) in active_assign.items()
                        },
                        "v_f": np.asarray(v_f, dtype=np.complex128),
                        "M_at_freeze": M_np, "S_at_freeze": S_np,
                        "N_TOTAL": N_TOTAL,
                    }
                    freezer.freeze(lvl, v_f, params_snapshot, meta)
                    snap_path = save_freeze_snapshot(snap_dir, freezer, lvl, step, cfg)
                    logger.info(
                        f"[Freeze@{step}] 能级{lvl} 达到化学精度（λ={lam_i:.8f}, err={err_i:.2e}Ha, "
                        f"E_f_exact={meta['E_f_exact_at_freeze']:.8f}）→ snapshot v_f + θ_0..θ_{K - 1} → {snap_path}"
                    )

        # ---------- 7. 周期性 gauge reset（与 baseline 一致）----------
        if (step + 1) % cfg["RESET_PERIOD"] == 0:
            g_current = col_mean_fn(total_params, x_batch)

        # ---------- 8. 日志 / 历史 ----------
        log_Psi_batch = total_machine(total_params, x_batch, g_current)
        e_L_eigs = np.asarray(jnp.linalg.eigvalsh(E_L_mean))

        def _level_str(i):
            if freezer.is_frozen(i):
                records = hist["E_frozen_exact"].get(i)
                e_f = records[-1][1] if records else freezer.frozen_meta[i]["E_f_exact_at_freeze"]
                return f"E{i}=FROZEN({e_f:.6f})"
            return f"E{i}={lam_levels[i]:.8f}"

        logger.info(
            f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.6f} | tr_std={float(stats['tr_std']):.4f}"
            f" | " + " | ".join(_level_str(i) for i in range(K))
        )
        logger.info(
            f"           grad raw={float(grad_norm_raw):.4f} natural={float(grad_norm_natural):.4f}"
            f" | condΨ={float(stats['cond_psi_mean']):.2e} | n_valid={int(finite.sum())}/{x_batch.shape[0]}"
            + (f" | QGTcond={float(jnp.linalg.cond(qgt_reg_mat)):.2e}" if qgt_reg_mat is not None and (step + 1) % cfg["QGT_COND_INTERVAL"] == 0 else "")
        )

        hist["steps"].append(step)
        hist["loss"].append(float(jnp.real(loss_mean)))
        hist["loss_std"].append(float(stats["tr_std"]))
        hist["tr_mean"].append(float(stats["tr_mean"]))
        hist["grad_norm_raw"].append(float(grad_norm_raw))
        hist["grad_norm_natural"].append(float(grad_norm_natural))
        hist["grad_norm_per_param"].append(float(grad_norm_per_param))
        hist["logpsi_mean"].append(float(jnp.real(log_Psi_batch.mean())))
        hist["logpsi_min"].append(float(jnp.real(log_Psi_batch.min())))
        hist["logpsi_max"].append(float(jnp.real(log_Psi_batch.max())))
        hist["cond_psi_mean"].append(float(stats["cond_psi_mean"]))
        hist["gauge_abs"].append(float(jnp.linalg.norm(g_current)))
        hist["lam_levels"].append(lam_levels.copy())
        hist["lam_MS_raw"].append(lam_new.copy())
        hist["E_L_eigvals"].append(e_L_eigs.copy())
        hist["V_history"].append(np.asarray(V_new))
        hist["M_history"].append(M_np)
        hist["S_history"].append(S_np)
        if qgt_reg_mat is not None and (step + 1) % cfg["QGT_COND_INTERVAL"] == 0:
            hist["qgt_cond"].append(float(jnp.linalg.cond(qgt_reg_mat)))
            hist["qgt_cond_steps"].append(step)

        if (step + 1) % cfg["SAVE_INTERVAL"] == 0:
            hist["params_every"].append((step, jax.tree_util.tree_map(np.asarray, total_params)))
            _dump_history(hist, f"./data/{time_str}_history_progressive_freeze_K4_{mode}.pkl")
            logger.info(f"[保存] Step {step} → 中间 pickle 已更新（snapshot 目录: {snap_dir if freezer.frozen_levels else '尚无'}）")

    # ---------- 收尾 ----------
    hist["params_every"].append((cfg["N_ITER"] - 1, jax.tree_util.tree_map(np.asarray, total_params)))
    hist["freeze_events"] = [freezer.frozen_meta[l] | {"level": l} for l in freezer.frozen_levels]
    history_file = f"./data/{time_str}_history_progressive_freeze_K4_{mode}.pkl"
    _dump_history(hist, history_file)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"训练完成 | 耗时 {elapsed:.1f}s | mode={mode} | 冻结事件: {freezer.frozen_levels or '无'}")
    logger.info(f"history → {history_file}")
    if freezer.frozen_levels:
        logger.info(f"snapshots → {snap_dir}")
        for l in freezer.frozen_levels:
            m = freezer.frozen_meta[l]
            logger.info(
                f"  能级{l}: freeze@step={m['freeze_step']} | λ*={m['lambda_at_freeze']:.8f} "
                f"(err {m['error_at_freeze']:.2e}) | E_f_exact={m['E_f_exact_at_freeze']:.8f}"
            )
    logger.info("=" * 70)
    return hist, freezer


def _dump_history(hist, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(hist, f)


# ====================== 6. 绘图 ======================
def plot_energies(hist, fig_path):
    """能级曲线：live λ + 精确参考 + 冻结事件标记 + frozen 态精确能量叠加。"""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    steps = np.asarray(hist["steps"])
    lam = np.asarray(hist["lam_levels"], dtype=float)     # (T, K)，frozen 处为 nan
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(K):
        ax.plot(steps, lam[:, i], ".", markersize=2.5, color=colors[i], alpha=0.55,
                label=f"$E_{i}$ (live Ritz λ)")
        ax.axhline(hist["exact"][i], color=colors[i], linestyle="--", linewidth=1.0, alpha=0.8)
        if i in hist["E_frozen_exact"]:
            fs = np.array([s for s, _ in hist["E_frozen_exact"][i]])
            fe = np.array([e for _, e in hist["E_frozen_exact"][i]])
            ax.plot(fs, fe, "-", color=colors[i], linewidth=2.2, alpha=0.95,
                    label=f"$E_{i}$ (frozen, exact eval)")
            ax.axvline(fs[0], color=colors[i], linestyle=":", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"NES-VMC K=4 H2/6-31G — Progressive Freezing ({hist['mode']})")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_frozen_detail(hist, fig_path):
    """frozen 态细节：ΔE（log）、RQ 监控（当前统计下的广义本征值波动）、重叠波动。"""
    if not hist["E_frozen_exact"]:
        return None
    levels = sorted(hist["E_frozen_exact"])
    fig, axes = plt.subplots(len(levels), 2, figsize=(12, 3.6 * len(levels)), squeeze=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for row, lvl in enumerate(levels):
        fs = np.array([s for s, _ in hist["E_frozen_exact"][lvl]])
        fe = np.array([e for _, e in hist["E_frozen_exact"][lvl]])
        frq = np.array([e for _, e in hist["E_frozen_RQ"][lvl]])
        # 左：|E_f^exact − target|（log）+ RQ 波动
        ax = axes[row][0]
        ax.semilogy(fs, np.abs(fe - hist["exact"][lvl]) + 1e-12, "-", color=colors[lvl],
                    label=r"$|E_f^{exact}-E^{target}|$ (frozen)")
        live_err = np.asarray([
            (hist["lam_MS_raw"][s][lvl] if s < len(hist["lam_MS_raw"]) else np.nan) for s in fs
        ])
        ax.semilogy(fs, np.abs(live_err - hist["exact"][lvl]) + 1e-12, ".", markersize=2,
                    color="gray", alpha=0.5, label=r"$|\lambda^{MS}-E^{target}|$ (raw eigensolve)")
        ax.set_title(f"level {lvl}: energy drift (frozen vs raw)", fontsize=10)
        ax.set_ylabel("|ΔE| (Ha)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        # 右：S-重叠波动
        ax = axes[row][1]
        ovs = np.asarray(hist["overlap_frozen"][lvl])   # (T, K)
        for l in range(K):
            if np.all(np.isnan(ovs[:, l])):
                continue
            ax.plot(fs, ovs[:, l], ".", markersize=2, color=colors[l], label=f"$|\\langle\\varphi_f|\\varphi_{l}\\rangle|^2$")
        ax.set_title(f"level {lvl}: overlap fluctuation", fontsize=10)
        ax.set_ylabel("overlap²")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_diagnostics(hist, fig_path):
    """方案 §16 监控面板：Loss / MCMC variance / 梯度范数 / cond(Ψ) / QGT cond。"""
    steps = np.asarray(hist["steps"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0][0].plot(steps, hist["loss"], ".", markersize=2)
    axes[0][0].set_title("Loss = tr(E_L)")
    axes[0][1].plot(steps, hist["loss_std"], ".", markersize=2, color="tab:red")
    axes[0][1].set_title("MCMC variance: std of per-walker tr(E_L)")
    axes[1][0].semilogy(steps, hist["grad_norm_raw"], ".", markersize=2, label="raw")
    axes[1][0].semilogy(steps, hist["grad_norm_natural"], ".", markersize=2, label="natural")
    axes[1][0].set_title("gradient norms")
    axes[1][0].legend(fontsize=8)
    axes[1][1].semilogy(steps, hist["cond_psi_mean"], ".", markersize=2, label="cond(Ψ)")
    if hist["qgt_cond"]:
        axes[1][1].semilogy(np.asarray(hist["qgt_cond_steps"]), hist["qgt_cond"], ".", markersize=2, label="cond(QGT)")
    axes[1][1].set_title("condition numbers")
    axes[1][1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_mode_comparison(hist_frozen, hist_baseline, fig_path):
    """方案 §16 对比图：Baseline（持续训练）vs Frozen。"""
    steps = np.asarray(hist_baseline["steps"])
    lam_b = np.asarray(hist_baseline["lam_levels"], dtype=float)
    lam_f = np.asarray(hist_frozen["lam_levels"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    # 左：两模式 live 能级对比（应逐位一致 → Phase I 非侵入性验证）
    for i in range(K):
        axes[0].plot(steps, lam_b[:, i], "-", color=colors[i], linewidth=1.0, alpha=0.9,
                     label=f"baseline $E_{i}$")
        axes[0].plot(np.asarray(hist_frozen["steps"]), lam_f[:, i], ":", color=colors[i], linewidth=2.2,
                     label=f"frozen-live $E_{i}$")
        axes[0].axhline(hist_baseline["exact"][i], color=colors[i], linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_ylim(-1.45, 0.1)
    axes[0].set_title("live Ritz energies: baseline vs frozen (identical ⇒ non-invasive)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("Energy (Ha)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    # 右：已收敛能级长期能量漂移对比（frozen 态精确评估 vs baseline live）
    for i in range(K):
        axes[1].plot(steps, lam_b[:, i], ".", markersize=2, color=colors[i], alpha=0.4,
                     label=f"baseline live $E_{i}$")
        if i in hist_frozen["E_frozen_exact"]:
            fs = np.array([s for s, _ in hist_frozen["E_frozen_exact"][i]])
            fe = np.array([e for _, e in hist_frozen["E_frozen_exact"][i]])
            axes[1].plot(fs, fe, "-", color=colors[i], linewidth=2.4,
                         label=f"frozen $E_{i}$ (exact)")
        axes[1].axhline(hist_baseline["exact"][i], color=colors[i], linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_ylim(-1.45, 0.1)
    axes[1].set_title("converged levels: baseline live vs frozen snapshot")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def make_all_plots(hist, tag=None):
    os.makedirs("./日志", exist_ok=True)
    tag = tag or time.strftime("%y-%m-%d-%H-%M")
    paths = {}
    paths["energies"] = plot_energies(hist, f"./日志/[能量]{tag}_progressive_freeze_{hist['mode']}.png")
    p = plot_frozen_detail(hist, f"./日志/[冻结细节]{tag}_progressive_freeze_{hist['mode']}.png")
    if p:
        paths["frozen_detail"] = p
    paths["diagnostics"] = plot_diagnostics(hist, f"./日志/[诊断]{tag}_progressive_freeze_{hist['mode']}.png")
    for k, v in paths.items():
        print(f"[plot] {k}: {v}")
    return paths


def make_mode_comparison(hist_frozen, hist_baseline, tag=None):
    """Baseline vs Frozen 对比图封装（notebook 复用）。"""
    os.makedirs("./日志", exist_ok=True)
    tag = tag or time.strftime("%y-%m-%d-%H-%M")
    return plot_mode_comparison(
        hist_frozen, hist_baseline,
        f"./日志/[对比]{tag}_progressive_freeze_frozen_vs_baseline.png",
    )


# ====================== 7. CLI 入口 ======================
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "frozen"
    hist, freezer = run_training(mode=mode)
    make_all_plots(hist)
