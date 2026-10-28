"""
自适应 NES-VMC（Adaptive / 逐轮冻结-再探索）
【LiH / STO-3G / CAS(2e,4o) / 8 qubit · LiH分子-2.2】

算法（用户需求）
----------------
第 r 轮：
  1) 以 K_r 条 active 列 + 已冻结的 R 条物理列做 Matrix-Free deflation NES-VMC；
     第 1 轮无冻结列（R=0），即普通 NES-VMC（K_1 = K1 条列）。
  2) 训练结束后，用「单副本均匀测度 Ritz 值」读出本轮的各个能级 λ̂_k，
     并按升序把第 k 条 active 列看成剩余最低的第 k 个物理能级。
  3) 逐个判据：|λ̂_k − E_FCI(level_k)| < CHEM_TOL（默认 1 mHa，比化学精度 1.6 mHa 更严）
       → 该物理能级「达标」⇒ 其列被**冻结**，作为下一轮的新冻结组；
       → 未达标 ⇒ 该能级进入下一轮的 active 集合，重新计算（下一轮 K_{r+1}=失败能级数）。
  4) 若本轮全部达标本轮能级（等价于目标能级全部达标）⇒ active 清空 ⇒ 全部冻结，结束。
     否则继续下一轮，直到 active 为空或达到 MAX_ROUNDS。

冻结列的精确拼接（多轮链式，非近似）
------------------------------------
bundle 的冻结部分逐点表示为 φ_i = Σ_j ψ_j^old · V[j,i]（V 为 J×R 常量系数）。
某轮收敛的 active 列在缩放基下为 χ̃_k = Σ_m χ̃_m v[m,k]，其中
    χ̃ = Ψ̃ − Φ̃ c_f,  Ψ̃_m = e^{−g_m}ψ_m,  Φ̃_i = e^{−f}(Σ_j ψ_j^old V[j,i])
把它还原成「对旧网络 + 本轮网络」的系数：
    新冻结列 = Σ_m ψ_m (e^{−g_m} v[m,k])  −  Σ_j ψ_j^old (V @ e^{−f} c_f v)[j,k]
于是扩充后的全局系数矩阵严格分块：
    V_new = [[ V , V·D ], [ 0 , A ]] ,  D = −e^{−f} c_f v_keep ,  A = e^{−g} v_keep
这样多轮冻结组的 span 与逐轮精确投影完全一致（不是把投影残差丢掉）。

运行：
    python LiH_STO3G_CAS_8Qbit_adaptive_NESVMC.py
    NES_N_ITER=200 NES_K1=5 python ...            # 缩短每轮步数
    NES_N_ITER=5 NES_K1=2 NES_MAX_ROUNDS=2 python ...   # smoke

环境变量：
    NES_N_ITER        第 1 轮（K1 列并行）训练步数（默认 500）
    NES_N_ITER_LATER  后续轮次（残留硬能级）训练步数（默认 1500，沿用已验证的 V2 配置）
    NES_K1         第 1 轮的列数 K1（默认 5，目标最低能级 0..K1-1）
    NES_CHEM_TOL   冻结阈值 Ha（默认 1e-3 = 1 mHa；化学精度为 1.6e-3）
    NES_MAX_ROUNDS 最大轮数（默认 3）
    NES_AUX_M      辅助 Gram 采样数（默认 1024）
    NES_INIT_SEED  第 1 轮初始化种子（默认 11）
    NES_QGT_SHIFT / NES_QGT_SHIFT_LATER  自然梯度对角平移（默认 0.1 / 0.02）
"""
import logging
import os
import sys
import time
import pickle

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from jax.flatten_util import ravel_pytree
from scipy.linalg import eigh as scipy_eigh

this_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_file_dir = os.path.join(this_file_dir, "..")          # LiH分子-2.2/
gradparent_file_dir = os.path.join(parent_file_dir, "..")    # experiments/
sys.path.append(parent_file_dir)
sys.path.append(gradparent_file_dir)

from NES_VMC_V1 import (                       # noqa: E402
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
    Ham_psi_scaled,
)
from NES_VMC_deflation_mf import make_mf_bundle, update_scales   # noqa: E402
from LiH_CAS_STO3G_8Qbit import (              # noqa: E402
    SINGLE_SIZE, ha, hi, Hatree_Fock, single_edges, E_fcis,
)

E_FCI = np.asarray(E_fcis, dtype=float)


# ===================== Ansatz：实参数 + 复输出（与 K5 / K2 训练逐字一致） =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    """单列（自旋轨道占据组态 → 复对数振幅）前馈网络。"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)   # 实部头
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)   # 虚部头

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        z = (u + 1e-12) + 1j * (v + 1e-12)    # 防 log(0)
        return jnp.log(z)


class NESTotalAnsatz_stable(nnx.Module):
    """K 列总拟设：持有 K 个独立单列网络。"""

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8,
                 *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            self.single_ansatz_list.append(
                SingleStateAnsatz_SharedBackbone(
                    n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(params=sub_key))
            )


# ====================== 超参 ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER = int(os.environ.get("NES_N_ITER", 500))              # 第 1 轮（K1 列并行）步数
# 后续轮次只算「残留硬能级」，列数少但更难收敛：沿用已验证的 V2 配置（更多步 + 更小 diag_shift）
N_ITER_LATER = int(os.environ.get("NES_N_ITER_LATER", 1500))
K1 = int(os.environ.get("NES_K1", 5))
CHEM_TOL = float(os.environ.get("NES_CHEM_TOL", 1e-3))       # 1 mHa
MAX_ROUNDS = int(os.environ.get("NES_MAX_ROUNDS", 3))
clip_norm = 20.0
per_param_clip = 0.5
lr = 0.1
qgt_diag_shift = float(os.environ.get("NES_QGT_SHIFT", 0.1))          # 第 1 轮
qgt_diag_shift_later = float(os.environ.get("NES_QGT_SHIFT_LATER", 0.02))
SAVE_INTERVAL = 20
INIT_SEED = int(os.environ.get("NES_INIT_SEED", 11))
AUX_M = int(os.environ.get("NES_AUX_M", 1024))
AUX_CHAINS = 8
AUX_SWEEP = 20
SCALE_BETA = float(os.environ.get("NES_SCALE_BETA", 0.9))
EPS_MIX = 0.25
PINV_RCOND = float(os.environ.get("NES_PINV_RCOND", 1e-8))


# ====================== 日志 ======================
time_str = time.strftime("%y-%m-%d-%H-%M")
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
graph_dir = os.path.join(this_file_dir, "graph")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

logger = logging.getLogger("NES_VMC_adaptive_LiH")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
_log_path = f"{log_dir}/{time_str}_adaptive_nesvmc_K1={K1}_LiH_STO3G_CAS_8Qbit.log"
_fh = logging.FileHandler(_log_path, mode="w", encoding="utf-8")
_fh.setFormatter(_formatter)
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(_formatter)
logger.addHandler(_ch)


# ====================== 工具：冻结组的全局拼接 ======================
def build_frozen_global(frozen_sms, frozen_params, V_global):
    """返回 (sms 列表, params 列表, V 全局矩阵)，V 形状 (J_total, R_total)。

    V_global 已按「所有冻结网络逐点求值后的系数」存好：φ_i = Σ_j ψ_j V[j,i]。
    """
    return list(frozen_sms), list(frozen_params), np.asarray(V_global, dtype=np.complex128)


def make_ext_edges(K):
    """联合 walker 空间的 Hop 边：K 个副本各自偏移 k*SINGLE_SIZE。"""
    return jnp.asarray([(i + k * SINGLE_SIZE, j + k * SINGLE_SIZE)
                        for k in range(K) for (i, j) in single_edges])


def extract_deflated(bundle, total_params, f, g, sms, frozen_sms, frozen_params,
                     V, sigma_ext):
    """收尾提取：Matrix-Free MC Gram 广义本征，返回 (lam 升序, v_active, c_f)。"""
    r = len(frozen_sms)
    K = len(sms)
    c_f, _c2, _m1, _m2, aux_gf = bundle["estimate_grams"](total_params, sigma_ext, f, g)
    if not bool(aux_gf["grams_finite"]):
        logger.warning("提取批 grams 非有限，尝试继续（结果可能不稳）")

    Phi, Psi = bundle["cols"](total_params, sigma_ext, f, g)     # (M,R),(M,K)
    chi = Psi - Phi @ c_f                                       # (M,K)
    Hpsi = jnp.stack(
        [Ham_psi_scaled(ha=ha, single_machine=sms[m],
                        params=total_params["single_ansatz_list"][m],
                        x=sigma_ext, shift=g[m]) for m in range(K)], axis=-1)
    if r > 0:
        Hphi = jnp.stack(
            [Ham_psi_scaled(ha=ha, single_machine=frozen_sms[j],
                            params=frozen_params[j], x=sigma_ext, shift=f)
             for j in range(r)], axis=-1) @ jnp.asarray(V, dtype=jnp.complex128)
        Hchi = Hpsi - Hphi @ c_f
        q = jnp.sum(jnp.abs(Phi) ** 2, axis=-1) + EPS_MIX * jnp.sum(jnp.abs(Psi) ** 2, axis=-1)
    else:
        Hchi = Hpsi
        q = EPS_MIX * jnp.sum(jnp.abs(Psi) ** 2, axis=-1)

    wi = 1.0 / (q * sigma_ext.shape[0])
    S = np.asarray(jnp.conj(chi).T @ (wi[:, None] * chi), dtype=np.complex128)
    M = np.asarray(jnp.conj(chi).T @ (wi[:, None] * Hchi), dtype=np.complex128)
    S = 0.5 * (S + S.conj().T) + 1e-12 * np.eye(K)
    M = 0.5 * (M + M.conj().T)
    lam, v = scipy_eigh(M + 1e-14 * S, S)
    order = np.argsort(lam.real)
    lam, v = lam.real[order], v[:, order]
    return lam, v, c_f


# ====================== 主流程 ======================
if __name__ == "__main__":
    target_levels = list(range(K1))
    active_levels = list(target_levels)          # 本轮待算的物理能级（升序）
    frozen_levels = []                           # 已冻结的物理能级
    frozen_sms, frozen_params = [], []
    V_global = np.zeros((0, 0), dtype=np.complex128)
    frozen_errors = {}                           # level -> 冻结时的误差

    round_summary = []                           # 每轮结果
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.sgd(learning_rate=lr),
    )

    logger.info("\n" + "=" * 64)
    logger.info("自适应 NES-VMC | LiH / STO-3G / CAS(2e,4o) / 8 qubit | 逐轮 冻结-再探索")
    logger.info(f"目标最低能级 = {target_levels} | 第 1 轮 K1 = {K1}")
    logger.info(f"冻结判据 |λ̂ − E_FCI| < {CHEM_TOL:.1e} Ha（{CHEM_TOL * 1000:.2f} mHa）")
    logger.info(f"每轮步数 = 第1轮 {N_ITER} / 后续轮 {N_ITER_LATER} | 最大轮数 = {MAX_ROUNDS}")
    logger.info(f"精确 CAS 基准 E_FCI[:{K1}] = {np.round(E_FCI[:K1], 8)}")
    logger.info("=" * 64)

    total_start = time.time()

    for rnd in range(1, MAX_ROUNDS + 1):
        K = len(active_levels)
        if K == 0:
            logger.info("active 集合为空 ⇒ 全部目标能级已冻结，算法结束")
            break

        logger.info("\n" + "*" * 64)
        logger.info(f"第 {rnd} 轮开始 | active 能级 = {sorted(active_levels)} (K={K}) | "
                    f"已冻结能级 = {sorted(frozen_levels)} (R={V_global.shape[1]})")

        # ---------- 构建 deflation bundle ----------
        fsms, fparams, V = build_frozen_global(frozen_sms, frozen_params, V_global)
        seed = INIT_SEED + 17 * (rnd - 1)
        ansatz = NESTotalAnsatz_stable(
            n_spin_orbitals=SINGLE_SIZE, n_states=K,
            hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(seed),
        )
        total_params = nnx.split(ansatz)[1]
        sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
               for a in ansatz.single_ansatz_list]
        bundle = make_mf_bundle(ha, fsms, fparams, V, sms,
                                n_spin=SINGLE_SIZE, eps=EPS_MIX, pinv_rcond=PINV_RCOND)

        f = jnp.asarray(0.0)                 # 冻结列公共实尺度（R=0 时无作用）
        g = jnp.zeros(K)                     # active 列逐列实尺度

        def aux_machine(params_aux, sigma):
            return bundle["aux_logq"](params_aux, sigma)

        hi_ext = hi ** K
        ext_edges = make_ext_edges(K)

        def sample_machine(params, sigma):
            p, cc, cc2, ff, gg = params
            return bundle["machine_batch"](p, sigma.reshape(-1, hi_ext.size), cc, ff, gg)

        aux_rule = NESFermionHopRule(edges=jnp.asarray(single_edges), K=1, single_size=SINGLE_SIZE)
        aux_sampler = nk.sampler.MetropolisSampler(
            hilbert=hi, rule=aux_rule, n_chains=AUX_CHAINS, sweep_size=AUX_SWEEP, machine_pow=1)
        nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
        nes_sampler = nk.sampler.MetropolisSampler(
            hilbert=hi_ext, rule=nes_rule, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE,
            machine_pow=1)

        opt_state = optimizer.init(total_params)
        aux_rng = jax.random.PRNGKey(5 + rnd)
        aux_state = aux_sampler.init_state(aux_machine, (total_params, f, g), aux_rng)
        _s, aux_state = aux_sampler.sample(
            machine=aux_machine, parameters=(total_params, f, g),
            state=aux_state, chain_length=AUX_M // AUX_CHAINS)

        sampler_rng = jax.random.PRNGKey(21 + rnd)
        sampler_state = None
        c_cur = jnp.zeros((V.shape[0], K), dtype=jnp.complex128)
        c2_cur = jnp.zeros((V.shape[0], K), dtype=jnp.complex128)

        # ---------- 本轮训练 ----------
        loss_history, steps_history, lam_history = [], [], []
        diag_history = []
        n_skipped = 0
        n_round_start = time.time()

        n_iter_round = N_ITER if rnd == 1 else N_ITER_LATER
        shift_round = qgt_diag_shift if rnd == 1 else qgt_diag_shift_later

        for step in range(n_iter_round):
            # 1) aux 采样 → MC Gram 投影系数 c/c2
            s_aux, aux_state = aux_sampler.sample(
                machine=aux_machine, parameters=(total_params, f, g),
                state=aux_state, chain_length=AUX_M // AUX_CHAINS)
            sigma_aux = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
            c_mc, c2_mc, max_phi2, max_psi, aux_g = bundle["estimate_grams"](
                total_params, sigma_aux, f, g)
            if not bool(aux_g["grams_finite"]):
                logger.warning(f"[R{rnd} Step {step}] grams 非有限，跳过本步")
                n_skipped += 1
                loss_history.append(np.nan)
                lam_history.append(np.full(K, np.nan))
                steps_history.append(step)
                diag_history.append({"valid_ratio": np.nan, "grad_norm_raw": np.nan})
                continue

            # 2) EMA 尺度更新 + c/c2 同步变换
            beta = 1.0 if step == 0 else SCALE_BETA
            f_np, g_np = update_scales(float(f), np.asarray(g), max_phi2, max_psi, beta=beta)
            d_f = f_np - float(f)
            d_g = g_np - np.asarray(g)
            trans = np.exp(d_f - d_g)
            c_cur = jnp.asarray(np.asarray(c_mc) * trans[None, :])
            c2_cur = jnp.asarray(np.asarray(c2_mc) * trans[None, :])
            f = jnp.asarray(f_np)
            g = jnp.asarray(g_np)

            # 3) 主采样（联合 K walker）
            if sampler_state is None:
                sampler_state = nes_sampler.init_state(
                    sample_machine, (total_params, c_cur, c2_cur, f, g), sampler_rng)
            samples_raw, sampler_state = nes_sampler.sample(
                machine=sample_machine, parameters=(total_params, c_cur, c2_cur, f, g),
                state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN)
            sigma_flat = jnp.asarray(np.asarray(samples_raw).reshape(-1, hi_ext.size))

            # 4) 梯度 / 自然梯度
            grad_raw, loss_mean, _E_L_mean, aux = bundle["grad_fn"](
                total_params, sigma_flat, c_cur, c2_cur, f, g)
            grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
            grad_norm_raw = float(jnp.linalg.norm(grad_raw_flat))
            n_params = grad_raw_flat.size
            grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)

            if grad_norm_per_param > per_param_clip:
                scale = per_param_clip / grad_norm_per_param
                grad_raw_flat = grad_raw_flat * scale
                grad_raw = unravel_fn(grad_raw_flat)
                grad_norm_raw = float(jnp.linalg.norm(grad_raw_flat))
                grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)

            if bool(aux["should_skip"]):
                logger.warning(f"[R{rnd} Step {step}] should_skip=True "
                               f"(valid_ratio={float(aux['valid_ratio']):.3f})，跳过更新")
                n_skipped += 1
            else:
                S_qgt = bundle["qgt_fn"](total_params, sigma_flat, shift_round,
                                         c_cur, c2_cur, f, g)
                ng_flat = jnp.linalg.solve(S_qgt, grad_raw_flat)
                grad_update = unravel_fn(ng_flat)
                updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
                total_params = optax.apply_updates(total_params, updates)

            # 5) 监控（单副本均匀测度 Ritz 值）
            lam_hat = np.sort(np.asarray(aux_g["lam_uniform"]).real)
            if int(aux_g["pinv_n_keep"]) < K:
                lam_hat = np.full(K, np.nan)

            loss_history.append(float(jnp.real(loss_mean)))
            lam_history.append(lam_hat)
            steps_history.append(step)
            diag_history.append({
                "valid_ratio": float(aux["valid_ratio"]),
                "grad_norm_raw": grad_norm_raw,
                "grad_norm_per_param": grad_norm_per_param,
            })

            if step % max(1, n_iter_round // 10) == 0 or step == n_iter_round - 1:
                errs = [f"{abs(lam_hat[k] - E_FCI[active_levels[k]]) * 1000:.3f}"
                        if np.isfinite(lam_hat[k]) else "nan" for k in range(K)]
                logger.info(f"[R{rnd} Step {step:4d}] Loss={float(jnp.real(loss_mean)):.8f} | "
                            f"λ̂={np.round(lam_hat, 8)} | |err|(mHa)={errs}")
                logger.info(f"          f={float(f):.3f} g={np.round(np.asarray(g), 3)} | "
                            f"valid={float(aux['valid_ratio']):.3f} | "
                            f"grad_raw={grad_norm_raw:.4f}")

            if (step + 1) % SAVE_INTERVAL == 0 and n_iter_round > 20:
                with open(f"{data_dir}/{time_str}_adaptive_round{rnd}_K{K}.pkl", "wb") as fsave:
                    pickle.dump({"round": rnd, "K": K, "active_levels": active_levels,
                                 "steps": steps_history, "loss": loss_history,
                                 "lam": lam_history, "diag": diag_history,
                                 "params": [total_params],
                                 "scales": {"f": float(f), "g": np.asarray(g).tolist()}}, fsave)

        round_elapsed = time.time() - n_round_start
        logger.info(f"第 {rnd} 轮训练结束 | {n_iter_round} 步 (diag_shift={shift_round}) | "
                    f"跳过 {n_skipped} | 耗时 {round_elapsed:.1f}s")

        # ---------- 收尾提取 + 达标判定 ----------
        s_aux, aux_state = aux_sampler.sample(
            machine=aux_machine, parameters=(total_params, f, g),
            state=aux_state, chain_length=AUX_M // AUX_CHAINS)
        sigma_ext = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
        lam_extract, v_active, c_f = extract_deflated(
            bundle, total_params, f, g, sms, frozen_sms, frozen_params, V, sigma_ext)
        g_np = np.asarray(g, dtype=np.float64)
        f_np = float(f)

        lam_use = lam_extract
        errs = np.abs(lam_use - np.asarray([E_FCI[lv] for lv in active_levels]))
        passed = errs < CHEM_TOL
        logger.info(f"第 {rnd} 轮提取 λ = {np.round(lam_use, 8)}（对应能级 {active_levels}）")
        logger.info(f"第 {rnd} 轮 |λ − E_FCI| (mHa) = {np.round(errs * 1000, 4)} | "
                    f"判据 < {CHEM_TOL * 1000:.2f} mHa → "
                    f"达标能级 {[active_levels[k] for k in range(K) if passed[k]]} | "
                    f"未达标 {[active_levels[k] for k in range(K) if not passed[k]]}")
        round_summary.append({"round": rnd, "K": K, "active_levels": list(active_levels),
                              "lam": lam_use.copy(), "errs": errs.copy(),
                              "passed": passed.copy(), "elapsed_s": round_elapsed,
                              "steps": np.asarray(steps_history),
                              "lam_curve": np.asarray(lam_history, dtype=float)})

        keep = np.where(passed)[0]
        fail = np.where(~passed)[0]

        if len(keep) > 0:
            # 冻结本轮达标的列：新冻结组（本轮 K 条网络 + 达标列系数）
            A_new = (np.exp(-g_np)[:, None] * v_active[:, keep]).astype(np.complex128)  # (K,Rk)
            D = (-np.exp(-f_np) * (np.asarray(c_f, dtype=np.complex128) @ v_active[:, keep])
                 ).astype(np.complex128)                                                # (R,Rk)
            V_old = V                                                                    # (J,R)
            V_new = np.zeros((V_old.shape[0] + K, V_old.shape[1] + len(keep)), np.complex128)
            V_new[:V_old.shape[0], :V_old.shape[1]] = V_old
            V_new[:V_old.shape[0], V_old.shape[1]:] = V_old @ D
            V_new[V_old.shape[0]:, V_old.shape[1]:] = A_new
            frozen_sms = frozen_sms + list(sms)
            frozen_params = frozen_params + [total_params["single_ansatz_list"][m] for m in range(K)]
            V_global = V_new
            for k in keep:
                frozen_levels.append(int(active_levels[k]))
                frozen_errors[int(active_levels[k])] = float(errs[k])
            logger.info(f"第 {rnd} 轮冻结能级 {[active_levels[k] for k in keep]} "
                        f"⇒ 累计冻结能级 {sorted(frozen_levels)} (R={V_global.shape[1]})")

        active_levels = sorted(int(active_levels[k]) for k in fail)
        if len(active_levels) == 0:
            logger.info(f"★ 第 {rnd} 轮后所有目标能级均已达标 ⇒ 全部冻结，算法结束")
            break
        logger.info(f"第 {rnd} 轮未达标能级 {active_levels} 进入下一轮 (K_{rnd + 1}={len(active_levels)})")

    total_elapsed = time.time() - total_start

    # ====================== 总结 ======================
    logger.info("\n" + "=" * 64)
    logger.info(f"自适应 NES-VMC 完成 | 总耗时 {total_elapsed:.1f}s | 执行轮数 {len(round_summary)}")
    logger.info(f"冻结能级 = {sorted(frozen_levels)} | 最终 active = {sorted(active_levels)}")
    for lv in sorted(frozen_errors):
        logger.info(f"  能级 {lv}: 冻结误差 = {frozen_errors[lv] * 1000:.4f} mHa "
                    f"(E_FCI={E_FCI[lv]:.8f})")
    for lv in sorted(active_levels):
        logger.info(f"  能级 {lv}: 未达标，下一轮继续 (E_FCI={E_FCI[lv]:.8f})")
    all_ok = len(active_levels) == 0 and len(frozen_levels) == K1
    logger.info(f"目标能级 {target_levels} → {('全部达标并冻结' if all_ok else '仍有未达标能级')}")
    logger.info("=" * 64)

    # ====================== 保存 npz ======================
    out_npz = os.path.join(data_dir, f"{time_str}_adaptive_nesvmc_K1={K1}.npz")
    np.savez(
        out_npz,
        target_levels=np.asarray(target_levels),
        frozen_levels=np.asarray(sorted(frozen_levels)),
        remaining_levels=np.asarray(sorted(active_levels)),
        E_fci=np.asarray([E_FCI[lv] for lv in target_levels]),
        chem_tol=CHEM_TOL, K1=K1, n_iter_first=N_ITER, n_iter_later=N_ITER_LATER,
        rounds=len(round_summary),
        elapsed_s=total_elapsed,
        round_lam=np.asarray([r["lam"] for r in round_summary], dtype=object),
        round_err=np.asarray([r["errs"] for r in round_summary], dtype=object),
        V_global=np.asarray(V_global),
    )
    logger.info(f"已保存 npz: {out_npz}")

    # ====================== 绘图：每步 λ̂（各轮）+ 误差 ======================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_r = len(round_summary)
    if n_r > 0:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        fig, axs = plt.subplots(2, 1, figsize=(10, 9))
        # 上：各轮 λ̂ 训练曲线（按物理能级着色）+ FCI 基准虚线
        offset = 0.0
        for res in round_summary:
            steps = res["steps"]
            curve = res["lam_curve"]
            for k, lv in enumerate(res["active_levels"]):
                col = colors[lv % len(colors)]
                axs[0].plot(steps + offset, curve[:, k], lw=1.2, color=col,
                            label=f"level {lv} (R{res['round']})")
                axs[0].axhline(E_FCI[lv], ls=":", lw=1, color=col)
            offset += len(steps)
        axs[0].set_xlabel("cumulative training step")
        axs[0].set_ylabel("single-occupancy Ritz λ̂ (Ha)")
        axs[0].set_title(f"Adaptive NES-VMC (LiH STO-3G 8Qbit) | K1={K1} | "
                         f"tol={CHEM_TOL * 1000:.2f} mHa | dotted = E_FCI")
        axs[0].legend(fontsize=7, ncol=2)

        # 下：各轮收尾提取误差（条形按物理能级着色）+ 阈值
        width = 0.8 / max(1, max(res["K"] for res in round_summary))
        for res in round_summary:
            for k, lv in enumerate(res["active_levels"]):
                axs[1].bar(res["round"] + (k - res["K"] / 2) * width + width / 2,
                           res["errs"][k] * 1000, width=width,
                           color=colors[lv % len(colors)], alpha=0.85)
        axs[1].axhline(CHEM_TOL * 1000, ls="--", color="k", lw=1,
                       label=f"threshold {CHEM_TOL * 1000:.2f} mHa")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("round")
        axs[1].set_ylabel("|λ − E_FCI| (mHa, log)")
        axs[1].set_title("end-of-round level error (bars colored by physical level)")
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(graph_dir, f"{time_str}_adaptive_summary_K1={K1}.png")
        fig.savefig(out_png, dpi=150)
        logger.info(f"已保存图: {out_png}")

    logger.info(f"日志: {_log_path}")
