"""
方案 A' 纯净版 —— 真 Matrix-Free deflation NES-VMC（冻结 3 条物理列，探索 K=3 新能级）
【LiH / STO-3G / CAS(2e,4o) / 8 qubit · 冻结K3_探索K3 目录】

背景（为什么冻结的是能级 0、1、3，而不是 0、1、2）：
    LiH 8 比特精确谱存在成对简并（E_fcis）：
        E0 = -7.88235069
        E1 = E2 = -7.71498451   ← 二重简并
        E3 = E4 = -7.69407064   ← 二重简并
        E5 = -7.57515759
    K=4 NES-VMC 训练收敛到 λ = [-7.88234885, -7.71500477, -7.71233932, -7.69403689]：
    能级 0、1、3 已达可接受精度，能级 2（能级 1 的简并 partner）因简并子空间内旋转
    零梯度而收敛缓慢（偏 2.65 mHa）。
    本脚本只冻结 **能级 0、1、3** 三条物理列（v_rot 取列子集），active K=3 在补空间里探索。
    补空间里能量最低的三态 = 能级 2（1 的 partner）、能级 4（3 的 partner）、能级 5：
        active λ 预期 = [E2, E4, E5]，索引 ACTIVE_LEVELS = (2, 4, 5)。
    注意：这不是"跳过"简并能级——简并 partner 必须是冻结列或 active 列之一；
    本 run 让 active 把两个缺的 partner 各自重新收敛出来，并顺带拿到能级 5。
    冻结集 {0,1,3} ∪ active {2,4,5} 合起来覆盖能级 0..5。

算法与 H2分子/冻结K4_探索K2/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py 同构：
  ① 冻结源从 LiH 的 K=4 训练 pkl 读取（params[-1] + samples），v_rot 由脚本内
     样本版 MV=SVA 现算（NES_VMC_tool.compute_lam_v_from_samples），然后**取列子集**；
  ② 冻结列数 R=3 而旧网络个数 J=4：冻结列定义为 φ_k = Σ_{j<4} ψ_j^old · v_rot[j,k]，
     逐点求值仍需要 4 条旧网络；NES_VMC_deflation_mf.make_mf_bundle 已支持 V 为 (J,R)；
  ③ active K=3（探索 3 个新态），行列式 3×3；
  ④ 监控量 = 单副本均匀测度下的 Ritz 值 M v = λ S v（截断伪逆）；
     广义本征在收尾提取时用 scipy_eigh。
算法（前向、梯度、QGT、采样、监控、提取）全程逐点、不做任何精确全空间对角化；
E_fcis 只从体系模块读取、仅用作绘图参考真值与误差基准，不参与算法。

运行：
    python LiH_STO3G_CAS_8Qbit_K3_frozen3_deflation_MatrixFree_Pure.py               # 500 步
    NES_N_ITER=5 python LiH_STO3G_CAS_8Qbit_K3_frozen3_deflation_MatrixFree_Pure.py  # smoke
环境变量：NES_N_ITER / NES_AUX_M(默认1024) / NES_SCALE_BETA(默认0.9) / NES_INIT_SEED / NES_PINV_RCOND
"""
import logging
import os
import time
import pickle
import numpy as np                    # 经典数组：采样批转换、Gram 组装、绘图数据
import jax                            # 自动微分/编译后端；jax.config 控制精度
import jax.numpy as jnp               # jax 版 numpy：进入 jit/vmap 计算图的数组运算
import flax.nnx as nnx                # 神经网络模块库（NNX）：定义 FFNN 参数与 nnx.split
import netket as nk                   # 量子蒙特卡洛框架：希尔伯特空间、Metropolis 采样器
import optax                          # 优化器库：全局范数裁剪 + SGD
from jax.flatten_util import ravel_pytree   # 参数树 ⇄ 一维向量互转（SR 求解需要一维）
from scipy.linalg import eigh as scipy_eigh # scipy 广义 Hermitian 本征（收尾 3×3 提取用）
import sys

this_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_file_dir = os.path.join(this_file_dir, "..")          # LiH分子/：含 LiH_CAS_STO3G_8Qbit.py
gradparent_file_dir = os.path.join(parent_file_dir, "..")    # experiments/：含公共模块
sys.path.append(parent_file_dir)
sys.path.append(gradparent_file_dir)


# NES_VMC_V1：单列波函数 machine 与哈密顿量-波函数逐点乘法
from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
    Ham_psi_scaled,
)
# NES_VMC_deflation_mf：方案 A' 工具箱（MC Gram 投影系数的 matrix-free bundle）
from NES_VMC_deflation_mf import make_mf_bundle, update_scales
# NES_VMC_tool：样本版 MV=SVA block 对角化（算 v_rot 用）
from NES_VMC_tool import make_MS_estimator_fn, compute_lam_v_from_samples
# 体系定义文件（LiH/STO-3G，CAS(2e,4o) → 8 自旋轨道 = 8 qubit）
from LiH_CAS_STO3G_8Qbit import (
    SINGLE_SIZE, ha, hi, Hatree_Fock, single_edges, E_fcis,
)

# K：active 列数（补空间里探索的态数）；本案例 = 3
K = 3
# 联合 walker 空间与 Hop 边：直接由单组态 hi / single_edges 构造
hi_ext = hi ** K
ext_edges = jnp.asarray([(i + k * SINGLE_SIZE, j + k * SINGLE_SIZE)
                         for k in range(K) for (i, j) in single_edges])


jax.config.update("jax_enable_x64", True)   # 全链路 float64/complex128

# ===================== Ansatz：与 K4 训练逐字一致 =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    """单列（单自旋轨道占据组态 → 复对数振幅）前馈网络。"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)  # 实部头
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)  # 虚部头

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        z = (u + 1e-12) + 1j * (v + 1e-12)     # 防 log(0)
        return jnp.log(z)


class NESTotalAnsatz_stable(nnx.Module):
    """K 列总拟设：持有 K 个独立单列网络。"""
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            self.single_ansatz_list.append(
                SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(params=sub_key))
            )


# ====================== 日志配置 ======================
time_str = time.strftime("%y-%m-%d-%H-%M")
this_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

logger = logging.getLogger("NES_VMC_K3_frozen3_deflation_MatrixFree_Pure")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
log_path = f"{log_dir}/{time_str}_nes_vmc_K3_frozen3_deflation_MatrixFree_Pure.log"
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setFormatter(simple_formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(simple_formatter)
logger.addHandler(ch)

# ====================== 超参 ======================
# ---- 主采样（联合 NES walker）----
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER = int(os.environ.get("NES_N_ITER", 500))
Natural_Grad = True
clip_norm = 20.0
per_param_clip = 0.5
lr = 0.1
qgt_diag_shift = 0.1
SAVE_INTERVAL = 20
INIT_SEED = int(os.environ.get("NES_INIT_SEED", 11))
# ---- 辅助采样（单组态，估 MC Gram 用）----
AUX_M = int(os.environ.get("NES_AUX_M", 1024))
AUX_CHAINS = 8
AUX_SWEEP = 20
SCALE_BETA = float(os.environ.get("NES_SCALE_BETA", 0.9))
EPS_MIX = 0.25
PINV_RCOND = float(os.environ.get("NES_PINV_RCOND", 1e-8))


parent_dir = os.path.join(this_file_dir, '..')
# 冻结源：LiH K=4 训练收敛的 pkl（params[-1] + samples）
PKL_PATH = os.path.join(
    parent_dir, "NES-VMC K4", "data",
    "26-10-27-09-48_history_natural_gradient_LiH_molecule_K4_realFFNN.pkl",
)
HISTORY_FILE = f"{data_dir}/{time_str}_history_K3_frozen3_deflation_MatrixFree_Pure.pkl"

# ====================== 步骤 0：冻结列的逐点表示 ======================
OLD_K = 4                       # 冻结源训练的列数（K4 网络结构）
KEEP = (0, 1, 3)                # 保留的物理能级索引（0-based，按 λ_k4 升序）
N_FROZEN = len(KEEP)            # 冻结列数 R = 3
ACTIVE_LEVELS = (2, 4, 5)       # active 预期对应的能级索引（补空间最低三态）

with open(PKL_PATH, "rb") as f:
    old_hist = pickle.load(f)
old_params = old_hist["params"][-1]       # 旧 K4 训练最后一步的参数树（冻结权重）
# block 对角化用的 walker 样本：取倒数第二批（与提取脚本一致，最后一批可能不完整）
samples_k4 = np.asarray(old_hist["samples"][-2])
x_batch_k4 = samples_k4.reshape(samples_k4.shape[0], OLD_K, SINGLE_SIZE)

# 重建旧 4 列网络结构（hidden_dim=8+4=12，与 K4 训练一致）
old_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=OLD_K,
    hidden_dim=SINGLE_SIZE + OLD_K, rngs=nnx.Rngs(11),
)
# frozen_sms：4 条旧网络的 gauge-fixed callable（参数为常量，不参与梯度）
frozen_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
              for a in old_ansatz.single_ansatz_list]
frozen_params = [old_params["single_ansatz_list"][j] for j in range(OLD_K)]

# v_rot：(OLD_K,OLD_K) 列旋转矩阵，脚本内计算（样本版 MV=SVA 广义本征）：
#   φ_k = Σ_j ψ_j^old · v_rot[j,k]，列=能级升序
ms_estimator = make_MS_estimator_fn(ha, frozen_sms)
lam_k4, v_rot_full, n_valid_k4, _order = compute_lam_v_from_samples(
    ha=ha, single_machine_list=frozen_sms, total_params=old_params,
    x_batch=x_batch_k4, estimator=ms_estimator,
)
v_rot_full = np.asarray(v_rot_full, dtype=np.complex128)
# 关键：只保留 KEEP 指定的物理列 ⇒ v_rot 形状 (4,3)
v_rot = v_rot_full[:, list(KEEP)]

logger.info("\n" + "=" * 60)
logger.info(f"方案 A' 纯净版 真 Matrix-Free deflation NES-VMC（冻结物理能级 {KEEP} + 探索 K={K}）")
logger.info(f"冻结源: {PKL_PATH}")
logger.info(f"v_rot 计算完成 | 有效 walker {int(n_valid_k4)}/{x_batch_k4.shape[0]} | λ_k4 = {np.round(np.asarray(lam_k4), 8)}")
logger.info(f"保留冻结列 = {KEEP} | 冻结列数 R={N_FROZEN}（旧网络仍为 J={OLD_K}）| "
            f"active 预期能级 = {ACTIVE_LEVELS} → 参考 {np.round(np.asarray(E_fcis[list(ACTIVE_LEVELS)]), 8)}")
logger.info(f"超参: lr={lr} diag_shift={qgt_diag_shift} clip={clip_norm}/{per_param_clip} N_ITER={N_ITER} "
            f"main: chains={N_CHAINS}x{N_SAMPLES_PER_CHAIN} sweep={SWEEP_SIZE} | "
            f"aux: M={AUX_M} chains={AUX_CHAINS} sweep={AUX_SWEEP} eps={EPS_MIX} beta={SCALE_BETA}")
logger.info("=" * 60)

# ====================== fresh active 初始化 ======================
# active 必须 fresh：沿用旧列投影后 χ≈0，K×K 行列式奇异。
seed = INIT_SEED
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K,
    hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(seed),
)
sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list]
total_params = nnx.split(total_ansatz)[1]
logger.info(f"active 列 fresh 初始化 seed={seed}（K={K}，hidden_dim={SINGLE_SIZE + K}）")

# ====================== bundle + 采样器 ======================
# 注意：frozen_sms 传 4 条旧网络，v_rot 传 (4,3) ⇒ bundle 内部 R=3、J=4
bundle = make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms,
                        n_spin=SINGLE_SIZE, eps=EPS_MIX, pinv_rcond=PINV_RCOND)

f = jnp.asarray(0.0)          # 冻结列公共实尺度（φ̃=φ·e^{−f}）
g = jnp.zeros(K)              # active 列逐列实尺度（ψ̃_m=ψ_m·e^{−g_m}）


def aux_machine(params_aux, sigma):
    """辅助采样器 machine：返回 log q(σ)。"""
    return bundle["aux_logq"](params_aux, sigma)


def sample_machine(params, sigma):
    """主采样器 machine：返回联合 walker 批的 logΨ_F = log det[χ̃₁..χ̃_K]。"""
    p, cc, cc2, ff, gg = params
    return bundle["machine_batch"](p, sigma.reshape(-1, hi_ext.size), cc, ff, gg)


aux_rule = NESFermionHopRule(edges=jnp.asarray(single_edges), K=1, single_size=SINGLE_SIZE)
aux_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi, rule=aux_rule, n_chains=AUX_CHAINS, sweep_size=AUX_SWEEP, machine_pow=1,
)
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext, rule=nes_rule, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE, machine_pow=1,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)

aux_rng = jax.random.PRNGKey(5)
aux_state = aux_sampler.init_state(aux_machine, (total_params, f, g), aux_rng)
# aux 预热（f=g=0 尺度下先跑一轮）
_s, aux_state = aux_sampler.sample(machine=aux_machine, parameters=(total_params, f, g),
                                   state=aux_state, chain_length=AUX_M // AUX_CHAINS)

sampler_rng = jax.random.PRNGKey(21)
sampler_state = None
# 形状 (N_FROZEN, K)：行=保留的冻结列（3），列=active 列（K）；第 0 步即被覆盖
c_cur = jnp.zeros((N_FROZEN, K), dtype=jnp.complex128)
c2_cur = jnp.zeros((N_FROZEN, K), dtype=jnp.complex128)

# ====================== 历史容器 ======================
loss_history, steps_history = [], []
lam_history, params_history = [], []
diag_history = []
n_skipped = 0
n_grad_clip = 0

start_time = time.time()

# ====================== 训练循环 ======================
for step in range(N_ITER):
    params_history.append(total_params)

    # --- 1. aux 采样 → MC Gram 估计 c/c2 ---
    s_aux, aux_state = aux_sampler.sample(
        machine=aux_machine, parameters=(total_params, f, g),
        state=aux_state, chain_length=AUX_M // AUX_CHAINS,
    )
    sigma_aux = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
    c_mc, c2_mc, max_phi2, max_psi, aux_g = bundle["estimate_grams"](total_params, sigma_aux, f, g)
    grams_ok = bool(aux_g["grams_finite"])
    if not grams_ok:
        logger.warning(f"【Step {step}】grams 非有限，跳过整步")
        n_skipped += 1
        loss_history.append(np.nan)
        lam_history.append(np.full(K, np.nan))
        steps_history.append(step)
        diag_history.append({"valid_ratio": np.nan, "ess": np.nan, "grad_norm_raw": np.nan,
                             "grad_norm_natural": np.nan, "w_ess": np.nan, "max_w_ratio": np.nan})
        continue

    # --- 2. EMA 尺度更新 + c/c2 同步变换（χ̃ 方向不变） ---
    beta = 1.0 if step == 0 else SCALE_BETA
    f_np, g_np = update_scales(float(f), np.asarray(g), max_phi2, max_psi, beta=beta)
    d_f = f_np - float(f)
    d_g = g_np - np.asarray(g)
    trans = np.exp(d_f - d_g)                  # (K,) 每列 e^{Δf−Δg_m}
    c_cur = jnp.asarray(np.asarray(c_mc) * trans[None, :])     # 广播到 (3,K)
    c2_cur = jnp.asarray(np.asarray(c2_mc) * trans[None, :])
    f = jnp.asarray(f_np)
    g = jnp.asarray(g_np)

    # --- 3. 主采样（联合 K walker，machine_pow=1 按 |Ψ_F|） ---
    if sampler_state is None:
        sampler_state = nes_sampler.init_state(
            sample_machine, (total_params, c_cur, c2_cur, f, g), sampler_rng
        )
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine,
        parameters=(total_params, c_cur, c2_cur, f, g),
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    sigma_flat = jnp.asarray(np.asarray(samples_raw).reshape(-1, hi_ext.size))

    # --- 4. 梯度 / SR 更新 ---
    grad_raw, loss_mean, _E_L_mean, aux = bundle["grad_fn"](
        total_params, sigma_flat, c_cur, c2_cur, f, g
    )
    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = float(jnp.linalg.norm(grad_raw_flat))
    n_params = grad_raw_flat.size
    grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)

    if grad_norm_per_param > per_param_clip:
        grad_scale = per_param_clip / grad_norm_per_param
        grad_raw_flat = grad_raw_flat * grad_scale
        grad_raw = unravel_fn(grad_raw_flat)
        grad_norm_raw = float(jnp.linalg.norm(grad_raw_flat))
        grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)
        n_grad_clip += 1
        logger.info(f"【Step {step}】per-param 梯度裁剪 scale={grad_scale:.4f}")

    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw={grad_norm_raw:.2f}")

    grad_norm_natural = float("nan")
    if bool(aux["should_skip"]):
        logger.warning(f"【Step {step}】should_skip=True（valid_ratio={float(aux['valid_ratio']):.3f}），跳过更新")
        n_skipped += 1
    elif Natural_Grad:
        S = bundle["qgt_fn"](total_params, sigma_flat, qgt_diag_shift, c_cur, c2_cur, f, g)
        ng_flat = jnp.linalg.solve(S, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = float(jnp.linalg.norm(ng_flat))
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)
    else:
        updates, opt_state = optimizer.update(grad_raw, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

    # --- 5. 监控：单副本均匀测度 Ritz 值（截断伪逆） ---
    lam_hat = np.sort(np.asarray(aux_g["lam_uniform"]).real)
    pinv_n_keep = int(aux_g["pinv_n_keep"])
    pinv_cond = float(aux_g["pinv_cond"])
    if pinv_n_keep < K:
        logger.warning(
            f"【Step {step}】S 有效秩 {pinv_n_keep}/{K} < K，监控量整步屏蔽；"
            f"截断前伪逆值={np.round(lam_hat, 8)}"
        )
        lam_hat = np.full(K, np.nan)

    accept = getattr(sampler_state, "average_acceptance", None)
    logger.info(
        f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.8f} | "
        f"λ̂={np.round(lam_hat, 8)} | f={float(f):.3f} g={np.round(np.asarray(g),3)}"
    )
    logger.info(f"          λ̂(pinv keep={pinv_n_keep}/{K}, cond(S)={pinv_cond:.2e})")
    logger.info(
        f"诊断 | valid={float(aux['valid_ratio']):.3f} | ESS={float(aux['ess']):.0f}/{sigma_flat.shape[0]} | "
        f"aux_ess={float(aux_g['w_ess']):.0f}/{AUX_M} | aux_maxw={float(aux_g['max_w_ratio']):.2f} | "
        f"accept={float(accept) if accept is not None else float('nan'):.3f}"
    )
    logger.info(f"梯度 | raw={grad_norm_raw:.4f} per_param={grad_norm_per_param:.4f} natural={grad_norm_natural:.4f}")
    logger.info("#-----------------------------------------#")

    loss_history.append(float(jnp.real(loss_mean)))
    lam_history.append(lam_hat)
    steps_history.append(step)
    diag_history.append({
        "valid_ratio": float(aux["valid_ratio"]),
        "ess": float(aux["ess"]),
        "grad_norm_raw": grad_norm_raw,
        "grad_norm_natural": grad_norm_natural,
        "w_ess": float(aux_g["w_ess"]),
        "max_w_ratio": float(aux_g["max_w_ratio"]),
    })

    if (step + 1) % SAVE_INTERVAL == 0 and N_ITER > 5:
        with open(HISTORY_FILE, "wb") as fsave:
            pickle.dump({
                "steps": steps_history, "loss": loss_history, "lam": lam_history,
                "params": params_history[:: max(1, len(params_history) // 15)][-15:],
                "diag": diag_history, "seed": seed,
                "scales": {"f": float(f), "g": np.asarray(g).tolist()},
            }, fsave)

elapsed = time.time() - start_time
logger.info(f"\n训练结束：{N_ITER} 步 | 跳过 {n_skipped} | 梯度裁剪 {n_grad_clip} | 总耗时 {elapsed:.1f}s")

# ====================== 物理态提取（matrix-free：MC Gram 广义本征） ======================
s_aux, aux_state = aux_sampler.sample(
    machine=aux_machine, parameters=(total_params, f, g),
    state=aux_state, chain_length=AUX_M // AUX_CHAINS,
)
sigma_ext = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
c_f, _c2_f, _m1, _m2, aux_gf = bundle["estimate_grams"](total_params, sigma_ext, f, g)
if not bool(aux_gf["grams_finite"]):
    logger.warning("提取批 grams 非有限，改用训练末 c_cur")
    c_f = c_cur

V = jnp.asarray(v_rot, dtype=jnp.complex128)          # (4,3)
Phi, Psi = bundle["cols"](total_params, sigma_ext, f, g)   # (M,3), (M,K)
chi = Psi - Phi @ c_f                                   # (M,K)
Hpsi = jnp.stack(
    [Ham_psi_scaled(ha=ha, single_machine=sms[m], params=total_params["single_ansatz_list"][m],
                    x=sigma_ext, shift=g[m]) for m in range(K)], axis=-1)
Hphi = jnp.stack(
    [Ham_psi_scaled(ha=ha, single_machine=frozen_sms[j], params=frozen_params[j],
                    x=sigma_ext, shift=f) for j in range(OLD_K)], axis=-1) @ V
Hchi = Hpsi - Hphi @ c_f
q = jnp.sum(jnp.abs(Phi) ** 2, axis=-1) + EPS_MIX * jnp.sum(jnp.abs(Psi) ** 2, axis=-1)
wi = 1.0 / (q * sigma_ext.shape[0])
S_chi = np.asarray(jnp.conj(chi).T @ (wi[:, None] * chi), dtype=np.complex128)
M_chi = np.asarray(jnp.conj(chi).T @ (wi[:, None] * Hchi), dtype=np.complex128)
S_chi = 0.5 * (S_chi + S_chi.conj().T) + 1e-12 * np.eye(K)
M_chi = 0.5 * (M_chi + M_chi.conj().T)
lam_mc, v_active = scipy_eigh(M_chi + 1e-14 * S_chi, S_chi)
order = np.argsort(lam_mc.real)
lam_mc, v_active = lam_mc[order].real, v_active[:, order]

logger.info(f"\n================ 提取（matrix-free MC Gram） ================")
logger.info(f"λ_extract（预期对应能级 {ACTIVE_LEVELS}，升序）= {np.round(lam_mc, 8)}")
logger.info(f"Loss 末值 = {loss_history[-1]:.8f} | 监控量 = 每步单副本均匀测度 Ritz 值（M v = λ S v）")

# ====================== 保存 ======================
output_npz_dir = os.path.join(this_file_dir, "data")
os.makedirs(output_npz_dir, exist_ok=True)
out_npz = os.path.join(output_npz_dir, f"近似本征态_K{K}扩展_frozen{N_FROZEN}_MatrixFree_Pure_{time_str}.npz")
np.savez(
    out_npz,
    lam_extract=lam_mc.real,
    active_levels=np.asarray(ACTIVE_LEVELS),
    keep=np.asarray(KEEP),
    v_active=v_active,                 # φ̂_k = Σ_m χ̃_m v_active[m,k]（χ̃=Ψ̃−Φ̃c_f，缩放基）
    c_extract=c_f,                     # 提取用投影系数 (3,K)（缩放基；配 scales_f/g 还原原始基）
    v_rot_frozen=v_rot,                # 冻结列约定 (4,3)（复现所冻结的 3 条物理列用）
    v_rot_full=v_rot_full,             # 旧 4 列完整旋转（便于对照 λ_k4）
    lam_k4=np.asarray(lam_k4),
    scales_f=float(f), scales_g=np.asarray(g),
    seed=seed, N_ITER=N_ITER, elapsed_s=elapsed,
    n_skipped=n_skipped, n_grad_clip=n_grad_clip,
    loss_final=loss_history[-1],
)
with open(HISTORY_FILE, "wb") as fsave:
    pickle.dump({
        "steps": steps_history, "loss": loss_history, "lam": lam_history,
        "params": params_history, "diag": diag_history, "seed": seed,
        "elapsed_s": elapsed, "n_skipped": n_skipped, "n_grad_clip": n_grad_clip,
        "lam_extract": lam_mc, "v_active": v_active, "c_extract": c_f,
        "keep": KEEP, "active_levels": ACTIVE_LEVELS, "v_rot": v_rot,
        "scales": {"f": float(f), "g": np.asarray(g).tolist()},
    }, fsave)
logger.info(f"已保存 npz: {out_npz}\n已保存 history: {HISTORY_FILE}\n日志: {log_path}")

# ====================== 绘图 ======================
import matplotlib
import matplotlib.pyplot as plt

steps_arr = np.asarray(steps_history)
lam_arr = np.asarray(lam_history)
loss_arr = np.asarray(loss_history)
diag = {k: np.asarray([d[k] for d in diag_history]) for k in diag_history[0]}

fig, axs = plt.subplots(5, 1, figsize=(10, 20))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:K]
# 参照基准：active 预期能级对应的 FCI 值（非连续的 E_fcis[2],E_fcis[4],E_fcis[5]）
ref_targets = np.asarray(E_fcis[list(ACTIVE_LEVELS)], dtype=float)
lam_mc = np.asarray(lam_mc, dtype=float)

# 子图 1：每步 Ritz 值 λ̂ + FCI 基准虚线
for k in range(K):
    lvl = ACTIVE_LEVELS[k]
    axs[0].plot(steps_arr, lam_arr[:, k], lw=1.2, color=colors[k],
                label=f"NES-VMC level {lvl}")
    axs[0].axhline(ref_targets[k], lw=1.2, linestyle=':', color=colors[k],
                   label=f"FCI level {lvl}")
axs[0].set_ylabel("Ritz λ (uniform measure, Ha)")
axs[0].set_title(f"Scheme A' PURE matrix-free deflated NES-VMC, LiH STO-3G 8Qbit, "
                 f"frozen levels {KEEP} + active K={K} ({N_ITER} steps, {elapsed:.0f}s)")
axs[0].xaxis.set_major_formatter('{x:.0f}')
axs[0].legend(fontsize=8)

# 子图 2：Loss
m = np.isfinite(loss_arr)
axs[1].plot(steps_arr[m], loss_arr[m], lw=1.0)
axs[1].set_ylabel("Loss = Re tr(E_L)")
axs[1].xaxis.set_major_formatter('{x:.0f}')

# 子图 3：采样健康度
for lbl, key, norm in [("valid_ratio", "valid_ratio", 1.0), ("ESS frac", "ess", N_CHAINS * N_SAMPLES_PER_CHAIN),
                       ("aux w_ess frac", "w_ess", AUX_M)]:
    v = diag[key] / norm
    mk = np.isfinite(v)
    axs[2].plot(steps_arr[mk], v[mk], lw=1.0, label=lbl)
axs[2].set_ylabel("sampling health")
axs[2].set_xlabel("step")
axs[2].xaxis.set_major_formatter('{x:.0f}')
axs[2].legend(fontsize=8)

# 子图 4：梯度范数
for key, lbl in [("grad_norm_raw", "grad_norm_raw"), ("grad_norm_natural", "grad_norm_natural")]:
    v = diag[key]
    mk = np.isfinite(v) & (v > 0)
    axs[3].semilogy(steps_arr[mk], v[mk], lw=1.0, label=lbl)
axs[3].set_ylabel("gradient (log)")
axs[3].set_xlabel("step")
axs[3].legend(fontsize=8)
axs[3].xaxis.set_major_formatter('{x:.0f}')

# 子图 5：误差 |λ̂_k(step) − E_FCI,level|
for k in range(K):
    error_k = np.abs(lam_arr[:, k] - ref_targets[k])
    mk = np.isfinite(error_k) & (error_k > 0)
    axs[4].semilogy(steps_arr[mk], error_k[mk], lw=1.2, color=colors[k],
                    label=f"level {ACTIVE_LEVELS[k]} error")
axs[4].axhspan(0, 0.0016, facecolor="lightblue", alpha=0.3, zorder=0,
               label="chemical accuracy (1.6 mHa)")
axs[4].set_ylabel("Error = E - EFCI (Ha)")
axs[4].set_xlabel("step")
axs[4].set_yscale("log")
axs[4].xaxis.set_major_formatter('{x:.0f}')
axs[4].legend(fontsize=8)

fig.tight_layout()
os.makedirs(os.path.join(this_file_dir, "graph"), exist_ok=True)
out_png = os.path.join(this_file_dir, "graph", f"训练曲线_K{K}冻结{N_FROZEN}_MatrixFree_Pure_{time_str}.png")
fig.savefig(out_png, dpi=150)
logger.info(f"已保存图（监控量：单副本均匀测度 Ritz 值，截断伪逆）: {out_png}")

# 末步误差 vs FCI（仅日志参考）
err_end = np.abs(np.asarray(lam_arr[-1], dtype=float) - ref_targets) * 1000.0
logger.info(f"末步 |λ̂ − E_FCI| (mHa) = {np.round(err_end, 4)}（对应能级 {ACTIVE_LEVELS}）| "
            f"化学精度阈值 1.6 mHa → {('全部达标' if np.all(np.isfinite(err_end)) and np.all(err_end < 1.6) else ('无有效监控值' if not np.any(np.isfinite(err_end)) else '未达标: ' + str(np.where(np.isfinite(err_end) & (err_end >= 1.6))[0].tolist())))}")
logger.info(f"（参考）收尾提取 λ_extract = {np.round(np.asarray(lam_mc, dtype=float), 8)} | "
            f"其与 FCI 偏差 (mHa) = {np.round(np.abs(np.asarray(lam_mc, dtype=float) - ref_targets) * 1000.0, 4)}")
