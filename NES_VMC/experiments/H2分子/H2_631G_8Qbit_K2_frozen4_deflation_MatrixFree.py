"""
方案 A' —— 真 Matrix-Free deflation NES-VMC（冻结 K=4 + 剩余子空间 K=2，2×2 χ 行列式）

与精确版方案 A（H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py）的区别：
投影系数 c=F†A 不再用全空间内积，而是由**辅助 MCMC**（参考密度 q=Σ|φ̃|²+εΣ|ψ̃|²，
machine_pow=1 采样）的蒙特卡洛 Gram 估计 c=Ĝ⁻¹M̂、c2=Ĝ⁻¹(M̂2−ĤGG·c) 得到（Z 抵消引理，
见 NES_VMC_deflation_mf.py docstring）。训练循环全程不接触 2^N=16 维全空间数组；
稠密 H / 全空间振幅仅出现在 Step0 预检、每步监控与最终验收（诊断专用）。

尺度管理（防 0917/方案A 冻结分量漂移）：φ̃=φ·e^{−f}、ψ̃_m=ψ_m·e^{−g_m}，
每步由 aux 批 max 做 EMA（beta=NES_SCALE_BETA，首步 beta=1 fresh）；尺度变化后
c/c2 按 c[:,m]←c[:,m]·e^{Δf−Δg_m} 同步变换（χ̃ 方向不变，E_L 严格不变）。

运行：
    python H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py                 # 1500 步
    NES_N_ITER=5 python H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py    # smoke
环境变量：NES_N_ITER / NES_AUX_M(默认1024) / NES_SCALE_BETA(默认0.9) / NES_INIT_SEED
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
from jax.flatten_util import ravel_pytree
from scipy.linalg import eigh as scipy_eigh
import sys

sys.path.append('..')

from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
)
from NES_VMC_deflation import (
    build_full_stage,
    amplitudes_full,
    extract_physical_states,
)
from NES_VMC_deflation_real import (
    load_and_orthonormalize_frozen,
    deflation_oracle,
)
from NES_VMC_deflation_mf import make_mf_bundle, update_scales
from H2_6_31G_8Qbit_K2 import (
    SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock, single_edges,
)

jax.config.update("jax_enable_x64", True)
assert K == 2 and SINGLE_SIZE == 8, "本脚本按 K=2/SINGLE_SIZE=8 设计"

# ===================== Ansatz：与 K4/方案A RealFFNN 完全一致 =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        z = (u + 1e-12) + 1j * (v + 1e-12)
        return jnp.log(z)


class NESTotalAnsatz_stable(nnx.Module):
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

logger = logging.getLogger("NES_VMC_K2_frozen4_deflation_MatrixFree")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
log_path = f"{log_dir}/{time_str}_nes_vmc_K2_frozen4_deflation_MatrixFree.log"
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setFormatter(simple_formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(simple_formatter)
logger.addHandler(ch)

# ====================== 超参（沿用方案 A） ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER = int(os.environ.get("NES_N_ITER", 1500))
Natural_Grad = True
clip_norm = 20.0
per_param_clip = 0.5
lr = 0.1
qgt_diag_shift = 0.1
SAVE_INTERVAL = 20
INIT_SEED = int(os.environ.get("NES_INIT_SEED", 11))
MAX_INIT_COND = 1e6
AUX_M = int(os.environ.get("NES_AUX_M", 1024))
AUX_CHAINS = 8
AUX_SWEEP = 20
SCALE_BETA = float(os.environ.get("NES_SCALE_BETA", 0.9))
EPS_MIX = 0.25

NPZ_PATH = os.path.join(this_file_dir, "冻结K4", "近似本征态_K4_26-10-07-19-57.npz")
PKL_PATH = os.path.join(this_file_dir, "data", "26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl")
HISTORY_FILE = f"{data_dir}/{time_str}_history_K2_frozen4_deflation_MatrixFree.pkl"

# ====================== 步骤 0：冻结重建 + oracle（全空间仅诊断） ======================
H_dense, all_states = build_full_stage(ha, hi)
all_states_j = jnp.asarray(all_states)
F, phi_raw, gs_residuals = load_and_orthonormalize_frozen(NPZ_PATH, hi)
nu, u_vecs, qhq_evals = deflation_oracle(F, H_dense, n_keep=K)
E_exact, U_exact = np.linalg.eigh(H_dense)

npz = np.load(NPZ_PATH)
v_rot = np.asarray(npz["v_rot"], dtype=np.complex128)
with open(PKL_PATH, "rb") as f:
    old_params = pickle.load(f)["params"][-1]

old_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=4,
    hidden_dim=SINGLE_SIZE + 4, rngs=nnx.Rngs(11),
)
frozen_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
              for a in old_ansatz.single_ansatz_list]
frozen_params = [old_params["single_ansatz_list"][j] for j in range(4)]

# 冻结网络列 vs npz φ_k 保真度断言（gauge 约定，抄 B' L167-174）
A_old = np.exp(np.stack([np.asarray(frozen_sms[j](frozen_params[j], all_states_j), dtype=np.complex128)
                         for j in range(4)], axis=1))
Phi_net = A_old @ v_rot
Phi_net_n = Phi_net / np.linalg.norm(Phi_net, axis=0, keepdims=True)
fid_cols = [float(abs(np.vdot(phi_raw[:, k], Phi_net_n[:, k])) ** 2) for k in range(4)]
assert min(fid_cols) > 0.999, "gauge 约定不一致，禁止继续"

Phi_raw_all = Phi_net                                   # (16,4) 未缩放冻结列（监控用原始基）
Gphi_ex = Phi_raw_all.conj().T @ Phi_raw_all            # 原始 Gram（监控对照 c 用）

logger.info("\n" + "=" * 60)
logger.info(f"方案 A' 真 Matrix-Free deflation NES-VMC（冻结 K=4 + K={K}，MC Gram 投影系数）")
logger.info(f"冻结重建列保真度 = {np.round(fid_cols, 8)} | GS 残差 {np.round(gs_residuals, 4)}")
logger.info(f"oracle ν = {np.round(nu, 8)} | ν−E(mHa) = {np.round((nu - E_exact[4:4+K]) * 1000, 4)}")
logger.info(f"理论 Loss 目标 ≈ Σν = {nu.sum():.8f}")
logger.info(f"超参: lr={lr} diag_shift={qgt_diag_shift} clip={clip_norm}/{per_param_clip} N_ITER={N_ITER} "
            f"main: chains={N_CHAINS}x{N_SAMPLES_PER_CHAIN} sweep={SWEEP_SIZE} | "
            f"aux: M={AUX_M} chains={AUX_CHAINS} sweep={AUX_SWEEP} eps={EPS_MIX} beta={SCALE_BETA}")
logger.info("=" * 60)

# ====================== fresh active 初始化 + 投影可逆性预检（同方案 A） ======================
def _chi_full_np(params):
    A = np.asarray(amplitudes_full(sms, params, all_states_j)).astype(np.complex128)
    c = F.conj().T @ A
    return A, c, A - F @ c

single_machine_list = None
total_ansatz = None
total_params = None
seed = INIT_SEED
while True:
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE, n_states=K,
        hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(seed),
    )
    sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list]
    total_params = nnx.split(total_ansatz)[1]
    A, c0, chi = _chi_full_np(total_params)
    cond_chi = np.linalg.cond(chi)
    if cond_chi < MAX_INIT_COND:
        break
    logger.warning(f"seed={seed} 初始 cond(χ_full)={cond_chi:.2e} 过大，换 seed")
    seed += 1
logger.info(f"初始化 seed={seed} | cond(χ_full)={cond_chi:.3e}")

# ====================== bundle + 采样器 ======================
bundle = make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms,
                        n_spin=SINGLE_SIZE, eps=EPS_MIX)

f = jnp.asarray(0.0)
g = jnp.zeros(K)


def aux_machine(params_aux, sigma):
    return bundle["aux_logq"](params_aux, sigma)


def sample_machine(params, sigma):
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
# aux 预热（f=g=0 尺度下先跑一轮，避免首步拖尾偏差）
_s, aux_state = aux_sampler.sample(machine=aux_machine, parameters=(total_params, f, g),
                                    state=aux_state, chain_length=AUX_M // AUX_CHAINS)

sampler_rng = jax.random.PRNGKey(21)
sampler_state = None
c_cur = jnp.zeros((4, K), dtype=jnp.complex128)
c2_cur = jnp.zeros((4, K), dtype=jnp.complex128)

# ====================== 历史容器 ======================
loss_history, steps_history = [], []
lam_history, params_history = [], []
diag_history = []
n_skipped = 0
n_grad_clip = 0

start_time = time.time()

# ====================== 训练循环（matrix-free） ======================
TIMING = bool(int(os.environ.get("NES_TIMING", "0")))
_t_acc = {}

def _tick(name, t0):
    if TIMING:
        jax.block_until_ready(0)
        _t_acc[name] = _t_acc.get(name, 0.0) + (time.time() - t0)


for step in range(N_ITER):
    params_history.append(total_params)

    # --- 1. aux 采样 → MC Gram 估计 c/c2 ---
    _t0 = time.time()
    s_aux, aux_state = aux_sampler.sample(
        machine=aux_machine, parameters=(total_params, f, g),
        state=aux_state, chain_length=AUX_M // AUX_CHAINS,
    )
    sigma_aux = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
    c_mc, c2_mc, max_phi2, max_psi, aux_g = bundle["estimate_grams"](total_params, sigma_aux, f, g)
    _tick("aux+grams", _t0)
    grams_ok = bool(aux_g["grams_finite"])
    if not grams_ok:
        logger.warning(f"【Step {step}】grams 非有限，跳过整步")
        n_skipped += 1
        loss_history.append(np.nan)
        lam_history.append(np.array([np.nan, np.nan]))
        steps_history.append(step)
        diag_history.append({"valid_ratio": np.nan, "ess": np.nan, "c_err": np.nan,
                             "frozen_res": np.nan, "grad_norm_raw": np.nan,
                             "grad_norm_natural": np.nan, "cond_S": np.nan,
                             "max_absA": np.nan, "w_ess": np.nan, "max_w_ratio": np.nan})
        continue

    # --- 2. EMA 尺度更新 + c/c2 同步变换（χ̃ 方向不变） ---
    beta = 1.0 if step == 0 else SCALE_BETA
    f_np, g_np = update_scales(float(f), np.asarray(g), max_phi2, max_psi, beta=beta)
    d_f = f_np - float(f)
    d_g = g_np - np.asarray(g)
    trans = np.exp(d_f - d_g)                      # 每列 e^{Δf−Δg_m}
    c_cur = jnp.asarray(np.asarray(c_mc) * trans[None, :])
    c2_cur = jnp.asarray(np.asarray(c2_mc) * trans[None, :])
    f = jnp.asarray(f_np)
    g = jnp.asarray(g_np)

    # --- 3. 主采样（联合 2 walker，machine_pow=1 按 |Ψ_F|） ---
    if sampler_state is None:
        sampler_state = nes_sampler.init_state(
            sample_machine, (total_params, c_cur, c2_cur, f, g), sampler_rng
        )
    _t0 = time.time()
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine,
        parameters=(total_params, c_cur, c2_cur, f, g),
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    sigma_flat = jnp.asarray(np.asarray(samples_raw).reshape(-1, hi_ext.size))
    _tick("main_sample", _t0)

    # --- 4. 梯度 / SR 更新 ---
    _t0 = time.time()
    grad_raw, loss_mean, E_L_mean, aux = bundle["grad_fn"](
        total_params, sigma_flat, c_cur, c2_cur, f, g
    )
    _tick("grad", _t0)
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
        _t0 = time.time()
        S = bundle["qgt_fn"](total_params, sigma_flat, qgt_diag_shift, c_cur, c2_cur, f, g)
        ng_flat = jnp.linalg.solve(S, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = float(jnp.linalg.norm(ng_flat))
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)
        _tick("qgt+sr+update", _t0)
    else:
        updates, opt_state = optimizer.update(grad_raw, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

    # --- 5. 全空间监控（仅诊断，不进训练回路） ---
    _t0 = time.time()
    A, c_ex_raw, chi = _chi_full_np(total_params)
    # MC c 在原始基下的等价系数：c_raw[:,m] = c_cur[:,m]·e^{g_m−f}
    c_mc_raw = np.asarray(c_cur) * np.exp(np.asarray(g) - float(f))[None, :]
    c_ref_raw = np.linalg.solve(Gphi_ex + 1e-12 * np.eye(4), Phi_raw_all.conj().T @ A)
    c_err = float(np.abs(c_mc_raw - c_ref_raw).max())
    S_chi = chi.conj().T @ chi
    M_chi = chi.conj().T @ (H_dense @ chi)
    lam_sub, v_sub = scipy_eigh(M_chi + 1e-14 * S_chi, S_chi)
    lam_sub = np.sort(lam_sub.real)
    cond_S = float(np.linalg.cond(S_chi))
    chi_n = chi / np.maximum(np.linalg.norm(chi, axis=0, keepdims=True), 1e-30)
    frozen_res = float(np.abs(F.conj().T @ chi_n).max())
    max_absA = float(np.abs(A).max())
    if max_absA > 1e3:
        logger.warning(f"【Step {step}】漂移告警 max|A|={max_absA:.2e}")

    accept = getattr(sampler_state, "average_acceptance", None)
    logger.info(
        f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.8f} (目标Σν={nu.sum():.8f}) | "
        f"λ0={lam_sub[0]:.8f} λ1={lam_sub[1]:.8f} | f={float(f):.3f} g={np.round(np.asarray(g),3)}"
    )
    logger.info(
        f"诊断 | valid={float(aux['valid_ratio']):.3f} | ESS={float(aux['ess']):.0f}/{sigma_flat.shape[0]} | "
        f"c_err(MC vs 精确)={c_err:.2e} | 冻结剩余={frozen_res:.2e} | cond(S_χ)={cond_S:.2e} | "
        f"max|A|={max_absA:.2e} | aux_ess={float(aux_g['w_ess']):.0f}/{AUX_M} | "
        f"aux_maxw={float(aux_g['max_w_ratio']):.2f} | accept={float(accept) if accept is not None else float('nan'):.3f}"
    )
    logger.info(f"梯度 | raw={grad_norm_raw:.4f} per_param={grad_norm_per_param:.4f} natural={grad_norm_natural:.4f}")
    if TIMING:
        _tick("monitor+log", _t0)
        logger.info("TIMING | " + " | ".join(f"{k}={v/(step+1)*1000:.0f}ms" for k, v in _t_acc.items()))
    logger.info("#-----------------------------------------#")

    loss_history.append(float(jnp.real(loss_mean)))
    lam_history.append(lam_sub)
    steps_history.append(step)
    diag_history.append({
        "valid_ratio": float(aux["valid_ratio"]),
        "ess": float(aux["ess"]),
        "c_err": c_err,
        "frozen_res": frozen_res,
        "cond_S": cond_S,
        "max_absA": max_absA,
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
                "diag": diag_history, "seed": seed, "nu_oracle": nu,
                "scales": {"f": float(f), "g": np.asarray(g).tolist()},
            }, fsave)

elapsed = time.time() - start_time
logger.info(f"\n训练结束：{N_ITER} 步 | 跳过 {n_skipped} | 梯度裁剪 {n_grad_clip} | 总耗时 {elapsed:.1f}s")

# ====================== 提取 φ̂₄,φ̂₅ + 验收（全空间仅此处） ======================
A, c_ex_raw, chi = _chi_full_np(total_params)
phis, lams = extract_physical_states(chi, H_dense)

logger.info(f"\n================ φ̂₄..φ̂_{3+K} 验收（方案 A' Matrix-Free） ================")
logger.info(f"{'level':>5} | {'λ(Rayleigh)':>12} | {'ν(oracle)':>12} | {'E(FCI)':>12} | "
            f"{'λ-E(mHa)':>9} | {'Var[E_loc]':>10} | {'FCI重叠':>9} | {'冻结剩余重叠':>12}")
rows = []
for k in range(K):
    phi = phis[k]
    Hphi = H_dense @ phi
    E_ray = float(np.vdot(phi, Hphi).real)
    var = float(np.vdot(Hphi, Hphi).real - E_ray ** 2)
    ov_fci = float(np.abs(U_exact[:, 4 + k].conj() @ phi) ** 2)
    ov_frozen = float((np.abs(phi_raw.conj().T @ phi) ** 2).max())
    rows.append((E_ray, var, ov_fci, ov_frozen))
    logger.info(
        f"{4 + k:>5} | {E_ray:>12.8f} | {nu[k]:>12.8f} | {E_exact[4 + k]:>12.8f} | "
        f"{(E_ray - E_exact[4 + k]) * 1000:>9.4f} | {var:>10.2e} | {ov_fci:>9.5f} | {ov_frozen:>12.2e}"
    )

# 验收线：误差<2mHa、重叠>0.99、Var<1e-4、冻结剩余<1e-3、耗时<500s
ok = all(abs(r[0] - E_exact[4 + k]) * 1000 < 2.0 and r[2] > 0.99 and r[1] < 1e-4 and r[3] < 1e-3
         for k, r in enumerate(rows)) and elapsed < 500.0
logger.info(f"验收（误差<2mHa & 重叠>0.99 & Var<1e-4 & 冻结剩余<1e-3 & 耗时<500s）: {'PASS ✅' if ok else 'FAIL ❌'}")

# ====================== 保存 ======================
out_npz = os.path.join(this_file_dir, "冻结K4", f"近似本征态_K4冻结_K{K}扩展_MatrixFree_{time_str}.npz")
S_chi = chi.conj().T @ chi
M_chi = chi.conj().T @ (H_dense @ chi)
lam2, v2 = scipy_eigh(M_chi + 1e-14 * S_chi, S_chi)
v2 = v2[:, np.argsort(lam2.real)]
np.savez(
    out_npz,
    phi_vecs=np.stack(phis, axis=1),
    all_states=all_states,
    chi_full=chi,
    v_rot=v2,
    lam_subspace=np.array([r[0] for r in rows]),
    nu_oracle=nu,
    E_exact=E_exact[4:4+K],
    var_eloc=np.array([r[1] for r in rows]),
    overlap_fci=np.array([r[2] for r in rows]),
    overlap_frozen_residual=np.array([r[3] for r in rows]),
    cond_S_chi=np.linalg.cond(S_chi),
    seed=seed, N_ITER=N_ITER, elapsed_s=elapsed,
)
with open(HISTORY_FILE, "wb") as fsave:
    pickle.dump({
        "steps": steps_history, "loss": loss_history, "lam": lam_history,
        "params": params_history, "diag": diag_history, "seed": seed, "nu_oracle": nu,
        "elapsed_s": elapsed, "n_skipped": n_skipped, "n_grad_clip": n_grad_clip,
    }, fsave)
logger.info(f"已保存 npz: {out_npz}\n已保存 history: {HISTORY_FILE}\n日志: {log_path}")

# ====================== 绘图（两段式防 NaN；图内标签英文） ======================
import matplotlib
import matplotlib.pyplot as plt

steps_arr = np.asarray(steps_history)
lam_arr = np.asarray(lam_history)
diag = {k: np.asarray([d[k] for d in diag_history]) for k in diag_history[0]}
err_mha = (lam_arr - E_exact[None, 4:4+K]) * 1000.0
loss_arr = np.asarray(loss_history)

fig, axs = plt.subplots(6, 1, figsize=(10, 28))

axs[0].plot(steps_arr, lam_arr, lw=1.2, label=[f"lambda_{k} (phi_{4+k})" for k in range(K)])
for k in range(K):
    axs[0].axhline(nu[k], color=["red", "orange"][k], ls="--", alpha=0.5, label=f"oracle nu_{k}={nu[k]:.6f}")
    axs[0].axhline(E_exact[4 + k], color="gray", ls=":", alpha=0.7, label=f"FCI E{4+k}={E_exact[4+k]:.6f}")
axs[0].set_ylabel("Energy (Ha)")
axs[0].set_title(f"Scheme A' matrix-free deflated NES-VMC, frozen K=4 + K={K}, H2 6-31G ({N_ITER} steps, {elapsed:.0f}s)")
axs[0].legend(fontsize=8)

m = np.isfinite(loss_arr)
axs[1].plot(steps_arr[m], loss_arr[m], lw=1.0)
axs[1].axhline(nu.sum(), color="red", ls="--", alpha=0.6, label=f"target sum(nu)={nu.sum():.6f}")
axs[1].set_ylabel("Loss = Re tr(E_L)")
axs[1].legend(fontsize=8)

axs[2].axhspan(0, 1.6, facecolor="lightblue", alpha=0.3, zorder=0, label="chemical accuracy (1.6 mHa)")
for k in range(K):
    mk = np.isfinite(err_mha[:, k])
    axs[2].plot(steps_arr[mk], err_mha[mk, k], lw=1.2, label=f"E{4+k} error")
axs[2].set_yscale("symlog", linthresh=1e-2)
axs[2].set_ylabel("lambda - E_FCI (mHa)")
axs[2].legend(fontsize=8)

for lbl, key in [("valid_ratio", "valid_ratio"), ("ESS frac", "ess"), ("aux w_ess frac", "w_ess")]:
    v = diag[key].copy()
    v = v / (N_CHAINS * N_SAMPLES_PER_CHAIN if key == "ess" else (AUX_M if key == "w_ess" else 1.0))
    mk = np.isfinite(v)
    axs[3].plot(steps_arr[mk], v[mk], lw=1.0, label=lbl)
mk = np.isfinite(diag["frozen_res"])
axs[3].semilogy(steps_arr[mk], np.maximum(diag["frozen_res"][mk], 1e-32), lw=1.0, color="tab:red", label="frozen residual")
axs[3].set_ylabel("sampling / projection health")
axs[3].set_xlabel("step")
axs[3].legend(fontsize=8)

mk = np.isfinite(diag["c_err"])
axs[4].semilogy(steps_arr[mk], np.maximum(diag["c_err"][mk], 1e-32), lw=1.0)
axs[4].set_ylabel("max|c_MC - c_exact| (raw basis)")
axs[4].set_xlabel("step")

for key, lbl in [("grad_norm_raw", "grad_norm_raw"), ("grad_norm_natural", "grad_norm_natural"), ("max_absA", "max|A|")]:
    v = diag[key]
    mk = np.isfinite(v) & (v > 0)
    axs[5].semilogy(steps_arr[mk], v[mk], lw=1.0, label=lbl)
axs[5].set_ylabel("gradient / drift (log)")
axs[5].set_xlabel("step")
axs[5].legend(fontsize=8)

fig.tight_layout()
out_png = os.path.join(this_file_dir, "冻结K4", f"训练曲线_K2冻结4_MatrixFree_{time_str}.png")
fig.savefig(out_png, dpi=150)
logger.info(f"已保存图: {out_png}")
