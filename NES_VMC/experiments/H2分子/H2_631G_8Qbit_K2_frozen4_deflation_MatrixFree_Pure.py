"""
方案 A' 纯净版 —— 真 Matrix-Free deflation NES-VMC（无任何全空间自检）

与 H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py 的关系：
训练更新路径（aux 采样 → MC Gram → 尺度更新 → 主采样 → 梯度/SR → 更新）逐行相同；
删除了全部五处「小系统特权」的全空间诊断/定标代码（方案已验证有效，见
解读文档/1019-方案A-prime训练脚本逐函数解读-MatrixFree与build_full_stage.md §1）：
  ① build_full_stage（H_dense/all_states staging）
  ② deflation_oracle（ν 定标）与 FCI 对角化
  ③ 冻结重建保真度断言（16 维列比对）
  ④ 初始化 cond(χ_full) 全空间预检（改为固定 seed，参照运行已证 seed=11 健康）
  ⑤ 每步全空间监控（c_err/λ_sub/冻结剩余/max|A|）与收尾 extract_physical_states

删除后本脚本**不接触任何 2^N 维数组**：前向、梯度、QGT、采样、提取全部逐点。
监控改为 matrix-free 量：Loss=Re tr⟨E_L⟩、λ̂=eig(sym(⟨E_L⟩))（方法报告 §2.1，
2×2 小矩阵）、valid/ESS、aux 权重健康、尺度 f/g、梯度范数。

物理态提取（matrix-free）：训练结束后在最终尺度下采一批 aux 样本，用同一 q 的
重要性权重构造 ⟨χ|χ⟩、⟨χ|H|χ⟩ 的 MC Gram（Z 在广义本征中严格抵消），解 2×2
广义本征 → λ̂ 与 active 块旋转 v_active；φ̂_k = Σ_m χ̃_m·v_active[m,k] 由
（旧网络参数 + active 参数 + c/f/g/v_rot/v_active）完全决定，存 npz/pkl。

运行：
    python H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py                 # 1500 步
    NES_N_ITER=5 python H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py    # smoke
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
    Ham_psi_scaled,
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

logger = logging.getLogger("NES_VMC_K2_frozen4_deflation_MatrixFree_Pure")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
log_path = f"{log_dir}/{time_str}_nes_vmc_K2_frozen4_deflation_MatrixFree_Pure.log"
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setFormatter(simple_formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(simple_formatter)
logger.addHandler(ch)

# ====================== 超参（与已验证 A' 完全一致） ======================
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
AUX_M = int(os.environ.get("NES_AUX_M", 1024))
AUX_CHAINS = 8
AUX_SWEEP = 20
SCALE_BETA = float(os.environ.get("NES_SCALE_BETA", 0.9))
EPS_MIX = 0.25

# 冻结源：npz 只取 v_rot（4×4 列约定，非全空间对象）；pkl 取旧网络参数
NPZ_PATH = os.path.join(this_file_dir, "冻结K4", "近似本征态_K4_26-10-07-19-57.npz")
PKL_PATH = os.path.join(this_file_dir, "data", "26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl")
HISTORY_FILE = f"{data_dir}/{time_str}_history_K2_frozen4_deflation_MatrixFree_Pure.pkl"

# ====================== 步骤 0：冻结列的逐点表示（无全空间运算） ======================
npz = np.load(NPZ_PATH)
v_rot = np.asarray(npz["v_rot"], dtype=np.complex128)          # (4,4) 列约定 φ_k=Σ_j ψ_j^old v_rot[j,k]
with open(PKL_PATH, "rb") as f:
    old_params = pickle.load(f)["params"][-1]

old_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=4,
    hidden_dim=SINGLE_SIZE + 4, rngs=nnx.Rngs(11),
)
frozen_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
              for a in old_ansatz.single_ansatz_list]
frozen_params = [old_params["single_ansatz_list"][j] for j in range(4)]

logger.info("\n" + "=" * 60)
logger.info(f"方案 A' 纯净版 真 Matrix-Free deflation NES-VMC（冻结 K=4 + K={K}，无全空间自检）")
logger.info(f"冻结源: npz v_rot + pkl params[-1]（gauge 约定与保真度已在参照运行 26-10-16-02-23 验证）")
logger.info(f"超参: lr={lr} diag_shift={qgt_diag_shift} clip={clip_norm}/{per_param_clip} N_ITER={N_ITER} "
            f"main: chains={N_CHAINS}x{N_SAMPLES_PER_CHAIN} sweep={SWEEP_SIZE} | "
            f"aux: M={AUX_M} chains={AUX_CHAINS} sweep={AUX_SWEEP} eps={EPS_MIX} beta={SCALE_BETA}")
logger.info("=" * 60)

# ====================== fresh active 初始化（固定 seed，无 cond 预检） ======================
seed = INIT_SEED
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K,
    hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(seed),
)
sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list]
total_params = nnx.split(total_ansatz)[1]
logger.info(f"active 列 fresh 初始化 seed={seed}（参照运行同 seed，cond 预检已证健康）")

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

# ====================== 训练循环（matrix-free，与参照版逐行同构） ======================
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
        lam_history.append(np.array([np.nan, np.nan]))
        steps_history.append(step)
        diag_history.append({"valid_ratio": np.nan, "ess": np.nan, "grad_norm_raw": np.nan,
                             "grad_norm_natural": np.nan, "w_ess": np.nan, "max_w_ratio": np.nan})
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
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine,
        parameters=(total_params, c_cur, c2_cur, f, g),
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    sigma_flat = jnp.asarray(np.asarray(samples_raw).reshape(-1, hi_ext.size))

    # --- 4. 梯度 / SR 更新 ---
    grad_raw, loss_mean, E_L_mean, aux = bundle["grad_fn"](
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

    # --- 5. matrix-free 监控：λ̂ = eig(sym(⟨E_L⟩))（方法报告 §2.1，2×2 小矩阵） ---
    E_h = 0.5 * (E_L_mean + jnp.conj(E_L_mean.T))
    lam_hat = np.sort(np.linalg.eigvalsh(np.asarray(E_h)).real)

    accept = getattr(sampler_state, "average_acceptance", None)
    logger.info(
        f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.8f} | "
        f"λ̂0={lam_hat[0]:.8f} λ̂1={lam_hat[1]:.8f} | f={float(f):.3f} g={np.round(np.asarray(g),3)}"
    )
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

# ====================== 物理态提取（matrix-free：MC Gram 广义本征，Z 抵消） ======================
# 最终尺度下采一批新 aux 样本，estimate_grams 用同一 q 加权 ⇒ Ĝ⁻¹M̂ 等无归一化偏差；
# ⟨χ|χ⟩、⟨χ|H|χ⟩ 同带 1/Z，广义本征中严格抵消。
s_aux, aux_state = aux_sampler.sample(
    machine=aux_machine, parameters=(total_params, f, g),
    state=aux_state, chain_length=AUX_M // AUX_CHAINS,
)
sigma_ext = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
c_f, _c2_f, _m1, _m2, aux_gf = bundle["estimate_grams"](total_params, sigma_ext, f, g)
if not bool(aux_gf["grams_finite"]):
    logger.warning("提取批 grams 非有限，改用训练末 c_cur")
    c_f = c_cur

V = jnp.asarray(v_rot, dtype=jnp.complex128)
Phi, Psi = bundle["cols"](total_params, sigma_ext, f, g)
chi = Psi - Phi @ c_f                                   # (M, K) 缩放基下的 χ̃
Hpsi = jnp.stack(
    [Ham_psi_scaled(ha=ha, single_machine=sms[m], params=total_params["single_ansatz_list"][m],
                    x=sigma_ext, shift=g[m]) for m in range(K)], axis=-1)
Hphi = jnp.stack(
    [Ham_psi_scaled(ha=ha, single_machine=frozen_sms[j], params=frozen_params[j],
                    x=sigma_ext, shift=f) for j in range(4)], axis=-1) @ V
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
logger.info(f"λ̂ = {np.round(lam_mc, 8)} | 参照运行（已验证）φ̂₄/₅ 收敛区间 ≈ (-0.02802, -0.01448)")
logger.info(f"Loss 末值 = {loss_history[-1]:.8f} | 参照运行末段 Loss ≈ -3.2749（=ΣE_ray+ν，见方法报告 §2.3）")

# ====================== 保存 ======================
out_npz = os.path.join(this_file_dir, "冻结K4", f"近似本征态_K2扩展_MatrixFree_Pure_{time_str}.npz")
np.savez(
    out_npz,
    lam_extract=lam_mc,
    v_active=v_active,                 # φ̂_k = Σ_m χ̃_m v_active[m,k]（χ̃=Ψ̃−Φ̃c_f，缩放基）
    c_extract=c_f,
    v_rot_frozen=v_rot,
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
        "scales": {"f": float(f), "g": np.asarray(g).tolist()},
    }, fsave)
logger.info(f"已保存 npz: {out_npz}\n已保存 history: {HISTORY_FILE}\n日志: {log_path}")

# ====================== 绘图（仅 matrix-free 监控量；图内标签英文） ======================
import matplotlib
import matplotlib.pyplot as plt

steps_arr = np.asarray(steps_history)
lam_arr = np.asarray(lam_history)
loss_arr = np.asarray(loss_history)
diag = {k: np.asarray([d[k] for d in diag_history]) for k in diag_history[0]}

fig, axs = plt.subplots(4, 1, figsize=(10, 16))

axs[0].plot(steps_arr, lam_arr, lw=1.2, label=[f"lambda_hat_{k} (active block)" for k in range(K)])
axs[0].set_ylabel("eig(sym <E_L>) (Ha)")
axs[0].set_title(f"Scheme A' PURE matrix-free deflated NES-VMC, frozen K=4 + K={K}, H2 6-31G "
                 f"({N_ITER} steps, {elapsed:.0f}s, no full-space ops)")
axs[0].legend(fontsize=8)

m = np.isfinite(loss_arr)
axs[1].plot(steps_arr[m], loss_arr[m], lw=1.0)
axs[1].set_ylabel("Loss = Re tr(E_L)")

for lbl, key, norm in [("valid_ratio", "valid_ratio", 1.0), ("ESS frac", "ess", N_CHAINS * N_SAMPLES_PER_CHAIN),
                       ("aux w_ess frac", "w_ess", AUX_M)]:
    v = diag[key] / norm
    mk = np.isfinite(v)
    axs[2].plot(steps_arr[mk], v[mk], lw=1.0, label=lbl)
axs[2].set_ylabel("sampling health")
axs[2].set_xlabel("step")
axs[2].legend(fontsize=8)

for key, lbl in [("grad_norm_raw", "grad_norm_raw"), ("grad_norm_natural", "grad_norm_natural")]:
    v = diag[key]
    mk = np.isfinite(v) & (v > 0)
    axs[3].semilogy(steps_arr[mk], v[mk], lw=1.0, label=lbl)
axs[3].set_ylabel("gradient (log)")
axs[3].set_xlabel("step")
axs[3].legend(fontsize=8)

fig.tight_layout()
out_png = os.path.join(this_file_dir, "冻结K4", f"训练曲线_K2冻结4_MatrixFree_Pure_{time_str}.png")
fig.savefig(out_png, dpi=150)
logger.info(f"已保存图: {out_png}")
