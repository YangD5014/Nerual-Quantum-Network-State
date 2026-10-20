"""
冻结 K=4 近似本征态 + 剩余子空间训练 K=2（方案 A：Matrix-Free 投影态优化）

目标：复用 K=4 训练 300 步得到的 φ_0..φ_3（`冻结K4/近似本征态_K4_26-10-07-19-57.npz`，
全空间 16 维振幅向量），冻结为投影基 F (16,4)，新建 2 个实参数+复输出单态拟设
χ_0,χ_1，在 Q₄ = I − F F† 的 12 维剩余子空间中运行 NES-VMC（2×2 行列式联合采样，
machine_pow=1 + 重要性加权），收敛后精确提取 φ̂₄,φ̂₅，与 FCI E₄,E₅ 及
oracle ν₀,ν₁（QHQ 最低两本征值 = 冻结偏差下限）三方对照。

结构模板：H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.py（实参数梯度/QGT、裁剪、自然梯度）；
投影 bundle / 物理态提取复用 NES_VMC_deflation.py；实参数 deflated grad/QGT 来自
NES_VMC_deflation_real.py。

运行：
    python H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py           # 300 步
    NES_N_ITER=5 python H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py  # smoke
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
    make_deflation_bundle,
    extract_physical_states,
)
from NES_VMC_deflation_real import (
    load_and_orthonormalize_frozen,
    deflation_oracle,
    make_deflated_grad_fn_real,
    make_deflated_qgt_fn_real,
)
# 可扩展子空间维度：NES_K_EXTEND=2（默认，找 E4,E5）或 4（找 E4..E7）
import importlib
K_EXT = int(os.environ.get("NES_K_EXTEND", 2))
assert K_EXT in (2, 3, 4), "NES_K_EXTEND 支持 2/3/4"
_mod = importlib.import_module(f"H2_6_31G_8Qbit_K{K_EXT}")
assert _mod.K == K_EXT, "模块 K 与 NES_K_EXTEND 不一致"
SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock = (
    _mod.SINGLE_SIZE, _mod.ha, _mod.hi_ext, _mod.ext_edges, _mod.K, _mod.hi, _mod.Hatree_Fock,
)

jax.config.update("jax_enable_x64", True)


# ===================== Ansatz：与 K4 RealFFNN 完全一致（不可 import 该脚本，其 import 即训练） =====================
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
    """NES 总拟设：K 个「实参数 + 复输出」单态拟设组成的行列式波函数。"""

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=jnp.complex128)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))
            shift = jax.lax.stop_gradient(jnp.max(jnp.real(L)))
            L_stable = L - shift
            Psi_stable = jnp.exp(L_stable)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi_stable, L_stable, shift

        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 1:
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")


# ====================== 日志配置 ======================
time_str = time.strftime("%y-%m-%d-%H-%M")
this_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

logger = logging.getLogger(f"NES_VMC_K{K}_frozen4_deflation_realFFNN")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
log_path = f"{log_dir}/{time_str}_nes_vmc_K{K}_frozen4_deflation_realFFNN.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# ====================== 超参配置（沿用 K4 RealFFNN 模板） ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER = int(os.environ.get("NES_N_ITER", 300))
Natural_Grad = True
clip_norm = 20.0
per_param_clip = 0.5
lr = 0.1
qgt_diag_shift = 0.1
SAVE_INTERVAL = 20
INIT_SEED = int(os.environ.get("NES_INIT_SEED", 11))
MAX_INIT_COND = 1e6

NPZ_PATH = os.path.join(this_file_dir, "冻结K4", "近似本征态_K4_26-10-07-19-57.npz")
HISTORY_FILE = f"{data_dir}/{time_str}_history_frozen4_K{K}extend_realFFNN.pkl"

# ====================== 冻结态 + oracle（步骤 0 预检） ======================
H_dense, all_states = build_full_stage(ha, hi)
F, phi_raw, gs_residuals = load_and_orthonormalize_frozen(NPZ_PATH, hi)
nu, u_vecs, qhq_evals = deflation_oracle(F, H_dense, n_keep=K)
E_exact, U_exact = np.linalg.eigh(H_dense)

logger.info("\n" + "=" * 60)
logger.info(f"冻结 K=4 + 剩余子空间 K={K} Deflation NES-VMC（方案 A，实参数 RealFFNN）")
logger.info(f"冻结态来源: {NPZ_PATH}")
logger.info(f"GS 残差范数: {np.round(gs_residuals, 4)} | F 形状 {F.shape}")
logger.info(f"oracle ν (QHQ 最低 {K} 本征值) = {np.round(nu, 8)}  ← 方法收敛目标/偏差下限")
logger.info(f"FCI  E4.. = {np.round(E_exact[4:4+K], 8)} | ν-E 偏差(mHa) = {np.round((nu - E_exact[4:4+K]) * 1000, 4)}")
logger.info(f"理论 Loss 目标 = tr(S⁻¹M) 收敛值 ≈ Σν = {nu.sum():.8f}")
logger.info(f"超参: lr={lr} diag_shift={qgt_diag_shift} clip={clip_norm}/{per_param_clip} "
            f"N_ITER={N_ITER} machine_pow=1 (|Ψ_F|采样+w∝|Ψ_F|) N_CHAINS={N_CHAINS} "
            f"samples/chain={N_SAMPLES_PER_CHAIN} sweep={SWEEP_SIZE}")
logger.info("=" * 60)

# ====================== 模型（fresh 初始化 + 投影可逆性预检） ======================
all_states_j = jnp.asarray(all_states)


def _chi_full_np(params):
    """χ = A − F(F†A)：投影后的可训练列全空间振幅 (16,K)。"""
    A = np.asarray(amplitudes_full(single_machine_list, params, all_states_j)).astype(np.complex128)
    c = F.conj().T @ A
    return A, c, A - F @ c


single_machine_list = None
total_ansatz = None
total_params = None
seed = INIT_SEED
while True:
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(seed),
    )
    single_machine_list = [
        create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]
    total_params = nnx.split(total_ansatz)[1]
    A, c, chi = _chi_full_np(total_params)
    cond_chi = np.linalg.cond(chi)
    frozen_frac = np.linalg.norm(c) / max(np.linalg.norm(A), 1e-30)
    if cond_chi < MAX_INIT_COND:
        break
    logger.warning(f"seed={seed} 初始 cond(χ_full)={cond_chi:.2e} 过大，换 seed")
    seed += 1
logger.info(f"初始化 seed={seed} | cond(χ_full)={cond_chi:.3e} | 冻结分量占比 ‖F†A‖/‖A‖={frozen_frac:.4f}")

# ====================== Deflation bundle + 采样器 ======================
machine_single, machine_batch, el_fn, forward = make_deflation_bundle(
    single_machine_list, all_states, H_dense, F
)

grad_fn = make_deflated_grad_fn_real(machine_single, forward)
qgt_fn = make_deflated_qgt_fn_real(machine_single, machine_batch)

nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
    machine_pow=1,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)
sampler_rng = jax.random.PRNGKey(21)


def sample_machine(params, sigma):
    return machine_batch(params, sigma)


sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

# ====================== 历史容器 ======================
loss_history, steps_history = [], []
lam_history, params_history, samples_history = [], [], []
diag_history = []

start_time = time.time()

# ====================== 训练循环 ======================
for step in range(N_ITER):
    params_history.append(total_params)

    # 1. 采样（NES 联合 2 态，扩展空间 K*SINGLE_SIZE=16 比特）
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    sigma_flat = np.asarray(samples_raw).reshape(-1, hi_ext.size)

    # 2. deflated 梯度 + loss
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, sigma_flat)

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
        logger.info(f"【Step {step}】per-param 梯度裁剪 scale={grad_scale:.4f}")

    grad_update = grad_raw
    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw={grad_norm_raw:.2f}")

    grad_norm_natural = float("nan")
    if bool(aux["should_skip"]):
        logger.warning(f"【Step {step}】should_skip=True（valid_ratio={float(aux['valid_ratio']):.3f}），跳过更新")
    elif Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, sigma_flat, qgt_diag_shift)
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = float(jnp.linalg.norm(ng_flat))
    else:
        grad_norm_natural = grad_norm_raw

    if not bool(aux["should_skip"]):
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

    # 3. 全空间精确监控：χ → 2×2 广义本征 λ（无 MC 噪声）
    A, c, chi = _chi_full_np(total_params)
    S_chi = chi.conj().T @ chi
    M_chi = chi.conj().T @ (H_dense @ chi)
    lam_sub, v_sub = scipy_eigh(M_chi + 1e-14 * S_chi, S_chi)
    lam_sub = np.sort(lam_sub.real)
    cond_S = float(np.linalg.cond(S_chi))
    max_absA = float(np.abs(A).max())
    max_norm_c = float(np.abs(c).max())
    if max_absA > 1e3 or max_norm_c > 1e3:
        logger.warning(f"【Step {step}】冻结分量漂移告警 max|A|={max_absA:.2e} ‖F†A‖max={max_norm_c:.2e}")

    eig_EL = np.sort(np.linalg.eigvals(np.asarray(E_L_mean)).real)
    accept = getattr(sampler_state, "average_acceptance", None)

    logger.info(
        f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.8f} (目标Σν={nu.sum():.8f}) | "
        f"λ0={lam_sub[0]:.8f} λ1={lam_sub[1]:.8f} | E_L_mean eig={np.round(eig_EL, 6)}"
    )
    logger.info(
        f"诊断 | valid={float(aux['valid_ratio']):.3f} | ESS={float(aux['ess']):.0f}/{sigma_flat.shape[0]} | "
        f"cond(S_χ)={cond_S:.2e} | max|A|={max_absA:.2e} | ‖F†A‖max={max_norm_c:.2e} | "
        f"accept={float(accept) if accept is not None else float('nan'):.3f}"
    )
    logger.info(f"梯度 | raw={grad_norm_raw:.4f} per_param={grad_norm_per_param:.4f} natural={grad_norm_natural if Natural_Grad else float('nan'):.4f}")
    logger.info("#-----------------------------------------#")

    loss_history.append(float(jnp.real(loss_mean)))
    lam_history.append(lam_sub)
    steps_history.append(step)
    samples_history.append(sigma_flat)
    diag_history.append({
        "valid_ratio": float(aux["valid_ratio"]),
        "ess": float(aux["ess"]),
        "cond_S": cond_S,
        "max_absA": max_absA,
        "max_norm_c": max_norm_c,
        "grad_norm_raw": grad_norm_raw,
        "grad_norm_natural": grad_norm_natural,
    })

    if (step + 1) % SAVE_INTERVAL == 0 and N_ITER > 5:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump({
                "steps": steps_history,
                "loss": loss_history,
                "lam": lam_history,
                "params": params_history[:: max(1, len(params_history) // 15)][-15:],
                "samples": samples_history[-2:],
                "diag": diag_history,
                "seed": seed,
                "nu_oracle": nu,
            }, f)

# ====================== 提取 φ̂₄,φ̂₅ + 验收 ======================
A, c, chi = _chi_full_np(total_params)
phis, lams = extract_physical_states(chi, H_dense)
lam2, v2 = scipy_eigh(M_chi + 1e-14 * S_chi, S_chi)
v2 = v2[:, np.argsort(lam2.real)]

logger.info(f"\n================ φ̂₄..φ̂_{3+K} 验收（冻结 K=4 + 子空间 K={K} → K={4+K} 等效） ================")
logger.info(f"{'level':>5} | {'λ(子空间精确)':>14} | {'ν(oracle)':>12} | {'E(FCI)':>12} | "
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
        f"{4 + k:>5} | {E_ray:>14.8f} | {nu[k]:>12.8f} | {E_exact[4 + k]:>12.8f} | "
        f"{(E_ray - E_exact[4 + k]) * 1000:>9.4f} | {var:>10.2e} | {ov_fci:>9.5f} | {ov_frozen:>12.2e}"
    )

for k in range(K):
    idx = np.argsort(np.abs(chi @ v2[:, k]))[::-1][:3]
    comp = ", ".join(f"{all_states[i].tolist()}:{(chi @ v2[:, k])[i]:.3f}" for i in idx)
    logger.info(f"φ̂_{4 + k} 前3大分量: {comp}")

# ====================== 保存 ======================
out_npz = os.path.join(this_file_dir, "冻结K4", f"近似本征态_K4冻结_K{K}扩展_{time_str}.npz")
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
    seed=seed,
    N_ITER=N_ITER,
)
with open(HISTORY_FILE, "wb") as f:
    pickle.dump({
        "steps": steps_history,
        "loss": loss_history,
        "lam": lam_history,
        "params": params_history,
        "samples": samples_history[-2:],
        "diag": diag_history,
        "seed": seed,
        "nu_oracle": nu,
    }, f)

logger.info(f"\n总耗时 {time.time() - start_time:.1f}s")
logger.info(f"已保存 npz: {out_npz}")
logger.info(f"已保存 history: {HISTORY_FILE}")
logger.info(f"日志: {log_path}")

# ====================== 训练结束绘图（风格同 K4 RealFFNN 脚本，图内标签用英文） ======================
import matplotlib
import matplotlib.pyplot as plt

logger.info("绘制图片...")
steps_arr = np.asarray(steps_history)
lam_arr = np.asarray(lam_history)                      # (T,2) 子空间精确 λ
diag = {k: np.asarray([d[k] for d in diag_history]) for k in diag_history[0]}
err_mha = (lam_arr - E_exact[None, 4:4+K]) * 1000.0    # λ−E（mHa）

fig, axs = plt.subplots(5, 1, figsize=(10, 25))

axs[0].plot(steps_arr, lam_arr, lw=1.2, label=[f"lambda_{k} (phi_{4+k})" for k in range(K)])
_ref_colors = ["red", "orange", "crimson", "darkorange"]
for k in range(K):
    axs[0].axhline(nu[k], color=_ref_colors[k % 4], ls="--", alpha=0.5, label=f"oracle nu_{k}={nu[k]:.6f}")
    axs[0].axhline(E_exact[4 + k], color="gray", ls=":", alpha=0.7, label=f"FCI E{4+k}={E_exact[4+k]:.6f}")
axs[0].set_ylabel("Energy (Ha)")
axs[0].set_title(f"Frozen K=4 + subspace K={K} deflated NES-VMC, H2 6-31G (real FFNN, {N_ITER} steps)")
axs[0].legend(fontsize=8)

axs[1].plot(steps_arr, loss_history, lw=1.0)
axs[1].axhline(nu.sum(), color="red", ls="--", alpha=0.6, label=f"target sum(nu)={nu.sum():.6f}")
axs[1].set_ylabel("Loss = Re tr(E_L^F)")
axs[1].legend(fontsize=8)

axs[2].axhspan(0, 1.6, facecolor="lightblue", alpha=0.3, zorder=0, label="chemical accuracy (1.6 mHa)")
axs[2].plot(steps_arr, err_mha, lw=1.2, label=[f"E{4+k} error" for k in range(K)])
axs[2].set_yscale("symlog", linthresh=1e-2)
axs[2].set_ylabel("lambda - E_FCI (mHa)")
axs[2].set_xlabel("step")
axs[2].legend(fontsize=8)

axs[3].plot(steps_arr, diag["valid_ratio"], lw=1.0, label="valid_ratio")
axs[3].plot(steps_arr, diag["ess"] / (N_CHAINS * N_SAMPLES_PER_CHAIN), lw=1.0, label="ESS fraction")
axs[3].plot(steps_arr, diag["cond_S"] / 100.0, lw=1.0, alpha=0.7, label="cond(S_chi)/100")
axs[3].set_ylabel("sampling / subspace health")
axs[3].set_xlabel("step")
axs[3].legend(fontsize=8)

axs[4].plot(steps_arr, diag["grad_norm_raw"], lw=1.0, label="grad_norm_raw")
axs[4].plot(steps_arr, np.nan_to_num(diag["grad_norm_natural"], nan=np.nan), lw=1.0, label="grad_norm_natural")
axs[4].set_yscale("log")
axs[4].twinx().semilogy(steps_arr, diag["max_absA"], lw=1.0, color="tab:red", alpha=0.7, label="max|A| (frozen drift)")
axs[4].set_ylabel("gradient norm (log)")
axs[4].set_xlabel("step")
axs[4].legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig_path = f"{data_dir}/H2_631G_8Qbit_K{K}_frozen4_deflation_RealFFNN_{time_str}.png"
fig.savefig(fig_path, dpi=150)
plt.show()
logger.info(f"已保存图: {fig_path}")
