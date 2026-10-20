"""
真正的 Matrix-Free 冻结块 NES-VMC（方案 B'：行列式自动投影）— H2/6-31G/K=6=冻结4+可训练2

与精确版 H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py（方案 A，显式 16 维投影矩阵 F、
χ=A−FF†A）不同，本脚本训练全程**不接触 2^N=16 维全空间**：

  1. 冻结：载入 K=4 训练的 300 步参数（pkl history['params'][-1]），重建 4 个 gauge-fixed
     单列网络 ψ_j^old，用 npz v_rot 组成 4 个冻结列 φ_i = Σ_j ψ_j^old·v_rot[j,i]（常量参数，
     stop-gradient，只通过网络前向在采样构型上逐点求值）。
  2. 联合拟设：m=6 列 × 6 walker 行列式 Ψ = det[φ0..φ3 | χ1 χ2]，χ 为 2 个 fresh
     gauge-fixed 实参数 FFNN 列。
  3. 自动投影（B' 核心数学）：det 对列变换不变 ⇒ det[Φ|C] = det[Φ|Q_Φ C] 逐点成立，
     active 列中与冻结 span 相关的分量在行列式里自动消失；autodiff 过 slogdet 自动给出
     切空间投影 ∂χ=Q∂A。无需显式投影系数 c、Gram 矩阵或 ‖ψ‖² 绝对范数。
  4. 采样：NESFermionHopRule（K=6 副本，48 bit）MetropolisSampler，machine_pow=1
     按 |Ψ| 采样 + 重要性加权 w∝|Ψ|（1006 修复：防零方差冻结）。
  5. 梯度/QGT：实参数 Wirtinger（∇logΨ=∇Re+i∇Im）、力梯度 2Re⟨tr(E_L−⟨E_L⟩)(∇logΨ)*⟩_w、
     度量 Re(S_w)+diag_shift·I，float64，SR=solve，optax 全局裁剪 + per-param 裁剪。

验证/诊断阶段允许全空间 16 维稠密运算（仅提取 λ̂4,5 与重叠，绝不进入训练循环）。
对照基准：精确版 K=2 扩展 λ=(−0.02803225,−0.01448578)，误差 (0.0005, 0.018) mHa，
重叠 (0.99999, 0.99813)；oracle ν 为冻结偏差下限。

运行：
    python H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py                 # 1500 步
    NES_N_ITER=5 python H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py    # smoke
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

from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    Ham_psi_scaled,
    NESFermionHopRule,
    flatten_batched_pytree,
)
from NES_VMC_deflation_real import (
    load_and_orthonormalize_frozen,
    deflation_oracle,
    importance_weights_from_logpsi,
)
from H2_6_31G_8Qbit_K6 import (
    SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock,
)

jax.config.update("jax_enable_x64", True)

N_REPLICA = K      # 6 个 NES walker（行列式 6×6）
N_FROZEN = 4       # 冻结列 φ0..φ3（旧 K4 网络 + v_rot 线性组合）
N_ACTIVE = K - N_FROZEN   # 2 个可训练 fresh 列

assert SINGLE_SIZE == 8 and N_REPLICA == 6, "本脚本按 K=6/SINGLE_SIZE=8 设计"

# ===================== Ansatz（与 K4 RealFFNN / 测试各能级 完全一致） =====================
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
    """NES 总拟设：K 个「实参数 + 复输出」单态拟设（仅用于重建旧 K4 架构）。"""

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            self.single_ansatz_list.append(
                SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            )


# ====================== 日志配置 ======================
time_str = time.strftime("%y-%m-%d-%H-%M")
this_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

logger = logging.getLogger("NES_VMC_K6_frozen4_block_MatrixFree")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
log_path = f"{log_dir}/{time_str}_nes_vmc_K6_frozen4_block_MatrixFree.log"
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setFormatter(simple_formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(simple_formatter)
logger.addHandler(ch)

# ====================== 超参 ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER = int(os.environ.get("NES_N_ITER", 1500))
Natural_Grad = True
clip_norm = 20.0
per_param_clip = 0.5
lr = 0.1
qgt_diag_shift = 0.1
SAVE_INTERVAL = 50
DIAG_EVERY = 25
RESET_PERIOD = 10
INIT_SEED = int(os.environ.get("NES_INIT_SEED", 77))
MAX_INIT_COND = 1e6

NPZ_PATH = os.path.join(this_file_dir, "冻结K4", "近似本征态_K4_26-10-07-19-57.npz")
PKL_PATH = os.path.join(this_file_dir, "data", "26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl")
HISTORY_FILE = f"{data_dir}/{time_str}_history_K6_frozen4_block_MatrixFree.pkl"

# ====================== 步骤 0：冻结网络重建 + oracle 预检 ======================
npz = np.load(NPZ_PATH)
phi_raw = np.asarray(npz["phi_vecs"], dtype=np.complex128)   # (16,4) 列归一
v_rot = np.asarray(npz["v_rot"], dtype=np.complex128)         # (4,4) 列约定 φ_k=Σ_j ψ_j v_rot[j,k]
F, _, gs_res = load_and_orthonormalize_frozen(NPZ_PATH, hi)   # (16,4) 正交归一（仅诊断/验收用）

with open(PKL_PATH, "rb") as f:
    old_history = pickle.load(f)
old_params = old_history["params"][-1]

old_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=N_FROZEN,
    hidden_dim=SINGLE_SIZE + N_FROZEN, rngs=nnx.Rngs(11),
)
frozen_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
              for a in old_ansatz.single_ansatz_list]
frozen_params = [old_params["single_ansatz_list"][j] for j in range(N_FROZEN)]

# 全空间稠密 H（仅诊断）
all_states = np.asarray(hi.all_states())
all_states_j = jnp.asarray(all_states)
H_dense = np.asarray(ha.to_dense(), dtype=np.complex128)
E_exact, U_exact = np.linalg.eigh(H_dense)
nu, u_vecs, qhq_evals = deflation_oracle(F, H_dense, n_keep=N_ACTIVE)

# 冻结网络列 vs npz φ_k 一致性（gauge 约定必须吻合）
A_old = np.exp(np.stack([np.asarray(frozen_sms[j](frozen_params[j], all_states_j), dtype=np.complex128)
                         for j in range(N_FROZEN)], axis=1))  # (16,4), ψ_j(HF)=1（machine 输出 logψ）
Phi_net = A_old @ v_rot
Phi_net = Phi_net / np.linalg.norm(Phi_net, axis=0, keepdims=True)
fid_cols = [float(abs(np.vdot(phi_raw[:, k], Phi_net[:, k])) ** 2) for k in range(N_FROZEN)]
logger.info("=" * 60)
logger.info("冻结网络重建 vs npz φ_k 列保真度（应 ≈1）: " + str(np.round(fid_cols, 8)))
assert min(fid_cols) > 0.999, "gauge 约定不一致，禁止继续"

E_ray_frozen = np.asarray(npz["E_rayleigh"], dtype=np.float64)
logger.info(f"冻结列 Rayleigh E(φ0..φ3) = {np.round(E_ray_frozen, 8)}")
logger.info(f"oracle ν (QHQ 最低 {N_ACTIVE} 本征值) = {np.round(nu, 8)} | ν-E(mHa) = {np.round((nu - E_exact[4:4+N_ACTIVE]) * 1000, 4)}")
logger.info(f"Loss 收敛目标 ≈ Σ E_ray(frozen) + Σν = {E_ray_frozen.sum() + nu.sum():.8f}")
logger.info(f"超参: lr={lr} diag_shift={qgt_diag_shift} clip={clip_norm}/{per_param_clip} N_ITER={N_ITER} "
            f"machine_pow=1(|Ψ|采样+w∝|Ψ|) chains={N_CHAINS} samples/chain={N_SAMPLES_PER_CHAIN} sweep={SWEEP_SIZE}")
logger.info("=" * 60)

# ====================== B' 核心：冻结块前向（matrix-free） ======================
V_MAT = jnp.asarray(v_rot, dtype=jnp.complex128)   # (4,4) 闭包常量


def make_block_forward(ha):
    """流形 B=[φ0..φ3 | χ1 χ2]（φ_i=Σ_j ψ_j^old v_rot[j,i]）的前向。

    _one(p, xw, g) 对单个 walker xw (6, n_spin)：
        Lf[n,j] = logψ_j^old(x_n)（gauge-fixed，常量参数）
        La[n,m] = logχ_m(x_n) - g[m]
        s       = max Re(Lf ∪ La)（per-walker shift，stop-gradient）
        A = [exp(Lf-s)@v_rot | exp(La-s)]（6×6）
        Z = [HPf@v_rot | HPa·e^{-g}]（H 线性 ⇒ Hφ_i = Σ_j v_rot[j,i]·Hψ_j）
        logΨ = 6s + slogdet(A)；E_L = solve(A, Z)
    行列式列变换不变性 ⇒ det[Φ|C]=det[Φ|Q_ΦC]：投影自动实现，梯度只流向 active。
    """

    def _one(p, xw, g):
        Lf = jnp.stack([frozen_sms[j](frozen_params[j], xw) for j in range(N_FROZEN)], axis=1)  # (6,4)
        La = jnp.stack([sms[m](p[m], xw) for m in range(N_ACTIVE)], axis=1)                     # (6,2)
        La = La - jax.lax.stop_gradient(g)[None, :]

        s = jax.lax.stop_gradient(jnp.max(jnp.real(jnp.concatenate([Lf, La], axis=1))))
        Phi_scaled = jnp.exp(Lf - s) @ V_MAT                     # (6,4)
        A = jnp.concatenate([Phi_scaled, jnp.exp(La - s)], axis=1)   # (6,6)

        HPf = jnp.stack([Ham_psi_scaled(ha=ha, single_machine=frozen_sms[j], params=frozen_params[j],
                                        x=xw, shift=s) for j in range(N_FROZEN)], axis=1)  # (6,4)
        HPhi = HPf @ V_MAT
        HPa = jnp.stack([Ham_psi_scaled(ha=ha, single_machine=sms[m], params=p[m], x=xw, shift=s)
                         for m in range(N_ACTIVE)], axis=1)                                   # (6,2)
        HPa = HPa * jnp.exp(-jax.lax.stop_gradient(g))[None, :]
        Z = jnp.concatenate([HPhi, HPa], axis=1)               # (6,6)

        sign, log_abs = jnp.linalg.slogdet(A)
        logPsi = N_REPLICA * s + log_abs + 1j * jnp.angle(sign)
        E_L = jnp.linalg.solve(A, Z)
        loss = jnp.real(jnp.trace(E_L))
        valid = (jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(Z))
                 & jnp.all(jnp.isfinite(E_L)) & jnp.isfinite(loss))
        return logPsi, E_L, loss, valid

    def core(p, x_batch, g):
        logPsi, E_L, loss, valid = jax.vmap(lambda xw: _one(p, xw, g))(x_batch)
        return logPsi, E_L, loss, valid

    return core, _one


def dlogpsi_one(_one, p, xw, g):
    re_grad = jax.grad(lambda q: jnp.real(_one(q, xw, g)[0]))(p)
    im_grad = jax.grad(lambda q: jnp.imag(_one(q, xw, g)[0]))(p)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


def make_block_grad_fn(core, _one, min_valid_ratio=0.25):
    """实参数力梯度 + 重要性加权 w∝|Ψ|（machine_pow=1 采样）。"""

    @jax.jit
    def grad_fn(p, x_batch, g):
        logPsi, E_L, loss, valid = core(p, x_batch, g)
        valid = jax.lax.stop_gradient(valid)
        valid_f = valid.astype(jnp.float64)
        n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)
        valid_ratio = jnp.sum(valid_f) / x_batch.shape[0]

        wv = jax.lax.stop_gradient(importance_weights_from_logpsi(logPsi, valid))

        E_L_safe = jnp.where(jnp.isfinite(E_L), E_L, 0.0)
        E_L_mean = jnp.sum(E_L_safe * wv.reshape((-1, 1, 1)), axis=0)
        tr_centered = jnp.trace(E_L_safe - E_L_mean, axis1=-2, axis2=-1)
        tr_centered = jnp.where(valid, tr_centered, 0.0)

        dlog_batch = jax.vmap(lambda xw: dlogpsi_one(_one, p, xw, g))(x_batch)

        def weighted_mean(comp):
            comp_safe = jnp.where(jnp.isfinite(comp), comp, 0.0)
            w = (wv * tr_centered).reshape((-1,) + (1,) * (comp.ndim - 1))
            return 2.0 * jnp.real(jnp.sum(w * jnp.conj(comp_safe), axis=0))

        grad = jax.tree.map(weighted_mean, dlog_batch)
        loss_safe = jnp.where(jnp.isfinite(loss), loss, 0.0)
        loss_mean = jnp.sum(wv * loss_safe)

        grad_flat, _ = ravel_pytree(grad)
        ess = 1.0 / jnp.maximum(jnp.sum(wv * wv), 1e-30)
        aux = {
            "valid_ratio": valid_ratio, "n_valid": n_valid, "ess": ess,
            "should_skip": ~(jnp.all(jnp.isfinite(grad_flat)) & jnp.isfinite(loss_mean)
                             & (valid_ratio >= min_valid_ratio)),
        }
        return grad, loss_mean, E_L_mean, aux

    return grad_fn


def make_block_qgt_fn(core, _one):
    """QGT：O=∇logΨ（active 参数），重要性加权中心化，度量 Re(S_w)+diag_shift·I。"""

    @jax.jit
    def qgt_fn(p, x_batch, diag_shift, g):
        n_samples = x_batch.shape[0]
        logPsi, _, _, valid = core(p, x_batch, g)
        grad_tree_batch = jax.vmap(lambda xw: dlogpsi_one(_one, p, xw, g))(x_batch)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)          # (N,P) complex
        wv = jax.lax.stop_gradient(importance_weights_from_logpsi(logPsi, valid))
        O_safe = jnp.where(jnp.isfinite(O), O, 0.0)
        O_mean = jnp.sum(wv[:, None] * O_safe, axis=0, keepdims=True)
        O_centered = O_safe - O_mean
        S = (jnp.conj(O_centered) * wv[:, None]).T @ O_centered
        S = 0.5 * (S + jnp.conj(S.T))
        S_real = jnp.real(S)
        return S_real + diag_shift * jnp.eye(S_real.shape[0], dtype=S_real.dtype)

    return qgt_fn


# ====================== fresh active 列初始化（投影可逆性预检） ======================
def active_basis_full(p_list):
    """active 列全空间振幅 (16,2)，共享 shift（仅诊断/预检）。"""
    logs = np.stack([np.asarray(sms[m](p_list[m], all_states_j), dtype=np.complex128)
                     for m in range(N_ACTIVE)], axis=1)
    return np.exp(logs - logs.real.max())


def init_block_cond(p_list):
    Ba = active_basis_full(p_list)
    B = np.concatenate([Phi_net / np.linalg.norm(Phi_net, axis=0, keepdims=True), Ba], axis=1)
    return np.linalg.cond(B), Ba


sms = None
active_params = None
seed = INIT_SEED
while True:
    active_ansatz = []
    key = jax.random.PRNGKey(seed)
    for m in range(N_ACTIVE):
        key, sub = jax.random.split(key)
        active_ansatz.append(SingleStateAnsatz_SharedBackbone(SINGLE_SIZE, SINGLE_SIZE + N_ACTIVE,
                                                              rngs=nnx.Rngs(params=sub)))
    sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in active_ansatz]
    active_params = [nnx.split(a)[1] for a in active_ansatz]
    cond_B, _ = init_block_cond(active_params)
    if cond_B < MAX_INIT_COND:
        break
    logger.warning(f"seed={seed} 初始 cond(B_6列)={cond_B:.2e} 过大，换 seed")
    seed += 1
logger.info(f"active 初始化 seed={seed} | cond(B)={cond_B:.3e}")

core, _one = make_block_forward(ha)
grad_fn = make_block_grad_fn(core, _one)
qgt_fn = make_block_qgt_fn(core, _one)

# ====================== 采样器（matrix-free，48 bit NES 联合行走） ======================
nes_rule = NESFermionHopRule(edges=ext_edges, K=N_REPLICA, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext, rule=nes_rule, n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE, machine_pow=1,
)

g_current = [jnp.zeros(N_ACTIVE, dtype=jnp.complex128)]


def sample_machine(p, sigma):
    x = sigma.reshape(-1, N_REPLICA, SINGLE_SIZE)
    return core(p, x, g_current[0])[0]


optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.sgd(learning_rate=lr))
opt_state = optimizer.init(active_params)
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(sample_machine, active_params, sampler_rng)

# ====================== 训练循环 ======================
loss_history, steps_history, lam_hist = [], [], []
elam_hist, params_history, samples_history, diag_history = [], [], [], []
start_time = time.time()

for step in range(N_ITER):
    params_history.append(active_params)

    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine, parameters=active_params,
        state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN,
    )
    x_batch = np.asarray(samples_raw).reshape(-1, N_REPLICA, SINGLE_SIZE)

    grad_raw, loss_mean, E_L_mean, aux = grad_fn(active_params, x_batch, g_current[0])
    grad_flat, unravel = ravel_pytree(grad_raw)
    grad_norm_raw = float(jnp.linalg.norm(grad_flat))
    n_params = grad_flat.size
    grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)
    if grad_norm_per_param > per_param_clip:
        scale = per_param_clip / grad_norm_per_param
        grad_flat = grad_flat * scale
        grad_raw = unravel(grad_flat)
        grad_norm_raw = float(jnp.linalg.norm(grad_flat))
        grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)
        logger.info(f"【Step {step}】per-param 梯度裁剪 scale={scale:.4f}")

    grad_update = grad_raw
    grad_norm_natural = float("nan")
    if bool(aux["should_skip"]):
        logger.warning(f"【Step {step}】should_skip=True（valid_ratio={float(aux['valid_ratio']):.3f}），跳过更新")
    elif Natural_Grad:
        qgt = qgt_fn(active_params, x_batch, qgt_diag_shift, g_current[0])
        ng_flat = jnp.linalg.solve(qgt, ravel_pytree(grad_raw)[0])
        grad_update = unravel(ng_flat)
        grad_norm_natural = float(jnp.linalg.norm(ng_flat))

    if not bool(aux["should_skip"]):
        updates, opt_state = optimizer.update(grad_update, opt_state, active_params)
        active_params = optax.apply_updates(active_params, updates)

    # 周期列规范重置（仅 active 列）
    if (step + 1) % RESET_PERIOD == 0:
        La_means = jnp.stack([jnp.mean(sms[m](active_params[m], x_batch.reshape(-1, SINGLE_SIZE)))
                              for m in range(N_ACTIVE)])
        g_current[0] = jnp.asarray(La_means, dtype=jnp.complex128)

    eig_EL = np.sort(np.asarray(jnp.linalg.eigvals(np.asarray(E_L_mean)).real))
    elam_hist.append(eig_EL)

    # 全空间 block 诊断（仅诊断，不进训练）
    lam_blk = np.full(N_ACTIVE, np.nan)
    if (step + 1) % DIAG_EVERY == 0 or step == 0 or step == N_ITER - 1:
        Ba = active_basis_full(active_params)
        Bp = Ba - F @ (F.conj().T @ Ba)
        M = Bp.conj().T @ (H_dense @ Bp)
        S = 0.5 * (Bp.conj().T @ Bp + (Bp.conj().T @ Bp).conj().T)
        ridge = 1e-10 * max(np.trace(S).real / S.shape[0], 1.0)
        lam_blk, V_blk = scipy_eigh(M, S + ridge * np.eye(S.shape[0]))
        lam_blk = np.sort(lam_blk.real)

    accept = getattr(sampler_state, "average_acceptance", None)
    logger.info(f"[Step {step:3d}] Loss={float(jnp.real(loss_mean)):.8f} "
                f"(目标={E_ray_frozen.sum() + nu.sum():.8f}) | eig(E_L)={np.round(eig_EL[4:], 6)} | "
                f"block λ̂45={np.round(lam_blk, 6)}")
    logger.info(f"诊断 | valid={float(aux['valid_ratio']):.3f} | ESS={float(aux['ess']):.0f}/{x_batch.shape[0]} | "
                f"accept={float(accept) if accept is not None else float('nan'):.3f}")
    logger.info(f"梯度 | raw={grad_norm_raw:.4f} per_param={grad_norm_per_param:.4f} natural={grad_norm_natural:.4f}")
    logger.info("#-----------------------------------------#")

    loss_history.append(float(jnp.real(loss_mean)))
    steps_history.append(step)
    lam_hist.append(lam_blk)
    samples_history.append(x_batch.reshape(-1, hi_ext.size))
    diag_history.append({
        "valid_ratio": float(aux["valid_ratio"]), "ess": float(aux["ess"]),
        "grad_norm_raw": grad_norm_raw, "grad_norm_natural": grad_norm_natural,
        "accept": float(accept) if accept is not None else float("nan"),
    })

    if (step + 1) % SAVE_INTERVAL == 0 and N_ITER > 5:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump({
                "steps": steps_history, "loss": loss_history, "lam_block": lam_hist,
                "params": params_history[:: max(1, len(params_history) // 15)][-15:],
                "samples": samples_history[-2:], "diag": diag_history,
                "seed": seed, "nu_oracle": nu, "E_ray_frozen": E_ray_frozen,
            }, f)

logger.info(f"\n训练总耗时 {time.time() - start_time:.1f}s")

# ====================== 最终提取 φ̂₄,φ̂₅ + 验收（诊断用全空间） ======================
Ba = active_basis_full(active_params)
Bp = Ba - F @ (F.conj().T @ Ba)
S_b = Bp.conj().T @ Bp
M_b = Bp.conj().T @ (H_dense @ Bp)
ridge = 1e-10 * max(np.trace(S_b).real / S_b.shape[0], 1.0)
lam_f, V_f = scipy_eigh(M_b, S_b + ridge * np.eye(S_b.shape[0]))
order = np.argsort(lam_f.real)
lam_f, V_f = lam_f[order].real, V_f[:, order]

logger.info("\n================ Matrix-Free 冻结块验收（φ̂₄,φ̂₅） ================")
logger.info(f"{'level':>5} | {'λ(子空间精确)':>14} | {'ν(oracle)':>12} | {'E(FCI)':>12} | "
            f"{'λ-E(mHa)':>9} | {'Var[E_loc]':>10} | {'FCI重叠':>9} | {'冻结剩余重叠':>12}")
rows = []
final_phis = []
for k in range(N_ACTIVE):
    phi = Bp @ V_f[:, k]
    phi = phi / np.linalg.norm(phi)
    final_phis.append(phi)
    Hphi = H_dense @ phi
    E_ray = float(np.vdot(phi, Hphi).real)
    var = float(np.vdot(Hphi, Hphi).real - E_ray ** 2)
    ov_fci = float(abs(np.vdot(U_exact[:, 4 + k], phi)) ** 2)
    ov_frozen = float((np.abs(phi_raw.conj().T @ phi) ** 2).max())
    rows.append((E_ray, var, ov_fci, ov_frozen))
    logger.info(f"{4 + k:>5} | {E_ray:>14.8f} | {nu[k]:>12.8f} | {E_exact[4 + k]:>12.8f} | "
                f"{(E_ray - E_exact[4 + k]) * 1000:>9.4f} | {var:>10.2e} | {ov_fci:>9.5f} | {ov_frozen:>12.2e}")

# 冻结列保真度（构造上应严格不变）
frozen_fid = [float(abs(np.vdot(U_exact[:, k], phi_raw[:, k])) ** 2) for k in range(N_FROZEN)]
logger.info(f"冻结列对 FCI 重叠（构造上不变）= {np.round(frozen_fid, 5)}")
logger.info(f"最终 eig(E_L_mean)（matrix-free 监控）= {np.round(elam_hist[-1], 6)}")

# ====================== 保存 ======================
out_npz = os.path.join(this_file_dir, "冻结K4", f"近似本征态_K6冻结块_MatrixFree_{time_str}.npz")
np.savez(
    out_npz,
    phi_vecs=np.stack(final_phis, axis=1),
    all_states=all_states,
    active_full=Ba,
    v_rot=np.asarray(V_f),
    lam_subspace=np.array([r[0] for r in rows]),
    nu_oracle=nu,
    E_exact=E_exact[4:4 + N_ACTIVE],
    var_eloc=np.array([r[1] for r in rows]),
    overlap_fci=np.array([r[2] for r in rows]),
    overlap_frozen_residual=np.array([r[3] for r in rows]),
    E_ray_frozen=E_ray_frozen,
    seed=seed, N_ITER=N_ITER,
)
with open(HISTORY_FILE, "wb") as f:
    pickle.dump({
        "steps": steps_history, "loss": loss_history, "lam_block": lam_hist,
        "eig_EL": elam_hist, "params": params_history,
        "samples": samples_history[-2:], "diag": diag_history,
        "seed": seed, "nu_oracle": nu, "E_ray_frozen": E_ray_frozen,
    }, f)
logger.info(f"已保存 npz: {out_npz}")
logger.info(f"已保存 history: {HISTORY_FILE}")
logger.info(f"日志: {log_path}")

# ====================== 训练结束绘图（5 联图，图内标签英文） ======================
import matplotlib
import matplotlib.pyplot as plt

logger.info("绘制图片...")
steps_arr = np.asarray(steps_history)
lam_arr = np.asarray(lam_hist)                       # (steps,2)，仅诊断步有限
eig_arr = np.asarray(elam_hist)[:, N_FROZEN:]        # (steps,2) 每步 matrix-free 监控
diag = {k: np.asarray([d[k] for d in diag_history]) for k in diag_history[0]}
diag_mask = np.all(np.isfinite(lam_arr), axis=1)     # block 精确诊断点
err_mha = (lam_arr - E_exact[None, 4:4 + N_ACTIVE]) * 1000.0
err_mha_eig = (eig_arr - E_exact[None, 4:4 + N_ACTIVE]) * 1000.0

fig, axs = plt.subplots(5, 1, figsize=(10, 25))

# lam_arr 大部分为 NaN（每 DIAG_EVERY 步才算一次），直接 plot 线不连续会"消失"：
# 连续曲线用每步 eig(E_L) 监控，block 精确值用散点 marker 叠加
axs[0].plot(steps_arr, eig_arr, lw=1.0, alpha=0.8,
            label=[f"eig(E_L) monitor lambda_{k}" for k in range(N_ACTIVE)])
axs[0].plot(steps_arr[diag_mask], lam_arr[diag_mask], "o", ms=3.5,
            label=[f"block lambda_{k} (exact, every {DIAG_EVERY} steps)" for k in range(N_ACTIVE)])
for k in range(N_ACTIVE):
    axs[0].axhline(nu[k], color="red" if k == 0 else "orange", ls="--", alpha=0.5,
                   label=f"oracle nu_{k}={nu[k]:.6f}")
    axs[0].axhline(E_exact[4 + k], color="gray", ls=":", alpha=0.7,
                   label=f"FCI E{4+k}={E_exact[4+k]:.6f}")
axs[0].set_ylabel("Energy (Ha)")
axs[0].set_title(f"Matrix-free frozen-4 block NES-VMC (K=6, det-projection), H2 6-31G, {N_ITER} steps")
axs[0].legend(fontsize=8)

target_loss = E_ray_frozen.sum() + nu.sum()
axs[1].plot(steps_arr, loss_history, lw=1.0)
axs[1].axhline(target_loss, color="red", ls="--", alpha=0.6, label=f"target sum(E_ray)+sum(nu)={target_loss:.6f}")
axs[1].set_ylabel("Loss = Re tr(E_L)")
axs[1].legend(fontsize=8)

axs[2].axhspan(0, 1.6, facecolor="lightblue", alpha=0.3, zorder=0, label="chemical accuracy (1.6 mHa)")
axs[2].plot(steps_arr, err_mha_eig, lw=1.0, alpha=0.8,
            label=[f"E{4+k} error (eig(E_L) monitor)" for k in range(N_ACTIVE)])
axs[2].plot(steps_arr[diag_mask], err_mha[diag_mask], "o", ms=3.5,
            label=[f"E{4+k} error (block exact)" for k in range(N_ACTIVE)])
axs[2].set_yscale("symlog", linthresh=1e-2)
axs[2].set_ylabel("lambda - E_FCI (mHa)")
axs[2].set_xlabel("step")
axs[2].legend(fontsize=8)

axs[3].plot(steps_arr, diag["valid_ratio"], lw=1.0, label="valid_ratio")
axs[3].plot(steps_arr, diag["ess"] / (N_CHAINS * N_SAMPLES_PER_CHAIN), lw=1.0, label="ESS fraction")
axs[3].plot(steps_arr, diag["accept"], lw=1.0, alpha=0.7, label="acceptance")
axs[3].set_ylabel("sampling health")
axs[3].set_xlabel("step")
axs[3].legend(fontsize=8)

axs[4].plot(steps_arr, diag["grad_norm_raw"], lw=1.0, label="grad_norm_raw")
axs[4].plot(steps_arr, np.nan_to_num(diag["grad_norm_natural"], nan=np.nan), lw=1.0, label="grad_norm_natural")
axs[4].set_yscale("log")
axs[4].set_ylabel("gradient norm (log)")
axs[4].set_xlabel("step")
axs[4].legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig_path = f"{data_dir}/H2_631G_8Qbit_K6_frozen4_block_MatrixFree_{time_str}.png"
fig.savefig(fig_path, dpi=150)
plt.show()
logger.info(f"已保存图: {fig_path}")
