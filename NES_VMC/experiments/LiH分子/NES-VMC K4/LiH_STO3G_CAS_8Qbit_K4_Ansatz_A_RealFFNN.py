import logging
import os
import sys
import time
import pickle

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
import numpy as np
from jax.flatten_util import ravel_pytree

this_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_file_dir = os.path.join(this_file_dir, "..")       # 上级目录（LiH分子/，含 LiH_CAS_STO3G_8Qbit.py）
gradparent_file_dir = os.path.join(parent_file_dir, "..")  # 上上级目录（experiments/，含公共模块）
sys.path.append(parent_file_dir)
sys.path.append(gradparent_file_dir)
from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    Ham_Psi_scaled,
    NESFermionHopRule,
    flatten_batched_pytree,
)
from NES_VMC_tool import (
    create_gauge_reset_total_machines,
    NES_loss_energy_stable_gauge,
    _masked_mean_batch,
    make_gauge_fn,
    make_MS_estimator_fn,
    make_uniform_level_estimator_fn,
    compute_lam_v_from_samples_uniform,
)
from LiH_CAS_STO3G_8Qbit import (
    LiH_CAS_STO3G_8Qbit,
    SINGLE_SIZE,
    ha,
    hi,
    E_fcis,
    Hatree_Fock,
)

# 8 比特 = CAS(2e,4o)：冻结 Li 1s，4 个活性空间轨道（CASSCF 自然轨道基）
molecule = LiH_CAS_STO3G_8Qbit(K=4)
K = molecule.K
hi_ext = molecule.hi_ext
ext_edges = molecule.ext_edges


# ===================== Ansatz：双实输出拼装复数（实参数 + 复输出，替代振幅相位分离版） =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        # 全部网络参数依旧 float 实数，不使用复数权重
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        # ========== 替换原来 head_logA、head_phi → head_u、head_v 两个实输出头 ==========
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        # 两路实输出 u, v
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        # 不再 logA + iφ；拼装复数 z = u + iv，返回 log(z)
        # 小epsilon防止u+iv=0造成log奇点
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

            # 用 complex128 承接复输出，避免 complex128->complex64 的隐式降精度 scatter
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


# ============================================================
# 实参数 / 复输出 logΨ 的梯度兼容层
# ------------------------------------------------------------
# 原 Complex FFNN 参数是复数，直接可用
#     jax.grad(machine, argnums=0, holomorphic=True)
# 现在换成「实数参数 + 复数输出」，holomorphic=True 要求输入 dtype 为复数，直接报错。
# 处理办法（Wirtinger 拆分）：对实参数分别求 Re / Im 两个实值函数的梯度，再组合
#     ∇logΨ(θ) = ∇Re(logΨ) + i·∇Im(logΨ)
# 由此（实参数、复波函数）：
#   力梯度   ∇⟨L⟩ = 2·Re⟨ (tr E_L − ⟨tr E_L⟩) (∇logΨ)* ⟩
#   QGT      度量取 Re(S)，S_ij = ⟨(∂_i logΨ)* (∂_j logΨ)⟩
# ------------------------------------------------------------
# 重要性加权（配合 machine_pow=1 采样 p ∝ |Ψ|）：
#   期望定义在 |Ψ|² 测度下，而采样分布是 |Ψ|，故每个样本乘权重
#       w(σ) = |Ψ(σ)|² / p(σ) ∝ |Ψ(σ)|
#   所有 ⟨·⟩（E_L 均值、loss、力梯度、QGT）均改为 w 加权平均。
# ============================================================

def _grad_logpsi_real_single(total_machine, params, x_single, g):
    """单样本梯度（实参数、复输出）：∇logΨ = ∇Re + i·∇Im。"""
    re_grad = jax.grad(lambda p: jnp.real(total_machine(p, x_single, g)))(params)
    im_grad = jax.grad(lambda p: jnp.imag(total_machine(p, x_single, g)))(params)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


def _importance_weights(machine, params, x_batch, g, valid):
    """w_i ∝ |Ψ(σ_i)|（在 valid 样本上归一化），配合 p ∝ |Ψ| 采样（machine_pow=1）。
    总规范 g 是逐样本常数平移，归一化后抵消，不影响权重。"""
    log_psi = machine(params, x_batch, g)
    log_w = jnp.real(log_psi)
    valid_f = jnp.asarray(valid, dtype=jnp.float64)
    log_w_max = jnp.max(jnp.where(valid_f > 0, log_w, -jnp.inf))
    log_w_max = jax.lax.stop_gradient(jnp.where(jnp.isfinite(log_w_max), log_w_max, 0.0))
    w = jnp.exp(log_w - log_w_max)
    wv = w * valid_f
    return wv / jnp.maximum(jnp.sum(wv), 1e-12)


def nes_vmc_gradient_stable_gauge_real(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
    total_params,
    x_batch,
    g,
    min_valid_ratio: float = 0.25,
    return_aux: bool = False,
):
    """实参数版 NES-VMC 梯度（重要性加权）。结构对齐 nes_vmc_gradient_stable_gauge，
    ∇logΨ 用 Wirtinger 拆分，对实参数取实部 + 因子 2，均值估计全部改用 w∝|Ψ| 加权。"""
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable_gauge(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_batch,
        g=g,
        return_aux=True,
    )

    valid = jax.lax.stop_gradient(loss_aux["valid"])
    batch_size = x_batch.shape[0]
    n_valid = jnp.sum(valid.astype(jnp.float32))
    valid_ratio = n_valid / batch_size

    # 重要性权重（valid 上归一化，sum(wv)=1）
    wv = _importance_weights(total_machine, total_params, x_batch, g, valid)

    E_L_safe = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0)
    # 加权均值替代均匀 masked 均值：⟨E_L⟩ 定义在 |Ψ|² 测度下
    E_L_mean = jnp.sum(E_L_safe * wv.reshape((-1,) + (1,) * (E_L_safe.ndim - 1)), axis=0)

    E_L_centered = E_L_safe - E_L_mean

    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)
    tr_centered = jnp.where(valid, tr_centered, 0.0)

    # 每个样本的 ∇logΨ（复 pytree，batch 维在最前）
    dlogPsi_batch = jax.vmap(
        lambda x: _grad_logpsi_real_single(total_machine, total_params, x, g)
    )(x_batch)

    def weight_and_weighted_mean(grad_component):
        grad_safe = jnp.where(jnp.isfinite(grad_component), grad_component, 0.0)
        weights = (wv * tr_centered).reshape((-1,) + (1,) * (grad_component.ndim - 1))
        weighted = weights * jnp.conj(grad_safe)
        # 实参数：取实部，因子 2；wv 已归一化，无需再除分母
        return 2.0 * jnp.real(jnp.sum(weighted, axis=0))

    grad = jax.tree.map(weight_and_weighted_mean, dlogPsi_batch)

    loss_safe = jnp.where(jnp.isfinite(loss_batch), loss_batch, 0.0)
    loss_mean = jnp.sum(wv * loss_safe)

    grad_flat, _ = ravel_pytree(grad)
    grad_finite = jnp.all(jnp.isfinite(grad_flat))
    loss_finite = jnp.isfinite(loss_mean)
    enough_valid = valid_ratio >= min_valid_ratio

    aux = {
        "valid": valid,
        "n_valid": n_valid,
        "valid_ratio": valid_ratio,
        "grad_finite": grad_finite,
        "loss_finite": loss_finite,
        "enough_valid": enough_valid,
        "should_skip": ~(grad_finite & loss_finite & enough_valid),
        "loss_aux": loss_aux,
    }

    if return_aux:
        return grad, loss_mean, E_L_mean, aux

    return grad, loss_mean, E_L_mean


def make_grad_fn_gauge_real(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list):
    @jax.jit
    def grad_fn(total_params, x_batch, g):
        return nes_vmc_gradient_stable_gauge_real(
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

    return grad_fn


def make_qgt_fn_gauge_real(machine):
    """实参数版 QGT：∇logΨ 用 Wirtinger 拆分，度量取 Re(S)，
    均值/协方差用重要性权重 w∝|Ψ|（配合 machine_pow=1 采样）。"""

    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g):
        n_samples = sigma.shape[0]

        grad_tree_batch = jax.vmap(
            lambda s: _grad_logpsi_real_single(machine, params, s, g)
        )(sigma)

        O = flatten_batched_pytree(grad_tree_batch, n_samples)   # (N, P) 复数

        finite_row = jnp.all(jnp.isfinite(O), axis=1)
        wv = _importance_weights(machine, params, sigma, g, finite_row)

        O_safe = jnp.where(jnp.isfinite(O), O, 0.0)
        O_mean = jnp.sum(wv[:, None] * O_safe, axis=0, keepdims=True)
        O_centered = O_safe - O_mean

        S = (jnp.conj(O_centered) * wv[:, None]).T @ O_centered
        S = 0.5 * (S + jnp.conj(S.T))

        S_real = jnp.real(S)                                  # 实参数度量
        I = jnp.eye(S_real.shape[0], dtype=S_real.dtype)
        return S_real + diag_shift * I

    return qgt_fn


# ====================== 日志配置 ======================
time_str = time.strftime("%y-%m-%d-%H-%M")
logger = logging.getLogger(f"NES_VMC_K{K}_gauge_reset_realFFNN")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

# 获取当前py文件所在文件夹：日志/data/graph 三个同级目录都在脚本所在目录下
this_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
graph_dir = os.path.join(this_file_dir, "graph")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

log_path = f"{log_dir}/{time_str}_nes_vmc_K{K}_LiH_STO3G_CAS_8Qbit_gauge_reset_realFFNN.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# ====================== 超参配置 ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER = 500
Natural_Grad = True
clip_norm = 20.0
per_param_clip = 0.5
lr = 0.1
qgt_diag_shift = 0.1
RESET_PERIOD = 10
SAVE_INTERVAL = 20
# 能级监控：单副本均匀测度采样（与训练采样解耦，样本估计，不枚举全空间）
MONITOR_CHAINS = 32
MONITOR_SAMPLES_PER_CHAIN = 64
MONITOR_SWEEP_SIZE = 10
DATA_DIR = os.path.join(this_file_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = f"{DATA_DIR}/{time_str}_history_natural_gradient_LiH_molecule_K{K}_realFFNN.pkl"
PARAMS_NPZ = f"{DATA_DIR}/{time_str}_params_LiH_molecule_K{K}_realFFNN.npz"


# ====================== nnx.split 参数的 npz 存取 ======================
# nnx.split 得到的 params 是 flax.nnx.State（pytree，叶子是 jax 数组），可以直接存 npz。
# 需额外保存两项元信息才能无损还原：
#   __order__  ：每个叶子的可读路径；
#   __treedef__：pickle 后的 PyTreeDef，tree_unflatten 用它重建同一结构。
# 用法：
#   save_params_npz(total_params, path)
#   params = load_params_npz(path)          # 取回 State
#   model  = nnx.merge(graphdef, params)    # 需要时再合并成 nnx.Module
# 注意：读回前务必先 jax.config.update("jax_enable_x64", True)。
def save_params_npz(params, npz_path):
    """把 nnx.split 的参数树（State）逐叶子存进 npz，返回叶子数。"""
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    arrays = {f"p{i:04d}": np.asarray(leaf) for i, (_, leaf) in enumerate(leaves_with_path)}
    arrays["__order__"] = np.array(
        [jax.tree_util.keystr(p) for p, _ in leaves_with_path], dtype=str
    )
    arrays["__treedef__"] = np.frombuffer(pickle.dumps(treedef), dtype=np.uint8)
    np.savez(npz_path, **arrays)
    return len(leaves_with_path)


def load_params_npz(npz_path):
    """与 save_params_npz 配对：读回参数树（State），结构与 dtype/shape 完全一致。"""
    z = np.load(npz_path, allow_pickle=False)
    treedef = pickle.loads(z["__treedef__"].tobytes())
    leaves = [z[f"p{i:04d}"] for i in range(len(z["__order__"]))]
    return jax.tree_util.tree_unflatten(treedef, leaves)

# ====================== 模型与采样器 ======================
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)

g_current = jnp.zeros(K, dtype=jnp.complex128)   # 全局列规范 g，非训练参数，动态传入 jit

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

# 使用实参数版梯度 / QGT（原 make_grad_fn_gauge / make_qgt_fn_gauge 依赖复数参数）
grad_fn = make_grad_fn_gauge_real(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
)
qgt_fn = make_qgt_fn_gauge_real(total_machine)

gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
# machine_pow=1: 按 |Ψ|（而非 |Ψ|²）采样，配合梯度/QGT 中 w∝|Ψ| 的重要性加权
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
    machine_pow=1,
)

# CAS 活性空间 FCI 精确参考能量
exact_eigvals = E_fcis

# ====================== 优化器 ======================
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)

sampler_rng = jax.random.PRNGKey(21)


def sample_machine(params, sigma):
    """采样只需要 |Ψ|² 的转移率，全局列规范 g 是常数平移，不影响 ratio。"""
    return total_machine(params, sigma, g_current)


sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

# ====================== 能级监控：单副本均匀测度 ======================
# 物理能级是单副本希尔伯特空间计数测度下的 Ritz 值。训练用的扩展空间 |Ψ|² 测度不等于
# 这个内积，因此单独用单副本参考分布 q(σ)∝Σ_j|ψ_j(σ)|² 采样 + 1/q 自归一化重要性加权
# 还原均匀测度，全程 K×K、逐样本计算。
monitor_estimator, monitor_log_q = make_uniform_level_estimator_fn(
    ha=ha,
    single_machine_list=single_machine_list,
    total_graphdef=total_graphdef,
)


def monitor_machine(params, sigma):
    """machine_pow=1 时，采样分布 ∝ exp(log q) = q(σ) = Σ_j|ψ_j(σ)|²。"""
    return monitor_log_q(params, sigma)


monitor_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi,
    rule=molecule.single_rule,
    n_chains=MONITOR_CHAINS,
    sweep_size=MONITOR_SWEEP_SIZE,
    machine_pow=1,
)
monitor_sampler_state = monitor_sampler.init_state(
    monitor_machine, total_params, jax.random.PRNGKey(31)
)
# ====================== 训练循环 ======================
logger.info("\n" + "=" * 60)
logger.info("开始多链 NES-VMC 训练 | LiH / STO-3G / CAS(2e,4o) / 8 qubit | 实参数 FFNN | 双侧一致列规范 + 周期性 gauge reset")
logger.info(f"""超参数配置:\n\
N_CHAINS ={N_CHAINS}\n\
N_SAMPLES_PER_CHAIN ={N_SAMPLES_PER_CHAIN}\n\
SWEEP_SIZE = {SWEEP_SIZE}\n\
machine_pow = 1 (|Ψ| 采样 + 重要性加权 w∝|Ψ|)\n\
MONITOR_CHAINS = {MONITOR_CHAINS} | MONITOR_SAMPLES_PER_CHAIN = {MONITOR_SAMPLES_PER_CHAIN} (单副本均匀测度能级监控)\n\
N_ITER = {N_ITER}\n\
Natural_Grad = {Natural_Grad}\n\
clip_norm = {clip_norm}\n\
per_param_clip = {per_param_clip}\n\
lr = {lr}\n\
qgt_diag_shift = {qgt_diag_shift}\n\
RESET_PERIOD = {RESET_PERIOD}\n\
SAVE_INTERVAL = {SAVE_INTERVAL}\
""")

logger.info("=" * 60)
logger.info(
    f"精确CAS基准：基态={exact_eigvals[0]:.8f} Ha | 1激发={exact_eigvals[1]:.8f} Ha"
    f"|2激发={exact_eigvals[2]:.8f} Ha|3激发={exact_eigvals[3]:.8f} Ha|"
)
logger.info(f"理论 Loss 上限：{sum(exact_eigvals[0:K]):.8f} ")
logger.info(f"超参：clip_norm={clip_norm}, per_param_clip={per_param_clip}, lr={lr}, QGT diag_shift={qgt_diag_shift}, RESET_PERIOD={RESET_PERIOD}")

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

loss_history = []
logpsi_history = []
steps_history = []
logpsi_mean_history = []
logpsi_min_history = []
logpsi_max_history = []
grad_norm_raw_history = []
grad_norm_per_param_history = []
grad_norm_nat_history = []
E_L_real_history = []
E_L_imag_history = []
M_matrix_history = []
S_matrix_history = []
V_matrix_history = []
lam_real_history = []
params_history = []          # 只保留**最新一份**参数（滚动覆盖，不逐步累积）
Energy_levels_history = []
samples_history = []         # 只保留**最近 2 批**样本（滚动窗口）
gaugue_reset_history = []

start_time = time.time()

# ====================== 初始化/加载 pickle 历史文件 ======================
first_step = None
last_step = None
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "rb") as f:
            history = pickle.load(f)
        if history.get("steps"):
            last_step = int(history["steps"][-1])
            first_step = int(history["steps"][0])
            logger.info(f"检测到已有历史文件，上次保存 step={last_step}，将从 step={last_step + 1} 继续")
        else:
            logger.info("检测到历史文件但为空，将从头开始")
    except Exception as e:
        logger.warning(f"读取历史文件失败（可能损坏），将覆盖写入: {e}")
# ====================== 初始化结束 ======================

for step in range(N_ITER):
    params_history[:] = [total_params]     # 覆盖式保存：只留最新参数
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # 2. 梯度与损失（实参数版）
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g_current)

    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
    n_params = grad_raw_flat.size
    grad_norm_per_param = grad_norm_raw / jnp.sqrt(n_params)

    # 【裁剪】per-param 梯度范数超过上限时按比例整体缩放（发生在计算 QGT 之前）
    if float(grad_norm_per_param) > per_param_clip:
        grad_scale = per_param_clip / float(grad_norm_per_param)
        grad_raw_flat = grad_raw_flat * grad_scale
        grad_raw = unravel_fn(grad_raw_flat)
        grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
        grad_norm_per_param = grad_norm_raw / jnp.sqrt(n_params)
        logger.info(f"【Step {step}】per-param 梯度范数超限，已裁剪：scale={grad_scale:.4f} | 裁剪后 per_param={float(grad_norm_per_param):.4f}")

    grad_update = grad_raw

    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

    # 3. QGT 自然梯度（实对称系统，直接实数求解）
    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift, g_current)
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_norm_natural = grad_norm_raw

    # 4. 优化器内部完成梯度裁剪 + SGD 更新
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

    # 5. 能量监控：单副本均匀测度下的 Ritz 广义本征值  M v = λ S v
    monitor_samples_raw, monitor_sampler_state = monitor_sampler.sample(
        machine=monitor_machine,
        parameters=total_params,
        state=monitor_sampler_state,
        chain_length=MONITOR_SAMPLES_PER_CHAIN,
    )
    monitor_samples = monitor_samples_raw.reshape(-1, SINGLE_SIZE)
    lam_real, v, n_valid, Mhat, Shat = compute_lam_v_from_samples_uniform(
        estimator=monitor_estimator,
        total_params=total_params,
        x_batch=monitor_samples,
        return_MS=True,
    )
    eig_vals = lam_real          # 监控量：均匀测度 Ritz 值（能量升序）

    M_matrix_history.append(Mhat)
    S_matrix_history.append(Shat)
    V_matrix_history.append(v)
    lam_real_history.append(lam_real)

    # 6. 波函数与 Ψ 矩阵条件数监控
    log_Psi_batch = total_machine(total_params, x_batch, g_current)
    x_single = x_batch[0:1, ...]
    psi_mat = total_matrix_machine(total_params, x_single, g_current)[0]
    psi_cond = jnp.linalg.cond(psi_mat)
    s_cond = float(jnp.linalg.cond(Shat))

    gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))
    g_abs = float(jnp.linalg.norm(g_current))
    n_valid_train = int(jnp.sum(aux["valid"]))

    # 7. 日志
    logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
    logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | per_param={grad_norm_per_param:.4f} |  natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
    logger.info(f"原始规范坐标 G(Re Σrow_mean) = {gauge_now:+.4f} | 累计|g| = {g_abs:.4f}")
    logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e} | 监控S条件数 cond(S) = {s_cond:.2e}")
    logger.info(
        f"Loss={loss_mean:.6f} | 均匀测度 Ritz λ(升序): "
        + " | ".join(f"λ{i}={lam_real[i]:.8f}" for i in range(K))
        + f" | n_valid_monitor={n_valid} | n_valid_train={n_valid_train}"
    )
    logger.info("#-----------------------------------------#")

    loss_history.append(float(jnp.real(loss_mean)))
    logpsi_history.append(float(jnp.real(log_Psi_batch.mean())))
    steps_history.append(step)

    logpsi_mean_history.append(float(jnp.real(log_Psi_batch.mean())))
    logpsi_min_history.append(float(jnp.real(log_Psi_batch.min())))
    logpsi_max_history.append(float(jnp.real(log_Psi_batch.max())))
    grad_norm_raw_history.append(float(grad_norm_raw))
    grad_norm_nat_history.append(float(grad_norm_natural))
    grad_norm_per_param_history.append(float(grad_norm_per_param))
    E_L_real_history.append(float(jnp.real(jnp.trace(E_L_mean))))
    E_L_imag_history.append(float(jnp.imag(jnp.trace(E_L_mean))))
    Energy_levels_history.append(eig_vals[:K])
    # 样本只保留最近 2 批（滚动窗口）
    samples_history.append(samples)
    if len(samples_history) > 2:
        samples_history.pop(0)

    # 每 SAVE_INTERVAL 步保存/追加一次 pickle
    if (step + 1) % SAVE_INTERVAL == 0:
        history = {
            "steps": [int(s) for s in steps_history],
            "logpsi_mean": logpsi_mean_history,
            "logpsi_min": logpsi_min_history,
            "logpsi_max": logpsi_max_history,
            "grad_norm_raw": grad_norm_raw_history,
            "grad_norm_natural": grad_norm_nat_history,
            "grad_norm_per_param": grad_norm_per_param_history,
            "E_L_real": E_L_real_history,
            "E_L_imag": E_L_imag_history,
            "loss": loss_history,
            "first_step": first_step if first_step is not None else 0,
            "save_interval": SAVE_INTERVAL,
            "Energy_levels": Energy_levels_history,
            "params": params_history,
            "M_matrix": M_matrix_history,
            "S_matrix": S_matrix_history,
            "V_matrix": V_matrix_history,
            "lam_real": lam_real_history,
            "samples": samples_history,
        }

        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(history, f)
        logger.info(f"[保存] Step {step} → pickle 文件已更新 ({len(steps_history)} 条记录)")

    # 8. 周期性 gauge reset：把 raw L 的列均值吸收进全局 g（物理不变）
    if (step + 1) % RESET_PERIOD == 0:
        new_col_mean = col_mean_fn(total_params, x_batch)
        g_current = new_col_mean
        absorbed = float(jnp.sum(jnp.real(new_col_mean)))
        logger.info(
            f"[GaugeReset@{step + 1}] 吸收 Δg = Σ Re(col_mean) = {absorbed:+.4f} "
            f"| 新|g| = {float(jnp.linalg.norm(g_current)):.4f}"
        )

end_time = time.time()
logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
logger.info("\n" + "=" * 60)
logger.info("训练完成!")
logger.info("=" * 60)

# 训练结束后记录「最后一次优化更新之后」的参数，保证 history["params"][-1] 与末尾另存的 npz 同一份
params_history[:] = [total_params]

# 训练结束后最后一次写入，确保数据完整
history = {
    "steps": [int(s) for s in steps_history],
    "logpsi_mean": logpsi_mean_history,
    "logpsi_min": logpsi_min_history,
    "logpsi_max": logpsi_max_history,
    "grad_norm_raw": grad_norm_raw_history,
    "grad_norm_natural": grad_norm_nat_history,
    "grad_norm_per_param": grad_norm_per_param_history,
    "E_L_real": E_L_real_history,
    "E_L_imag": E_L_imag_history,
    "loss": loss_history,
    "first_step": first_step if first_step is not None else 0,
    "save_interval": SAVE_INTERVAL,
    "Energy_levels": Energy_levels_history,
    "M_matrix": M_matrix_history,
    "S_matrix": S_matrix_history,
    "V_matrix": V_matrix_history,
    "lam_real": lam_real_history,
    "params": params_history,
    "samples": samples_history,
}
with open(HISTORY_FILE, "wb") as f:
    pickle.dump(history, f)
logger.info(f"[保存] 训练结束，pickle 最终写入 {len(steps_history)} 条记录 → {HISTORY_FILE}")

# 最后一个轮次的参数另存一份 npz（nnx.State 逐叶子存储 + 结构元信息）
n_leaf = save_params_npz(total_params, PARAMS_NPZ)
logger.info(f"[保存] 最终轮次 params 另存 npz（{n_leaf} 个叶子）→ {PARAMS_NPZ}")

logger.info(f'绘制图片...')

# 监控量 = 单副本均匀测度下的 Ritz 广义本征值 λ（M v = λ S v）
levels = np.asarray(history["Energy_levels"], dtype=float)
fci_k = np.asarray(E_fcis[:K], dtype=float)
energy_level_error = np.abs(levels - fci_k)          # (N_step, K) 与 FCI 的偏差
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:K]

fig, axs = plt.subplots(5, 1, figsize=(10, 25))

# 子图 0：每步 λ（= 单副本均匀测度 Ritz 值）+ FCI 基准虚线（同能级同色）
for k in range(K):
    axs[0].plot(history["steps"], levels[:, k], lw=1.2, color=colors[k],
                label=f"NES-VMC level {k}")
    axs[0].axhline(fci_k[k], lw=1.0, ls=":", color=colors[k],
                   label=f"EFCI level {k}")
axs[0].set_ylabel("Ritz λ (uniform measure, Ha)")
axs[0].set_xlabel("step")
axs[0].set_title(f"NES-VMC LiH (CAS(2e,4o), 8 qubit, real-parameter FFNN) K={K}, monitoring = uniform-measure Ritz values")
axs[0].legend(fontsize=8)

# 子图 1：Loss（= Re tr⟨E_L⟩，迹估计无偏；收敛后与 Σλ 一致）
axs[1].plot(history["steps"], history["loss"])
axs[1].set_ylabel("Loss = Re tr(E_L)")
axs[1].set_xlabel("step")

# 子图 2：log|Ψ| 采样统计
axs[2].plot(history["steps"], history["logpsi_mean"])
axs[2].set_ylabel("logpsi_mean")
axs[2].set_xlabel("step")

# 子图 3：能级误差 |λ_k − E_FCI,k|（log 轴 + 化学精度带）
for k in range(K):
    err = energy_level_error[:, k]
    mk = np.isfinite(err) & (err > 0)              # 防 log 轴 NaN 断线
    axs[3].semilogy(np.asarray(history["steps"])[mk], err[mk], lw=1.2, color=colors[k],
                    label=f"level {k} error")
axs[3].axhspan(1e-9, 0.0016, facecolor="lightblue", alpha=0.3, zorder=0,
               label="chemical accuracy (1.6 mHa)")
axs[3].set_ylabel("|λ - E_FCI| (Ha)")
axs[3].set_xlabel("step")
axs[3].set_yscale("log")
axs[3].legend(fontsize=8)

# 子图 4：梯度
axs[4].plot(history["steps"], history["grad_norm_per_param"])
axs[4].set_ylabel("grad_norm_per_param")
axs[4].set_xlabel("step")

fig.tight_layout()
out_png = os.path.join(graph_dir, f"LiH_STO3G_CAS_8Qbit_K{K}_Ansatz_A_RealFFNN_{time_str}.png")
fig.savefig(out_png, dpi=150)
logger.info(f"已保存图: {out_png}")
plt.show()
