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
import numpy as np                    # 经典数组：采样批转换、Gram 组装、绘图数据
import jax                            # 自动微分/编译后端；jax.config 控制精度
import jax.numpy as jnp               # jax 版 numpy：进入 jit/vmap 计算图的数组运算
import flax.nnx as nnx                # 神经网络模块库（NNX）：定义 FFNN 参数与 nnx.split
import netket as nk                   # 量子蒙特卡洛框架：希尔伯特空间、Metropolis 采样器
import optax                          # 优化器库：全局范数裁剪 + SGD
from jax.flatten_util import ravel_pytree   # 参数树 ⇄ 一维向量互转（SR 求解需要一维）
from scipy.linalg import eigh as scipy_eigh # scipy 广义 Hermitian 本征（收尾 2×2 提取用）
import sys

sys.path.append('..')                 # 上级目录含公共模块 NES_VMC_V1.py 等

# NES_VMC_V1：单列波函数 machine 与哈密顿量-波函数逐点乘法
#   create_single_machine_gauge_fixed：把 nnx 网络包装成 machine，并把规范（gauge）
#     固定在 HF 行列式上（logψ(HF) 的实部/虚部归零约定），保证不同运行可复现比较
#   NESFermionHopRule：费米子 Hop 跳跃规则（在轨道对之间移动电子，保持粒子数）
#   Ham_psi_scaled：逐点计算 (Hψ)(x)（稀疏局域连接求和），shift 参数提出公共实指数
#     因子 e^{shift} 以控制数值尺度
from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
    Ham_psi_scaled,
)
# NES_VMC_deflation_mf：方案 A' 工具箱（MC Gram 投影系数的 matrix-free bundle）
#   make_mf_bundle：构造 aux_logq / estimate_grams / machine_batch / grad_fn / qgt_fn / cols
#   update_scales：由每批 aux 样本的 max 做 EMA，更新冻结尺度 f 与 active 尺度 g
from NES_VMC_deflation_mf import make_mf_bundle, update_scales
# 体系定义文件（H2/6-31G，CAS(2e,4o) → 8 自旋轨道 = 8 qubit）：
#   SINGLE_SIZE=8：单组态比特数（自旋轨道个数）
#   ha：netket 费米子哈密顿量（稀疏，get_conn_padded 路径）
#   hi：单组态希尔伯特空间（8 bit）；hi_ext：K 副本联合 walker 空间（K×8 bit）
#   K=2：active 列数（剩余子空间维数）；Hatree_Fock：HF 行列式（gauge 锚点）
#   single_edges：单组态 Hop 边的轨道对列表；ext_edges：联合 walker 的边（每副本内）
from H2_6_31G_8Qbit_K2 import (
    SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock, single_edges,
)

jax.config.update("jax_enable_x64", True)   # 全链路 float64/complex128（防 0917 float32 崩溃）
assert K == 2 and SINGLE_SIZE == 8, "本脚本按 K=2/SINGLE_SIZE=8 设计"

# ===================== Ansatz：与 K4/方案A RealFFNN 完全一致 =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    """单列（单自旋轨道占据组态 → 复对数振幅）前馈网络。

    输出 logψ(x) = log(u + i·v)：u/v 是两个线性头，实参数网络产生复数输出，
    相位支路 v 使激发态（需变号/复相位）可表示——实参数单振幅网络只能正定。
    """
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals   # n_so：number of spin-orbitals（输入维 = 8）
        # 主干两层 tanh：8 → hidden → hidden；param_dtype=float 即实参数
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)  # 实部头 u=Re z
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)  # 虚部头 v=Im z

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        u = jnp.squeeze(self.head_u(h))   # 标量：复振幅实部
        v = jnp.squeeze(self.head_v(h))   # 标量：复振幅虚部
        # +1e-12 防 z 落在原点（log 奇异/梯度爆炸）；约定 u,v 同时非零
        z = (u + 1e-12) + 1j * (v + 1e-12)
        return jnp.log(z)                 # 返回 logψ（machine 惯例：输出对数振幅）


class NESTotalAnsatz_stable(nnx.Module):
    """K 列总拟设：持有 K 个独立单列网络（NES = Neural-network Quantum State
    多 Slater 展开，每列对应一个单粒子轨道组态波函数 ψ_m）。"""
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states             # 列数（active 子空间维数）
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()   # nnx.List 使子模块参数被注册、可 split
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)   # 每列独立子种子，初始化互不相关
            self.single_ansatz_list.append(
                SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(params=sub_key))
            )


# ====================== 日志配置 ======================
time_str = time.strftime("%y-%m-%d-%H-%M")   # 运行时间戳，用于所有产物文件名（ts）
this_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

logger = logging.getLogger("NES_VMC_K2_frozen4_deflation_MatrixFree_Pure")
logger.setLevel(logging.INFO)
logger.propagate = False              # 不向 root logger 冒泡，避免重复输出
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
log_path = f"{log_dir}/{time_str}_nes_vmc_K2_frozen4_deflation_MatrixFree_Pure.log"
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")   # fh：file handler（mode="w" 每次覆盖）
fh.setFormatter(simple_formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()          # ch：console handler（同步打印）
ch.setFormatter(simple_formatter)
logger.addHandler(ch)

# ====================== 超参（与已验证 A' 完全一致） ======================
# ---- 主采样（联合 NES walker）----
N_CHAINS = 16                         # MCMC 并行链数
N_SAMPLES_PER_CHAIN = 200             # 每链每步采样点数（总样本 = 16×200 = 3200）
SWEEP_SIZE = 20                       # 每步做多少个全链扫描（一次 sweep=每 walker 尝试 n_move 次移动）
N_ITER = int(os.environ.get("NES_N_ITER", 1500))   # 训练步数（smoke 用 NES_N_ITER=5）
Natural_Grad = True                   # True=SR 自然梯度（解 QGT 线性系统）；False=原始梯度
clip_norm = 20.0                      # 全局梯度范数裁剪上限（optax clip_by_global_norm）
per_param_clip = 0.5                  # "每参数平均范数"裁剪阈值（先于 SR 施加在原始梯度上）
lr = 0.1                              # SGD 学习率（SR 方向已含度量信息，故用较大步长）
qgt_diag_shift = 0.1                  # diag_shift：QGT 对角正则 η，S+ηI 防奇异（SR 稳定项）
SAVE_INTERVAL = 20                    # 每 20 步落盘一次 history pkl（断点可查）
INIT_SEED = int(os.environ.get("NES_INIT_SEED", 11))   # active 列随机初始化种子（参照运行已证健康）
# ---- 辅助采样（单组态，估 MC Gram 用；aux = auxiliary）----
AUX_M = int(os.environ.get("NES_AUX_M", 1024))         # 每步辅助样本批大小 M（Gram 估计精度 ∝ M^{-1/2}）
AUX_CHAINS = 8                        # 辅助采样链数
AUX_SWEEP = 20                        # 辅助采样每步扫描数
SCALE_BETA = float(os.environ.get("NES_SCALE_BETA", 0.9))  # β：尺度 f/g 的 EMA 平滑系数（首步取 1=fresh）
EPS_MIX = 0.25                        # ε：参考密度 q=Σ|φ̃|²+ε·Σ|ψ̃|² 中 active 项混合系数
                                      #（保证 q 覆盖 active 列支撑，防稀有组态梯度信号丢失）

# 冻结源：npz 只取 v_rot（4×4 列约定，非全空间对象）；pkl 取旧网络参数
NPZ_PATH = os.path.join(this_file_dir, "冻结K4", "近似本征态_K4_26-10-07-19-57.npz")
PKL_PATH = os.path.join(this_file_dir, "data", "26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl")
HISTORY_FILE = f"{data_dir}/{time_str}_history_K2_frozen4_deflation_MatrixFree_Pure.pkl"

# ====================== 步骤 0：冻结列的逐点表示（无全空间运算） ======================
npz = np.load(NPZ_PATH)
# v_rot：(4,4) 列旋转矩阵。K4 训练收尾 block 分析给出的列约定：
#   φ_k = Σ_j ψ_j^old · v_rot[j,k]，即冻结物理态 = 旧网络 4 列的固定线性组合。
#   只取这个 4×4 小矩阵 ⇒ 无需存储任何 16 维向量（matrix-free 关键）。
v_rot = np.asarray(npz["v_rot"], dtype=np.complex128)
with open(PKL_PATH, "rb") as f:
    old_params = pickle.load(f)["params"][-1]   # 旧 K4 训练最后一步的参数树（冻结权重）

# 重建旧 4 列网络结构（hidden_dim=8+4=12，与当年 K4 训练一致；rngs=11 仅为建结构，
# 参数随后由 old_params 覆盖 ⇒ 初始化种子无关紧要）
old_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=4,
    hidden_dim=SINGLE_SIZE + 4, rngs=nnx.Rngs(11),
)
# frozen_sms：frozen single machines —— 4 个 gauge-fixed 旧网络 callable，
#   签名 (params, sigma) → logψ；参数为常量，不参与梯度（天然 stop-gradient）
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
# 工程惯例：Stage1（剩余子空间）必须 fresh 初始化，不能沿用 Stage0 收敛列，
# 否则投影后向量落在低维子空间使 2×2 行列式奇异。hidden_dim=8+2=10。
seed = INIT_SEED
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K,
    hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(seed),
)
# sms：single machines（active 列）—— 可训练的 gauge-fixed machine 列表
sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in total_ansatz.single_ansatz_list]
# nnx.split(模块) → (变量视图, 参数pytree)；[1] 取纯参数树（jax 函数只收 pytree）
total_params = nnx.split(total_ansatz)[1]
logger.info(f"active 列 fresh 初始化 seed={seed}（参照运行同 seed，cond 预检已证健康）")

# ====================== bundle + 采样器 ======================
# bundle：方案 A' 全部计算对象的字典（aux_logq/estimate_grams/machine_single/
#   machine_batch/forward/grad_fn/qgt_fn/cols），内部只按 walker 逐点求值
bundle = make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms,
                        n_spin=SINGLE_SIZE, eps=EPS_MIX)

# f：冻结列公共实尺度（φ̃=φ·e^{−f}）；g：active 列逐列实尺度（ψ̃_m=ψ_m·e^{−g_m}）
# 初值 0 ⇒ 训练第一步的 aux 批先做 fresh 归一（β=1）
f = jnp.asarray(0.0)
g = jnp.zeros(K)


def aux_machine(params_aux, sigma):
    """辅助采样器 machine：返回 log q(σ)（未归一参考密度的对数）。
    params_aux = (total_params, f, g)；netket 只要求标量 log 幅，q 不是波函数但接口兼容。"""
    return bundle["aux_logq"](params_aux, sigma)


def sample_machine(params, sigma):
    """主采样器 machine：返回联合 walker 批的 logΨ_F = log det[χ̃₁|χ̃₂]。
    params = (total_params, c, c2, f, g) —— c/c2/f/g 必须显式入参，
    若走闭包会被 jax.jit 焊死成常量（每步换值需重编译）。"""
    p, cc, cc2, ff, gg = params
    return bundle["machine_batch"](p, sigma.reshape(-1, hi_ext.size), cc, ff, gg)


# 辅助采样器：单组态空间 hi（8 bit），K=1 Hop 规则，machine_pow=1 ⇒ 按 |√q| 行走
aux_rule = NESFermionHopRule(edges=jnp.asarray(single_edges), K=1, single_size=SINGLE_SIZE)
aux_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi, rule=aux_rule, n_chains=AUX_CHAINS, sweep_size=AUX_SWEEP, machine_pow=1,
)
# 主采样器：联合空间 hi_ext（K×8=16 bit），ext_edges 只在各副本内部移动 ⇒
# 每个 walker 是 K 个独立单组态的元组，链间不交换副本
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext, rule=nes_rule, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE, machine_pow=1,
    # machine_pow=1：按 |Ψ| 而非 |Ψ|² 采样 —— 10-06 修复的延续，配合梯度端
    # w∝|Ψ| 重要性加权恢复精确期望公式，防零方差冻结（稀有组态信号 ∝ ψη 非 ψη²）
)

# 原始梯度先全局范数裁剪，再 SGD；SR 解出的自然梯度也走同一链（clip 通常不触发）
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
sampler_state = None                  # 主采样状态延迟初始化（需等首个 c_cur 算出）
c_cur = jnp.zeros((4, K), dtype=jnp.complex128)    # c：Φ̃→Ψ̃ 投影系数（4 冻结 × K active），步内常量
c2_cur = jnp.zeros((4, K), dtype=jnp.complex128)   # c2：二次投影系数（project→H→project）

# ====================== 历史容器 ======================
loss_history, steps_history = [], []          # Loss=Re tr⟨E_L⟩；步号
lam_history, params_history = [], []          # 每步 λ̂（2×2 本征）；每步参数快照
diag_history = []                              # 每步诊断字典（valid/ESS/梯度/aux 权重）
n_skipped = 0                                  # 跳过更新计数（grams 非有限 / should_skip）
n_grad_clip = 0                                # per-param 裁剪触发计数

start_time = time.time()

# ====================== 训练循环（matrix-free，与参照版逐行同构） ======================
for step in range(N_ITER):
    params_history.append(total_params)

    # --- 1. aux 采样 → MC Gram 估计 c/c2 ---
    # s_aux：(AUX_CHAINS, chain_length, SINGLE_SIZE) 单组态样本 ~ p=q/Z
    s_aux, aux_state = aux_sampler.sample(
        machine=aux_machine, parameters=(total_params, f, g),
        state=aux_state, chain_length=AUX_M // AUX_CHAINS,
    )
    # sigma_aux：(M, 8) 展平批；np.asarray 双转换是 netket 状态对象 → jax 数组的惯例
    sigma_aux = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))
    # estimate_grams：同一批样本、同一 q 加权 ⇒ Ĝ⁻¹M̂ 中未知归一化 Z 严格抵消（Z 抵消引理）
    #   c_mc/c2_mc：缩放基 (Φ̃,Ψ̃) 下的投影系数（stop-grad，视作步内常量）
    #   max_phi2/max_psi：批内 max(|Φ̃|² 行和, |ψ̃_m|)，供 EMA 尺度更新
    #   aux_g：{"w_ess"(权重有效样本数), "max_w_ratio"(权重峰值/均值), "grams_finite"}
    c_mc, c2_mc, max_phi2, max_psi, aux_g = bundle["estimate_grams"](total_params, sigma_aux, f, g)
    grams_ok = bool(aux_g["grams_finite"])
    if not grams_ok:                    # Gram 出现 NaN/Inf ⇒ 本步无安全 c 可用，整步作废
        logger.warning(f"【Step {step}】grams 非有限，跳过整步")
        n_skipped += 1
        loss_history.append(np.nan)
        lam_history.append(np.array([np.nan, np.nan]))
        steps_history.append(step)
        diag_history.append({"valid_ratio": np.nan, "ess": np.nan, "grad_norm_raw": np.nan,
                             "grad_norm_natural": np.nan, "w_ess": np.nan, "max_w_ratio": np.nan})
        continue

    # --- 2. EMA 尺度更新 + c/c2 同步变换（χ̃ 方向不变） ---
    # 首步 β=1（fresh 归一），其后 β=SCALE_BETA 平滑；update_scales 返回新 (f,g)：
    #   f += (β/2)·ln maxΣ|φ̃_i|²，g_m += β·ln max|ψ̃_m| ⇒ 目标两项同 O(1)
    beta = 1.0 if step == 0 else SCALE_BETA
    f_np, g_np = update_scales(float(f), np.asarray(g), max_phi2, max_psi, beta=beta)
    d_f = f_np - float(f)               # Δf：本步冻结尺度增量
    d_g = g_np - np.asarray(g)          # Δg：K 维 active 尺度增量向量
    # 尺度变了：新基 φ̃'=φ̃·e^{Δf}、ψ̃'=ψ̃·e^{Δg}，要 χ̃'=Ψ̃'−Φ̃'c' 与旧 χ̃ 只差常量因子，
    # 须 c'[:,m] = c[:,m]·e^{Δf−Δg_m}（恒等变形，E_L 严格不变）
    trans = np.exp(d_f - d_g)           # (K,) 每列变换因子 e^{Δf−Δg_m}
    c_cur = jnp.asarray(np.asarray(c_mc) * trans[None, :])     # 广播到 (4,K) 的列
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
    # sigma_flat：(3200, hi_ext.size=16) 联合组态批；bundle 内部再 reshape 成 (n,K,8)
    sigma_flat = jnp.asarray(np.asarray(samples_raw).reshape(-1, hi_ext.size))

    # --- 4. 梯度 / SR 更新 ---
    # grad_fn：实参数 Wirtinger 力梯度 ∇=∇Re+i∇Im，配 w∝|Ψ_F| 重要性加权
    #   grad_raw：力梯度 pytree；loss_mean：加权 Re tr⟨E_L⟩（优化目标）
    #   E_L_mean：加权 K×K 局域能量矩阵 ⟨E_L⟩；aux：{valid_ratio, n_valid, ess,
    #   grad_finite, loss_finite, should_skip}
    grad_raw, loss_mean, E_L_mean, aux = bundle["grad_fn"](
        total_params, sigma_flat, c_cur, c2_cur, f, g
    )
    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)   # 参数树展平为一维（裁剪/SR 都在一维做）
    grad_norm_raw = float(jnp.linalg.norm(grad_raw_flat))
    n_params = grad_raw_flat.size
    grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)   # RMS 每参数梯度范数

    # per-param 裁剪：把 RMS 压到 0.5 ⇒ 等效把整梯度缩放，防单步过大（0917 教训的兜底）
    if grad_norm_per_param > per_param_clip:
        grad_scale = per_param_clip / grad_norm_per_param
        grad_raw_flat = grad_raw_flat * grad_scale
        grad_raw = unravel_fn(grad_raw_flat)             # 还原成树（供 else 分支/记录）
        grad_norm_raw = float(jnp.linalg.norm(grad_raw_flat))
        grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)
        n_grad_clip += 1
        logger.info(f"【Step {step}】per-param 梯度裁剪 scale={grad_scale:.4f}")

    # 异常预警：NaN 或原始梯度范数 >5000（0917 梯度爆炸的早期信号，仅告警不干预）
    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw={grad_norm_raw:.2f}")

    grad_norm_natural = float("nan")
    if bool(aux["should_skip"]):        # valid 比例 <0.25 或 loss/梯度非有限 ⇒ 不更新
        logger.warning(f"【Step {step}】should_skip=True（valid_ratio={float(aux['valid_ratio']):.3f}），跳过更新")
        n_skipped += 1
    elif Natural_Grad:
        # SR（stochastic reconfiguration）自然梯度：
        #   S = Re(QGT) + diag_shift·I（qgt_fn 内部已加正则，float64 稠密 P×P，P≈290）
        #   δθ = S⁻¹·g（solve 而非显式求逆）⇒ 沿波函数流形度量最速下降方向
        S = bundle["qgt_fn"](total_params, sigma_flat, qgt_diag_shift, c_cur, c2_cur, f, g)
        ng_flat = jnp.linalg.solve(S, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = float(jnp.linalg.norm(ng_flat))   # 监控量：收敛时应降至 ~1e-2
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)
    else:
        updates, opt_state = optimizer.update(grad_raw, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

    # --- 5. matrix-free 监控：λ̂ = eig(sym(⟨E_L⟩))（方法报告 §2.1，2×2 小矩阵） ---
    # E_h：E_L_mean 的 Hermitian 对称化（采样噪声使 ⟨E_L⟩ 非严格 Herm，对称化后取实本征值）
    # λ̂ 是 active block 能量的实时估计（有采样噪声；精确值看收尾广义本征）
    E_h = 0.5 * (E_L_mean + jnp.conj(E_L_mean.T))
    lam_hat = np.sort(np.linalg.eigvalsh(np.asarray(E_h)).real)

    accept = getattr(sampler_state, "average_acceptance", None)   # 主采样平均接受率（健康 ~0.1–0.5）
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
        "valid_ratio": float(aux["valid_ratio"]),   # 本步有效 walker 比例
        "ess": float(aux["ess"]),                   # 主采样重要性权重有效样本数
        "grad_norm_raw": grad_norm_raw,
        "grad_norm_natural": grad_norm_natural,     # SR 方向范数（收敛指标）
        "w_ess": float(aux_g["w_ess"]),             # aux 批 1/q 权重有效样本数（Gram 噪声水平）
        "max_w_ratio": float(aux_g["max_w_ratio"]), # aux 权重峰值/均值（重尾预警）
    })

    # 周期性落盘（smoke 的 N_ITER≤5 不存，防覆盖正式产物）；params 只留均匀采样的最近 15 份
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
sigma_ext = jnp.asarray(np.asarray(s_aux).reshape(-1, SINGLE_SIZE))   # 提取批样本 (M,8)
# c_f：提取用投影系数（复用 estimate_grams 只取第一个返回值；其余丢弃）
c_f, _c2_f, _m1, _m2, aux_gf = bundle["estimate_grams"](total_params, sigma_ext, f, g)
if not bool(aux_gf["grams_finite"]):
    logger.warning("提取批 grams 非有限，改用训练末 c_cur")
    c_f = c_cur

V = jnp.asarray(v_rot, dtype=np.complex128)
# cols：逐点求 (Φ̃, Ψ̃)，形状均 (M, ·)——冻结列由旧网络+v_rot 组合、active 列由当前参数
Phi, Psi = bundle["cols"](total_params, sigma_ext, f, g)
chi = Psi - Phi @ c_f                                   # (M, K) 缩放基下的 χ̃ = Ψ̃ − Φ̃·c
# Hpsi/Hphi：H 作用在缩放列上（Ham_psi_scaled 的 shift 参数吸收 e^{−g_m}/e^{−f}，
# 仍为逐点稀疏运算，每组态 ≤13 个连接）；Hphi 再按 v_rot 组合成冻结列
Hpsi = jnp.stack(
    [Ham_psi_scaled(ha=ha, single_machine=sms[m], params=total_params["single_ansatz_list"][m],
                    x=sigma_ext, shift=g[m]) for m in range(K)], axis=-1)
Hphi = jnp.stack(
    [Ham_psi_scaled(ha=ha, single_machine=frozen_sms[j], params=frozen_params[j],
                    x=sigma_ext, shift=f) for j in range(4)], axis=-1) @ V
Hchi = Hpsi - Hphi @ c_f                                # (M,K) Hχ̃（χ 已投影，Hχ 再减冻结成分）
# q/wi：与训练同一参考密度、同一 1/q 重要性权重（Z 抵消的前提：所有 Gram 用同一 q）
q = jnp.sum(jnp.abs(Phi) ** 2, axis=-1) + EPS_MIX * jnp.sum(jnp.abs(Psi) ** 2, axis=-1)
wi = 1.0 / (q * sigma_ext.shape[0])                     # 每样本权重 1/(M·q)
S_chi = np.asarray(jnp.conj(chi).T @ (wi[:, None] * chi), dtype=np.complex128)   # ⟨χ̃|χ̃⟩/Z (K,K)
M_chi = np.asarray(jnp.conj(chi).T @ (wi[:, None] * Hchi), dtype=np.complex128)  # ⟨χ̃|H|χ̃⟩/Z (K,K)
S_chi = 0.5 * (S_chi + S_chi.conj().T) + 1e-12 * np.eye(K)   # Hermitian 化 + 对角正则
M_chi = 0.5 * (M_chi + M_chi.conj().T)                        # 实能量本征要求 H 块 Herm
# 广义 Hermitian 本征 M·v = λ·S·v：1e-14·S 微扰防 S 病态；λ 带采样噪声但 Z 严格消掉
lam_mc, v_active = scipy_eigh(M_chi + 1e-14 * S_chi, S_chi)
order = np.argsort(lam_mc.real)                          # 升序排列（λ̂₄ < λ̂₅）
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
    c_extract=c_f,                     # 提取用投影系数 (4,K)（缩放基；配 scales_f/g 还原原始基）
    v_rot_frozen=v_rot,                # 冻结列约定（复现 φ₀..φ₃ 用）
    scales_f=float(f), scales_g=np.asarray(g),   # 最终尺度（φ̃=φe^{−f}、ψ̃=ψe^{−g}）
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
# diag：把每步诊断字典列表转成"键→数组"字典，便于整列绘图
diag = {k: np.asarray([d[k] for d in diag_history]) for k in diag_history[0]}

fig, axs = plt.subplots(4, 1, figsize=(10, 16))

# 子图 1：active block 能量 λ̂（每步，含采样噪声）——应收敛到 oracle ν≈(-0.0280,-0.0145)
axs[0].plot(steps_arr, lam_arr, lw=1.2, label=[f"lambda_hat_{k} (active block)" for k in range(K)])
axs[0].set_ylabel("eig(sym <E_L>) (Ha)")
axs[0].set_title(f"Scheme A' PURE matrix-free deflated NES-VMC, frozen K=4 + K={K}, H2 6-31G "
                 f"({N_ITER} steps, {elapsed:.0f}s, no full-space ops)")
axs[0].legend(fontsize=8)

# 子图 2：Loss（跳过步为 NaN，先掩码再画，防 1015 记录的"NaN 断线"坑）
m = np.isfinite(loss_arr)
axs[1].plot(steps_arr[m], loss_arr[m], lw=1.0)
axs[1].set_ylabel("Loss = Re tr(E_L)")

# 子图 3：采样健康度（均归一为比例）：valid_ratio、主采样 ESS 占比、aux 权重 ESS 占比
for lbl, key, norm in [("valid_ratio", "valid_ratio", 1.0), ("ESS frac", "ess", N_CHAINS * N_SAMPLES_PER_CHAIN),
                       ("aux w_ess frac", "w_ess", AUX_M)]:
    v = diag[key] / norm
    mk = np.isfinite(v)
    axs[2].plot(steps_arr[mk], v[mk], lw=1.0, label=lbl)
axs[2].set_ylabel("sampling health")
axs[2].set_xlabel("step")
axs[2].legend(fontsize=8)

# 子图 4：梯度范数（log 轴）：raw（裁剪后）与 natural（SR 解出方向）
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
