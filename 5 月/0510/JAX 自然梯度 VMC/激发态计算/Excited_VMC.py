import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx
import optax
from tqdm import tqdm
from functools import partial
from jax import flatten_util
import orbax.checkpoint as ocp
from pathlib import Path


# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
bond_length = 1.4
geometry = [("H", (0.0, 0.0, 0.0)), ("H", (bond_length, 0.0, 0.0))]
mol = gto.M(atom=geometry, basis="STO-3G", verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("=" * 60)
print("H₂ FCI 基准能量")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

ha = nkx.operator.from_pyscf_molecule(mol)
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1 / 2,
    n_fermions_per_spin=(1, 1),
)


# ==============================================================================
# 2. 神经网络 Ansatz
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(
            n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex
        )
        self.linear2 = nnx.Linear(
            hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex
        )
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


# ==============================================================================
# 4. 初始化模型、采样器、优化器
# ==============================================================================
excited_model = SingleStateAnsatz(4, 12, rngs=nnx.Rngs(21))
ground_model = SingleStateAnsatz(4, 12, rngs=nnx.Rngs(22))
# 采样器
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(
    hi, rule=single_rule, n_chains=100, sweep_size=32
)
optimizer = nk.optimizer.Sgd(learning_rate=0.1)

# ================================

# 1. 获取当前工作目录的绝对路径
current_dir = Path.cwd().parent  # 或者用 Path(__file__).parent 如果在脚本中
ckpt_dir = current_dir / "暂存态"
# 2. 使用 PyTreeCheckpointer 而不是 StandardCheckpointer
checkpointer = ocp.PyTreeCheckpointer()

# 3. 保存模型（使用绝对路径）
graphdef, gs_state = nnx.split(ground_model)
save_path = str(ckpt_dir / "ground_state")  # 转为字符串
restore_state = checkpointer.restore(save_path, gs_state)
ground_model = nnx.merge(graphdef, restore_state)


# ===================== 4. 包装模型为 machine 函数 =====================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state


# ===================== 5. 纯 JAX 实现的 force-based 梯度计算 =====================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)

    这对应 NetKet 的 local_value_kernel
    """
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    核心：复刻 NetKet 的 forces_expect_hermitian 函数

    使用 force-based 梯度计算：
    ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩

    关键：对于复数值网络，使用 holomorphic=True
    """
    # 1. 计算局部能量
    O_loc = compute_local_energies(machine, params, sigma)

    # 2. 统计能量均值
    O_mean, O_std = statistics(O_loc)

    # 3. 中心化局部能量
    O_centered = O_loc - O_mean

    # 4. 计算 ∇log ψ 对每个样本
    # 使用 jax.grad 计算复数梯度（holomorphic=True）
    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    # 5. 计算 force-based 梯度
    # grad = ⟨(E_loc - E_mean) ∇log ψ⟩ = (1/N) Σ (E_loc[i] - E_mean) ∇log ψ(σ[i])
    # grad_matrix 已经是 PyTree 结构，每个元素的形状是 (n_samples, ...)
    # 关键修复：O_centered 形状为 (n_samples,)，需要正确广播到梯度张量的每个维度
    # 使用 reshape 将 O_centered 变为 (n_samples, 1, 1, ..., 1) 以匹配梯度张量
    def weight_and_mean(grad_component):
        # grad_component 形状：(n_samples, d1, d2, ...)
        # O_centered 形状：(n_samples,)
        # 需要广播相乘后沿 axis=0 求平均
        weights = O_centered.reshape(
            (O_centered.shape[0],) + (1,) * (grad_component.ndim - 1)
        )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)

    return O_mean, O_std, grad


# @partial(jax.jit, static_argnames=("machine",))
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）/ F 矩阵

    QGT 定义：
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩

    这就是 NetKet SR 的核心

    参数：
    - machine: 波函数机器
    - params: 网络参数
    - sigma: 样本 (n_samples, n_orbitals)
    - diag_shift: 对角线正则化参数 λ

    返回：
    - qgt_reg: 正则化后的 QGT 矩阵 (n_params, n_params)
    - unravel_fn: 用于将展平的向量恢复为 PyTree 结构的函数
    """
    n_samples = sigma.shape[0]

    # 步骤 1: 计算每个样本的 ∇log ψ
    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    # grad_matrix 是 PyTree，每个元素形状为 (n_samples, ...)
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    # 步骤 2: 将 PyTree 展平为矩阵 (n_samples, n_params)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    # 步骤 3: 中心化（减去均值）
    # 这对应 QGT 定义中的第二项：- ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)  # (1, n_params)
    grad_centered = grad_flat - grad_mean  # (n_samples, n_params)

    # 步骤 4: 计算 QGT = (1/N) * Σ ∇log ψ* ∇log ψ^T
    # 注意：对于复数，需要使用共轭
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered

    # 步骤 5: 添加正则化
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

    return qgt_reg, unravel_fn


def exact_energy_efficient(machine, params, hi, ha):
    """
    使用局部能量公式精确计算，避免构建整个H矩阵
    """
    all_states = hi.all_states()
    n_states = all_states.shape[0]

    # 计算所有ψ(σ)
    logpsi_all = machine(params, all_states)
    psi_all = jnp.exp(logpsi_all)

    # 计算每个基态的局部能量
    E_loc_all = []
    for i, sigma in enumerate(all_states):
        # 获取连接和矩阵元
        eta, H_eta = ha.get_conn_padded(sigma.reshape(1, -1))
        eta = eta[0]  # (n_conn, n_sites)
        H_eta = H_eta[0]  # (n_conn,)

        # 计算ψ(η)
        logpsi_eta = machine(params, eta)
        psi_eta = jnp.exp(logpsi_eta)

        # 局部能量
        E_loc = jnp.sum(H_eta * psi_eta / psi_all[i])
        E_loc_all.append(E_loc)

    E_loc_all = jnp.array(E_loc_all)

    # 期望值
    prob = psi_all**2 / jnp.sum(psi_all**2)
    energy = jnp.sum(prob * E_loc_all)

    return energy.real


@partial(
    jax.jit,
    static_argnames=(
        "machine_gs",
        "machine_es",
    ),
)
def forces_expect_with_penalty(
    machine_gs, machine_es, params_gs, params_es, sigma, lam
):
    # 1. 能量项
    Eloc = compute_local_energies(machine_es, params_es, sigma)
    E, std = statistics(Eloc)
    E_cent = Eloc - E

    # 2. 波函数值
    log_psi_es = machine_es(params_es, sigma)  # 激发态
    log_psi_gs = machine_gs(params_gs, sigma)  # 基态

    # ===================== 【核心修复】正确重叠积分 =====================
    # 正确 VMC 重叠：O = ⟨ Ψ₀ / Ψₑₛ ⟩
    O = jnp.mean(jnp.exp(log_psi_gs - log_psi_es))
    overlap2 = jnp.abs(O) ** 2  # 一定 ≤ 1

    # 3. 对数梯度 ∇logΨ_es
    def glog(p, s):
        return machine_es(p, s)

    def grad_single(s):
        return jax.grad(lambda p: glog(p, s), holomorphic=True)(params_es)

    grad_log = jax.vmap(grad_single)(sigma)

    # ---------------------
    # 4. 能量梯度
    # ---------------------
    def gradE(g):
        w = E_cent.reshape(-1, *((1,) * (g.ndim - 1)))
        return 2.0 * jnp.mean(w * jnp.conj(g), axis=0)

    dE = jax.tree_map(gradE, grad_log)

    # ---------------------
    # 5. 正交惩罚梯度（正确公式）
    # ---------------------
    def gradP(g):
        # 权重 = Ψ₀* / Ψₑₛ
        weight = jnp.exp(jnp.conj(log_psi_gs) - log_psi_es)
        w = weight.reshape(-1, *((1,) * (g.ndim - 1)))
        return 2 * lam * jnp.real(O * jnp.mean(jnp.conj(w) * jnp.conj(g), axis=0))

    dP = jax.tree_map(gradP, grad_log)

    # 总梯度
    grad_total = jax.tree_map(lambda a, b: a + b, dE, dP)
    loss = E.real + lam * overlap2
    return loss, E, std, grad_total, overlap2


# 辅助函数：计算统计量（均值+标准差）
def statistics(x):
    mean = jnp.mean(x)
    std = jnp.std(x)
    return mean, std


def forces_expect_with_multiple_penalty(
    machine_list,  # 低能态machine列表 [machine_gs, machine_es1, machine_es2, ...]
    params_list,  # 低能态参数列表 [params_gs, params_es1, params_es2, ...]
    target_machine,  # 当前待优化的激发态machine
    target_params,  # 当前待优化的激发态参数
    sigma,  # 采样样本
    lam_list,  # 各低能态对应的惩罚系数 [lam_gs, lam_es1, lam_es2, ...]
):
    """
    多激发态带惩罚的梯度计算函数
    输入：
        machine_list: 所有更低能态的machine列表（按能量从低到高排序）
        params_list: 对应更低能态的参数列表
        target_machine: 当前要优化的激发态machine
        target_params: 当前要优化的激发态参数
        sigma: VMC采样样本 (n_samples, n_spin_orbitals)
        lam_list: 各低能态对应的惩罚系数列表
    输出：
        total_loss: 总损失 (能量 + 所有惩罚项)
        E_mean: 能量期望
        E_std: 能量标准差
        total_grad: 总梯度
        overlap_sq_list: 与各低能态的重叠平方列表
    """

    # ===================== 1. 计算能量项 =====================
    # 计算局部能量
    def compute_local_energies(machine, params, sigma):
        eta, H_eta = ha.get_conn_padded(sigma)
        logpsi_sigma = machine(params, sigma)
        logpsi_eta = machine(params, eta)
        logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
        return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

    Eloc = compute_local_energies(target_machine, target_params, sigma)
    E_mean, E_std = statistics(Eloc)
    E_cent = Eloc - E_mean

    # 能量项梯度 - 使用 vmap + tree_map 模式（与 forces_expect_with_penalty 一致）
    def glog(p, s):
        return target_machine(p, s)

    def grad_single(s):
        return jax.grad(lambda p: glog(p, s), holomorphic=True)(target_params)

    grad_log = jax.vmap(grad_single)(sigma)

    def gradE(g):
        w = E_cent.reshape(-1, *((1,) * (g.ndim - 1)))
        return 2.0 * jnp.mean(w * jnp.conj(g), axis=0)

    energy_grad = jax.tree_map(gradE, grad_log)

    # ===================== 2. 计算多态惩罚项 =====================
    # 初始化惩罚项总和和惩罚梯度总和
    total_penalty = 0.0
    total_penalty_grad = jax.tree_util.tree_map(jnp.zeros_like, energy_grad)
    overlap_sq_list = []

    # 遍历所有更低能态，计算正交惩罚
    for idx, (machine_low, params_low, lam) in enumerate(
        zip(machine_list, params_list, lam_list)
    ):
        # 计算与当前低能态的重叠积分 ⟨Ψ_low | Ψ_target⟩
        log_psi_target = target_machine(target_params, sigma)
        log_psi_low = machine_low(params_low, sigma)

        # 重叠积分的蒙特卡洛估计
        overlap = jnp.mean(jnp.exp(log_psi_low - log_psi_target))
        overlap_sq = jnp.abs(overlap) ** 2
        overlap_sq_list.append(overlap_sq)

        # 累加惩罚项
        total_penalty += lam * overlap_sq

        # 惩罚项梯度 - 使用 tree_map 模式（与 forces_expect_with_penalty 一致）
        def gradP(g):
            weight = jnp.exp(jnp.conj(log_psi_low) - log_psi_target)
            w = weight.reshape(-1, *((1,) * (g.ndim - 1)))
            return 2 * lam * jnp.real(overlap * jnp.mean(jnp.conj(w) * jnp.conj(g), axis=0))

        penalty_grad = jax.tree_map(gradP, grad_log)
        total_penalty_grad = jax.tree_util.tree_map(
            lambda a, b: a + b, total_penalty_grad, penalty_grad
        )

    # ===================== 3. 总损失和总梯度 =====================
    total_loss = E_mean + total_penalty
    total_grad = jax.tree_util.tree_map(
        lambda a, b: a + b, energy_grad, total_penalty_grad
    )

    return total_loss, E_mean, E_std, total_grad, overlap_sq_list
