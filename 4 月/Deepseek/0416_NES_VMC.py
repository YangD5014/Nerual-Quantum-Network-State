import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax

# =============================================================================
# 0. 设置 JAX 浮点精度 (可选，提升稳定性)
# =============================================================================
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 1. 分子定义与 PySCF 哈密顿量生成
# =============================================================================
bond_length = 1.4  # Bohr
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确解作为基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("=" * 60)
print("H₂ FCI 基准能量 (STO-3G)")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

# 将 PySCF 分子转为 NetKet 离散算符
ha = nkx.operator.from_pyscf_molecule(mol)

# 获取自旋轨道数
n_spin_orbitals = ha.hilbert.size  # STO-3G for H2: 4 orbitals (2 spatial * 2 spins)

# =============================================================================
# 2. NES‑VMC 参数与扩展 Hilbert 空间
# =============================================================================
K = 2  # 需要计算的态数（基态 + 1 个激发态）

# 原始 Hilbert 空间：每个 H 原子提供一个 1s 轨道，共 2 个空间轨道，每个轨道可容纳自旋上/下
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_spin_orbitals // 2,  # 空间轨道数 = 2
    s=1/2,
    n_fermions_per_spin=(1, 1),      # 一个上自旋电子，一个下自旋电子
)

# 扩展 Hilbert 空间：K 个副本的直积
hi_ensemble = hi ** K

print(f"\n原始 Hilbert 空间大小: {hi.size}")
print(f"扩展 Hilbert 空间大小 (K={K}): {hi_ensemble.size}")

# =============================================================================
# 3. 构建采样器
# =============================================================================
# 为每个副本构造独立的费米子跃迁规则
edges = [(0, 1), (2, 3)] #edges 是对 hi 的 edges
g = nk.graph.Graph(edges=edges)

single_rule = nk.sampler.rules.FermionHopRule(
    hilbert=hi,
    graph=nk.graph.Graph(edges=edges)  
)

# 组合成张量规则，作用在扩展空间上
tensor_rule = nk.sampler.rules.TensorRule(
    hilbert=hi_ensemble,
    rules=[single_rule] * K
)

sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ensemble,
    rule=tensor_rule,
    n_chains=16,          # 并行链数
    sweep_size=16,        # 每步 MCMC 尝试次数
    reset_chains=True,    # 初始化时重置链状态
)

# =============================================================================
# 4. 神经网络模型定义
# =============================================================================
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz: 输入单个副本的 Fock 态，输出复数波函数值"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        # x: (n_spin_orbitals,) 的 one-hot 表示
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)   # 标量复数


class NESTotalAnsatz(nnx.Module):
    """NES 总 Ansatz: 输入 K 个副本状态，返回 det[ψ_j(x^i)]"""
    def __init__(self, n_spin_orbitals: int, n_states: int = K, hidden_dim: int = 16):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals

        # 为每个单态 Ansatz 生成不同的随机种子，避免对称性
        key = jax.random.key(42)
        keys = jax.random.split(key, n_states)

        self.single_ansatz_list = [
            SingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim,
                rngs=nnx.Rngs(keys[i])
            )
            for i in range(n_states)
        ]

    def __call__(self, x_batch):
        """
        x_batch: (n_states, n_spin_orbitals)  —— 即 K 个副本的状态
        返回: (总波函数值 Ψ, 矩阵 M)  其中 M_{ij} = ψ_j(x^i)
        """
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)                     # (K, K)
        psi_total = jnp.linalg.det(M)        # 行列式
        return psi_total, M


# 实例化总 Ansatz
total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=n_spin_orbitals,
    n_states=K,
    hidden_dim=4 * 3,   # 隐藏层宽度
)

# 将模型参数提取为可微分状态 (nnx.split)
graph_def, params = nnx.split(total_ansatz)

# =============================================================================
# 5. 辅助函数：哈密顿量作用与局部能量矩阵
# =============================================================================
def apply_hamiltonian_to_single_state(ha, single_model, x):
    """
    计算 Ĥ ψ_j(x) 的值
    ha: NetKet 离散算符
    single_model: SingleStateAnsatz 实例
    x: 单个 Fock 态 (长度 n_spin_orbitals 的一维数组)
    返回: Σ_{x'} H_{x,x'} ψ_j(x')
    """
    # 获取所有连接态及矩阵元
    x_conn, mels = ha.get_conn(x)
    # 计算每个连接态的波函数值
    psi_conn = jax.vmap(lambda xc: single_model(xc))(x_conn)
    # 求和
    return jnp.sum(mels * psi_conn)


# 向量化：对多个副本同时计算
apply_hamiltonian_to_single_state_batch = jax.vmap(
    apply_hamiltonian_to_single_state,
    in_axes=(None, None, 0)   # ha, model 不变，x 沿第0维批量
)



def compute_H_Psi_matrix(ha, total_ansatz, x_batch):
    K = total_ansatz.n_states
    _, M = total_ansatz(x_batch)                      # M: (K, K)
    H_Psi = jnp.zeros((K, K), dtype=jnp.complex128)

    # 对每个副本 i 施加哈密顿量
    for i in range(K):
        xi = x_batch[i]
        conn_states, mels = ha.get_conn_padded(xi)           # 获取 xi 的所有连接态及矩阵元
        # 计算替换第 i 行后的行列式
        for xp, mel in zip(conn_states, mels):
            new_row = jnp.array([total_ansatz.single_ansatz_list[j](xp) for j in range(K)])
            M_modified = M.at[i, :].set(new_row)
            det_modified = jnp.linalg.det(M_modified)
            # 累加 mel * det_modified 到 H_Psi 的第 i 行（因为是对第 i 个副本作用）
            H_Psi = H_Psi.at[i, :].add(mel * det_modified)
    return H_Psi
 

def compute_local_energy_matrix(ha, total_ansatz, x_batch):
    """
    计算局部能量矩阵 E_L = M^{-1} (ĤΨ)
    返回: (迹, 矩阵 E_L)
    """
    psi_total, M = total_ansatz(x_batch)
    # 防止奇异
    eps = 1e-6
    M_reg = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    H_Psi = compute_H_Psi_matrix(ha, total_ansatz, x_batch)
    # 求解线性方程组
    E_L = jnp.linalg.solve(M_reg, H_Psi)
    return jnp.trace(E_L), E_L


# 批量版本（用于多个样本）
def compute_local_energy_single(ha, total_ansatz, x_batch):
    tr, mat = compute_local_energy_matrix(ha, total_ansatz, x_batch)
    return tr.real, mat


compute_local_energy_batch = jax.vmap(
    compute_local_energy_single,
    in_axes=(None, None, 0),
    out_axes=(0, 0)
)


def compute_average_local_energy(ha, total_ansatz, samples):
    """
    samples: (n_samples, K, n_spin_orbitals)
    返回: 平均迹，平均 E_L 矩阵
    """
    traces, E_mats = compute_local_energy_batch(ha, total_ansatz, samples)
    return traces.mean(), E_mats.mean(axis=0)


# =============================================================================
# 6. 损失函数与梯度
# =============================================================================
def loss_fn(params, graph, ha, x_batch):
    """单样本损失 = 局部能量矩阵的迹"""
    model = nnx.merge(graph, params)
    tr, _ = compute_local_energy_single(ha, model, x_batch)
    return tr


def loss_fn_batch(params, graph, ha, x_batch):
    """批量平均损失"""
    model = nnx.merge(graph, params)
    tr, _ = compute_average_local_energy(ha, model, x_batch)
    return tr


value_and_grad_batch = jax.value_and_grad(loss_fn_batch)


# =============================================================================
# 7. 用于采样的 forward 函数 (NetKet 要求返回 log|ψ|)
# =============================================================================
def forward(params, x_batch):
    """
    NetKet 采样器接口：
    x_batch: (n_chains, chain_length, n_features) 其中 n_features = n_spin_orbitals * K
    返回: log|ψ| (实数)
    """
    n_chains = x_batch.shape[0]
    # 重塑为 (n_chains, K, n_spin_orbitals)
    x_reshaped = x_batch.reshape(n_chains, K, n_spin_orbitals)

    def single_logpsi(p, x):
        model = nnx.merge(graph_def, p)
        psi, _ = model(x)
        # 返回 log|ψ|，用于 MCMC 接受率计算
        return jnp.log(jnp.abs(psi) + 1e-12)

    log_psi_batch = jax.vmap(single_logpsi, in_axes=(None, 0))(params, x_reshaped)
    return log_psi_batch


# =============================================================================
# 8. 初始化采样器状态
# =============================================================================
sampler_state = sampler.init_state(forward, params, seed=1)
sampler_state = sampler.reset(forward, params, sampler_state)

# =============================================================================
# 9. 训练设置
# =============================================================================
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

n_epochs = 100
chain_length = 64   # 每个链每次采样的长度

print("\n开始训练 NES‑VMC ...")
print("-" * 60)

for epoch in range(n_epochs):
    # 采样
    sampler_state = sampler.reset(forward, params, sampler_state)
    samples, sampler_state = sampler.sample(
        forward,
        params,
        state=sampler_state,
        chain_length=chain_length
    )
    # samples 形状: (n_chains * chain_length, n_features) = (n_chains * chain_length, n_spin_orbitals * K)
    n_total_samples = samples.shape[0]
    x_batch = samples.reshape(-1, K, n_spin_orbitals)

    # 计算损失和梯度
    loss, grads = value_and_grad_batch(params, graph_def, ha, x_batch)

    # 更新参数
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # 打印日志
    if epoch % 20 == 0:
        # 使用当前样本估算各态能量
        _, E_mat_avg = compute_average_local_energy(ha, nnx.merge(graph_def, params), x_batch)
        eigvals = jnp.linalg.eigvals(E_mat_avg).real
        eigvals_sorted = jnp.sort(eigvals)
        print(f"Epoch {epoch:4d} | Loss = {loss:.6f} | Energies: {eigvals_sorted}")

print("训练完成！")

# =============================================================================
# 10. 最终评估与对比
# =============================================================================
# 使用更多样本做最终估计
samples_final, _ = sampler.sample(
    forward,
    params,
    state=sampler_state,
    chain_length=500
)
x_batch_final = samples_final.reshape(-1, K, n_spin_orbitals)
final_model = nnx.merge(graph_def, params)
_, E_mat_final = compute_average_local_energy(ha, final_model, x_batch_final)
eigvals_final = jnp.linalg.eigvals(E_mat_final).real
eigvals_final_sorted = jnp.sort(eigvals_final)

print("\n" + "=" * 60)
print("NES‑VMC 计算结果 (排序后)")
print("=" * 60)
for i, e in enumerate(eigvals_final_sorted):
    exc_eV = (e - eigvals_final_sorted[0]) * 27.2114
    print(f"State {i}: E = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

print("\nFCI 基准:")
for i, e in enumerate(E_fcis[:K]):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"State {i}: E = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")