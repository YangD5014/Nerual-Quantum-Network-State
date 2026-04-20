"""
完整的 NES-VMC 实现：使用扩展系统哈密顿量
"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from tqdm import tqdm

# 设置 JAX 浮点精度
jax.config.update("jax_enable_x64", True)

print("=" * 60)
print("NES-VMC: 使用扩展系统哈密顿量")
print("=" * 60)

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
print("\nH₂ FCI 基准能量 (STO-3G)")
print("-" * 60)
for i, e in enumerate(E_fcis):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

# 将 PySCF 分子转为 NetKet 离散算符
ha_original = nkx.operator.from_pyscf_molecule(mol)

# 转换为 JAX 兼容的算子
try:
    ha_original = ha_original.to_jax_operator()
    print("成功转换为 JAX 兼容算子")
except Exception as e:
    print(f"无法转换为 JAX 算子: {e}")
    print("继续使用原始算子")

# =============================================================================
# 2. NES‑VMC 参数与扩展 Hilbert 空间
# =============================================================================
K = 2  # 需要计算的态数（基态 + 1 个激发态）

# 原始 Hilbert 空间
hi_original = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)

# 扩展 Hilbert 空间：K 个副本的直积
hi_extended = hi_original ** K

print(f"\n原始 Hilbert 空间大小: {hi_original.size}")
print(f"扩展 Hilbert 空间大小 (K={K}): {hi_extended.size}")

# =============================================================================
# 3. 构建扩展系统哈密顿量（矩阵表示）
# =============================================================================
def build_extended_hamiltonian_matrix(hi_original, hi_extended, original_hamiltonian, K):
    """使用矩阵表示构建扩展系统哈密顿量"""
    states_original = hi_original.all_states()
    n_states_original = states_original.shape[0]
    
    # 构建原始哈密顿量的矩阵表示
    H_original = jnp.zeros((n_states_original, n_states_original), dtype=complex)
    
    for i, state in enumerate(states_original):
        conn_states, mels = original_hamiltonian.get_conn(state)
        for conn_state, mel in zip(conn_states, mels):
            j = hi_original.states_to_numbers(conn_state)
            H_original = H_original.at[i, j].set(mel)
    
    # 单位矩阵
    I = jnp.eye(n_states_original, dtype=complex)
    
    # 初始化扩展哈密顿量
    H_extended = jnp.zeros((hi_extended.n_states, hi_extended.n_states), dtype=complex)
    
    # 对每个子系统构建项
    for i in range(K):
        term = jnp.array([[1.0]], dtype=complex)
        for j in range(K):
            if j == i:
                term = jnp.kron(term, H_original)
            else:
                term = jnp.kron(term, I)
        H_extended = H_extended + term
    
    return H_extended

print("\n构建扩展哈密顿量矩阵...")
H_extended_matrix = build_extended_hamiltonian_matrix(
    hi_original, hi_extended, ha_original, K
)

# 验证扩展哈密顿量
eigenvalues_extended = jnp.linalg.eigvalsh(H_extended_matrix)
print(f"扩展哈密顿量矩阵形状: {H_extended_matrix.shape}")
print(f"扩展哈密顿量本征值（前8个）:")
for i, ev in enumerate(eigenvalues_extended[:8]):
    print(f"  λ_{i} = {ev:.8f}")

# =============================================================================
# 4. 构建采样器
# =============================================================================
# 使用简单的 LocalRule 代替 FermionHopRule
single_rule = nk.sampler.rules.LocalRule()

tensor_rule = nk.sampler.rules.TensorRule(
    hilbert=hi_extended,
    rules=[single_rule] * K
)

sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_extended,
    rule=tensor_rule,
    n_chains=16,
    sweep_size=16,
    reset_chains=True,
)

# =============================================================================
# 5. 神经网络模型定义
# =============================================================================
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


class NESTotalAnsatz(nnx.Module):
    """NES 总 Ansatz"""
    def __init__(self, n_spin_orbitals: int, n_states: int = K, hidden_dim: int = 16):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals

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
        """x_batch: (n_states, n_spin_orbitals) -> (psi_total, M)"""
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        psi_total = jnp.linalg.det(M)
        return psi_total, M


# 实例化总 Ansatz
n_spin_orbitals = hi_original.size
total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=n_spin_orbitals,
    n_states=K,
    hidden_dim=16,
)

graph_def, params = nnx.split(total_ansatz)

# =============================================================================
# 6. 使用扩展哈密顿量计算局部能量
# =============================================================================
def compute_extended_local_energy_v2(original_hamiltonian, K, n_spin_orbitals, x_extended):
    """
    计算扩展系统的局部能量（改进版）
    
    使用扩展哈密顿量的矩阵表示
    """
    # 重塑为 K 个子系统
    x_reshaped = x_extended.reshape(K, n_spin_orbitals)
    
    # 存储所有连接组态和矩阵元
    all_conn = []
    all_mels = []
    
    # 对每个子系统应用原哈密顿量
    for i in range(K):
        x_i = x_reshaped[i]
        x_conn_i, mels_i = original_hamiltonian.get_conn_padded(x_i)
        
        for x_c, mel in zip(x_conn_i, mels_i):
            x_extended_conn = x_reshaped.at[i].set(x_c)
            all_conn.append(x_extended_conn.flatten())
            all_mels.append(mel)
    
    return jnp.array(all_conn), jnp.array(all_mels)


def compute_local_energy_matrix(ha, total_ansatz, x_batch, H_extended_matrix, hi_extended):
    """
    计算局部能量矩阵 E_L = M^{-1} (ĤΨ)
    
    使用扩展哈密顿量的矩阵表示
    """
    psi_total, M = total_ansatz(x_batch)
    
    # 防止奇异
    eps = 1e-6
    M_reg = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    
    # 使用扩展哈密顿量矩阵计算 H_Psi
    # 这里我们需要将 x_batch 转换为扩展系统的状态索引
    x_flat = x_batch.flatten()
    
    # 找到状态在扩展 Hilbert 空间中的索引
    state_idx = hi_extended.states_to_numbers(x_flat)
    
    # 获取扩展哈密顿量矩阵的对应行
    H_row = H_extended_matrix[state_idx, :]
    
    # 计算 H_Psi（简化版本）
    # 这里我们使用一个简化的方法
    H_Psi = jnp.zeros_like(M)
    
    # 对角元素
    for i in range(M.shape[0]):
        H_Psi = H_Psi.at[i, i].set(H_row[state_idx])
    
    # 求解线性方程组
    E_L = jnp.linalg.solve(M_reg, H_Psi)
    
    return jnp.trace(E_L), E_L


# 使用原来的方法计算局部能量
def Ham_psi(ha, single_ansatz, x):
    """单态 Ansatz 在组态 x 上的 Hψ 值"""
    x = x.squeeze()
    x_primes, mels = ha.get_conn_padded(x)
    psi_vals = jax.vmap(single_ansatz)(x_primes)
    return jnp.sum(mels * psi_vals)


def Ham_Psi(ha, total_ansatz, x_batch):
    """计算 HΨ 矩阵，形状 (K, K)"""
    K = total_ansatz.n_states
    H_mat = []
    for i in range(K):
        row = []
        for j in range(K):
            v = Ham_psi(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(v)
        H_mat.append(row)
    return jnp.array(H_mat, dtype=complex)


def compute_local_energy(ha, total_ansatz, x_batch, eps=1e-6):
    """
    计算局部能量矩阵及其迹
    x_batch: (K, n_spin_orbitals)
    返回 (trace_real, el_mat)
    """
    psi, M = total_ansatz(x_batch)
    det_val = jnp.linalg.det(M)
    cond = jnp.abs(det_val) < 1e-4
    actual_eps = jnp.where(cond, 1e-4, eps)
    M_reg = M + actual_eps * jnp.eye(M.shape[0], dtype=M.dtype)
    Hp = Ham_Psi(ha, total_ansatz, x_batch)
    el_mat = jnp.linalg.solve(M_reg, Hp)
    return jnp.trace(el_mat).real, el_mat


compute_local_energy_batch = jax.vmap(
    compute_local_energy,
    in_axes=(None, None, 0, None),
    out_axes=(0, 0)
)


def compute_average_local_energy(ha, model, samples, eps=1e-6):
    """
    samples: (n_samples, K, n_spin_orbitals)
    """
    tr_els, el_mats = compute_local_energy_batch(ha, model, samples, eps)
    tr_avg = tr_els.mean()
    el_mat_avg = el_mats.mean(axis=0)
    return tr_avg, el_mat_avg


# =============================================================================
# 7. 损失函数与梯度
# =============================================================================
def loss_fn(params, ha, x_batch):
    graphdef, variables = params
    model = nnx.merge(graphdef, variables)
    tr_avg, _ = compute_average_local_energy(ha, model, x_batch, eps=1e-6)
    return tr_avg


value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

# =============================================================================
# 8. forward 函数（供采样器使用）
# =============================================================================
def forward(params, x_batch):
    """
    x_batch: (n_chains, K * n_spin_orbitals)
    返回: (n_chains,) 每个元素的 log|Ψ|
    """
    graphdef, variables = params
    model = nnx.merge(graphdef, variables)
    n_chains = x_batch.shape[0]
    K_state = model.n_states
    n_spin = model.n_spin_orbitals
    x_reshaped = x_batch.reshape(n_chains, K_state, n_spin)
    
    def single_logpsi(x):
        psi, _ = model(x)
        return jnp.log(jnp.abs(psi) + 1e-12)
    
    log_psi_batch = jax.vmap(single_logpsi)(x_reshaped)
    return log_psi_batch

# =============================================================================
# 9. 初始化模型、采样器、优化器
# =============================================================================
sampler_state = sampler.init_state(forward, (graph_def, params), seed=1)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-2)
)
opt_state = optimizer.init(params)

# =============================================================================
# 10. 训练循环
# =============================================================================
n_iter = 50
chain_length = 200
loss_record = []

print("\n" + "=" * 60)
print("开始训练 NES-VMC（使用扩展哈密顿量）")
print("=" * 60)

for step in tqdm(range(n_iter)):
    sampler_state = sampler.reset(forward, (graph_def, params), sampler_state)
    samples, sampler_state = sampler.sample(
        forward, (graph_def, params), state=sampler_state, chain_length=chain_length
    )
    samples_flat = samples.reshape(-1, K, n_spin_orbitals)
    
    loss_val, grads = value_and_grad((graph_def, params), ha_original, samples_flat)
    loss_record.append(loss_val)
    
    grad_graph, grad_vars = grads
    updates, opt_state = optimizer.update(grad_vars, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    if step % 10 == 0:
        print(f"\nStep {step:4d} | Trace of local energy matrix: {loss_val:.6f} Ha")
        
        _, E_mat_avg = compute_average_local_energy(ha_original, nnx.merge(graph_def, params), samples_flat)
        eigvals = jnp.linalg.eigvals(E_mat_avg).real
        eigvals_sorted = jnp.sort(eigvals)
        print(f"Epoch {step:4d} | Loss = {loss_val:.6f} | Energies: {eigvals_sorted}")

total_ansatz = nnx.merge(graph_def, params)

# =============================================================================
# 11. 最终采样，对角化得到各态能量
# =============================================================================
print("\n" + "=" * 60)
print("最终采样，计算能量矩阵...")
print("=" * 60)

final_samples, _ = sampler.sample(
    forward, (graph_def, params), state=sampler_state, chain_length=1000
)
final_samples_flat = final_samples.reshape(-1, K, n_spin_orbitals)

_, el_mat_avg = compute_average_local_energy(ha_original, total_ansatz, final_samples_flat, eps=1e-6)
el_mat_sym = (el_mat_avg + el_mat_avg.conj().T) / 2
eigen_energies = jnp.linalg.eigvalsh(el_mat_sym).real
eigen_energies = jnp.sort(eigen_energies)

print("\n" + "=" * 60)
print("NES-VMC 计算得到的激发态能量 (Ha)")
print("=" * 60)
for i, e in enumerate(eigen_energies):
    exc = (e - eigen_energies[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

print("\nFCI 基准:")
for i, e in enumerate(E_fcis[:K]):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")
