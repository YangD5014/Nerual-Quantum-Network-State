"""
NES-VMC: 完整实现（自定义训练循环）
使用扩展哈密顿量 + TotalAnsatz
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
print("NES-VMC: 完整实现")
print("=" * 60)

# =============================================================================
# 1. 分子定义
# =============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print("\nH₂ FCI 基准能量 (STO-3G)")
print("-" * 60)
for i, e in enumerate(E_fcis):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

# =============================================================================
# 2. 定义哈密顿量和 Hilbert 空间
# =============================================================================
ha_original = nkx.operator.from_pyscf_molecule(mol)

# 转换为 JAX 兼容算子
try:
    ha_original_jax = ha_original.to_jax_operator()
    print("\n✓ 成功转换为 JAX 兼容算子")
except:
    ha_original_jax = ha_original

hi_original = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)

K = 2
n_spin_orbitals = hi_original.size
hi_extended = hi_original ** K

print(f"\n原始 Hilbert 空间大小: {hi_original.size}")
print(f"扩展 Hilbert 空间大小: {hi_extended.size}")

# =============================================================================
# 3. 定义 TotalAnsatz（NNX 版本）
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


class TotalAnsatz(nnx.Module):
    """NES 总 Ansatz"""
    def __init__(self, n_spin_orbitals: int, n_states: int = K, hidden_dim: int = 16):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals

        key = jax.random.key(42)
        keys = jax.random.split(key, n_states)

        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(keys[i]))
            for i in range(n_states)
        ]

    def __call__(self, x_batch):
        """
        x_batch: (n_states, n_spin_orbitals) -> (psi_total, M)
        """
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        psi_total = jnp.linalg.det(M)
        return psi_total, M


# 实例化 TotalAnsatz
total_ansatz = TotalAnsatz(
    n_spin_orbitals=n_spin_orbitals,
    n_states=K,
    hidden_dim=16,
)

graph_def, params = nnx.split(total_ansatz)
print(f"\n✓ 创建 TotalAnsatz")
print(f"参数数量: {sum(p.size for p in jax.tree.leaves(params))}")

# =============================================================================
# 4. 定义局部能量计算函数
# =============================================================================
def compute_H_psi_for_single_state(ha, single_ansatz, x):
    """计算单个态的 H|ψ>"""
    x_conn, mels = ha.get_conn_padded(x)
    psi_conn = jax.vmap(single_ansatz)(x_conn)
    return jnp.sum(mels * psi_conn)


def compute_H_Psi_matrix(ha, total_ansatz, x_batch):
    """计算 HΨ 矩阵"""
    K = total_ansatz.n_states
    H_mat = []
    
    for i in range(K):
        row = []
        for j in range(K):
            v = compute_H_psi_for_single_state(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(v)
        H_mat.append(row)
    
    return jnp.array(H_mat, dtype=complex)


def compute_local_energy_matrix(ha, total_ansatz, x_batch, eps=1e-6):
    """计算局部能量矩阵"""
    psi, M = total_ansatz(x_batch)
    M_reg = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    H_Psi = compute_H_Psi_matrix(ha, total_ansatz, x_batch)
    E_matrix = jnp.linalg.solve(M_reg, H_Psi)
    E_loc = jnp.trace(E_matrix)
    return E_loc, E_matrix

# =============================================================================
# 5. 定义损失函数和梯度
# =============================================================================
def loss_fn(params, ha, x_batch):
    graph_def, variables = params
    model = nnx.merge(graph_def, variables)
    
    def single_loss(x):
        E_loc, _ = compute_local_energy_matrix(ha, model, x)
        return E_loc.real
    
    E_locs = jax.vmap(single_loss)(x_batch)
    return E_locs.mean()


value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

# =============================================================================
# 6. 定义采样器
# =============================================================================
def forward(params, x_batch):
    graph_def, variables = params
    model = nnx.merge(graph_def, variables)
    
    n_chains = x_batch.shape[0]
    x_reshaped = x_batch.reshape(n_chains, K, n_spin_orbitals)
    
    def single_logpsi(x):
        psi, _ = model(x)
        return jnp.log(jnp.abs(psi) + 1e-12)
    
    log_psi_batch = jax.vmap(single_logpsi)(x_reshaped)
    return log_psi_batch


sampler = nk.sampler.MetropolisLocal(hi_extended, n_chains=16)
sampler_state = sampler.init_state(forward, (graph_def, params), seed=1)

print(f"\n✓ 创建采样器")

# =============================================================================
# 7. 训练循环（参考 NetKet 标准流程）
# =============================================================================
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.01)
)
opt_state = optimizer.init(params)

n_iter = 300
chain_length = 200
loss_record = []

print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

for step in tqdm(range(n_iter)):
    # 重置采样器并采样
    sampler_state = sampler.reset(forward, (graph_def, params), sampler_state)
    samples, sampler_state = sampler.sample(
        forward, (graph_def, params), state=sampler_state, chain_length=chain_length
    )
    
    # 重塑样本
    samples_flat = samples.reshape(-1, K, n_spin_orbitals)
    
    # 计算损失和梯度
    loss_val, grads = value_and_grad(
        (graph_def, params), ha_original_jax, samples_flat
    )
    loss_record.append(loss_val)
    
    # 更新参数
    grad_graph, grad_vars = grads
    updates, opt_state = optimizer.update(grad_vars, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # 打印日志
    if step % 50 == 0:
        print(f"\nStep {step:4d} | Loss = {loss_val:.6f} Ha")

# 合并最终模型
total_ansatz = nnx.merge(graph_def, params)
print("\n✓ 训练完成")

# =============================================================================
# 8. 计算局域能量矩阵
# =============================================================================
print("\n" + "=" * 60)
print("计算局域能量矩阵...")
print("=" * 60)

final_samples, _ = sampler.sample(
    forward, (graph_def, params), state=sampler_state, chain_length=1000
)
final_samples_flat = final_samples.reshape(-1, K, n_spin_orbitals)

print(f"样本数: {final_samples_flat.shape[0]}")


def compute_average_energy_matrix(total_ansatz, samples, ha):
    def single_E_matrix(x):
        _, E_matrix = compute_local_energy_matrix(ha, total_ansatz, x)
        return E_matrix
    
    E_matrices = jax.vmap(single_E_matrix)(samples)
    E_matrix_avg = E_matrices.mean(axis=0)
    E_avg = jnp.trace(E_matrix_avg)
    
    return E_avg, E_matrix_avg


E_avg, E_matrix_avg = compute_average_energy_matrix(
    total_ansatz, final_samples_flat, ha_original_jax
)

print(f"\n平均局部能量: {E_avg:.6f} Ha")
print(f"\n平均局域能量矩阵:\n{E_matrix_avg}")

# =============================================================================
# 9. 对角化得到激发态能量
# =============================================================================
E_matrix_sym = (E_matrix_avg + E_matrix_avg.conj().T) / 2
eigenvalues_vmc = jnp.linalg.eigvalsh(E_matrix_sym)
eigenvalues_vmc = jnp.sort(eigenvalues_vmc)

print("\n" + "=" * 60)
print("NES-VMC 计算得到的激发态能量")
print("=" * 60)
for i, e in enumerate(eigenvalues_vmc):
    exc = (e - eigenvalues_vmc[0]) * 27.2114
    error = abs(e - E_fcis[i])
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV  |  误差: {error:.6f} Ha")

print("\nFCI 基准:")
for i, e in enumerate(E_fcis[:K]):
    exc_eV = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc_eV:.4f} eV")

# =============================================================================
# 10. 总结
# =============================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("\n✅ 成功实现 NES-VMC 算法：")
print("1. 使用扩展哈密顿量（原始哈密顿量的直和）")
print("2. 使用 TotalAnsatz（行列式形式的波函数）")
print("3. 使用 NetKet 采样器进行采样")
print("4. 训练了 300 次迭代")
print("5. 计算了局域能量矩阵并提取激发态能量")

print("\n📊 结果：")
for i, e in enumerate(eigenvalues_vmc):
    error = abs(e - E_fcis[i])
    print(f"E{i} = {e:.8f} Ha  |  误差: {error:.6f} Ha")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
