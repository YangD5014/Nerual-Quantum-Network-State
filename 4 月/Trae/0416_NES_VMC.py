import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from tqdm import tqdm

try:
    from netket.hilbert import SpinOrbitalFermions
except ImportError:
    from netket.experimental.hilbert import SpinOrbitalFermions

try:
    from netket.sampler import MetropolisFermionHop
except (ImportError, AttributeError):
    try:
        from netket.experimental.sampler import MetropolisFermionHop
    except (ImportError, AttributeError):
        from netket.sampler import MetropolisExchange as MetropolisFermionHop

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
K = 2  # NES-VMC 要计算的低激发态数量（基态+1个激发态）
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准（4个态）
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# 转为 NetKet 哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)
# 转换为 JAX 兼容的算符
ha = ha.to_jax_operator()

# 定义原始 Hilbert 空间
hi = SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
# 扩展 Hilbert 空间 - 使用 TensorHilbert
hi_K = nk.hilbert.TensorHilbert(hi**K)

# 采样器设置 - 使用 MetropolisFermionHop
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
sampler = MetropolisFermionHop(hi_K, graph=g, n_chains=16, sweep_size=32)

# ==============================================================================
# 2. 定义神经网络 Ansatz
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
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
    def __init__(self, n_spin_orbitals, n_states=K, hidden_dim=16):
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
        # x_batch: (K, n_spin_orbitals)
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        psi_total = jnp.linalg.det(M)
        return psi_total, M

# ==============================================================================
# 3. 辅助函数：计算 Hψ 和局部能量矩阵
# ==============================================================================
def Ham_psi(ha, single_ansatz, x):
    """计算单个单态 Ansatz 在组态 x 上的 Hψ 值"""
    x_conn, mels = ha.get_conn(x)
    psi_vals = jax.vmap(single_ansatz)(x_conn)
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

def compute_local_energy(ha, total_ansatz, x_batch):
    """计算局部能量矩阵及其迹"""
    psi, M = total_ansatz(x_batch)
    eps = 1e-8
    M_reg = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    Hp = Ham_Psi(ha, total_ansatz, x_batch)
    el_mat = jnp.linalg.solve(M_reg, Hp)
    return jnp.trace(el_mat).real, el_mat

compute_local_energy_batch = jax.vmap(
    compute_local_energy,
    in_axes=(None, None, 0),
    out_axes=(0, 0)
)

def compute_average_local_energy(ha, model, samples):
    """samples: (n_samples, K, n_spin_orbitals)"""
    tr_els, el_mats = compute_local_energy_batch(ha, model, samples)
    tr_avg = tr_els.mean()
    el_mat_avg = el_mats.mean(axis=0)
    return tr_avg, el_mat_avg

def loss_fn(params, ha, x_batch):
    graphdef, variables = params
    model = nnx.merge(graphdef, variables)
    tr_avg, _ = compute_average_local_energy(ha, model, x_batch)
    return tr_avg

value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

# ==============================================================================
# 4. forward 函数（供 NetKet 采样器使用）
# ==============================================================================
def forward(params, x_batch):
    """返回 log(Ψ(x_batch))，x_batch 形状 (n_chains, K * n_spin)"""
    graphdef, variables = params
    model = nnx.merge(graphdef, variables)
    n_chains = x_batch.shape[0]
    K = model.n_states
    n_spin = 4
    x_reshaped = x_batch.reshape(n_chains, K, n_spin)
    
    def single_logpsi(x):
        psi, _ = model(x)
        return jnp.log(psi + 1e-10)
    
    log_psi_batch = jax.vmap(single_logpsi)(x_reshaped)
    return log_psi_batch

# ==============================================================================
# 5. 初始化模型、采样器状态、优化器
# ==============================================================================
total_ansatz = NESTotalAnsatz(n_spin_orbitals=4, n_states=K, hidden_dim=32)
graphdef, variables = nnx.split(total_ansatz)
sampler_state = sampler.init_state(forward, (graphdef, variables), seed=1)
optimizer = optax.adam(learning_rate=5e-3)
opt_state = optimizer.init(variables)

# ==============================================================================
# 6. 训练循环
# ==============================================================================
loss_record = []
n_iter = 500
chain_length = 500

for step in tqdm(range(n_iter)):
    # 重置采样器（可选，但能保证每次从随机起点开始）
    sampler_state = sampler.reset(forward, (graphdef, variables), sampler_state)
    samples, sampler_state = sampler.sample(
        forward, (graphdef, variables), state=sampler_state, chain_length=chain_length
    )
    # samples 形状: (n_chains, chain_length, K*4)
    samples_flat = samples.reshape(-1, K, 4)  # (n_samples, K, 4)
    
    loss_val, grads = value_and_grad((graphdef, variables), ha, samples_flat)
    loss_record.append(loss_val)
    
    if step % 20 == 0:
        print(f"Step {step:3d} | Trace of local energy matrix: {loss_val:.6f} Ha")
    
    grad_graph, grad_vars = grads
    updates, opt_state = optimizer.update(grad_vars, opt_state, variables)
    variables = optax.apply_updates(variables, updates)

# 合并最终参数
total_ansatz = nnx.merge(graphdef, variables)

# ==============================================================================
# 7. 最终采样，计算激发态能量
# ==============================================================================
# 重新采样一批独立样本（不 reset 也可以）
#sampler_state = sampler.reset(forward, (graphdef, variables), sampler_state)
final_samples, _ = sampler.sample(
    forward, (graphdef, variables), state=sampler_state, chain_length=1000
)
final_samples_flat = final_samples.reshape(-1, K, 4)

# 计算平均局部能量矩阵
_, el_mat_avg = compute_average_local_energy(ha, total_ansatz, final_samples_flat)

# 对角化得到各态能量
eigen_energies = jnp.linalg.eigvalsh((el_mat_avg + el_mat_avg.conj().T) / 2).real
print("\n" + "="*60)
print("NES-VMC 计算得到的激发态能量 (Ha)")
print("="*60)
for i, e in enumerate(eigen_energies):
    exc = (e - eigen_energies[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")