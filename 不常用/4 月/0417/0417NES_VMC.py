import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from tqdm import tqdm

# ==============================================================================
# 这是 Natural-Excited State VMC 的二次量子化下的算法复现
# 
# 
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
K = 2                      # 同时计算的态数（基态 + 第一激发态）
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# NetKet 哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)

# 原始 Hilbert 空间（2个轨道，每个自旋1个电子）
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
hi_ensemble = hi ** K          # 扩展空间

# 采样器：使用 TensorRule 组合 K 个单链规则
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hilbert=hi_ensemble, rules=[single_rule]*K)
sampler = nk.sampler.MetropolisSampler(hi_ensemble, rule=tensor_rule, n_chains=16, sweep_size=32)

# ==============================================================================
# 2. 神经网络 Ansatz
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
        """x_batch: (K, n_spin_orbitals) -> (psi_total, M)"""
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        psi_total = jnp.linalg.det(M)
        return psi_total, M

# ==============================================================================
# 3. 辅助函数：Hψ 与局部能量矩阵
# ==============================================================================
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
    # 动态正则化：若行列式接近零则增大 eps
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

def loss_fn(params, ha, x_batch):
    graphdef, variables = params
    model = nnx.merge(graphdef, variables)
    tr_avg, _ = compute_average_local_energy(ha, model, x_batch, eps=1e-6)
    return tr_avg

value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

# ==============================================================================
# 4. forward 函数（供采样器使用，返回 log|Ψ|）
# ==============================================================================
def forward(params, x_batch):
    """
    x_batch: (n_chains, K * n_spin_orbitals)
    返回: (n_chains,) 每个元素的 log|Ψ|
    """
    graphdef, variables = params
    model = nnx.merge(graphdef, variables)
    n_chains = x_batch.shape[0]
    K_state = model.n_states
    n_spin = model.n_spin_orbitals   # 4
    x_reshaped = x_batch.reshape(n_chains, K_state, n_spin)
    
    def single_logpsi(x):
        psi, _ = model(x)
        return jnp.log(jnp.abs(psi))
    
    log_psi_batch = jax.vmap(single_logpsi)(x_reshaped)
    return log_psi_batch

# ==============================================================================
# 5. 初始化模型、采样器、优化器（带梯度裁剪）
# ==============================================================================
total_ansatz = NESTotalAnsatz(n_spin_orbitals=4, n_states=K, hidden_dim=16)
graphdef, variables = nnx.split(total_ansatz)

# 采样器状态（只初始化一次，训练中不 reset）
sampler_state = sampler.init_state(forward, (graphdef, variables), seed=1)

# 优化器：Adam + 全局梯度裁剪
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-1)
)
opt_state = optimizer.init(variables)

# ==============================================================================
# 6. 训练循环
# ==============================================================================
n_iter = 100
chain_length = 200          # 每条链采样步数
loss_record = []

print("\n开始训练 NES-VMC...")
for step in tqdm(range(n_iter)):
    # 重要：不要 reset 采样器！只使用上一次的 state 继续采样
    sampler_state = sampler.reset(forward, (graphdef, variables), sampler_state)
    samples, sampler_state = sampler.sample(
        forward, (graphdef, variables), state=sampler_state, chain_length=chain_length
    )
    # samples 形状: (n_chains, chain_length, K*4)
    samples_flat = samples.reshape(-1, K, 4)   # (n_samples, K, 4)
    
    loss_val, grads = value_and_grad((graphdef, variables), ha, samples_flat)
    loss_record.append(loss_val)
    
    # 更新参数
    grad_graph, grad_vars = grads
    updates, opt_state = optimizer.update(grad_vars, opt_state, variables)
    variables = optax.apply_updates(variables, updates)
    
    if step % 5 == 0:
        print(f"Step {step:4d} | Trace of local energy matrix: {loss_val:.6f} Ha")
        
        _, E_mat_avg = compute_average_local_energy(ha, nnx.merge(graphdef, variables), samples_flat)
        eigvals = jnp.linalg.eigvals(E_mat_avg).real
        eigvals_sorted = jnp.sort(eigvals)
        print(f"Epoch {step:4d} | Loss = {loss_val:.6f} | Energies: {eigvals_sorted}")

# 合并最终模型
total_ansatz = nnx.merge(graphdef, variables)

# ==============================================================================
# 7. 最终采样，对角化得到各态能量
# ==============================================================================
print("\n最终采样，计算能量矩阵...")
final_samples, _ = sampler.sample(
    forward, (graphdef, variables), state=sampler_state, chain_length=2000
)
final_samples_flat = final_samples.reshape(-1, K, 4)

_, el_mat_avg = compute_average_local_energy(ha, total_ansatz, final_samples_flat, eps=1e-6)
# 对称化保证 Hermitian
el_mat_sym = (el_mat_avg + el_mat_avg.conj().T) / 2
eigen_energies = jnp.linalg.eigvalsh(el_mat_sym).real
# 按能量升序排列
eigen_energies = jnp.sort(eigen_energies)

print("\n" + "="*60)
print("NES-VMC 计算得到的激发态能量 (Ha)")
print("="*60)
for i, e in enumerate(eigen_energies):
    exc = (e - eigen_energies[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")