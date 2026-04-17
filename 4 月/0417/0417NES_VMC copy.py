import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from tqdm import tqdm
from netket.optimizer import SR  # 导入SR优化器

# ==============================================================================
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

def loss_fn(variables, graphdef, ha, x_batch):
    """
    重构loss_fn：仅对variables求导（graphdef是结构，无梯度）
    """
    model = nnx.merge(graphdef, variables)
    tr_avg, _ = compute_average_local_energy(ha, model, x_batch, eps=1e-6)
    return tr_avg

# 仅对variables求导的梯度函数
value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

# ==============================================================================
# 4. forward 函数（供采样器/SR使用，返回 log|Ψ|）
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

# 定义log_psi的梯度函数（SR需要∇logψ）
def log_psi_gradient(variables, graphdef, samples):
    """
    计算每个样本的log|Ψ|对variables的梯度
    samples: (n_samples, K*4)
    """
    def _log_psi(var):
        return forward((graphdef, var), samples)
    # vmap计算每个样本的梯度（输出：(n_samples,) + variables结构）
    grad_log_psi = jax.vmap(jax.grad(_log_psi))(variables)
    return grad_log_psi

# ==============================================================================
# 5. 初始化模型、采样器、SR+SGD优化器
# ==============================================================================
total_ansatz = NESTotalAnsatz(n_spin_orbitals=4, n_states=K, hidden_dim=16)
graphdef, variables = nnx.split(total_ansatz)

# 采样器状态
sampler_state = sampler.init_state(forward, (graphdef, variables), seed=1)

# SR + SGD 优化器配置
sgd = optax.sgd(learning_rate=0.01)  # SGD基础优化器（学习率可调）
sr = SR(
    sgd,
    diag_shift=1e-4,    # Fisher矩阵对角移位（防止奇异，核心参数）
    centered=True,      # 中心化Fisher矩阵，提升稳定性
    holomorphic=False   # 非全纯函数（我们用了abs，所以设为False）
)
sr_state = sr.init(variables)  # 初始化SR状态

# ==============================================================================
# 6. 训练循环（SR+SGD）
# ==============================================================================
n_iter = 100
chain_length = 200          # 每条链采样步数
loss_record = []

print("\n开始训练 NES-VMC (SR+SGD)...")
for step in tqdm(range(n_iter)):
    # 采样（不reset，保持马尔可夫链连续性）
    sampler_state = sampler.reset(forward, (graphdef, variables), sampler_state)
    samples, sampler_state = sampler.sample(
        forward, (graphdef, variables), state=sampler_state, chain_length=chain_length
    )
    # samples 形状: (n_chains, chain_length, K*4)
    samples_flat = samples.reshape(-1, K, 4)   # (n_samples, K, 4)
    samples_1d = samples.reshape(-1, K*4)      # SR需要原始1D样本（n_samples, K*4）
    
    # 1. 计算能量梯度（dE/dθ）
    loss_val, energy_grad = value_and_grad(variables, graphdef, ha, samples_flat)
    loss_record.append(loss_val)
    
    # 2. 计算logψ的梯度（d(log|Ψ|)/dθ），用于SR的Fisher矩阵
    log_psi_grad = log_psi_gradient(variables, graphdef, samples_1d)
    
    # 3. SR修正梯度 + SGD更新
    # SR需要：能量梯度、logψ梯度、样本权重（默认全1）
    updates, sr_state = sr.update(
        energy_grad,       # 原始能量梯度
        sr_state,          # SR状态
        params=variables,  # 当前参数
        grad_logpsi=log_psi_grad  # logψ的样本梯度
    )
    # 应用SR修正后的更新
    variables = optax.apply_updates(variables, updates)
    
    # 打印训练信息
    if step % 5 == 0:
        # 计算平均局部能量矩阵
        current_model = nnx.merge(graphdef, variables)
        _, E_mat_avg = compute_average_local_energy(ha, current_model, samples_flat)
        # 对称化保证Hermitian，避免虚数特征值
        E_mat_sym = (E_mat_avg + E_mat_avg.conj().T) / 2
        eigvals = jnp.linalg.eigvalsh(E_mat_sym).real
        eigvals_sorted = jnp.sort(eigvals)
        
        print(f"\nStep {step:4d} | Loss (Trace): {loss_val:.6f} Ha")
        print(f"Step {step:4d} | Energies (sorted): {eigvals_sorted}")

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
final_samples_1d = final_samples.reshape(-1, K*4)

_, el_mat_avg = compute_average_local_energy(ha, total_ansatz, final_samples_flat, eps=1e-6)
# 对称化保证Hermitian
el_mat_sym = (el_mat_avg + el_mat_avg.conj().T) / 2
eigen_energies = jnp.linalg.eigvalsh(el_mat_sym).real
# 按能量升序排列
eigen_energies = jnp.sort(eigen_energies)

print("\n" + "="*60)
print("NES-VMC (SR+SGD) 计算得到的激发态能量 (Ha)")
print("="*60)
for i, e in enumerate(eigen_energies):
    exc = (e - eigen_energies[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# 对比FCI基准
print("\n" + "="*60)
print("FCI 基准能量对比 (Ha)")
print("="*60)
for i in range(min(K, len(E_fcis))):
    nes_e = eigen_energies[i]
    fci_e = E_fcis[i]
    err = abs(nes_e - fci_e)
    print(f"E{i} | NES-VMC: {nes_e:.8f} | FCI: {fci_e:.8f} | 误差: {err:.8f} Ha")