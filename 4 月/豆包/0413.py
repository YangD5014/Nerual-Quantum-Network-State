import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx

#==============================================================================
# 1. 全局参数 & H₂ 分子定义
#==============================================================================
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

#==============================================================================
# 2. Hilbert 空间定义
#==============================================================================
n_orbitals = 2
n_spin_orbitals = 4
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,
    s=1/2,
    n_fermions_per_spin=(1, 1)
)

#==============================================================================
# 3. NES-VMC 神经网络模型（复数 FFNN）
#==============================================================================
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
    def __init__(self, n_spin_orbitals, n_states=K, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(42+i))
            for i in range(n_states)
        ]

    def __call__(self, x_batch, return_log_psi=False):
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        psi_total = jnp.linalg.det(M)
        log_psi_total = jnp.log(psi_total)

        if return_log_psi:
            return log_psi_total
        else:
            return psi_total, M

# 初始化模型
rngs = nnx.Rngs(42)
total_ansatz = NESTotalAnsatz(n_spin_orbitals, n_states=K, hidden_dim=16, rngs=rngs)

#==============================================================================
# 4. 核心：哈密顿作用 + 局部能量矩阵（论文标准）
#==============================================================================
def Ham_psi(ha: nk.operator.DiscreteOperator, single_ansatz, x: jnp.array):
    x_primes, mels = ha.get_conn(x)
    psi_vals = jax.vmap(single_ansatz)(x_primes)
    return jnp.sum(mels * psi_vals)

def Ham_Psi(ha, total_ansatz: NESTotalAnsatz, x_batch):
    K_state = total_ansatz.n_states
    H_mat = []
    for i in range(K_state):
        row = []
        for j in range(K_state):
            v = Ham_psi(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(v)
        H_mat.append(row)
    return jnp.array(H_mat, dtype=complex)

def compute_local_energy(ha, total_ansatz: NESTotalAnsatz, x_batch):
    psi, M = total_ansatz(x_batch, return_log_psi=False)
    Hp = Ham_Psi(ha, total_ansatz, x_batch)
    el_mat = jnp.linalg.solve(M, Hp)
    return jnp.trace(el_mat), el_mat


def compute_local_energy(ha, total_ansatz: NESTotalAnsatz, x_batch):
    psi, M = total_ansatz(x_batch, return_log_psi=False)
    
    # 🔥 修复 1：对角加载，防止矩阵奇异
    eps = 1e-6
    M = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    
    Hp = Ham_Psi(ha, total_ansatz, x_batch)
    el_mat = jnp.linalg.solve(M, Hp)
    return jnp.trace(el_mat), el_mat

#==============================================================================
# 5. 损失函数 & 训练步骤（JAX 自动求导）
#==============================================================================
@jax.jit
def loss_fn(model_state, samples):
    nnx.update(total_ansatz, model_state)
    total_energy = 0.0 + 0j
    n_samples = samples.shape[0]
    for xb in samples:
        tr_EL, _ = compute_local_energy(ha, total_ansatz, xb)
        total_energy += tr_EL
    avg_energy = total_energy.real / n_samples
    return avg_energy

@jax.jit
def train_step(model_state, samples):
    loss, grads = jax.value_and_grad(loss_fn)(model_state, samples)
    model_state = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, model_state, grads)
    return model_state, loss

# 采样器 n_chains=K
g = nk.graph.Graph(edges=[(0,1), (2,3)])
sampler = nk.sampler.MetropolisFermionHop(
    hi, graph=g, n_chains=K, spin_symmetric=True, sweep_size=64
)

# 完全符合论文：采样 |TotalAnsatz|²
def forward(state, x_batch):
    # x_batch.shape = (K, n_spin)
    # 返回标量 log|Ψ|，对应联合分布
    log_psi_total, new_state = nnx.call(state)(x_batch, return_log_psi=True)
    # 关键：包装成 (1,) 或 (K,) 来骗过 NetKet 的形状检查
    return jnp.expand_dims(log_psi_total, 0).repeat(x_batch.shape[0])

parameters = nnx.split(total_ansatz)

sampler_state = sampler.init_state(forward, parameters, seed=1)
sampler_state = sampler.reset(forward, parameters, sampler_state)

#==============================================================================
# 6. 训练循环 + 采样 + 能量对角化（最后一部分！）
#==============================================================================
print("\n" + "="*60)
print(f"开始 NES-VMC 训练（K={K} 个态）| 采样 = Total Ansatz Ψ²")
print("="*60)

# 训练参数
n_steps = 800
lr = 0.005
model_state = nnx.state(total_ansatz)

# 主训练循环
for step in range(n_steps):
    # 1. 采样：从 Total Ansatz 联合分布采样
    samples, sampler_state = sampler.sample(
        forward, parameters, state=sampler_state, chain_length=80
    )
    
    # 2. 训练一步
    model_state, loss = train_step(model_state, samples)
    
    # 3. 日志输出
    if step % 50 == 0:
        print(f"Step {step:4d} | Loss (Tr(EL)) = {loss:.8f} Ha")

#==============================================================================
# 7. 最终计算：平均局部能量矩阵 + 对角化得到激发态
#==============================================================================
print("\n" + "="*60)
print("最终结果：对角化局部能量矩阵 → 激发态能量")
print("="*60)

# 加载最新模型参数
nnx.update(total_ansatz, model_state)

# 批量采样用于平均能量矩阵
samples_final, _ = sampler.sample(
    forward, parameters, state=sampler_state, chain_length=200
)

# 计算平均局部能量矩阵
el_mat_sum = jnp.zeros((K, K), dtype=complex)
n_sample_final = samples_final.shape[0]

for xb in samples_final:
    _, el_mat = compute_local_energy(ha, total_ansatz, xb)
    el_mat_sum += el_mat

el_mat_avg = el_mat_sum / n_sample_final

# 对角化 → 得到 K 个本征态能量（已排序）
eigen_energies = jnp.linalg.eigvalsh(el_mat_avg).real

# 输出结果
for i, e in enumerate(eigen_energies):
    exc_ev = (e - eigen_energies[0]) * 27.2114  # 转换为 eV
    print(f"NES-VMC  E{i} = {e:.8f} Ha  |  激发能: {exc_ev:.4f} eV")

# FCI 对比
print("\n" + "-"*60)
print("FCI 精确结果对比：")
for i in range(K):
    exc_ev = (E_fcis[i] - E_fcis[0]) * 27.2114
    print(f"FCI      E{i} = {E_fcis[i]:.8f} Ha  |  激发能: {exc_ev:.4f} eV")
