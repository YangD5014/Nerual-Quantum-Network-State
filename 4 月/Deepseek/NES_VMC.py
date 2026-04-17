import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax

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
# 1. 定义原始 Hilbert 空间 (同之前)
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
# 2. 定义 NES-VMC 参数
K = 2  # 想要同时采样的组态数
# 3. 修正点：使用指数运算符 ** 来构建扩展 Hilbert 空间
hi_ensemble = hi ** K
# 4. 后续的采样器设置保持不变
edges = [(0, 1), (2, 3)] #edges 是对 hi 的 edges
g = nk.graph.Graph(edges=edges)
# hi_ensemble.all_states()[0].shape # (12,)

hi_K = nk.hilbert.TensorHilbert(hi**K)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
rules_tuple = tuple(single_rule for _ in range(K))
tensor_rule = nk.sampler.rules.TensorRule(hilbert=hi_K, rules=[single_rule]*K)
sampler = nk.sampler.MetropolisSampler(hi_ensemble, rule=tensor_rule, n_chains=16, sweep_size=32)
#==============================================================================    
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
        #log_out = jnp.log(out)
        return jnp.squeeze(out)
    
class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals, n_states=K, hidden_dim=16):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals
        
        # 1. 生成一串不同的 JAX 随机 key
        key = jax.random.key(42)          # 主种子
        keys = jax.random.split(key, n_states)  # 拆成 n_states 个不同 key

        # 2. 给每个 ansatz 分配不同的 key → 不同的 nnx.Rngs
        # ✅ 这是 NNX 官方支持的写法
        self.single_ansatz_list = [
            SingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim,
                rngs=nnx.Rngs(keys[i])  # 每个用不同 key
            )
            for i in range(n_states)
        ]

    def __call__(self, x_batch):
        '''
        x_batch: (K, n_spin_orbitals)
        '''
        K_state = self.n_states
        M = []
        for i in range(K_state):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K_state)]
            M.append(jnp.array(row))
        M = jnp.stack(M)
        psi_total = jnp.linalg.det(M)
        return psi_total, M
    
K=2
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=4*3)

def Ham_psi(ha: nk.operator.DiscreteOperator, single_ansatz:SingleStateAnsatz, x: jnp.array):
    # 🔥 关键：用 flattened 版本支持批量/自动处理
    x_primes, mels = ha.get_conn_padded(x)
    psi_vals = jax.vmap(single_ansatz)(x_primes)
    return jnp.sum(mels * psi_vals)


def Ham_Psi(ha:nk.operator.DiscreteOperator, total_ansatz: NESTotalAnsatz, x_batch):
    K_state = total_ansatz.n_states
    H_mat = []
    for i in range(K_state):
        row = []
        for j in range(K_state):
            v = Ham_psi(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(v)
        H_mat.append(row)
    return jnp.array(H_mat, dtype=complex)

def compute_local_energy(ha:nk.operator.DiscreteOperator, total_ansatz: NESTotalAnsatz, x_batch):
    psi, M = total_ansatz(x_batch)
    # 🔥 修复 1：对角加载，防止矩阵奇异
    eps = 1e-3
    M = M + eps * jnp.eye(M.shape[0], dtype=M.dtype)
    Hp = Ham_Psi(ha, total_ansatz, x_batch)
    el_mat = jnp.linalg.solve(M, Hp)
    return jnp.trace(el_mat), el_mat

def compute_local_energy_single(ham: nk.operator.DiscreteOperator, 
                               model: NESTotalAnsatz, 
                               x_batch):
    # x_batch: (K, n_spin_orbitals)
    tr_el, el_mat = compute_local_energy(ham, model, x_batch)
    return tr_el.real, el_mat  # 确保返回实数迹

compute_local_energy_batch = jax.vmap(
    compute_local_energy_single,
    # 对应 4 个入参：ham, model, x_batch, return_log_psi
    in_axes=(None, None, 0),
    out_axes=(0, 0),
    axis_name='samples_batch'
)

def compute_average_local_energy(ham: nk.operator.DiscreteOperator, 
                                 model: NESTotalAnsatz, 
                                 samples, 
                                 ):
    
    '''
    samples.shape = (n_samples, K, n_spin_orbitals)
    '''
    tr_els, el_mats = compute_local_energy_batch(
        ham, model, samples
    )
    
    tr_avg = tr_els.mean()
    el_mat_avg = el_mats.mean(axis=0)
    
    return tr_avg, el_mat_avg

def loss_fn_single(params, graph, ham, x_batch):
    # 把参数放回模型
    model = nnx.merge(graph, params)
    # 计算 loss = trace(local energy matrix)
    tr_el, _ = compute_local_energy_single(ham, model, x_batch)
    return tr_el  # 我们对这个标量求导！


def loss_fn_batch(params, graph, ham, x_batch):
    # 把参数放回模型
    model = nnx.merge(graph, params)
    # 计算 loss = trace(local energy matrix)
    tr_el, _ = compute_average_local_energy(ham, model, x_batch)
    return tr_el  # 我们对这个标量求导！

def forward(params, x_batch):
    #print(x_batch.shape)
    # x_batch: (n_chains, chain_length, n_spin_orbitals*K)
    n_chains = x_batch.shape[0]
    K = 2
    n_spin = 4
    # 重塑为 (n_chains, K, n_spin)
    x_reshaped = x_batch.reshape(n_chains, K, n_spin)
    
    # 定义单个联合样本的计算
    def single_logpsi(params, x):
        (psi, _), _ = nnx.call(params)(x)
        return jnp.log(psi)  # 标量复数
    
    # 批量映射
    log_psi_batch = jax.vmap(single_logpsi, in_axes=(None, 0))(params, x_reshaped)
    return log_psi_batch  # 形状 (n_chains,)


value_and_grad_single = jax.value_and_grad(loss_fn_single)
value_and_grad_batch = jax.value_and_grad(loss_fn_batch)


K=2
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=4*3)
parameters = nnx.split(total_ansatz)
# 在Neket里使用Sampler的标准流程: init_state -> sample 如果模型参数变化 使用 reset
sampler_state = sampler.init_state(forward, parameters, seed=1)
sampler_state = sampler.reset(forward, parameters, sampler_state)
samples, sampler_state = sampler.sample(
    forward, parameters, state=sampler_state, chain_length=500
)


# value_and_grad_batch(paramters,ha,samples.reshape(-1,K,4)) #成功 
# compute_local_energy_single(ha, total_ansatz, samples.reshape(-1,K,4)[0]) #成功
# loss_fn_single(paramters, ha, samples.reshape(-1,K,4)[0]) #成功
# loss_fn_batch(paramters, ha, samples.reshape(-1,K,4)) #成功
# value_and_grad_batch(paramters,ha,samples.reshape(-1,K,4)) #成功


# parameters = nnx.split(total_ansatz)
# x_batch = jnp.array([[1,0,1,0],[0,1,0,1]])
# (log_psi_total_ansatz,M),state = nnx.call(parameters)(x_batch) #成功运行

    