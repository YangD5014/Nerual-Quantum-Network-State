# ===================== 环境配置 =====================
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from functools import partial
from jax import flatten_util
import matplotlib.pyplot as plt
from tqdm import tqdm
from jax import vmap,jit,grad
from NES_VMC import create_machine,hi_ext,hi,sampler,SingleStateAnsatz,\
    NESTotalAnsatz,ha,compute_loss_and_grad,E_fcis, \
    make_get_all_next_states,make_metropolis_hastings_step,mcmc_sampler_multichain,init_sampler_state

tensor_edges = [(0,1),(2,3),(4,5),(6,7)]
total_ansatz = NESTotalAnsatz(n_spin_orbitals=4,n_states=2,hidden_dim=12,rngs=nnx.Rngs(13))
machine, graphdef, state = create_machine(total_ansatz)


def Ham_psi(ha: nk.operator.DiscreteOperator, model:SingleStateAnsatz, x):
    """计算 Hψ(x)，model 输出 log_psi 时完全正确"""
    x_primes, mels = ha.get_conn_padded(x)
    # 1. 计算所有 σ' 的 log_psi
    log_psi_vals = jax.vmap(model)(x_primes)
    # 2. 指数还原成 ψ(σ')
    psi_vals = jnp.exp(log_psi_vals)
    # 3. 求和得到 Hψ(x)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x

def Ham_Psi(ha, total_ansatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    hilber_size = total_ansatz.n_spin
    k = total_ansatz.K
    x = x.reshape(k, hilber_size)
    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)

    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi


def NES_loss_energy(ha:nk.operator.DiscreteOperator,total_ansatz:NESTotalAnsatz,x:jnp.ndarray):
    """计算NES-VMC的损失函数  
    \mathcal{L} = \mathrm{Tr}\left(\Psi(\mathbf{x})^{-1} \hat{H}\Psi(\mathbf{x})\right)
    参数:
    ha: 哈密顿量
    total_ansatz: 总Ansatz
    x: 输入状态
    返回:
    损失函数值
    """
    #首先计算一下\Psi(\mathbf{x})
    value, log_M = total_ansatz(x)
    Psi_Matrix = jnp.exp(log_M)
    # 接下来计算 \hat{H}\Psi(\mathbf{x})
    H_psi_x = Ham_Psi(ha, total_ansatz, x)
    # \mathbf{x})^{-1} @ \hat{H}\Psi(\mathbf{x})
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix,H_psi_x)
    return jnp.trace(Psi_Matrix_inv),Psi_Matrix_inv


import jax
import jax.numpy as jnp
from functools import partial


def create_machine(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix = m(sigma)
        return log_psi_total

    return machine, graphdef, state

machine, graphdef, state = create_machine(total_ansatz)
# ===========================================================================
# 1. 【修复】单独定义 仅使用参数的 logΨ 函数（NNX 标准求导方式）
# ===========================================================================
# 构建 【可求导函数】

grad_logPsi = jax.grad(machine, argnums=0, holomorphic=True)

# 向量化（批量 walker）
vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

# ===========================================================================
# 2. 批量 E_L 矩阵（沿用你的逻辑，无修改）
# ===========================================================================
@partial(jax.vmap, in_axes=(None, None, 0))
def compute_local_energy_matrix_batch(ha, total_ansatz, x_batch):
    loss_val, E_L = NES_loss_energy(ha, total_ansatz, x_batch)
    return E_L

def nes_vmc_gradient(ha, total_ansatz:NESTotalAnsatz, x_batch):
    static, params = nnx.split(total_ansatz)
    #K = total_ansatz.K

    # 1. 批量局域能量矩阵
    E_L_batch = compute_local_energy_matrix_batch(ha, total_ansatz, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    E_L_diff = E_L_batch - E_L_mean[None, ...]  # (N,K,K)

    # 2. 【✅ 关键】先对每个样本求迹 → 得到标量！
    tr_diff_batch = jnp.trace(E_L_diff, axis1=1, axis2=2)  # (N,)  标量！

    # 3. 计算 ∇logΨ
    dlogPsi_batch = vmap_grad_logPsi(params, x_batch)  # PyTree

    # 4. 【✅ 最终正确】梯度 = tr_diff * ∇logΨ
    grad_tree = jax.tree.map(lambda g: tr_diff_batch[:, None, None] * g, dlogPsi_batch)

    # 5. 平均 ×2
    grad = jax.tree.map(lambda g: 2.0 * jnp.mean(g, axis=0), grad_tree)

    # 损失
    loss_mean = jnp.mean(jnp.trace(E_L_batch, axis1=1, axis2=2))
    return grad, loss_mean, E_L_mean


def generate_random_initial_states(hi, n_chains: int, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_chains)
    return jax.vmap(lambda k: hi.random_state(k))(keys)


def init_sampler_state(hi, n_chains, seed=42):
    init_states = generate_random_initial_states(hi, n_chains, seed)
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, n_chains)  # 每条链独立随机数
    return (init_states, chain_keys)


from NES_VMC import compute_qgt
import time
# ======================
# 超参数
# ======================
N_CHAINS = 16
N_WARMUP = 32
N_SAMPLES_PER_CHAIN = 100
SWEEP_SIZE = 32
N_ITER =300

# ======================
# 初始化 ONCE
# ======================
rngs = nnx.Rngs(21)
model = NESTotalAnsatz(4,2,12,rngs=rngs)
machine, graphdef, params = create_machine(model)

optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params)

# ===================== 7. 训练循环（多链版本） =====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (自然梯度下降法)")
print("="*60)

history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}
sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=21)  # 每次迭代换种子避免初始状态固定
start_time = time.time()
for step in range(N_ITER):
    # 1. 生成多链随机初始状态（模仿NetKet，无需手动指定单个initial_state）
    # 2. 多链采样（总样本数=16*63=1008，和原单链一致）
    samples,sampler_state = mcmc_sampler_multichain(
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        n_warmup=N_WARMUP,
        sampler_state=sampler_state,
        edges=((0, 1), (2, 3),(4,5),(6,7)),
        machine=machine,
        params=params,
    )

    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(machine, params, samples)
    #grad = jax.tree_map(lambda x: x*2, grad)
    qgt_reg,qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001) 
    grad_flat , grad_unravel_fn = flatten_util.ravel_pytree(grad)
    
    # 自然梯度求解
    natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    natural_grad = grad_unravel_fn(natural_grad)
    grad = natural_grad
        
    # 4. 更新参数
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # 5. 记录历史
    if step % 50 == 0 or step == N_ITER - 1:
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        # history['energy'].append(float(energy.real))
        # history['energy_std'].append(float(energy_std))
        # history['error'].append(float(error))
        print(f"Step {step:3d} | E: {E_L_mean:.8f}")

end_time = time.time()
print(f"训练耗时：{end_time - start_time:.2f} 秒")
# 最终结果
print("\n" + "="*60)
print(f"训练完成!")
print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
print(f"绝对误差：{final_error:.6f} Ha")
print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print("="*60)