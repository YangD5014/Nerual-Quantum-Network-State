import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx
import sys
sys.path.append('..')
from NES_VMC import NESTotalAnsatz, create_machine,init_sampler_state,\
    generate_random_initial_states,ha,SingleStateAnsatz,create_single_machine,\
        create_machine_matrix,Ham_psi,Ham_Psi,NES_loss_energy,nes_vmc_gradient,hi,E_fcis,mcmc_sampler_multichain
import optax
from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
import time
from netket.utils import struct  # NetKet 专用 dataclass（兼容 JAX）

# ========== 你原有全局参数（直接复用） ==========
# 单系统希尔伯特空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2  # NES 扩展副本数
hi_ext = hi ** K  # 扩展希尔伯特空间
SINGLE_SIZE = hi.size  # 单个子系统维度 = 4
single_edges = ((0, 1), (2, 3))  # 费米子跃迁边
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)

@struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    edges: jnp.ndarray

    def _check_duplicate(self, sigma_ext):
        """NES约束：子组态不重复"""
        sub = sigma_ext.reshape((*sigma_ext.shape[:-1], K, SINGLE_SIZE))
        return jnp.any(jnp.all(sub[...,1:,:] == sub[...,0:1,:], axis=-1), axis=-1)

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        batch_size = sigma.shape[0]
        key1, _ = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:,0], sel_e[:,1]

        sigma_cand = sigma.at[jnp.arange(batch_size),i].set(sigma[jnp.arange(batch_size),j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size),j].set(sigma[jnp.arange(batch_size),i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    # ✅【修复】官方标准接口：删除 sigma 参数！！！
    def random_state(self, sampler, machine, parameters, state, rng):
        """
        正确写法：
        1. 无 sigma 参数
        2. 从 state.σ 获取初始样本的形状
        3. 生成合法的初始态（无重复子组态）
        """
        # 从采样器状态中获取当前链的形状 (16,8)
        sigma_shape = state.σ.shape
        
        def gen_single(key):
            def cond(c): return c[1]
            def body(c):
                k,_,_=c
                s = hi_ext.random_state(k)
                return k, self._check_duplicate(s), s
            return jax.lax.while_loop(cond, body, (rng, True, hi_ext.random_state(rng)))[2]
        
        # 生成对应链数的合法初始态
        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)
    



ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # 转为jax数组（关键修复）

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=32
)



N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =50

rngs = nnx.Rngs(42)
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=12, rngs=rngs)
single_ansatz = SingleStateAnsatz(4, hidden_dim=8, rngs=rngs)
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef, total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

# 采样器状态初始化（替代原 init_sampler_state）
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ==================== 训练循环（仅替换采样部分） ====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (NetKet 自定义采样器 + 朴素梯度下降)")
print("="*60)
print(f"基态能量={E_fcis[0]:.8f} Ha| 第一激发态能量={E_fcis[1]:.8f} Ha| 第二激发态能量={E_fcis[2]:.8f} Ha")

history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'loss': [],
    'params': [],
    'E_Lmatrix':[],
    'natural_grad':[],
    'grad_flat':[],
    'samples':[],
    'log_Psi':[],
    'log_M':[]
}

start_time = time.time()
for step in range(N_ITER):
    # 1. Warmup 烧链
    _, sampler_state = nes_sampler.sample(
        total_machine, total_params, sampler_state, chain_length=N_WARMUP
    )
    # 2. 正式采样
    samples_raw, sampler_state = nes_sampler.sample(
        total_machine, total_params, sampler_state, chain_length=N_SAMPLES_PER_CHAIN
    )
        # 3. 维度重塑，适配梯度函数输入
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, 4)
    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))
    #grad = jax.tree_util.tree_map(lambda x: x * 2, grad)
    
    grad_flat , grad_unravel_fn = ravel_pytree(grad)
    # qgt_reg, unravel_fn = compute_qgt(total_machine, total_params, samples.reshape(-1,2,4), diag_shift=0.1)
    
    # # # 自然梯度求解
    # natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    # natural_grad = grad_unravel_fn(natural_grad_flat)
    # grad = natural_grad
        
    # 4. 更新参数
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
    
    # 5. 记录历史
    if step % 5 == 0 or step == N_ITER - 1:
        # --------------------- 【NES-VMC 监控模板】直接用 ---------------------
        # 1. 监控 log_Psi
        log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
        print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")

        # 2. 监控梯度范数
        grad_norm = jnp.linalg.norm(grad_flat)
        print(f"grad norm = {grad_norm:.4f}")

        # 5. 局域能量矩阵
        print(f"E_L mean =\n{E_L_mean}")
    
        eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
        history['step'].append(step)
        history['E_Lmatrix'].append(E_L_mean)
        history['samples'].append(samples)
        history['loss'].append(loss_mean)
        # #history['natural_grad'].append(natural_grad)
        # history['grad_flat'].append(grad_flat)
        # history['log_Psi'].append(log_Psi)
        # history['log_M'].append(log_M)
        history['params'].append(total_params)
        print(f"Step {step:3d} | Loss: {loss_mean}|0st能量={eig_vals[0]:.8f} Ha| 1st能量={eig_vals[1]:.8f} Ha")
        # print(f'grad={grad_flat[30:31]}')
        print('#-----------------------------------------#')


end_time = time.time()
print(f"训练耗时：{end_time - start_time:.2f} 秒")
# 最终结果
print("\n" + "="*60)
print(f"训练完成!")
# print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
# print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
# print(f"绝对误差：{final_error:.6f} Ha")
# print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print("="*60)

