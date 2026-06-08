# 关于NES-VMC 的实验测试
## 1. 基于纯梯度的 NES-VMC 算法 + 自制采样器 
以下代码是基于纯梯度下降的 NES-VMC 算法求解 H2 分子基态、第一激发态能量的例子(K=2)
```python
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
K=2
hi_ext = hi**K

# ======================
# 超参数
# ======================
N_CHAINS = 32
N_WARMUP = 50
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =20

rngs = nnx.Rngs(42)
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
single_ansatz = SingleStateAnsatz(4, hidden_dim=8, rngs=rngs)
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef, total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

# ===================== 7. 训练循环（多链版本） =====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (自然梯度下降法)")

print("超参数")
print("="*60)

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
print(f"基态能量={E_fcis[0]:.8f} Ha| 第一激发态能量={E_fcis[1]:.8f} Ha| 第二激发态能量={E_fcis[2]:.8f} Ha")
sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=21)  # 每次迭代换种子避免初始状态固定
start_time = time.time()
for step in range(N_ITER):
    # 1. 生成多链随机初始状态（模仿NetKet，无需手动指定单个initial_state）
    # 2. 多链采样（总样本数=16*63=1008，和原单链一致）
    samples,sampler_state = mcmc_sampler_multichain(
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        n_warmup=N_WARMUP,
        sampler_state=sampler_state,
        edges=((0,1),(2,3),(4,5),(6,7)),
        machine=total_machine,
        params=total_params,
    )
    #samples = samples.reshape(-1,2,4)

    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))
    grad = jax.tree_util.tree_map(lambda x: x * 2, grad)
    #model_output = log(\Psi(X)) 
    # qgt_reg,qgt_unravel_fun = compute_nes_qgt(total_machine, total_params, samples.reshape(-1,K,4), diag_shift=0.01) 
    grad_flat , grad_unravel_fn = ravel_pytree(grad)
    
    # # # 自然梯度求解
    # natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    # natural_grad = grad_unravel_fn(natural_grad)
    # grad = natural_grad
        
    # 4. 更新参数
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
    
    # 5. 记录历史
    if step % 5 == 0 or step == N_ITER - 1:
        # total_model =  nnx.merge(graphdef,total_params)
        # log_Psi,log_M  = total_model(samples.reshape(-1,2,4))
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
        print(f"Step {step:3d} | Loss: {loss_mean}|基态能量={eig_vals[0]:.8f} Ha| 第一激发态能量={eig_vals[1]:.8f} Ha Ha")
        print(f'grad={grad_flat[30:31]}')


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
>>
============================================================
开始多链 NES-VMC 训练 (自然梯度下降法)
超参数
============================================================
基态能量=-1.01546825 Ha| 第一激发态能量=-0.87542794 Ha| 第二激发态能量=-0.42938376 Ha
Step   0 | Loss: -1.3071755633244622|基态能量=-1.05577951 Ha| 第一激发态能量=-0.25139605 Ha Ha
grad=[0.00488799+0.0880935j]
Step   5 | Loss: -1.3399609683557112|基态能量=-1.02943806 Ha| 第一激发态能量=-0.31052291 Ha Ha
grad=[-0.01788345+0.02106104j]
Step  10 | Loss: -1.3553283766293875|基态能量=-1.02671814 Ha| 第一激发态能量=-0.32861024 Ha Ha
grad=[-0.01056921+0.00407642j]
Step  15 | Loss: -1.3666896093211887|基态能量=-1.05565305 Ha| 第一激发态能量=-0.31103656 Ha Ha
grad=[-0.01458677-0.00712241j]
Step  19 | Loss: -1.3971857643655523|基态能量=-1.10513069 Ha| 第一激发态能量=-0.29205507 Ha Ha
grad=[-0.03699766-0.01224924j]
训练耗时：21.96 秒

============================================================
训练完成!
============================================================
```
可以看出 Loss 确实在下降   

## 2. 基于自然梯度下降法的NES-VMC 算法 + 自制采样器

```python
def compute_nes_qgt(total_machine, params, samples, diag_shift=0.01):
    # 1. 单样本梯度
    def _single_grad(x):
        return jax.grad(lambda p: total_machine(p, x), holomorphic=True)(params)

    # 2. 对每个样本求导
    grads = [_single_grad(x) for x in samples]  # 列表 [B]

    # 3. 展平每个梯度 → list of [P]
    flat_grads = [ravel_pytree(g)[0] for g in grads]

    # 4. 堆叠成 → (B, P)
    grads_flat = jnp.stack(flat_grads)  

    # 5. 计算 QGT
    g = grads_flat
    g_conj = g.conj()

    term1 = jnp.mean(g_conj[:, :, None] * g[:, None, :], axis=0)
    g_mean = jnp.mean(g, axis=0)
    term2 = g_mean.conj()[:, None] * g_mean[None, :]
    S = term1 - term2

    S_reg = S + diag_shift * jnp.eye(S.shape[0], dtype=complex)
    unravel = ravel_pytree(params)[1]

    return S_reg, unravel


# ======================
# 超参数
# ======================
N_CHAINS = 32
N_WARMUP = 50
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =20

rngs = nnx.Rngs(42)
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
single_ansatz = SingleStateAnsatz(4, hidden_dim=8, rngs=rngs)
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef, total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

# ===================== 7. 训练循环（多链版本） =====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (自然梯度下降法)")

print("超参数")
print("="*60)

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
print(f"基态能量={E_fcis[0]:.8f} Ha| 第一激发态能量={E_fcis[1]:.8f} Ha| 第二激发态能量={E_fcis[2]:.8f} Ha")
sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=21)  # 每次迭代换种子避免初始状态固定
start_time = time.time()
for step in range(N_ITER):
    # 1. 生成多链随机初始状态（模仿NetKet，无需手动指定单个initial_state）
    # 2. 多链采样（总样本数=16*63=1008，和原单链一致）
    samples,sampler_state = mcmc_sampler_multichain(
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        n_warmup=N_WARMUP,
        sampler_state=sampler_state,
        edges=((0,1),(2,3),(4,5),(6,7)),
        machine=total_machine,
        params=total_params,
    )
    #samples = samples.reshape(-1,2,4)

    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))
    grad = jax.tree_util.tree_map(lambda x: x * 2, grad)
    #model_output = log(\Psi(X)) 
    qgt_reg,qgt_unravel_fun = compute_nes_qgt(total_machine, total_params, samples.reshape(-1,K,4), diag_shift=0.01) 
    grad_flat , grad_unravel_fn = ravel_pytree(grad)
    
    # # # 自然梯度求解
    natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    natural_grad = grad_unravel_fn(natural_grad)
    grad = natural_grad
        
    # 4. 更新参数
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
    
    # 5. 记录历史
    if step % 5 == 0 or step == N_ITER - 1:
        # total_model =  nnx.merge(graphdef,total_params)
        # log_Psi,log_M  = total_model(samples.reshape(-1,2,4))
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
        print(f"Step {step:3d} | Loss: {loss_mean}|基态能量={eig_vals[0]:.8f} Ha| 第一激发态能量={eig_vals[1]:.8f} Ha Ha")
        print(f'grad={grad_flat[30:31]}')


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
>>
============================================================
开始多链 NES-VMC 训练 (自然梯度下降法)
超参数
============================================================
基态能量=-1.01546825 Ha| 第一激发态能量=-0.87542794 Ha| 第二激发态能量=-0.42938376 Ha
Step   0 | Loss: -1.3071755633244622|基态能量=-1.05577951 Ha| 第一激发态能量=-0.25139605 Ha Ha
grad=[0.00488799+0.0880935j]
Step   5 | Loss: -1.3168079983958807|基态能量=-1.05503781 Ha| 第一激发态能量=-0.26177018 Ha Ha
grad=[0.00128115+0.1037072j]
Step  10 | Loss: -1.3244099336102908|基态能量=-1.04662430 Ha| 第一激发态能量=-0.27778563 Ha Ha
grad=[0.00710281+0.11332878j]
Step  15 | Loss: -1.3281164017930105|基态能量=-1.04500897 Ha| 第一激发态能量=-0.28310743 Ha Ha
grad=[0.009731+0.12894392j]
Step  19 | Loss: -1.341174031036991|基态能量=-1.04689195 Ha| 第一激发态能量=-0.29428208 Ha Ha
grad=[0.01469552+0.14928237j]
训练耗时：285.27 秒

============================================================
训练完成!
============================================================

```
你可以看出 非常耗时、并且梯度下降速度变慢！这应该可以说明 QGT 的梯度求解错误！ 为什么？分析原因