
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx
import sys
sys.path.append('..')
from NES_VMC import NESTotalAnsatz, create_machine,init_sampler_state,\
    generate_random_initial_states,ha,SingleStateAnsatz,create_single_machine,\
        create_machine_matrix,Ham_psi,Ham_Psi,NES_loss_energy,nes_vmc_gradient,hi,E_fcis,mcmc_sampler_multichain,\
            compute_qgt
import optax
from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
import time
from collections import Counter
import numpy as np
K=2
hi_ext = hi**K

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

# ===================== 7. 训练循环（多链版本） =====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (朴素梯度下降法)")
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