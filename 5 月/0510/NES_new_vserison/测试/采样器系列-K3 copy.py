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
        NESFermionHopRule,compute_qgt,sampler_info,NESFermionHopRule
import optax
from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
import time
import pickle
import os   

# ========== 你原有全局参数（直接复用） ==========
# 单系统希尔伯特空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 3  # NES 扩展副本数
hi_ext = hi ** K  # 扩展希尔伯特空间
SINGLE_SIZE = hi.size  # 单个子系统维度 = 4
single_edges = ((0, 1), (2, 3))  # 费米子跃迁边
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)

total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    

if __name__ == '__main__':
    N_CHAINS = 16
    N_WARMUP = 100
    N_SAMPLES_PER_CHAIN = 200
    SWEEP_SIZE = 30
    N_ITER =400
    SINGLE_SIZE = hi.size  # 单个子系统维度 = 4
    Natural_Grad = False


    total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
    total_machine, total_graphdef,total_params = create_machine(total_ansatz)
    total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

    single_machine_list = []
    for ansatz in total_ansatz.single_ansatz_list:
        m, g, p = create_single_machine(ansatz)
        single_machine_list.append(m)
        
        

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(total_params)

    ext_edges = []
    for k in range(K):
        offset = k * SINGLE_SIZE
        for (i, j) in single_edges:
            ext_edges.append((i + offset, j + offset))
    ext_edges = jnp.array(ext_edges)  # 转为jax数组（关键修复）

    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    nes_sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext,
        rule=nes_rule,
        n_chains=16,
        sweep_size=20
    )


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
        'energy_0st': [],
        'energy_1st': [],
        'energy_2st': [],
        'energy_std': [],
        'loss': [],
        'params': [],
        'E_Lmatrix':[],
        'natural_grad':[],
        'grad_flat':[],
        'samples':[],
        'log_Psi':[],
        'log_M':[],
        'log_Psi_mean':[],
        'log_Psi_min':[],
        'log_Psi_max':[],
        'grad_norm':[],
    }

    start_time = time.time()
    for step in range(N_ITER):
        # 2. 正式采样
        samples_raw, sampler_state = nes_sampler.sample(
            machine=total_machine, parameters=total_params, 
            state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN
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
        if Natural_Grad == True:
            grad_flat , grad_unravel_fn = ravel_pytree(grad)
            qgt_reg, unravel_fn = compute_qgt(total_machine, total_params, samples.reshape(-1,K,4), diag_shift=0.1)
            
            # # 自然梯度求解
            natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
            natural_grad = grad_unravel_fn(natural_grad_flat)
            grad = natural_grad
            
        # 4. 更新参数
        updates, opt_state = optimizer.update(grad, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)
        
        
        
        log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
        eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
        grad_norm = jnp.linalg.norm(grad_flat)
        
        
        history['step'].append(step)
        history['E_Lmatrix'].append(E_L_mean)
        history['samples'].append(samples)
        history['loss'].append(loss_mean)
        history['log_Psi_mean'].append(log_Psi_batch.mean())
        history['log_Psi_min'].append(log_Psi_batch.min())
        history['log_Psi_max'].append(log_Psi_batch.max())
        history['grad_norm'].append(grad_norm)
        history['energy_0st'].append(eig_vals[0])
        history['energy_1st'].append(eig_vals[1])
        history['energy_2st'].append(eig_vals[2])
        history['params'].append(total_params)
        # 5. 记录历史
        if step % 50 == 0 or step == N_ITER - 1:
            # --------------------- 【NES-VMC 监控模板】直接用 ---------------------
            # 1. 监控 log_Psi
            #log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
            print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")

            # 2. 监控梯度范数
            
            print(f"grad norm = {grad_norm:.4f}")
            print(f"Step {step:3d} | Loss: {loss_mean}|0st能量={eig_vals[0]:.8f} Ha｜1st能量={eig_vals[1]:.8f} Ha｜2st能量={eig_vals[2]:.8f} Ha")
            # print(f'grad={grad_flat[30:31]}')
            print('#-----------------------------------------#')


    end_time = time.time()
    print(f"训练耗时：{end_time - start_time:.2f} 秒")
    # 最终结果
    print("\n" + "="*60)
    print(f"训练完成!")
    print("="*60)
    print(f"#----------开始保存历史记录----------#")

    # 自动创建 data 文件夹（关键修复）
    os.makedirs('./data-K3', exist_ok=True)
    if Natural_Grad == True:
        print('保存自然梯度历史记录')
        # 保存 history
        with open('./data-K3/history_natural_gradient_K3.pkl', 'wb') as f:
            pickle.dump(history, f)
    else:
        # 保存 history
        with open('./data-K3/history_plain_gradient_K3.pkl', 'wb') as f:
            pickle.dump(history, f)

    print("保存成功！")