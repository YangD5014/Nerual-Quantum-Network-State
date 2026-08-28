from NES_VMC import NESTotalAnsatz_gauge_stable,create_single_machine_gauge_fixed
from LiH import Hatree_Fock
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
import flax.nnx as nnx
import optax
from functools import partial
from jax import flatten_util
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import jax
import time





def rebuild_active_machine(total_ansatz:NESTotalAnsatz_gauge_stable, total_params, freeze_idx):
    """
    执行方案4：物理拆分 frozen 和 active 子图
    """
    # 1. 拆分 NNX 模型状态
    graphdef, state = nnx.split(total_ansatz)
    
    # 2. 提取冻结的参数 (假设结构是 state['single_ansatz_list'][idx])
    frozen_state = state['single_ansatz_list'][freeze_idx]
    frozen_ansatz = total_ansatz.single_ansatz_list[freeze_idx]
    
    # 3. 构建冻结机器 (仅用于前向计算，不参与求导)
    frozen_machine, _, _ = create_single_machine_gauge_fixed(frozen_ansatz, Hatree_Fock)
    frozen_params = frozen_state
    
    # 4. 提取活跃的参数 (剔除已冻结的)
    active_state = {
        'single_ansatz_list': [s for i, s in enumerate(state['single_ansatz_list']) if i != freeze_idx]
    }
    # 注意：这里你需要重新构建一个只包含剩余 sub-ansatz 的 NESTotalAnsatz 对象
    # 或者修改你的 NESTotalAnsatz 使其支持动态长度的 list。
    active_ansatz = NESTotalAnsatz_gauge_stable(n_states=) # 传入剩余的 ansatz
    
    # 5. 重建活跃机器的组件
    active_machine_log_Psi, active_graphdef, active_params = create_machine_gauge_stable(active_ansatz)
    active_machine_L_stable, active_machine_shift, active_machine_L_gauge = create_machine_gauge_synthesis(active_ansatz)
    
    # 6. 重建梯度函数和 QGT 函数
    active_single_machine_list = []
    for ansatz in active_ansatz.single_ansatz_list:
        m, _, _ = create_single_machine_gauge_fixed(ansatz, Hatree_Fock)
        active_single_machine_list.append(m)
        
    active_grad_fn = make_grad_fn(
        ha, active_machine_L_stable, active_machine_shift, 
        active_machine_log_Psi, active_single_machine_list
    )
    active_qgt_fn = make_qgt_fn(active_machine_log_Psi)
    
    return {
        'frozen_machine': frozen_machine,
        'frozen_params': frozen_params,
        'active_ansatz': active_ansatz,
        'active_params': active_params,
        'active_grad_fn': active_grad_fn,
        'active_qgt_fn': active_qgt_fn,
        'active_optimizer_state': optimizer.init(active_params) # 重新初始化优化器状态
    }