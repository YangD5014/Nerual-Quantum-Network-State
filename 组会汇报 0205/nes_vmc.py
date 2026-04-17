"""
NES-VMC 改进实现

策略：
1. 先独立优化基态到收敛
2. 固定基态参数，优化激发态（与基态正交）
3. 计算哈密顿量矩阵对角化得到最终能量

这种方法更稳定，因为基态已经收敛，激发态只需要与基态正交
"""

from functools import partial
from typing import Any, Callable, Tuple, List
import jax
from jax import numpy as jnp
import netket as nk
from netket import jax as nkjax
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree, Array
from netket.operator import AbstractOperator
from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
)
from netket.vqs.mc.mc_state.state import MCState


def penalty_kernel(logpsi: Callable, pars1: PyTree, pars2: PyTree, σ: Array):
    """计算 ψ_1(σ)/ψ_2(σ)"""
    return jnp.exp(logpsi(pars1, σ) - logpsi(pars2, σ))


def compute_matrices(
    vstate_list: List[MCState],
    hamiltonian: AbstractOperator,
) -> Tuple[Array, Array]:
    """计算哈密顿量矩阵和重叠矩阵"""
    n_states = len(vstate_list)
    
    local_estimator_fun = get_local_kernel(vstate_list[0], hamiltonian)
    model_apply_fun = vstate_list[0]._apply_fun
    
    samples_list = []
    args_list = []
    pars_list = []
    model_state_list = []
    
    for i, vstate in enumerate(vstate_list):
        vstate.reset()
        σ, args = get_local_kernel_arguments(vstate, hamiltonian)
        samples_list.append(σ)
        args_list.append(args)
        pars_list.append(vstate.parameters)
        model_state_list.append(vstate.model_state)
    
    H_matrix = jnp.zeros((n_states, n_states), dtype=complex)
    S_matrix = jnp.zeros((n_states, n_states), dtype=complex)
    
    for i in range(n_states):
        σ_i = samples_list[i]
        σ_i_shape = σ_i.shape
        if jnp.ndim(σ_i) != 2:
            σ_i = σ_i.reshape((-1, σ_i_shape[-1]))
        
        pars_i = {"params": pars_list[i], **model_state_list[i]}
        
        for j in range(n_states):
            pars_j = {"params": pars_list[j], **model_state_list[j]}
            
            ratio = penalty_kernel(model_apply_fun, pars_j, pars_i, σ_i)
            S_ij = jnp.mean(ratio)
            S_matrix = S_matrix.at[i, j].set(S_ij)
            
            E_local_j = local_estimator_fun(
                model_apply_fun,
                pars_j,
                σ_i,
                args_list[j],
            )
            H_ij = jnp.mean(ratio * E_local_j)
            H_matrix = H_matrix.at[i, j].set(H_ij)
    
    return H_matrix, S_matrix


def diagonalize_generalized_eigenvalue_problem(
    H_matrix: Array, 
    S_matrix: Array
) -> Tuple[Array, Array]:
    """求解广义特征值问题: H c = E S c"""
    H_herm = (H_matrix + H_matrix.conj().T) / 2
    S_herm = (S_matrix + S_matrix.conj().T) / 2
    
    reg = 1e-6
    S_reg = S_herm + reg * jnp.eye(S_herm.shape[0], dtype=complex)
    
    try:
        L = jnp.linalg.cholesky(S_reg)
        L_inv = jnp.linalg.inv(L)
        H_prime = L_inv @ H_herm @ L_inv.conj().T
        energies, coeffs_prime = jnp.linalg.eigh(H_prime)
        coefficients = L_inv.conj().T @ coeffs_prime
    except:
        S_inv = jnp.linalg.inv(S_reg)
        H_eff = S_inv @ H_herm
        H_eff = (H_eff + H_eff.conj().T) / 2
        energies, coefficients = jnp.linalg.eigh(H_eff)
    
    return energies, coefficients


@partial(jax.jit, static_argnums=(0, 1, 2))
def grad_expect_with_penalty(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: Array,
    local_value_args: PyTree,
    fixed_pars_list: List[PyTree],
    fixed_model_state_list: List[PyTree],
    fixed_σ_list: List[Array],
    penalty_strength: float,
) -> Tuple[Stats, PyTree]:
    """
    计算单个态的能量和梯度，包含与固定态的正交化约束
    
    用于激发态优化：固定态是已经收敛的低能态
    """
    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))
    
    n_samples = σ.shape[0] * mpi.n_nodes
    
    pars = {"params": parameters, **model_state}
    
    O_loc = local_value_kernel(
        model_apply_fun,
        pars,
        σ,
        local_value_args,
    )
    
    E_loc = O_loc.copy()
    O_stats = statistics(O_loc.reshape(σ_shape[:-1]).T)
    O_loc -= O_stats.mean
    
    n_fixed = len(fixed_pars_list)
    for k in range(n_fixed):
        fixed_pars = fixed_pars_list[k]
        fixed_model_state = fixed_model_state_list[k]
        fixed_σ = fixed_σ_list[k]
        
        fixed_σ_shape = fixed_σ.shape
        if jnp.ndim(fixed_σ) != 2:
            fixed_σ = fixed_σ.reshape((-1, fixed_σ_shape[-1]))
        
        fixed_pars_dict = {"params": fixed_pars, **fixed_model_state}
        
        psi_loc_1 = penalty_kernel(
            model_apply_fun,
            pars,
            fixed_pars_dict,
            fixed_σ,
        )
        psi_1 = statistics(psi_loc_1.reshape(fixed_σ_shape[:-1]).T)
        
        psi_loc_2 = penalty_kernel(
            model_apply_fun,
            fixed_pars_dict,
            pars,
            σ,
        )
        psi_2 = statistics(psi_loc_2.reshape(σ_shape[:-1]).T)
        
        E_loc = E_loc + penalty_strength * psi_1.mean * psi_2.mean
        
        psi_loc_2_centered = psi_loc_2 - psi_2.mean
        psi_loc_2_centered = psi_loc_2_centered * penalty_strength * psi_1.mean
        O_loc = O_loc + psi_loc_2_centered
    
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    
    O_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]
    
    O_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(target.dtype),
        O_grad,
        parameters,
    )
    
    E_stats = statistics(E_loc.reshape(σ_shape[:-1]).T)
    
    new_model_state = new_model_state[0] if is_mutable else None
    
    return E_stats, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], O_grad), new_model_state


def expect_and_grad_nes(
    vstate_list: List[MCState],
    hamiltonian: AbstractOperator,
    use_covariance: bool = True,
    mutable: Any = False,
    penalty_strength: float = 0.3,
    fixed_state_indices: List[int] = None,
) -> Tuple[List[Stats], List[PyTree], Array, Array]:
    """
    计算NES-VMC的能量和梯度
    
    支持两种模式：
    1. 同时优化所有态（fixed_state_indices=None）
    2. 固定某些态，优化其他态（fixed_state_indices指定哪些态已固定）
    """
    n_states = len(vstate_list)
    
    local_estimator_fun = get_local_kernel(vstate_list[0], hamiltonian)
    model_apply_fun = vstate_list[0]._apply_fun
    
    samples_list = []
    args_list = []
    pars_list = []
    model_state_list = []
    
    for i, vstate in enumerate(vstate_list):
        vstate.reset()
        σ, args = get_local_kernel_arguments(vstate, hamiltonian)
        samples_list.append(σ)
        args_list.append(args)
        pars_list.append(vstate.parameters)
        model_state_list.append(vstate.model_state)
    
    energy_stats_list = []
    grad_list = []
    
    for i in range(n_states):
        σ_i = samples_list[i]
        args_i = args_list[i]
        pars_i = pars_list[i]
        model_state_i = model_state_list[i]
        
        if fixed_state_indices is not None and i in fixed_state_indices:
            E_stats = vstate_list[i].expect(hamiltonian)
            grad_i = jax.tree_map(lambda x: jnp.zeros_like(x), pars_i)
        else:
            other_indices = [j for j in range(n_states) if j != i]
            other_pars_list = [pars_list[j] for j in other_indices]
            other_model_state_list = [model_state_list[j] for j in other_indices]
            other_σ_list = [samples_list[j] for j in other_indices]
            
            E_stats, grad_i, new_model_state = grad_expect_with_penalty(
                local_estimator_fun,
                model_apply_fun,
                mutable,
                pars_i,
                model_state_i,
                σ_i,
                args_i,
                other_pars_list,
                other_model_state_list,
                other_σ_list,
                penalty_strength,
            )
            
            if mutable is not False and new_model_state is not None:
                vstate_list[i].model_state = new_model_state
        
        energy_stats_list.append(E_stats)
        grad_list.append(grad_i)
    
    H_matrix, S_matrix = compute_matrices(vstate_list, hamiltonian)
    
    return energy_stats_list, grad_list, H_matrix, S_matrix
