"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现

本文件实现基于原生 JAX 和部分 NetKet 的 NES-VMC 算法，用于计算量子多体系统的激发态能量。
"""
import jax
import jax.numpy as jnp
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
from functools import partial
from jax.flatten_util import ravel_pytree
from collections import Counter

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
# ===================== H₂ 分子定义 & FCI 基准 =====================
# bond_length = 1.4
# geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
# mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
# mf = scf.RHF(mol).run(verbose=0)

# # FCI 精确基准
# cisolver = fci.FCI(mf)
# cisolver.nroots = 4
# E_fcis, fcivec = cisolver.kernel()
# print("="*60)
# print("H₂ FCI 基准能量")
# print("="*60)
# for i, e in enumerate(E_fcis):
#     exc = (e - E_fcis[0]) * 27.2114
#     print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")
# # ===================== NetKet 哈密顿量和采样器 =====================
# ha = nkx.operator.from_pyscf_molecule(mol)

# hi = nkx.hilbert.SpinOrbitalFermions(
#     n_orbitals=2,
#     s=1/2,
#     n_fermions_per_spin=(1,1),
# )
#K=3
# hi_ext = hi**K
# edges = [(0, 1), (2, 3),(4, 5),(6,7)]

class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            
            ansatz = SingleStateAnsatz(
                n_spin_orbitals, 
                hidden_dim, 
                rngs=sub_rngs
            )
            self.single_ansatz_list.append(ansatz)
    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            # 形状：[K, n_spin]
            #print(f'x_single.shape: {x_single.shape}')
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
            log_Psi = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi, L  
        
        # 安全的批量处理
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            # 直接处理单个样本
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin*self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            # 直接处理批量样本
            return jax.vmap(_forward_single)(x)
        
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim ==1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f'不支持的输入形状: {x.shape}')
            
class NESTotalAnsatz_stable(nnx.Module):
    def __init__(
        self,
        n_spin_orbitals: int,
        n_states: int = 2,
        hidden_dim: int = 8,
        *,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()

        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)

            ansatz = SingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim,
                rngs=sub_rngs
            )
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)

            L = jnp.zeros((self.K, self.K), dtype=jnp.complex64)

            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )

            # 关键修正：只取实部最大值
            shift = jnp.max(jnp.real(L))

            # 关键修正：不要让梯度穿过 max(real(L))
            shift = jax.lax.stop_gradient(shift)

            # 稳定化
            L_stable = L - shift

            Psi_stable = jnp.exp(L_stable)

            sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)

            return log_Psi_stable, L_stable, shift

        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)

        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)

        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)

        elif x.ndim == 1:
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)

        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")        
    
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

def create_single_machine(model: SingleStateAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state

def create_machine_matrix(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix = m(sigma)
        return log_M_matrix

    return machine, graphdef, state

def create_machine_stable(model: NESTotalAnsatz_stable):
    graphdef, state = nnx.split(model)
    K = model.K
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, shift = m(sigma)

        log_psi_raw = log_stable + K * shift
        return log_psi_raw

    return machine, graphdef, state



def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])

def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    🔥 同时支持：
    - 单个态 x: (n_spin,)
    - 批量态 x: (batch_size, n_spin)
    """
    # ======================
    # 核心：自动给单个样本增加 batch 维度
    # ======================
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    # ======================
    # 向量化计算（批处理）
    # ======================
    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    # 批量处理
    H_psi_batch = jax.vmap(_single_hpsi)(x)

    # ======================
    # 如果是单个输入，就压回单个输出
    # ======================
    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
   
def Ham_psi_scaled(
    ha: nk.operator.DiscreteOperator,
    single_machine,
    params,
    x,
    shift,
):
    """
    稳定版单态哈密顿量作用。

    计算：
        e^{-shift} Hψ(x)
      = Σ_{x'} H[x, x'] * exp(logψ(x') - shift)

    参数：
    - ha: NetKet 离散哈密顿量
    - single_machine: 单态 logψ 机器函数
    - params: 对应 single ansatz 的参数
    - x: 单个构型 (n_spin,) 或批量构型 (batch, n_spin)
    - shift: 当前 NES walker 的全局 shift。
             单样本时是 scalar；
             批量时可以是 (batch,) 或 (batch, 1, 1)。

    返回：
    - 单样本：标量 complex
    - 批量：形状 (batch,)
    """
    is_single = (x.ndim == 1)

    if is_single:
        x = x[None, :]

    batch_size = x.shape[0]

    shift = jnp.asarray(shift)

    if shift.ndim == 0:
        shift_batch = jnp.broadcast_to(shift, (batch_size,))
    else:
        shift_batch = shift.reshape(-1)

    def _single_hpsi_scaled(x_single, c_single):
        # x_primes: (n_conn, n_spin)
        # mels:     (n_conn,)
        x_primes, mels = ha.get_conn_padded(x_single)

        # logψ_j(x')
        log_psi_vals = single_machine(params, x_primes)

        # 关键：这里直接缩放，不再 raw exp(logψ)
        psi_vals_scaled = jnp.exp(log_psi_vals - c_single)

        return jnp.sum(mels * psi_vals_scaled)

    H_psi_batch = jax.vmap(_single_hpsi_scaled)(x, shift_batch)

    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
    
def Ham_Psi(ha, single_machine_list, total_params, x):
    K = len(single_machine_list)
    # ======================
    # 核心：单样本 与 批处理 自动兼容
    # ======================
    if x.ndim == 2:
        # 输入形状：(K, n_spin) → 单个扩展态 → 返回 (K,K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]  # 单态：(4,)
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        return _single_HamPsi(x)

    elif x.ndim == 3:
        # 输入形状：(batch, K, n_spin) → 批量 → 返回 (batch, K, K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        # 自动批处理！
        return jax.vmap(_single_HamPsi)(x)

    else:
        raise ValueError(f"不支持的输入形状: {x.shape}")

def Ham_Psi_scaled(
    ha,
    single_machine_list,
    total_params,
    x,
    shift,
):
    """
    稳定版总 Ansatz 哈密顿量作用矩阵。

    计算：
        HPsi_stable[i, j]
        = e^{-shift} Hψ_j(x^i)
        = Σ_{x'} H[x^i, x'] * exp(logψ_j(x') - shift)

    输入：
    - x:
        单个 NES walker: (K, n_spin)
        批量 NES walkers: (batch, K, n_spin)

    - shift:
        单样本：scalar
        批量：shape 可以是 (batch,) 或 (batch, 1, 1)

    输出：
    - 单样本: (K, K)
    - 批量:   (batch, K, K)
    """
    K = len(single_machine_list)

    def _single_HamPsi_scaled(x_single, c_single):
        """
        x_single: (K, n_spin)
        c_single: scalar
        """
        HPsi = jnp.zeros((K, K), dtype=jnp.complex64)

        for i in range(K):
            xi = x_single[i]

            for j in range(K):
                machine_j = single_machine_list[j]
                params_j = total_params["single_ansatz_list"][j]

                val = Ham_psi_scaled(
                    ha=ha,
                    single_machine=machine_j,
                    params=params_j,
                    x=xi,
                    shift=c_single,
                )

                HPsi = HPsi.at[i, j].set(val)

        return HPsi

    if x.ndim == 2:
        # 单个 NES walker: (K, n_spin)
        shift_scalar = jnp.asarray(shift).reshape(-1)[0]
        return _single_HamPsi_scaled(x, shift_scalar)

    elif x.ndim == 3:
        # 批量 NES walkers: (batch, K, n_spin)
        shift_batch = jnp.asarray(shift).reshape(-1)

        return jax.vmap(
            _single_HamPsi_scaled,
            in_axes=(0, 0)
        )(x, shift_batch)

    else:
        raise ValueError(f"不支持的输入形状: {x.shape}")



def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 添加正则化项，防止矩阵奇异
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

def NES_loss_energy_stable(
    ha,
    total_matrix_machine,
    total_max_machine,
    single_machine_list,
    total_params,
    x,
    return_aux: bool = False,
):
    """
    稳定版 NES 局域能量矩阵计算。

    计算目标：
        E_L(X) = Psi(X)^{-1} · H Psi(X)

    稳定化做法：
        Psi_stable = exp(L - shift)
        HPsi_stable = exp(-shift) · HPsi

    其中 shift = max_{i,j} Re(L_ij)，
    由 total_max_machine 给出。

    参数：
    - ha:
        NetKet 离散 Hamiltonian
    - total_matrix_machine:
        稳定版矩阵机器函数，返回 L_stable = L - shift
    - total_max_machine:
        返回 shift，也就是 max real(L)
    - single_machine_list:
        每个单态 ansatz 的 machine 列表
    - total_params:
        总 ansatz 参数
    - x:
        单个 NES walker: (K, n_spin)
        或 batch: (batch, K, n_spin)
    - return_aux:
        是否返回调试信息

    返回：
    - 默认：
        loss_batch, E_L_matrix

    - 若 return_aux=True：
        loss_batch, E_L_matrix, aux
    """

    # 1. 稳定化后的 log 矩阵
    #    L_stable = L - shift
    L_stable = total_matrix_machine(total_params, x)

    # 2. 当前 walker 的 shift
    #    这里的 shift 应该是 max(real(L))，不是 complex L 的 max
    shift = total_max_machine(total_params, x)

    # 3. 构造稳定化后的 Psi 矩阵
    #    Psi_stable = exp(L - shift)
    Psi_Matrix_stable = jnp.exp(L_stable)

    # 4. 构造稳定化后的 HPsi 矩阵
    #    HPsi_stable[i,j] = sum_x' H[x_i,x'] * exp(logψ_j(x') - shift)
    #
    #    注意：
    #    这里不能用旧的 Ham_Psi。
    #    必须使用你刚刚新增的 Ham_Psi_scaled。
    HPsi_stable = Ham_Psi_scaled(
        ha=ha,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x,
        shift=shift,
    )

    # 5. 求局域能量矩阵
    #    E_L = Psi_stable^{-1} HPsi_stable
    E_L_matrix = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)

    # 6. loss 是 trace(E_L) 的实部
    loss_batch = jnp.real(
        jnp.trace(E_L_matrix, axis1=-2, axis2=-1)
    )

    if not return_aux:
        return loss_batch, E_L_matrix

    # 7. 调试信息：先返回，不在这里强行 mask
    #    mask 应该在下一步 nes_vmc_gradient_stable 里处理
    valid = (
        jnp.all(jnp.isfinite(Psi_Matrix_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(HPsi_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(E_L_matrix), axis=(-2, -1))
        & jnp.isfinite(loss_batch)
    )

    aux = {
        "L_stable": L_stable,
        "shift": shift,
        "Psi_Matrix_stable": Psi_Matrix_stable,
        "HPsi_stable": HPsi_stable,
        "valid": valid,
        "cond_Psi": jax.vmap(jnp.linalg.cond)(Psi_Matrix_stable)
        if Psi_Matrix_stable.ndim == 3
        else jnp.linalg.cond(Psi_Matrix_stable),
    }

    return loss_batch, E_L_matrix, aux




def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. 批量局域能量矩阵
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    #print(f'E_L_batch.shape={E_L_batch.shape}')
    
    E_L_centered = E_L_batch - E_L_mean
    
    tr_centered =  jnp.trace(E_L_centered, axis1=-2, axis2=-1) 

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. 计算 ∇logΨs
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 5. 核心加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean



def nes_vmc_gradient_stable(
    ha: nk.operator.DiscreteOperator,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
    total_params,
    x_batch,
    min_valid_ratio: float = 0.25,
    return_aux: bool = False,
):
    """
    稳定版 NES-VMC 梯度计算。

    与旧版 nes_vmc_gradient 的区别：
    1. 调用 NES_loss_energy_stable，而不是 NES_loss_energy。
    2. 使用 valid mask 过滤 NaN / inf walker。
    3. E_L_mean、loss_mean、gradient 都用 masked mean。
    4. invalid walker 的权重强制设为 0。
    5. 返回 aux，方便训练循环判断是否 skip update。

    参数：
    - ha:
        NetKet Hamiltonian
    - total_matrix_machine:
        create_machine_matrix_stable 得到的函数，返回 L_stable
    - total_max_machine:
        create_machine_max_stable 得到的函数，返回 shift
    - total_machine:
        create_machine_stable 得到的函数，返回 raw logΨ
    - single_machine_list:
        单态 machine 列表
    - total_params:
        总参数
    - x_batch:
        shape = (batch, K, n_spin)
    - min_valid_ratio:
        有效 walker 比例低于这个值时，建议训练循环 skip update
    - return_aux:
        是否返回辅助信息

    返回：
    - 默认：
        grad, loss_mean, E_L_mean

    - return_aux=True:
        grad, loss_mean, E_L_mean, aux
    """

    # ============================================================
    # 1. 稳定版局域能量矩阵
    # ============================================================
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_batch,
        return_aux=True,
    )

    valid = loss_aux["valid"]
    valid = jax.lax.stop_gradient(valid)

    batch_size = x_batch.shape[0]
    n_valid = jnp.sum(valid.astype(jnp.float32))
    valid_ratio = n_valid / batch_size

    # ============================================================
    # 2. masked mean: E_L_mean
    # ============================================================
    E_L_mean = _masked_mean_batch(E_L_batch, valid)

    # ============================================================
    # 3. 中心化 local energy matrix
    # ============================================================
    E_L_safe = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0)

    E_L_centered = E_L_safe - E_L_mean

    tr_centered = jnp.trace(
        E_L_centered,
        axis1=-2,
        axis2=-1,
    )

    # invalid walker 的权重设为 0
    tr_centered = jnp.where(valid, tr_centered, 0.0)

    # ============================================================
    # 4. 计算每个样本的 ∇logΨ
    # ============================================================
    grad_logPsi = jax.grad(
        total_machine,
        argnums=0,
        holomorphic=True,
    )

    vmap_grad_logPsi = jax.vmap(
        grad_logPsi,
        in_axes=(None, 0),
    )

    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # ============================================================
    # 5. masked weighted mean
    # ============================================================
    def weight_and_masked_mean(grad_component):
        """
        grad_component: shape = (batch, ...)
        """
        # 清掉 NaN / inf 梯度分量
        grad_safe = jnp.where(
            jnp.isfinite(grad_component),
            grad_component,
            0.0,
        )

        weights = tr_centered.reshape(
            (-1,) + (1,) * (grad_component.ndim - 1)
        )

        valid_f = valid.astype(jnp.float32).reshape(
            (-1,) + (1,) * (grad_component.ndim - 1)
        )

        weighted = weights * jnp.conj(grad_safe) * valid_f

        denom = jnp.maximum(n_valid, 1.0)

        return jnp.sum(weighted, axis=0) / denom

    grad = jax.tree.map(weight_and_masked_mean, dlogPsi_batch)

    # ============================================================
    # 6. masked loss mean
    # ============================================================
    loss_mean = _masked_mean_batch(loss_batch, valid)

    # ============================================================
    # 7. 诊断信息
    # ============================================================
    grad_flat, _ = ravel_pytree(grad)

    grad_finite = jnp.all(jnp.isfinite(grad_flat))
    loss_finite = jnp.isfinite(loss_mean)
    enough_valid = valid_ratio >= min_valid_ratio

    aux = {
        "valid": valid,
        "n_valid": n_valid,
        "valid_ratio": valid_ratio,
        "grad_finite": grad_finite,
        "loss_finite": loss_finite,
        "enough_valid": enough_valid,
        "should_skip": ~(grad_finite & loss_finite & enough_valid),
        "loss_aux": loss_aux,
    }

    if return_aux:
        return grad, loss_mean, E_L_mean, aux

    return grad, loss_mean, E_L_mean


#@partial(jax.jit, static_argnames=("machine",))
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）/ F 矩阵
    
    QGT 定义：
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    
    这就是 NetKet SR 的核心
    
    参数：
    - machine: 波函数机器
    - params: 网络参数
    - sigma: 样本 (n_samples, n_orbitals)
    - diag_shift: 对角线正则化参数 λ
    
    返回：
    - qgt_reg: 正则化后的 QGT 矩阵 (n_params, n_params)
    - unravel_fn: 用于将展平的向量恢复为 PyTree 结构的函数
    """
    n_samples = sigma.shape[0]
    
    # 步骤 1: 计算每个样本的 ∇log ψ
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    # grad_matrix 是 PyTree，每个元素形状为 (n_samples, ...)
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    # 步骤 2: 将 PyTree 展平为矩阵 (n_samples, n_params)
    grad_flat, unravel_fn = ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    
    # 步骤 3: 中心化（减去均值）
    # 这对应 QGT 定义中的第二项：- ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)  # (1, n_params)
    grad_centered = grad_flat - grad_mean  # (n_samples, n_params)
    
    # 步骤 4: 计算 QGT = (1/N) * Σ ∇log ψ* ∇log ψ^T
    # 注意：对于复数，需要使用共轭
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    
    # 步骤 5: 添加正则化
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    
    return qgt_reg, unravel_fn

def sampler_info(samples:jnp.array,K:int):
    test_samples = np.array(samples.reshape(-1, 4*K))
    count = Counter(tuple(each_row.tolist()) for each_row in test_samples)
    for tpl, count_ in count.items():
        print(f"元组 {tpl} 出现了 {count_} 次")
    return count



@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    edges: jnp.ndarray
    K: int = nk.utils.struct.static_field()
    single_size: int = nk.utils.struct.static_field()

    def _check_duplicate(self, sigma_ext):
        """NES约束：检测任意两个子组态重复
        兼容一维单样本(返回标量) / 二维批量(返回batch数组)
        """
        one_d_input = (sigma_ext.ndim == 1)
        if one_d_input:
            sigma_ext = sigma_ext[None, :]
        
        batch_dim = sigma_ext.shape[0]
        sub = sigma_ext.reshape((batch_dim, self.K, self.single_size))
        # 全部子组态两两比对
        pair_equal = jnp.all(sub[:, :, None, :] == sub[:, None, :], axis=-1)
        diag_mask = jnp.eye(self.K, dtype=jnp.bool_)[None, :, :]
        off_diag_dup = jnp.where(diag_mask, False, pair_equal)
        batch_dup = jnp.any(off_diag_dup, axis=(-2, -1))
        
        if one_d_input:
            return batch_dup.squeeze()
        return batch_dup

    # 修复：补齐完整7个形参：self, sampler, machine, parameters, state, rng, sigma
    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """跃迁规则"""
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:,0], sel_e[:,1]

        sigma_cand = sigma.at[jnp.arange(batch_size),i].set(sigma[jnp.arange(batch_size),j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size),j].set(sigma[jnp.arange(batch_size),i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        """随机态生成（完全不变）"""
        sigma_shape = state.σ.shape
        hilbert = sampler.hilbert

        def gen_single(key):
            max_tries = 100
            def cond(c): 
                return (c[0] < max_tries) & c[2]
            
            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)
                s = hilbert.random_state(k_new)
                is_dup = self._check_duplicate(s)  # 一维输入自动返回标量
                return (tries + 1, k, is_dup, s)
            
            # 初始值 c[2] = True（标量布尔值），匹配while_loop
            init_c = (0, key, True, hilbert.random_state(key))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            return jax.lax.cond(is_dup, lambda: hilbert.random_state(key), lambda: s)
        
        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)
     
import time
# ======================
# 超参数
# ======================

if __name__ == '__main__':
    N_CHAINS = 16 
    N_WARMUP = 32
    N_SAMPLES_PER_CHAIN = 100
    SWEEP_SIZE = 32
    N_ITER =100

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
        #samples = samples.reshape(-1,2,4)

        # 3. 计算能量和自然梯度（逻辑和原代码一致）
        grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                    graphdef=graphdef,
                                                    params=params,
                                                    x_batch=samples.reshape(-1,2,4))
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
        if step % 5 == 0 or step == N_ITER - 1:
            eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
            history['step'].append(step)
            print(f"Step {step:3d} | Loss: {loss_mean}｜eig_vals: {eig_vals}")

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
