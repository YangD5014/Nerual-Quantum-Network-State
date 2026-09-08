"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现

本文件实现基于原生 JAX 和部分 NetKet 的 NES-VMC 算法，用于计算量子多体系统的激发态能量。
"""
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
import flax.nnx as nnx
import optax
from functools import partial
from jax import flatten_util
import orbax.checkpoint as ocp
from typing import Union
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import jax
import jax.lax as lax
import time

from functools import partial
from jax.flatten_util import ravel_pytree
from collections import Counter


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

class NESTotalAnsatz_gauge_stable_full(nnx.Module):
    """
    完整版 gauge-fixed NES total ansatz。
    同时修复列规范（column gauge）和行规范（row gauge）自由度。

    列规范修复：L[i,j] -= logψ_j(ref)，消除每列的常数偏移
    行规范修复：L[i,j] -= mean_j(L[i,j])，消除每行的常数偏移

    返回四元组：
        log_Psi_gauge:    采样器用的完整 logΨ（含 K*shift + det correction）
        L_stable:         稳定化后的矩阵（供 loss 使用）
        shift:            max(Re(L_centered))
        L_gauge:          原始 gauge-fixed 矩阵（供诊断用）
    """
    def __init__(
        self,
        n_spin_orbitals: int,
        n_states: int,
        hidden_dim: int,
        ref_state,
        *,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.ref_state = jnp.asarray(ref_state, dtype=jnp.complex64)

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


    def _forward_single(self, x_single: jax.Array):
        x_single = x_single.reshape(self.K, self.n_spin)

        # ========== 1. 列规范修复（保留梯度）==========
        ref_logs = []
        for j in range(self.K):
            ans_j = self.single_ansatz_list[j]
            ref_val = ans_j(self.ref_state)
            ref_logs.append(ref_val)
        ref_logs = jnp.array(ref_logs, dtype=jnp.complex64)

        L_centered = jnp.zeros((self.K, self.K), dtype=jnp.complex64)
        for i in range(self.K):
            for j in range(self.K):
                log_x = self.single_ansatz_list[j](x_single[i])
                L_centered = L_centered.at[i, j].set(log_x - ref_logs[j])

        # ========== 2. 行规范修复（保留梯度）==========
        # row_mean[i] = mean_j(L_centered[i,j])
        row_mean = jnp.mean(L_centered, axis=1, keepdims=True)  # (K,1) 含梯度
        L_rowfixed_full = L_centered - row_mean                 # (K,K) 含梯度

        # ========== 3. 带梯度的完整 logΨ（供梯度计算使用）==========
        # shift_full 无 stop_gradient，使 log_Psi_gauge 获得完整梯度
        shift_full = jnp.max(jnp.real(L_rowfixed_full), axis=(-2, -1))
        L_stable_full = L_rowfixed_full - shift_full
        Psi_stable_full = jnp.exp(L_stable_full)
        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable_full)
        log_Psi_stable_full = log_abs_det + 1j * jnp.angle(sign)
        det_correction_full = jnp.sum(row_mean[:, 0])
        # 完整 logΨ = slogdet(Ψ_stable) + K*shift + sum(row_mean)
        log_Psi_gauge = log_Psi_stable_full + self.K * shift_full + det_correction_full

        # ========== 4. 数值稳定化版本（stop_gradient，供 loss 用）==========
        shift = jax.lax.stop_gradient(shift_full)
        row_mean_stop = jax.lax.stop_gradient(row_mean)
        L_rowfixed = L_centered - row_mean_stop
        L_stable = L_rowfixed - shift
        Psi_stable = jnp.exp(L_stable)
        sign_s, log_abs_det_s = jnp.linalg.slogdet(Psi_stable)
        log_Psi_stable_s = log_abs_det_s + 1j * jnp.angle(sign_s)
        det_correction_s = jax.lax.stop_gradient(jnp.sum(row_mean_stop[:, 0]))
        log_Psi_gauge_stable = log_Psi_stable_s + self.K * shift + det_correction_s

        return log_Psi_gauge, L_stable, shift, L_rowfixed

    def __call__(self, x: jax.Array):
        """自动兼容 1d/2d单样本 / 2d/3d批量输入"""
        def single_wrapper(x_mat):
            return self._forward_single(x_mat)

        if x.ndim == 1:
            flat_size = self.K * self.n_spin
            assert x.shape[0] == flat_size
            x_mat = x.reshape(self.K, self.n_spin)
            return single_wrapper(x_mat)
        elif x.ndim == 2:
            if x.shape == (self.K, self.n_spin):
                return single_wrapper(x)
            else:
                batch = x.reshape(-1, self.K, self.n_spin)
                return jax.vmap(single_wrapper)(batch)
        elif x.ndim == 3:
            return jax.vmap(single_wrapper)(x)
        else:
            raise ValueError(f"不支持输入维度 x.shape={x.shape}")


class NESTotalAnsatz_gauge_stable(nnx.Module):
    def __init__(
        self,
        n_spin_orbitals: int,
        n_states: int,
        hidden_dim: int,
        ref_state,
        *,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.ref_state = jnp.asarray(ref_state, dtype=jnp.complex64)
        self.single_ansatz_list = nnx.List()

        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            # 修复：关键字 rngs 匹配 SingleStateAnsatz 入参
            ans = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ans)

    def _forward_single(self, x_single: jax.Array):
        x_single = x_single.reshape(self.K, self.n_spin)

        # 预计算所有列参考态 log，避免双重循环重复调用网络，加速计算
        ref_logs = []
        for j in range(self.K):
            ans_j = self.single_ansatz_list[j]
            ref_val = jax.lax.stop_gradient(ans_j(self.ref_state))
            ref_logs.append(ref_val)
        ref_logs = jnp.array(ref_logs, dtype=jnp.complex64)

        # 构造 gauge-fixed L 矩阵 L_ij = logψ_j(x_i) - stop_grad(logψ_j(ref))
        L = jnp.zeros((self.K, self.K), dtype=jnp.complex64)
        for i in range(self.K):
            for j in range(self.K):
                ans_j = self.single_ansatz_list[j]
                log_x = ans_j(x_single[i])
                log_ref = ref_logs[j]
                L = L.at[i, j].set(log_x - log_ref)

        # 数值稳定化 L - max(Re(L))
        shift = jnp.max(jnp.real(L), axis=(-2, -1))
        shift = jax.lax.stop_gradient(shift)
        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        # 稳定行列式对数
        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
        # 采样专用完整logΨ，补回全局shift消除样本基准偏移
        log_Psi_gauge = log_Psi_stable + self.K * shift

        # 返回四元组供外部machine分发
        # 1. 采样专用完整logΨ，补回全局shift消除样本基准偏移->log_Psi_gauge
        # 2. 稳定化后的 L 矩阵 -> Lij 稳定版
        # 3. 全局shift -> L_max
        # 4. 减去参考态的L矩阵 -> Lij 原始版 = logψ_j(x_i)-logψ_j(ref_state)
        return log_Psi_gauge, L_stable, shift, L

    def __call__(self, x: jax.Array):
        """自动兼容 1d/2d单样本 / 2d/3d批量输入"""
        def single_wrapper(x_mat):
            return self._forward_single(x_mat)

        # 分支处理各类输入形状
        if x.ndim == 1:
            flat_size = self.K * self.n_spin
            assert x.shape[0] == flat_size
            x_mat = x.reshape(self.K, self.n_spin)
            return single_wrapper(x_mat)
        elif x.ndim == 2:
            if x.shape == (self.K, self.n_spin):
                # 单个K×K矩阵构型
                return single_wrapper(x)
            else:
                # 批量展平 (batch, K*n_spin)
                batch = x.reshape(-1, self.K, self.n_spin)
                return jax.vmap(single_wrapper)(batch)
        elif x.ndim == 3:
            # 批量矩阵 (batch, K, n_spin)
            return jax.vmap(single_wrapper)(x)
        else:
            raise ValueError(f"不支持输入维度 x.shape={x.shape}")
        





class NESTotalAnsatz_gauge_stable_frozen(nnx.Module):
    def __init__(
        self,
        n_spin_orbitals: int,
        n_states: int,
        hidden_dim: int,
        ref_state,
        *,
        rngs: nnx.Rngs,
        freeze_indices: list[int] = None  # 新增：初始冻结下标列表
    ):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.ref_state = jnp.asarray(ref_state, dtype=jnp.complex64)
        # 保存冻结下标，默认空列表=不冻结任何ansatz
        self.freeze_indices = freeze_indices if freeze_indices is not None else []
        self.single_ansatz_list = nnx.List()

        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ans = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ans)

    def set_freeze(self, new_freeze: list[int]):
        """外部接口：训练循环动态切换冻结集合"""
        self.freeze_indices = new_freeze

    def _forward_single(self, x_single: jax.Array, freeze_indices: list[int]):
        x_single = x_single.reshape(self.K, self.n_spin)

        # 预计算参考态log，冻结j则截断ref_val梯度
        ref_logs = []
        for j in range(self.K):
            ans_j = self.single_ansatz_list[j]
            ref_val = ans_j(self.ref_state)
            # 动态截断冻结网络的参考态梯度
            if j in freeze_indices:
                ref_val = lax.stop_gradient(ref_val)
            ref_logs.append(ref_val)
        ref_logs = jnp.array(ref_logs, dtype=jnp.complex64)

        # 构造 gauge-fixed L 矩阵
        L = jnp.zeros((self.K, self.K), dtype=jnp.complex64)
        for i in range(self.K):
            for j in range(self.K):
                ans_j = self.single_ansatz_list[j]
                log_x = ans_j(x_single[i])
                # 动态截断冻结网络构型输出梯度
                if j in freeze_indices:
                    log_x = lax.stop_gradient(log_x)
                log_ref = ref_logs[j]
                L = L.at[i, j].set(log_x - log_ref)

        # 数值稳定化逻辑完全保留原样
        shift = jnp.max(jnp.real(L), axis=(-2, -1))
        shift = lax.stop_gradient(shift)
        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
        log_Psi_gauge = log_Psi_stable + self.K * shift

        return log_Psi_gauge, L_stable, shift, L

    def __call__(self, x: jax.Array, freeze_indices: list[int] = None):
        """
        自动兼容 1d/2d单样本 / 2d/3d批量输入
        :param x: 构型数组
        :param freeze_indices: 临时指定冻结列表；不传则使用 self.freeze_indices
        """
        # 优先使用传入的冻结列表，否则用类内部保存的
        current_freeze = freeze_indices if freeze_indices is not None else self.freeze_indices

        def single_wrapper(x_mat):
            return self._forward_single(x_mat, current_freeze)

        # 原有维度分发逻辑完全不变
        if x.ndim == 1:
            flat_size = self.K * self.n_spin
            assert x.shape[0] == flat_size
            x_mat = x.reshape(self.K, self.n_spin)
            return single_wrapper(x_mat)
        elif x.ndim == 2:
            if x.shape == (self.K, self.n_spin):
                return single_wrapper(x)
            else:
                batch = x.reshape(-1, self.K, self.n_spin)
                return jax.vmap(single_wrapper)(batch)
        elif x.ndim == 3:
            return jax.vmap(single_wrapper)(x)
        else:
            raise ValueError(f"不支持输入维度 x.shape={x.shape}")
    
       
        
def get_ccsd_excitations_and_sampler_edges_from_hf(hf_state):
    """
    返回：
    1. sampler_edges: 给 NetKet Graph 用，格式为 [(i, a), ...]
    2. singles:       物理 single excitation，格式为 [((i, a),), ...]
    3. doubles:       物理 double excitation，格式为 [((i,a), (j,b)), ...]
    """
    hf_state = np.asarray(hf_state)
    n_spin_orbitals = hf_state.size
    assert n_spin_orbitals % 2 == 0

    nmo = n_spin_orbitals // 2

    sampler_edges = []

    # alpha block: 0 ~ nmo-1
    alpha_sites = np.arange(0, nmo)
    alpha_occ = alpha_sites[hf_state[alpha_sites] == 1]
    alpha_vir = alpha_sites[hf_state[alpha_sites] == 0]

    for i in alpha_occ:
        for a in alpha_vir:
            sampler_edges.append((int(i), int(a)))

    # beta block: nmo ~ 2*nmo-1
    beta_sites = np.arange(nmo, 2 * nmo)
    beta_occ = beta_sites[hf_state[beta_sites] == 1]
    beta_vir = beta_sites[hf_state[beta_sites] == 0]

    for i in beta_occ:
        for a in beta_vir:
            sampler_edges.append((int(i), int(a)))

    # 物理意义上的 single excitation
    singles = [(edge,) for edge in sampler_edges]

    # 物理意义上的 double excitation
    doubles = []
    for move1, move2 in combinations(sampler_edges, 2):
        i, a = move1
        j, b = move2

        # 不能动同一个电子，也不能占到同一个虚轨道
        if i != j and a != b:
            doubles.append((move1, move2))

    return sampler_edges, singles, doubles


# 原版
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
# 原版
def create_single_machine(model: SingleStateAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state
# 原版
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


# ====================== 新增 Stable 系列包装（适配 NESTotalAnsatz_stable） ======================
def create_machine_stable(model: NESTotalAnsatz_stable):
    """
    【采样器专用】
    输入stable模型三元输出，内部还原真实原始 logΨ_raw = log_Psi_stable + K * L_max
    采样器直接调用此函数，不受全局偏移影响
    """
    graphdef, state = nnx.split(model)
    K = model.K  # 从模型读取态数量

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        # 还原无偏移原始lnΨ，给Metropolis采样使用
        log_psi_raw = log_stable + K * L_max
        return log_psi_raw

    return machine, graphdef, state

def create_machine_gauge_stable(model: Union[NESTotalAnsatz_gauge_stable,NESTotalAnsatz_gauge_stable_full]):
    'Gauge Fixing- 使用列规范、行规范的版本'
    
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)

        # 采样器用 gauge-fixed logΨ
        return log_Psi_gauge

    return machine, graphdef, state


def create_machine_matrix_stable(model: Union[NESTotalAnsatz_stable,NESTotalAnsatz_gauge_stable_full]):
    """【损失函数专用】输出稳定化 L_stable = L - L_max"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        return L_stable

    return machine, graphdef, state

def create_machine_max_stable(model: NESTotalAnsatz_stable):
    """【损失函数专用】输出每样本全局最大值 L_max，用于HamΨ尺度对齐"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        return L_max

    return machine, graphdef, state


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

def _masked_mean_batch(x, valid, eps=1e-12):
    """
    对 batch 维度做 masked mean。

    x:     (batch, ...) 或 (batch,)
    valid: (batch,)
    """
    valid = jax.lax.stop_gradient(valid)
    valid_f = valid.astype(jnp.float32)

    # 把 NaN / inf 清掉，防止 invalid walker 污染
    x_safe = jnp.where(jnp.isfinite(x), x, 0.0)

    if x.ndim == 1:
        numerator = jnp.sum(x_safe * valid_f)
    else:
        shape = (valid_f.shape[0],) + (1,) * (x.ndim - 1)
        valid_f_reshaped = valid_f.reshape(shape)
        numerator = jnp.sum(x_safe * valid_f_reshaped, axis=0)

    denominator = jnp.maximum(jnp.sum(valid_f), eps)

    return numerator / denominator


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

def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 添加正则化项，防止矩阵奇异
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

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


def sampler_info(samples: jnp.array, K: int, hilbert_size: int) -> Counter:
    # jax数组转numpy，reshape成 (batch, K, hilbert_size)
    test_samples = np.array(samples.reshape(-1, K, hilbert_size))
    # each_row: (K, hilbert_size) 二维数组，先展平再转可哈希一维元组
    count = Counter(tuple(each_row.flatten().tolist()) for each_row in test_samples)
    
    for tpl, count_ in count.items():
        print(f"一维展平元组 {tpl} 出现了 {count_} 次")
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


import logging
import optax
def create_gauge_fixed_total_machines(total_model, ref_state):
    """
    Gauge-fixed total machine wrappers.

    支持输入:
        sigma.shape = (K * n_spin,)             单个 flattened walker
        sigma.shape = (K, n_spin)               单个 matrix walker
        sigma.shape = (batch, K * n_spin)       NetKet batch flattened walkers
        sigma.shape = (batch, K, n_spin)        batch matrix walkers

    返回:
        total_machine(params, sigma)        -> logΨ_centered(sigma)
        total_matrix_machine(params, sigma) -> L_stable
        total_max_machine(params, sigma)    -> shift
    """

    graphdef, state = nnx.split(total_model)

    K = total_model.K
    n_spin = total_model.n_spin
    flat_size = K * n_spin

    ref_state = jnp.asarray(ref_state)

    def _compute_L_centered_single(m, x_single):
        """
        x_single: shape (K, n_spin)

        返回:
            L_centered: shape (K, K)
            L[i, j] = logψ_j(x_i) - logψ_j(ref)
        """
        cols = []

        for j in range(K):
            ansatz_j = m.single_ansatz_list[j]

            # x_single: (K, n_spin)
            # log_col: (K,)
            log_col = ansatz_j(x_single)

            # log_ref: scalar
            log_ref = ansatz_j(ref_state)

            # 不要 stop_gradient(log_ref)
            log_col_centered = log_col - log_ref

            cols.append(log_col_centered)

        L_centered = jnp.stack(cols, axis=1)  # (K, K)
        return L_centered

    def _stable_from_L(L):
        """
        L: shape (K, K)
        """
        shift = jnp.max(jnp.real(L))
        shift = jax.lax.stop_gradient(shift)

        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_det_stable = log_abs_det + 1j * jnp.angle(sign)

        log_det_centered = log_det_stable + K * shift

        return log_det_centered, L_stable, shift

    def _one_from_matrix(m, x_single):
        """
        x_single: shape (K, n_spin)
        """
        L = _compute_L_centered_single(m, x_single)
        return _stable_from_L(L)

    def _as_single_matrix(sigma):
        """
        把单个 walker 统一成 (K, n_spin)。

        只处理:
            (flat_size,)
            (K, n_spin)
        """
        sigma = jnp.asarray(sigma)

        if sigma.ndim == 1:
            assert sigma.shape[0] == flat_size, (
                f"single flat sigma.shape={sigma.shape}, expected ({flat_size},)"
            )
            return sigma.reshape(K, n_spin)

        elif sigma.ndim == 2:
            assert sigma.shape == (K, n_spin), (
                f"single matrix sigma.shape={sigma.shape}, expected {(K, n_spin)}"
            )
            return sigma

        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to single walker")

    def _as_batch_matrix(sigma):
        """
        把 batch walker 统一成 (batch, K, n_spin)。

        支持:
            (batch, flat_size)
            (batch, K, n_spin)
        """
        sigma = jnp.asarray(sigma)

        if sigma.ndim == 2:
            # NetKet sampler.init_state / sample 最常见:
            # sigma.shape = (n_batches, hi_ext.size) = (batch, K*n_spin)
            assert sigma.shape[-1] == flat_size, (
                f"batch flat sigma.shape={sigma.shape}, expected last dim {flat_size}"
            )
            return sigma.reshape(-1, K, n_spin)

        elif sigma.ndim == 3:
            assert sigma.shape[1:] == (K, n_spin), (
                f"batch matrix sigma.shape={sigma.shape}, expected (-1, {K}, {n_spin})"
            )
            return sigma

        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to batch walkers")

    @jax.jit
    def total_machine(params, sigma):
        """
        给 NetKet sampler 用。
        必须支持:
            sigma: (batch, flat_size)
        并返回:
            logΨ: (batch,)
        """
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)

        # 单个 flattened walker: (flat_size,)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            log_det_centered, _, _ = _one_from_matrix(m, x_single)
            return log_det_centered

        # 二维有两种可能:
        #   (K, n_spin)       单个 matrix walker
        #   (batch, flat_size) NetKet batch flattened walkers
        elif sigma.ndim == 2:
            if sigma.shape == (K, n_spin):
                x_single = _as_single_matrix(sigma)
                log_det_centered, _, _ = _one_from_matrix(m, x_single)
                return log_det_centered
            else:
                x_batch = _as_batch_matrix(sigma)

                def _one(x_single):
                    log_det_centered, _, _ = _one_from_matrix(m, x_single)
                    return log_det_centered

                return jax.vmap(_one)(x_batch)

        # batch matrix walkers: (batch, K, n_spin)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)

            def _one(x_single):
                log_det_centered, _, _ = _one_from_matrix(m, x_single)
                return log_det_centered

            return jax.vmap(_one)(x_batch)

        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_matrix_machine(params, sigma):
        """
        给 loss 用。
        返回 L_stable。

        支持:
            sigma: (flat_size,)       -> (K, K)
            sigma: (K, n_spin)        -> (K, K)
            sigma: (batch, flat_size) -> (batch, K, K)
            sigma: (batch, K, n_spin) -> (batch, K, K)
        """
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)

        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            _, L_stable, _ = _one_from_matrix(m, x_single)
            return L_stable

        elif sigma.ndim == 2:
            if sigma.shape == (K, n_spin):
                x_single = _as_single_matrix(sigma)
                _, L_stable, _ = _one_from_matrix(m, x_single)
                return L_stable
            else:
                x_batch = _as_batch_matrix(sigma)

                def _one(x_single):
                    _, L_stable, _ = _one_from_matrix(m, x_single)
                    return L_stable

                return jax.vmap(_one)(x_batch)

        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)

            def _one(x_single):
                _, L_stable, _ = _one_from_matrix(m, x_single)
                return L_stable

            return jax.vmap(_one)(x_batch)

        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_max_machine(params, sigma):
        """
        返回 shift。
        """
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)

        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            _, _, shift = _one_from_matrix(m, x_single)
            return shift

        elif sigma.ndim == 2:
            if sigma.shape == (K, n_spin):
                x_single = _as_single_matrix(sigma)
                _, _, shift = _one_from_matrix(m, x_single)
                return shift
            else:
                x_batch = _as_batch_matrix(sigma)

                def _one(x_single):
                    _, _, shift = _one_from_matrix(m, x_single)
                    return shift

                return jax.vmap(_one)(x_batch)

        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)

            def _one(x_single):
                _, _, shift = _one_from_matrix(m, x_single)
                return shift

            return jax.vmap(_one)(x_batch)

        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    return total_machine, total_matrix_machine, total_max_machine, graphdef, state


def flatten_batched_pytree(pytree, n_samples):
    """
    pytree: 每个 leaf 的第 0 维都是 n_samples
    返回:
        mat: (n_samples, n_params)
    """
    leaves = jax.tree.leaves(pytree)

    flat_leaves = [
        jnp.reshape(leaf, (n_samples, -1))
        for leaf in leaves
    ]

    return jnp.concatenate(flat_leaves, axis=1)

def compute_qgt_fixed(machine, params, sigma, diag_shift=0.1, return_aux=False):
    n_samples = sigma.shape[0]

    def compute_grad_for_sample(s):
        return jax.grad(
            lambda p: machine(p, s),
            holomorphic=True
        )(params)

    grad_tree_batch = jax.vmap(compute_grad_for_sample)(sigma)

    # 关键修正：不要 ravel_pytree 后 reshape
    O = flatten_batched_pytree(grad_tree_batch, n_samples)

    O_mean = jnp.mean(O, axis=0, keepdims=True)
    O_centered = O - O_mean

    S = (jnp.conj(O_centered).T @ O_centered) / n_samples

    # 数值对称化，防止浮点误差破坏 Hermitian
    S = 0.5 * (S + jnp.conj(S.T))

    I = jnp.eye(S.shape[0], dtype=S.dtype)
    S_reg = S + diag_shift * I

    if not return_aux:
        return S_reg

    eigvals = jnp.linalg.eigvalsh(S.real)

    aux = {
        "O_norm_mean": jnp.mean(jnp.linalg.norm(O, axis=1)),
        "O_norm_max": jnp.max(jnp.linalg.norm(O, axis=1)),
        "S_trace": jnp.real(jnp.trace(S)),
        "S_eig_min": eigvals[0],
        "S_eig_max": eigvals[-1],
        "S_cond": eigvals[-1] / (eigvals[0] + 1e-12),
    }

    return S_reg, aux


def make_qgt_fn(machine):
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift):
        n_samples = sigma.shape[0]

        grad_tree_batch = vmap_grad_logpsi(params, sigma)

        O = flatten_batched_pytree(grad_tree_batch, n_samples)

        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean

        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))

        I = jnp.eye(S.shape[0], dtype=S.dtype)
        return S + diag_shift * I

    return qgt_fn


def make_grad_fn(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
):
    @jax.jit
    def grad_fn(total_params, x_batch):
        return nes_vmc_gradient_stable(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_max_machine=total_max_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
            return_aux=True,
        )

    return grad_fn

######################### gauge dirft mitigation  ########################

def create_single_machine_gauge_fixed(single_model:SingleStateAnsatz, ref_state):
    """
    对单个 SingleStateAnsatz 做 reference gauge fixing。

    返回:
        machine(params_j, sigma) = logψ_j(sigma) - logψ_j(ref_state)

    注意:
        不要 stop_gradient(log_ref)，否则只能前向居中，
        不能真正从梯度里去掉 gauge direction。
    """
    graphdef, state = nnx.split(single_model)
    ref_state = jnp.asarray(ref_state)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_raw = m(sigma)
        log_ref = m(ref_state)
        return log_raw - log_ref
    return machine, graphdef, state


def create_machine_matrix_gauge_stable(model: Union[NESTotalAnsatz_gauge_stable,NESTotalAnsatz_gauge_stable_full]):
    """
    loss 里面构造 Ψ_stable = exp(L_stable) 用。
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)

        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)

        return L_stable

    return machine, graphdef, state


def create_machine_max_gauge_stable(model: Union[NESTotalAnsatz_gauge_stable,NESTotalAnsatz_gauge_stable_full]):
    """
    loss 里面构造 HPsi_stable = exp(logψ_gauge(x') - shift) 用。
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)

        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)

        return shift

    return machine, graphdef, state

def create_machine_gauge_matrix(model: NESTotalAnsatz_gauge_stable):
    """
    loss 里面构造 Ψ_stable = exp(L_stable) 用。
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)

        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)

        return L_gauge

    return machine, graphdef, state

def create_machine_gauge_synthesis(model: NESTotalAnsatz_gauge_stable):
    machine_log_Psi_gauge =  create_machine_gauge_stable(model)
    machine_L_stable =  create_machine_matrix_gauge_stable(model)
    machine_shift = create_machine_max_gauge_stable(model)
    machine_L_gauge = create_machine_gauge_matrix(model)
    return machine_log_Psi_gauge[0], machine_L_stable[0], machine_shift[0], machine_L_gauge[0]
