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

total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
    
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
    
    
if __name__ == '__main__':
    N_CHAINS = 16
    N_WARMUP = 100
    N_SAMPLES_PER_CHAIN = 200
    SWEEP_SIZE = 30
    N_ITER =50

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

    sampler_state = nes_sampler.init_state(total_machine, total_params, seed=1)
    samples_raw, sampler_state = nes_sampler.sample(
        total_machine, total_params, state=sampler_state, chain_length=2
    )
    samples_raw.shape
    samples = samples_raw.reshape(-1, hi_ext.size)[0:2]
    print(samples.shape)
    print(samples)