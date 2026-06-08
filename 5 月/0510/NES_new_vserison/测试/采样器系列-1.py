
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
machine,graphdef,params = create_machine(total_ansatz)

sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)
sampler_state = sampler.init_state(machine, params, seed=1)


# ===================== 【修复版】NES 跃迁规则 =====================
@struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    edges: jnp.ndarray  # 跃迁边

    def _check_duplicate_substate(self, sigma_ext: jnp.ndarray) -> jnp.ndarray:
        """校验子组态是否重复（核心约束）"""
        sub_states = jnp.reshape(sigma_ext, (*sigma_ext.shape[:-1], K, SINGLE_SIZE))
        first_sub = sub_states[..., 0, :]
        duplicate = False
        for k in range(1, K):
            curr_sub = sub_states[..., k, :]
            eq = jnp.all(first_sub == curr_sub, axis=-1)
            duplicate = jnp.logical_or(duplicate, eq)
        return duplicate

    def _fermion_hop(self, sigma: jnp.ndarray, edge: jnp.ndarray) -> jnp.ndarray:
        """
        【手动实现费米子跳跃】替代废弃的hop_state
        输入：sigma (n_chains, 8) | edge (n_chains, 2) 跃迁边
        输出：跳跃后的态
        """
        # 批量交换两个位点的占据数（费米子标准跳跃）
        i, j = edge[:, 0], edge[:, 1]
        sigma_cand = sigma.copy()
        # 批量索引替换（JAX 向量化）
        sigma_cand = sigma_cand.at[jnp.arange(sigma.shape[0]), i].set(sigma[jnp.arange(sigma.shape[0]), j])
        sigma_cand = sigma_cand.at[jnp.arange(sigma.shape[0]), j].set(sigma[jnp.arange(sigma.shape[0]), i])
        return sigma_cand

    def transition(
        self,
        sampler,
        machine,
        parameters,
        state,
        rng: jax.random.PRNGKey,
        sigma: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """
        【修复后】核心跃迁方法
        sigma 输入形状：(n_chains, hilbert_size) → 必须是多链批量！
        """
        print(sigma.shape)
        n_chains, hilbert_size = sigma.shape
        key_edge, key_acc = jax.random.split(rng, 2)

        # 1. 随机选择跃迁边
        n_edges = len(self.edges)
        edge_idx = jax.random.randint(key_edge, (n_chains,), 0, n_edges)
        selected_edges = self.edges[edge_idx]  # (n_chains, 2)

        # 2. 【修复】手动费米子跳跃
        sigma_cand = self._fermion_hop(sigma, selected_edges)

        # 3. NES约束：过滤重复子组态
        is_duplicate = self._check_duplicate_substate(sigma_cand)
        sigma_cand = jnp.where(is_duplicate[:, None], sigma, sigma_cand)

        # 4. Metropolis 接受概率
        log_prob_curr = 2 * jnp.real(machine(parameters, sigma))
        log_prob_cand = 2 * jnp.real(machine(parameters, sigma_cand))
        log_accept_ratio = log_prob_cand - log_prob_curr

        # 5. 接受/拒绝
        u = jax.random.uniform(key_acc, (n_chains,))
        accept = jnp.log(u) < jnp.minimum(0.0, log_accept_ratio)
        new_sigma = jnp.where(accept[:, None], sigma_cand, sigma)

        return new_sigma, None

    def random_state(
        self, sampler, machine, parameters, state, rng, sigma: jnp.ndarray
    ) -> jnp.ndarray:
        """生成合法初始态（无重复子组态）"""
        hilb = sampler.hilbert
        n_chains = sigma.shape[0]

        def _gen_valid_state(key):
            def cond(carry): return carry[2]
            def body(carry):
                s, k, _ = carry
                k_new, k_rand = jax.random.split(k)
                s_new = hilb.random_state(k_rand, size=1)
                dup = self._check_duplicate_substate(s_new)[0]
                return (s_new, k_new, dup)

            s_init = hilb.random_state(key, size=1)
            dup_init = self._check_duplicate_substate(s_init)[0]
            valid_state, _, _ = jax.lax.while_loop(cond, body, (s_init, key, dup_init))
            return valid_state[0]

        keys = jax.random.split(rng, n_chains)
        return jax.vmap(_gen_valid_state)(keys)
    
if __name__ == '__main__':
    rule =NESFermionHopRule(edges=tuple(edges))
    rule._check_duplicate_substate(hi_ext.all_states()) #正常！
    
    rule.transition(sampler=sampler,machine=machine,
                    parameters=params,state=sampler_state,
                    rng= jax.random.key(1),
                    sigma=jnp.array([[1,0,1,0,0,1,1,0]])) #报错！
