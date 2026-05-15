"""
NES-VMC 调试脚本 - 检查维度问题
"""

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx

# H₂ 分子定义
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

ha = nkx.operator.from_pyscf_molecule(mol)

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

K = 2
hi_ext = hi ** K
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)

try:
    single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
except AttributeError:
    from netket.experimental.sampler.rules.fermion_2nd import ParticleExchangeRule
    single_rule = ParticleExchangeRule(hilbert=hi, graph=g)

tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)

# 模型定义
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=rngs)
            for _ in range(self.K)
        ]

    def __call__(self, x: jax.Array):
        print(f"\nDEBUG: 输入x.shape = {x.shape}")
        
        def _forward_single(x_single):
            print(f"  DEBUG _forward_single: x_single.shape = {x_single.shape}")
            M = []
            for i in range(self.K):
                row = []
                for j in range(self.K):
                    val = self.single_ansatz_list[j](x_single[i])
                    print(f"    M[{i},{j}] = {val}")
                    row.append(val)
                M.append(jnp.stack(row))
            M = jnp.stack(M)
            print(f"  DEBUG: M.shape = {M.shape}")
            log_det = jnp.linalg.det(M)
            print(f"  DEBUG: log_det = {log_det}")
            return log_det, M

        if len(x.shape) == 2 and x.shape[0] == self.K and x.shape[1] == self.n_spin:
            print("DEBUG: 单样本路径")
            log_psi, log_M = _forward_single(x)
        else:
            print("DEBUG: 批量路径")
            x_reshaped = x.reshape(-1, K, self.n_spin)
            print(f"DEBUG: 重塑后x_reshaped.shape = {x_reshaped.shape}")
            log_psi, log_M = jax.vmap(_forward_single)(x_reshaped)
        
        print(f"DEBUG: 最终输出 log_psi.shape = {log_psi.shape}, log_M.shape = {log_M.shape}")
        return log_psi, log_M


# 创建machine函数
def create_machine(model: NESTotalAnsatz):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_psi_total

    return machine, graphdef, state


# 测试
rngs = nnx.Rngs(21)
total_ansatz = NESTotalAnsatz(4, K, 16, rngs=rngs)
machine, graphdef, params = create_machine(total_ansatz)

sampler_state = sampler.init_state(machine, params, seed=21)

print("\n" + "="*70)
print("开始采样...")
print("="*70)

samples, sampler_state = sampler.sample(
    machine, params, state=sampler_state, chain_length=20
)

print(f"\nsamples.shape = {samples.shape}")
samples = samples.reshape(-1, K, 4)
print(f"重塑后 samples.shape = {samples.shape}")

print("\n" + "="*70)
print("测试模型前向传播...")
print("="*70)

log_psi, M = total_ansatz(samples)
print(f"\n最终结果:")
print(f"  log_psi.shape = {log_psi.shape}")
print(f"  M.shape = {M.shape}")
