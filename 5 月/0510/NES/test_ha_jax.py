"""测试ha.to_jax_operator()后的行为"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from pyscf import gto, scf, fci
import flax.nnx as nnx
import numpy as np

# H₂ 分子定义
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# 哈密顿量
hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
ha = nkx.operator.from_pyscf_molecule(mol)

print("原始ha:")
print(f"  类型: {type(ha)}")
print(f"  希尔伯特空间大小: {ha.hilbert.size}")

# 测试原始ha
x_test = hi.random_state(jax.random.PRNGKey(0))
print(f"\n测试向量: {x_test}")
x_primes, mels = ha.get_conn_padded(x_test)
print(f"原始ha.get_conn_padded:")
print(f"  x_primes形状: {x_primes.shape}")
print(f"  mels: {mels}")

# 转换为JAX版本
ha_jax = ha.to_jax_operator()
print(f"\n转换后ha_jax:")
print(f"  类型: {type(ha_jax)}")

# 测试ha_jax
try:
    x_primes_jax, mels_jax = ha_jax.get_conn_padded(x_test)
    print(f"ha_jax.get_conn_padded:")
    print(f"  x_primes形状: {x_primes_jax.shape}")
    print(f"  mels: {mels_jax}")

    # 比较结果
    print(f"\n比较:")
    print(f"  x_primes相同: {np.allclose(x_primes, x_primes_jax)}")
    print(f"  mels相同: {np.allclose(mels, mels_jax)}")
except Exception as e:
    print(f"ha_jax.get_conn_padded 出错: {e}")

# 测试用JAX版本计算Hψ
class SimpleAnsatz(nnx.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nnx.Linear(n, 1, rngs=nnx.Rngs(0), param_dtype=complex)

    def __call__(self, x):
        return jnp.sum(self.w(x))

ansatz = SimpleAnsatz(4)
params = nnx.state(ansatz)

def Ham_psi_test(ha, model, x):
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = jax.vmap(model)(x_primes)
    psi_vals = jnp.exp(log_psi_vals)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x

print("\n测试Ham_psi (原始ha):")
H_psi_original = Ham_psi_test(ha, ansatz, x_test)
print(f"  Hψ = {H_psi_original}")

print("\n测试Ham_psi (ha_jax):")
try:
    H_psi_jax = Ham_psi_test(ha_jax, ansatz, x_test)
    print(f"  Hψ = {H_psi_jax}")
    print(f"  两者相同: {np.allclose(H_psi_original, H_psi_jax)}")
except Exception as e:
    print(f"  出错: {e}")

# 检查ha_jax在vmap中的行为
print("\n测试在jax.vmap中使用ha_jax:")
try:
    x_batch = jnp.stack([x_test, x_test])
    def Ham_psi_vmap(ha, x):
        x_primes, mels = ha.get_conn_padded(x)
        log_psi_vals = jax.vmap(ansatz)(x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    H_psi_batch = jax.vmap(lambda x: Ham_psi_vmap(ha_jax, x))(x_batch)
    print(f"  批量Hψ = {H_psi_batch}")
except Exception as e:
    print(f"  出错: {e}")
