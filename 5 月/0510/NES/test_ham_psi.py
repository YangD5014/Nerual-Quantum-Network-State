"""测试Ham_psi函数"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from pyscf import gto, scf, fci
import flax.nnx as nnx
from flax import linen as nn
import numpy as np

# H₂ 分子定义
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print(f"FCI基态能量: {E_fcis[0]:.8f} Ha")
print(f"FCI第一激发态: {E_fcis[1]:.8f} Ha")

# 哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)
print(f"\nha希尔伯特空间大小: {ha.hilbert.size}")

# 希尔伯特空间
hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2
hi_ext = hi ** K
print(f"hi_ext大小: {hi_ext.size}")

# 定义单态Ansatz
class SingleStateAnsatz(nnx.Module):
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

# Ham_psi函数
def Ham_psi(ha: nk.operator.DiscreteOperator, model, x):
    """计算 Hψ(x)"""
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = jax.vmap(model)(x_primes)
    psi_vals = jnp.exp(log_psi_vals)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x

# 创建ansatz
rngs = nnx.Rngs(0)
ansatz = SingleStateAnsatz(4, hidden_dim=8, rngs=rngs)
graphdef, state = nnx.split(ansatz)

@jax.jit
def machine(params, sigma):
    m = nnx.merge(graphdef, params)
    return m(sigma)

params = state

# 生成样本
key = jax.random.PRNGKey(0)
samples = []
for _ in range(10):
    s = hi.random_state(key)
    samples.append(s)
    key = jax.random.split(key)[1]
samples = jnp.stack(samples)

print(f"\n生成样本形状: {samples.shape}")
print(f"样本示例: {samples[0]}")

# 测试Ham_psi
print("\n测试Ham_psi函数:")
for i, s in enumerate(samples[:3]):
    print(f"\n样本 {i}: {s}")
    H_psi = Ham_psi(ha, ansatz, s)
    print(f"  Hψ(s) = {H_psi}")
    log_psi = machine(params, s)
    print(f"  log ψ(s) = {log_psi}")

# 测试扩展态上的Ham_Psi
print("\n" + "="*60)
print("测试Ham_Psi在扩展态上")
print("="*60)

class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=rngs)
            for _ in range(self.K)
        ]

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            Psi_matrix = jnp.exp(L)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_matrix)
            log_Psi = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi, L
        return _forward_single(x)

def Ham_Psi(ha: nk.operator.DiscreteOperator, total_ansatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    hilber_size = total_ansatz.n_spin
    k = total_ansatz.K
    x = x.reshape(k, hilber_size)
    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)
    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi

# 创建NES ansatz
total_ansatz = NESTotalAnsatz(4, 2, 8, rngs=nnx.Rngs(12))
graphdef_total, state_total = nnx.split(total_ansatz)

@jax.jit
def machine_total(params, sigma):
    m = nnx.merge(graphdef_total, params)
    return m(sigma)

params_total = state_total

# 生成扩展态样本
ext_samples = []
for _ in range(5):
    s1 = hi.random_state(key)
    s2 = hi.random_state(jax.random.split(key)[1])
    key = jax.random.split(key)[2]
    ext_state = jnp.concatenate([s1, s2])
    ext_samples.append(ext_state)
ext_samples = jnp.stack(ext_samples)

print(f"\n扩展态样本形状: {ext_samples.shape}")
print(f"扩展态示例: {ext_samples[0]}")

# 测试Ham_Psi
print("\n测试Ham_Psi函数:")
for i, s in enumerate(ext_samples[:3]):
    print(f"\n扩展态 {i}: {s}")
    print(f"  x1 = {s[:4]}")
    print(f"  x2 = {s[4:]}")

    # 手动计算H_Psi
    x = s.reshape(2, 4)
    print(f"  x reshaped: {x}")

    # 对每个ansatz计算Hψ
    for j in range(2):
        H_psi = Ham_psi(ha, total_ansatz.single_ansatz_list[j], x[0])
        print(f"  Hψ_{j}(x[0]) = {H_psi}")
        H_psi2 = Ham_psi(ha, total_ansatz.single_ansatz_list[j], x[1])
        print(f"  Hψ_{j}(x[1]) = {H_psi2}")

    # 调用Ham_Psi
    H_Psi = Ham_Psi(ha, total_ansatz, s)
    print(f"  H_Psi矩阵:\n{H_Psi}")

    # 计算E_L = Ψ^{-1} * H_Psi
    log_psi, L = machine_total(params_total, s)
    Psi = jnp.exp(L)
    print(f"  Ψ矩阵:\n{Psi}")

    try:
        E_L = jnp.linalg.solve(Psi, H_Psi)
        print(f"  E_L矩阵:\n{E_L}")
        trace = jnp.trace(E_L)
        print(f"  Tr(E_L) = {trace}")
    except Exception as e:
        print(f"  计算E_L出错: {e}")
