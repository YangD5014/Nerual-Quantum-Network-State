"""诊断扩展哈密顿量的构造"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from pyscf import gto, scf, fci
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

# 哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)

# 希尔伯特空间
hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2
hi_ext = hi ** K

print("检查哈密顿量的结构：")
print(f"ha类型: {type(ha)}")
print(f"ha希尔伯特空间: {ha.hilbert}")
print(f"ha希尔伯特大小: {ha.hilbert.size}")

# 检查ha的矩阵表示
print("\n检查ha是否只在4维空间：")
x_test = hi.random_state(jax.random.PRNGKey(0))
print(f"测试向量形状: {x_test.shape}")

x_primes, mels = ha.get_conn_padded(x_test)
print(f"ha.get_conn_padded(x_test):")
print(f"  x_primes形状: {x_primes.shape}")
print(f"  mels形状: {mels.shape}")

# 检查如果用hi_ext的向量会怎样
print("\n检查hi_ext的向量：")
x_ext = hi_ext.random_state(jax.random.PRNGKey(0))
print(f"扩展向量形状: {x_ext.shape}")

try:
    x_primes_ext, mels_ext = ha.get_conn_padded(x_ext)
    print(f"ha.get_conn_padded(x_ext):")
    print(f"  x_primes_ext形状: {x_primes_ext.shape}")
    print(f"  mels_ext形状: {mels_ext.shape}")
    print(f"  这说明ha可以处理8维向量！")
except Exception as e:
    print(f"ha.get_conn_padded(x_ext) 出错: {e}")

# 关键问题：ha在8维空间上的作用是什么？
# 扩展哈密顿量应该是 H ⊕ H，而不是作用在整个8维空间上
print("\n" + "="*60)
print("关键问题：扩展哈密顿量的定义")
print("="*60)
print("""
根据NES-VMC论文：
- 扩展哈密顿量 H̃ = H ⊗ I ⊗ ... ⊗ I + I ⊗ H ⊗ ... ⊗ I + ...
- 这是一个块对角矩阵，每个块是H
- 作用在 [x1, x2, ..., xK] 上得到 [Hx1, Hx2, ..., HxK]

但 from_pyscf_molecule 构建的 ha 可能是：
1. 只作用在4维空间（正确但需要分别调用K次）
2. 作用在8维空间（错误，应该是块对角的）

让我检查ha.get_conn_padded在8维向量上的行为...
""")

# 创建8维的测试向量，格式为 [x1, x2]
x1 = jnp.array([0., 1., 1., 0.])  # 一个合法的H2组态
x2 = jnp.array([0., 1., 0., 1.])  # 另一个合法的H2组态
x_ext_test = jnp.concatenate([x1, x2])

print(f"扩展态: {x_ext_test}")
print(f"  x1 = {x_ext_test[:4]}")
print(f"  x2 = {x_ext_test[4:]}")

x_primes, mels = ha.get_conn_padded(x_ext_test)
print(f"\nha.get_conn_padded(x_ext_test):")
print(f"  x_primes形状: {x_primes.shape}")
print(f"  x_primes示例:\n{x_primes[:5]}")
print(f"  mels形状: {mels.shape}")
print(f"  mels示例: {mels[:5]}")

# 检查x_primes是否包含8维向量
print(f"\n关键检查：x_primes中的向量是4维还是8维？")
print(f"如果x_primes[i]是4维 → ha只在4维空间工作")
print(f"如果x_primes[i]是8维 → ha作用在整个8维空间（可能不正确）")
print(f"x_primes[i]长度: {len(x_primes[0])}")

if len(x_primes[0]) == 4:
    print("\n✓ ha只在4维空间工作（正确）")
    print("  在Ham_Psi中，需要对每个副本分别调用ha")
elif len(x_primes[0]) == 8:
    print("\n✗ ha作用在8维空间（可能不正确）")
    print("  这会导致错误的哈密顿量定义")
    print("  应该修改为对每个4维副本分别作用")
