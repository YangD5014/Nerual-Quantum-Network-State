from pyscf import gto, scf, fci
import netket as nk
import netket.experimental as nkx
import jax.numpy as jnp
import flax.nnx as nnx
import jax

# 设置H4分子的几何构型
# 使用线性构型，H-H键长为1.0埃
bond_length = 1.0
geometry = [
    ('H', (0., 0., 0.)),
    ('H', (bond_length, 0., 0.)),
    ('H', (2*bond_length, 0., 0.)),
    ('H', (3*bond_length, 0., 0.))
]

# 创建分子对象，使用STO-3G基组
mol = gto.M(atom=geometry, basis='STO-3G')

# 进行Hartree-Fock计算
mf = scf.RHF(mol).run(verbose=0)
E_hf = sum(mf.scf_summary.values())
print(f"Hartree-Fock能量: {E_hf:.8f} Ha")

# 进行FCI计算作为参考
E_fci = fci.FCI(mf).kernel()[0]
print(f"FCI能量: {E_fci:.8f} Ha")

# 使用NetKet创建哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)
# 使用Lanczos方法计算精确基态能量
E0 = float(nk.exact.lanczos_ed(ha))
print(f"NetKet精确对角化能量: {E0:.8f} Ha")
print(f"NetKet与FCI能量差: {abs(E0 - E_fci):.8f} Ha")

# 创建Hilbert空间 - 使用与哈密顿量匹配的费米子希尔伯特空间
hi = ha.hilbert

# 定义自定义前馈神经网络类
class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        Construct a Feed-Forward Neural Network with a single hidden layer.

        Args:
            N: The number of input nodes (number of spins in the chain).
            alpha: The density of the hidden layer. The hidden layer will have
                N*alpha nodes.
            rngs: The random number generator seed.
        """
        self.alpha = alpha

        # We define a linear (or dense) layer with `alpha` times the number of input nodes
        # as output nodes.
        # We must pass forward the rngs object to the dense layer.
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):

        # we apply the linear layer to the input
        y = self.linear(x)

        # the non-linearity is a simple ReLu
        y = nnx.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)

# 创建变分量子态（使用自定义FFN前馈神经网络）
N = hi.size  # 获取Hilbert空间的大小作为输入节点数
ma = FFN(N=N, alpha=4, rngs=nnx.Rngs(2))

# 创建变分态 - 使用费米子系统的特殊采样器
vs = nk.vqs.MCState(sampler=nk.sampler.MetropolisLocal(hi), model=ma, n_samples=2000)

# 创建优化器 - 降低学习率
op = nk.optimizer.Sgd(learning_rate=0.01)

# 创建变分驱动器
gs = nk.VMC(hamiltonian=ha, optimizer=op, variational_state=vs)

# 运行优化 - 增加迭代次数
gs.run(n_iter=1000, out='H4_log')

# 获取优化后的能量
E_vmc = gs.energy.mean.real
print(f"VMC优化后能量: {E_vmc:.8f} Ha")
print(f"VMC与FCI能量差: {abs(E_vmc - E_fci):.8f} Ha")
print(f"VMC与精确对角化能量差: {abs(E_vmc - E0):.8f} Ha")
print(f"相对误差: {abs(E_vmc - E0)/abs(E0)*100:.4f}%")