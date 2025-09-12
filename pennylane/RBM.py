import netket as nk
import numpy as np
import jax.numpy as jnp
from flax import nnx
import jax

# 设置随机种子
np.random.seed(42)
jkey = jax.random.PRNGKey(42)

# 定义前馈神经网络类
class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        构建一个具有单隐藏层的前馈神经网络。
        
        参数:
            N: 输入节点数（自旋链中的自旋数）。
            alpha: 隐藏层的密度。隐藏层将有 N*alpha 个节点。
            rngs: 随机数生成器种子。
        """
        self.alpha = alpha
        
        # 定义一个线性（或密集）层，输出节点数为输入节点数的 alpha 倍
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)
        
    def __call__(self, x: jax.Array):
        # 对输入应用线性层
        y = self.linear(x)
        
        # 非线性激活函数使用 ReLu
        y = nnx.relu(y)
        
        # 对输出求和
        return jnp.sum(y, axis=-1)

# 定义系统参数
nqubits = 4

# 创建 Hilbert 空间
hi = nk.hilbert.Spin(s=0.5, N=nqubits)

# 创建图结构
graph = nk.graph.Hypercube(nqubits)

# 创建哈密顿量（Heisenberg模型）
ha = nk.operator.Heisenberg(hilbert=hi, graph=graph)

# 使用Lanczos方法计算精确基态能量
E0_exact = nk.exact.lanczos_ed(ha)
E0 = float(E0_exact.mean()) if hasattr(E0_exact, 'mean') else float(E0_exact)
print(f"精确基态能量: {E0:.8f}")

# 创建变分量子态（使用FFNN，前馈神经网络）
ma = FFN(N=nqubits, alpha=4, rngs=nnx.Rngs(jkey))

# 创建采样器
sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=graph)

# 创建变分态
vs = nk.vqs.MCState(sampler=sampler, model=ma, n_samples=2000)

# 创建优化器
op = nk.optimizer.Sgd(learning_rate=0.01)

# 创建变分驱动器
gs = nk.VMC(hamiltonian=ha, optimizer=op, variational_state=vs)

# 运行优化
gs.run(n_iter=3000, out='FFNN_log')

# 获取优化后的能量
E_vmc = gs.energy.mean.real
print(f"VMC优化后能量: {E_vmc:.8f}")
print(f"VMC与精确能量差: {abs(E_vmc - E0):.8f}")
print(f"相对误差: {abs(E_vmc - E0)/abs(E0)*100:.4f}%")