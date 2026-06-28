import netket as nk
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import field

# 设置随机种子
key = jax.random.PRNGKey(42)

# 定义系统参数
L = 20  # 1D链的长度
nqubits = L

# 定义量子设备（PennyLane）
dev = qml.device("default.qubit", wires=nqubits)

# 创建1D链图结构（周期性边界条件）
g = nk.graph.Chain(length=L, pbc=True)

# 创建希尔伯特空间（总自旋为0）
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=0)

# 创建哈密顿量（Heisenberg模型）
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

# 定义自定义量子Ansatz: f[s; U(θ)] = ∑Nq i=1 ci⟨s|U†(θ)ZiU(θ)|s⟩
@qml.qnode(dev, interface="jax")
def custom_quantum_ansatz(params, nqubits):
    # 将参数重塑为合适的形状
    params = params.reshape((nqubits, 2))  # 每个量子比特2个参数（旋转门）
    
    # 应用单量子比特旋转门构成U(θ)
    for i in range(nqubits):
        qml.RX(params[i, 0], wires=i)
        qml.RY(params[i, 1], wires=i)
    
    # 应用纠缠门构成U(θ)
    for i in range(nqubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # 测量所有量子比特的Pauli Z期望值 ⟨s|U†(θ)ZiU(θ)|s⟩
    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

# 定义 Flax 模型
class QuantumModel(nn.Module):
    nqubits: int = field(default=4, metadata={'immutable': True})
    
    @nn.compact
    def __call__(self, x):
        # 获取批次大小
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        
        # 初始化量子参数
        quantum_params = self.param(
            'quantum_params',
            nn.initializers.normal(stddev=0.1),
            (self.nqubits * 2,)  # 每个量子比特2个参数
        )
        
        # 初始化可训练系数
        coeffs = self.param(
            'coeffs',
            nn.initializers.normal(stddev=0.1),
            (self.nqubits,)
        )
        
        # 计算量子期望值 ⟨s|U†(θ)ZiU(θ)|s⟩
        expvals = custom_quantum_ansatz(quantum_params, self.nqubits)
        expvals = jnp.array(expvals)  # 现在expvals是数值数组
        
        # 计算波函数对数 f[s; U(θ)] = ∑Nq i=1 ci⟨s|U†(θ)ZiU(θ)|s⟩
        log_psi = jnp.dot(coeffs, expvals)
        
        # 返回与批次大小匹配的形状
        if batch_size > 1:
            return jnp.ones(batch_size) * log_psi
        else:
            return log_psi

# 创建模型实例
model = QuantumModel(nqubits=nqubits)

# 初始化模型变量
dummy_input = jnp.ones(nqubits)
variables = model.init(key, dummy_input)

# 提取参数
model_params = variables['params']

# 创建采样器
sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, n_chains=16)

# 创建 MCState
vqs = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    variables=variables,
    n_samples=1008,
    n_discard_per_chain=10
)

# 定义 VMC 优化器和预条件器
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.1)
vmc = nk.VMC(hamiltonian=ha, optimizer=optimizer, variational_state=vqs, preconditioner=sr)

# 打印参数结构
print(f"# variational parameters: {vmc.state.n_parameters}")

# 使用Lanczos方法计算精确基态能量
E0_exact = nk.exact.lanczos_ed(ha)
E0 = float(E0_exact.mean()) if hasattr(E0_exact, 'mean') else float(E0_exact)
print(f"精确基态能量: {E0:.8f}")

# 运行优化
vmc.run(n_iter=300, out="output")

# 获取优化后的能量
E_vmc = vmc.energy.mean.real
print(f"VMC优化后能量: {E_vmc:.8f}")
print(f"VMC与精确能量差: {abs(E_vmc - E0):.8f}")
print(f"相对误差: {abs(E_vmc - E0)/abs(E0)*100:.4f}%")
