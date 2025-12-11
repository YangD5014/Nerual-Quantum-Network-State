import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np
from QNN_jax import initialize_parameters

# 修复后的量子神经网络函数
def quantum_neural_network(x, params, n_qubits, n_layers):
    """
    兼容Batch的量子电路核心逻辑：
    - x: 输入张量，形状为 (batch_size, n_qubits)（Batch维度在前）
    - params: 量子电路参数，形状为 (n_layers, 2 * n_qubits)
    - 所有量子门操作自动沿Batch维度向量化，无需手动循环
    """
    # 1. 强制将输入转为JAX张量（兼容np.array/其他格式），并确保是二维（batch, n_qubits）
    x = jnp.atleast_2d(x)
    # 校验特征维度（Batch维度不校验，由NNX自动兼容）
    if x.shape[-1] != n_qubits:
        raise ValueError(f"输入特征维度需为{n_qubits}，当前为{x.shape[-1]}")
    
    # 2. 数据编码：向量化RX门（自动兼容Batch）
    # qml.RX支持批量角度输入，会自动为每个Batch样本应用对应角度的门
    for i in range(n_qubits):
        qml.RX(x[:, i] * jnp.pi, wires=i)  # x[:, i] 取所有Batch样本的第i个特征
    
    # 3. 变分层：向量化旋转/纠缠门（Batch维度自动兼容）
    for layer in range(n_layers):
        # 纠缠层（CNOT无参数，Batch不影响）
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        # 旋转层：参数向量化，自动适配Batch
        # params[layer] 形状为 (2 * n_qubits,)，其中前n_qubits个为RX参数，后n_qubits个为RZ参数
        for i in range(n_qubits):
            qml.RX(params[layer, 2*i], wires=i)  # RX参数
            qml.RZ(params[layer, 2*i+1], wires=i)  # RZ参数
    
    # 4. 测量：返回每个量子比特的期望值
    # 注意：在PennyLane中，直接返回测量值，不需要手动转换为jnp.array
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 测试代码
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)
# 封装QNode：interface="jax" 确保兼容JAX/NNX的Batch计算
qnode = qml.QNode(quantum_neural_network, dev, interface="jax")

# 测试单个输入
params = initialize_parameters(n_layers, n_qubits)
result = qnode(x=[1,0,1,0], params=params, n_layers=n_layers, n_qubits=n_qubits)
print("单个输入结果:", result)

# 测试批量输入
batch_x = jnp.array([[1,0,1,0], [0,1,0,1], [1,1,0,0]])
batch_result = qnode(x=batch_x, params=params, n_layers=n_layers, n_qubits=n_qubits)
print("批量输入结果类型:", type(batch_result))
print("批量输入结果:")
print(batch_result)

# 如果需要转换为JAX数组，可以这样做
batch_array = jnp.stack([jnp.array(r) for r in batch_result]).T
print("转换为JAX数组后的形状:", batch_array.shape)
print("转换为JAX数组后:")
print(batch_array)