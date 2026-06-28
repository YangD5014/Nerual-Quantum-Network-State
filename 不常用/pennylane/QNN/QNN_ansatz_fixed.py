import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np
from QNN_jax import initialize_parameters
import jax
jax.config.update("jax_enable_x64", False)
# -------------------------- 第一步：改造量子电路，原生支持Batch输入 --------------------------
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
        qml.Barrier(wires=range(n_qubits))
        # 旋转层：参数向量化，自动适配Batch
        for i in range(n_qubits):
            qml.RX(params[layer, 2*i], wires=i)  # RX参数
            qml.RZ(params[layer, 2*i+1], wires=i)  # RZ参数
    # 4. 测量：返回每个量子比特的期望值

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qnn_circuit(n_qubits:int,n_layers:int):
    dev = qml.device('default.qubit',wires=n_qubits)
    pqc_node = qml.QNode(func=quantum_neural_network,device=dev,interface='jax')
    qnn_node_circuit = partial(pqc_node,n_qubits=n_qubits,n_layers=n_layers)
    return qnn_node_circuit

# 简化的QNN模型，完全避免PRNGKey存储
class SimpleQNNModel(nnx.Module):
    def __init__(self, n_qubits: int, n_layers: int, rngs: nnx.Rngs):
        # 直接初始化参数，不存储rngs
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 初始化QNN参数
        qnn_key = rngs.params()
        self.qnn_params = nnx.Param(
            jax.random.normal(qnn_key, (n_layers, 2*n_qubits), dtype=jnp.float32)
        )
        
        # 初始化线性层权重
        linear_key = rngs.params()
        self.linear_weights = nnx.Param(
            jax.random.normal(linear_key, (n_qubits, n_qubits), dtype=jnp.float32)
        )
        
        # 创建QNN函数
        self.qnn_func = qnn_circuit(n_qubits=n_qubits, n_layers=n_layers)
    
    def __call__(self, x):
        # 确保输入是float32类型
        x = jnp.asarray(x, dtype=jnp.float32)
        
        # 获取QNN输出
        qnn_output = self.qnn_func(x=x, params=self.qnn_params)
        qnn_output = jnp.asarray(qnn_output, dtype=jnp.float32)
        
        # 如果是单个样本，确保形状正确
        if qnn_output.ndim == 1:
            qnn_output = qnn_output.reshape(1, -1)
        
        # 线性变换
        y = jnp.dot(qnn_output, self.linear_weights)
        y = nnx.relu(y)
        
        # 返回标量输出
        return jnp.sum(y, axis=-1)