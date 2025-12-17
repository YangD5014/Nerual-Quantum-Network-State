import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np
import netket as nk
import netket.experimental as nkx
import matplotlib.pyplot as plt

# 禁用JAX的x64模式，使用float32
jax.config.update("jax_enable_x64", False)

def quantum_neural_network(x, params, n_qubits, n_layers):
    """
    兼容Batch的量子电路核心逻辑：
    - x: 输入张量，形状为 (batch_size, n_qubits)
    - params: 量子电路参数，形状为 (n_layers, 2 * n_qubits)
    - 所有量子门操作自动沿Batch维度向量化
    """
    # 确保输入是二维张量
    x = jnp.atleast_2d(x)
    
    # 校验特征维度
    if x.shape[-1] != n_qubits:
        raise ValueError(f"输入特征维度需为{n_qubits}，当前为{x.shape[-1]}")
    
    # 数据编码：向量化RX门
    for i in range(n_qubits):
        qml.RX(x[:, i] * jnp.pi, wires=i)
    
    # 变分层：向量化旋转/纠缠门
    for layer in range(n_layers):
        # 纠缠层
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.Barrier(wires=range(n_qubits))
        
        # 旋转层
        for i in range(n_qubits):
            qml.RX(params[layer, 2*i], wires=i)
            qml.RZ(params[layer, 2*i+1], wires=i)
    
    # 测量：返回每个量子比特的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qnn_circuit(n_qubits, n_layers):
    """创建量子电路函数"""
    dev = qml.device('default.qubit', wires=n_qubits)
    qnode = qml.QNode(
        func=quantum_neural_network, 
        device=dev, 
        interface='jax'
    )
    return partial(qnode, n_qubits=n_qubits, n_layers=n_layers)

# 使用更简单的模型定义，避免nnx.Param
class SimpleQNNModel:
    """简化的QNN模型，完全避免PRNGKey存储问题"""
    
    def __init__(self, n_qubits: int, n_layers: int, rngs: nnx.Rngs):
        # 直接初始化参数，不存储rngs
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 初始化QNN参数
        qnn_key = rngs.params()
        self.qnn_params = jax.random.normal(qnn_key, (n_layers, 2*n_qubits), dtype=jnp.float32)
        
        # 初始化线性层权重
        linear_key = rngs.params()
        self.linear_weights = jax.random.normal(linear_key, (n_qubits, n_qubits), dtype=jnp.float32)
        
        # 创建QNN函数
        self.qnn_func = qnn_circuit(n_qubits=n_qubits, n_layers=n_layers)
    
    def __call__(self, x):
        # 确保输入是float32类型
        x = jnp.asarray(x, dtype=jnp.float32)
        
        # 获取QNN输出
        qnn_output = self.qnn_func(x=x, params=self.qnn_params)
        qnn_output = jnp.asarray(qnn_output, dtype=jnp.float32)
        
        # 确保形状正确
        if qnn_output.ndim == 1:
            qnn_output = qnn_output.reshape(1, -1)
        
        # 线性变换
        y = jnp.dot(qnn_output, self.linear_weights)
        y = jax.nn.relu(y)
        
        # 返回标量输出
        return jnp.sum(y, axis=-1)
    
    def init(self, rng):
        """为NetKet兼容性提供的初始化方法"""
        return {"qnn_params": self.qnn_params, "linear_weights": self.linear_weights}
    
    def apply(self, params, x):
        """为NetKet兼容性提供的应用方法"""
        # 确保输入是float32类型
        x = jnp.asarray(x, dtype=jnp.float32)
        
        # 获取QNN输出
        qnn_output = self.qnn_func(x=x, params=params["qnn_params"])
        qnn_output = jnp.asarray(qnn_output, dtype=jnp.float32)
        
        # 确保形状正确
        if qnn_output.ndim == 1:
            qnn_output = qnn_output.reshape(1, -1)
        
        # 线性变换
        y = jnp.dot(qnn_output, params["linear_weights"])
        y = jax.nn.relu(y)
        
        # 返回标量输出
        return jnp.sum(y, axis=-1)

# 为NetKet创建兼容的模型包装器
def create_qnn_model(n_qubits, n_layers):
    """创建与NetKet兼容的QNN模型"""
    model = SimpleQNNModel(n_qubits=n_qubits, n_layers=n_layers, rngs=nnx.Rngs(42))
    
    # 创建NetKet兼容的模型函数
    def model_func(params, x):
        return model.apply(params, x)
    
    # 初始化参数
    rng = jax.random.PRNGKey(42)
    init_params = model.init(rng)
    
    return model_func, init_params