from flax.nnx.bridge import nnx_in_bridge_mdl
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image
from jax import numpy as jnp

def quantum_neural_network(x, n_qubits, params, n_layers):
    """完全JAX兼容的量子神经网络实现"""
    wires = range(n_qubits)
    
    # 数据编码 - 使用JAX友好的操作
    for i in range(n_qubits):
        qml.RX(jnp.pi * x[i], wires=i)
    
    # 变分层
    for layer in range(n_layers):
        # 单量子比特旋转
        for i in range(n_qubits):
            qml.RX(params[layer, 2*i], wires=i)
            qml.RZ(params[layer, 2*i+1], wires=i)
        
        # 纠缠门
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
    
    # 测量所有量子比特的PauliZ期望值
    return [qml.expval(qml.PauliZ(i)) for i in wires]
