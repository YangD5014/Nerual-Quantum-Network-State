from flax.nnx.bridge import nnx_in_bridge_mdl
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image
from jax import numpy as jnp
# 设置随机种子以确保结果可重现
np.random.seed(42)
# 定义量子设备
# n_qubits = 4  # 量子比特数量



# 在 QNN_jax.py 文件中

def data_encoding(x, n_qubits):
    """
    使用 Pauli-X 旋转进行数据编码，完全兼容 JAX 的追踪模型。
    
    Args:
        x (jnp.ndarray): 输入的二进制数据，形状为 (n_qubits,)。
        n_qubits (int): 量子比特的数量。
    """
    if len(x) != n_qubits:
        print(f"输入数据的长度为 {x.shape}，但预期长度为 {n_qubits}")
        raise ValueError("输入数据的长度必须等于量子比特数量")

    # 遍历每一个量子比特
    for i in range(n_qubits):
        # 对第 i 个比特应用 RX 旋转。
        # 旋转角度由输入 x 的第 i 个值决定。
        # 如果 x[i] 是 0, RX(0) 是单位操作。
        # 如果 x[i] 是 1, RX(pi) 等价于一个 X 门。
        # 这种方式是 JAX 友好的，因为它不涉及动态索引或控制流。
        qml.RX(jnp.pi * x[i], wires=i)


    
def variational_layer(params, wires):
    """变分层，包含单量子比特旋转门和纠缠门
    
    Args:
        params (array): 可训练参数
        wires (list): 量子比特索引列表
        entanglement (str): 纠缠策略，'linear'或'full'
    """
    n_wires = len(wires)
    
    # 单量子比特旋转门
    for i, wire in enumerate(wires):
        qml.RX(params[2*i], wires=wire)
        qml.RZ(params[2*i+1], wires=wire)
    for i in range(n_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    qml.Barrier(wires=range(n_wires))
    
def quantum_neural_network(x:np.array, n_qubits:int,params:np.array, n_layers=2):
    """量子神经网络
    
    Args:
        x (array): 输入数据
        params (array): 可训练参数
        n_layers (int): 变分层数量
        entanglement (str): 纠缠策略
    """
    
    wires = range(len(x))
    
    # 数据编码
    data_encoding(x,n_qubits)
    
    # 变分层
    for layer in range(n_layers):
        variational_layer(params[layer], wires)
    
    # 测量
    return [qml.expval(qml.PauliZ(i)) for i in wires]


def qnn_circuit(x:np.array,n_qubits:int,params:np.array, n_layers=2, entanglement='linear'):
    dev = qml.device('default.qubit',wires=n_qubits)
    pqc_node = qml.QNode(func=quantum_neural_network,device=dev,interface='jax')
    qnn_node_circuit = pqc_node(x,n_qubits,params,n_layers,entanglement)
    return qnn_node_circuit


def initialize_parameters(n_layers, n_qubits):
    """初始化量子神经网络参数
    
    Args:
        n_layers (int): 变分层数量
        n_qubits (int): 量子比特数量
        
    Returns:
        array: 初始化的参数
    """
    # 每个变分层有3*n_qubits个参数（每个量子比特有RX, RY, RZ三个旋转门）
    params_shape = (n_layers, 2 * n_qubits)
    return np.random.uniform(0, 2*np.pi, size=params_shape)

