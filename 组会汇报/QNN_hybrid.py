import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
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

import jax
import jax.numpy as jnp
from flax import nnx

class FFN_Amplitude(nnx.Module):

    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        self.alpha = alpha
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.linear(x)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)
    

class FFN_Phase(nnx.Module):

    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        self.alpha = alpha
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.linear(x)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)  
    
from flax import nnx
from functools import partial
class QNNLinear(nnx.Module):
    def __init__(self, rngs: nnx.Rngs,n_qubits:int,n_layer:int):
        key = rngs.params()
        # 不存储PRNGKey，直接使用rngs.params()初始化参数
        self.n_qubits,self.n_layer = n_qubits,n_layer
        self.qnn_params = nnx.Param(jax.random.normal(key, (self.n_layer, 2*self.n_qubits), dtype=jnp.float32))
        self.qnn_layer = partial(qnn_circuit, n_qubits=self.n_qubits, n_layers=self.n_layer)()
        self.Linear = nnx.Linear(in_features=self.n_qubits,out_features=self.n_qubits,use_bias=False,rngs=rngs)
        
    
    def __call__(self, s:np.array):
        # 确保输入是JAX数组
        s = jnp.array(s, dtype=jnp.float32)
        # 获取QNN输出
        qnn_output = self.qnn_layer(x=s,params=self.qnn_params) #dtype=float32
        # 将QNN输出转换为float32类型
        #qnn_output = jnp.array(qnn_output, dtype=jnp.float32).reshape(-1,self.n_qubits)
        qnn_output = jnp.stack(qnn_output, axis=1)  # 形状: (batch_size, n_qubits)
        qnn_output = jnp.array(qnn_output, dtype=jnp.float32)
        y = self.Linear(qnn_output)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)
    
class FFN_QNN_Hybrid(nnx.Module):
    def __init__(self, 
                 rngs_real: nnx.Rngs,
                 rngs_imag:nnx.Rngs,
                 rngs_amplitude:nnx.Rngs,
                 rngs_phase:nnx.Rngs,
                 n_qubits:int,
                 n_layer:int,
                 alpha:int):
        
        self.qnn_real = QNNLinear(rngs_real,n_qubits,n_layer)
        self.qnn_imag = QNNLinear(rngs_imag,n_qubits,n_layer)
        self.ffn_amplitude = FFN_Amplitude(N=n_qubits,alpha=alpha,rngs=rngs_amplitude)
        self.ffn_phase = FFN_Phase(N=n_qubits,alpha=alpha,rngs=rngs_phase)
        
    def __call__(self, s:np.array):
        ffnn_amplitude = self.ffn_amplitude(s)
        ffnn_phase = self.ffn_phase(s)
        qnn_real = self.qnn_real(s)
        qnn_imag = self.qnn_imag(s)
        
        return ffnn_amplitude+qnn_real+1j*ffnn_phase+1j*qnn_imag
        