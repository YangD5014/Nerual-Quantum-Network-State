#!/usr/bin/env python
import sys
sys.path.append('/Users/yangjianfei/mac_vscode/神经网络量子态/pennylane/QNN')

import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np
from QNN_jax import initialize_parameters

# 定义量子神经网络函数（从notebook中复制）
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
    # 注意：在PennyLane中，直接返回测量值，不需要手动转换为jnp.array
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qnn_circuit(n_qubits:int,n_layers:int):
    dev = qml.device('default.qubit',wires=n_qubits)
    pqc_node = qml.QNode(func=quantum_neural_network,device=dev,interface='jax')
    qnn_node_circuit = partial(pqc_node,n_qubits=n_qubits,n_layers=n_layers)
    return qnn_node_circuit

# 修复后的QNNLinear类
class QNNLinear(nnx.Module):
    def __init__(self, rngs: nnx.Rngs,n_qubits:int,n_layer:int):
        self.rngs = rngs
        self.key = rngs.params()
        self.n_qubits,self.n_layer = n_qubits,n_layer
        # 修复参数形状：应该是 (n_layer, 2 * n_qubits) 而不是 (n_qubits, n_layer)
        self.qnn_params = nnx.Param(initialize_parameters(self.n_layer, self.n_qubits))
        self.qnn_layer = qnn_circuit(n_qubits=self.n_qubits,n_layers=self.n_layer)
        self.Linear = nnx.Linear(in_features=self.n_qubits,out_features=self.n_qubits,use_bias=False,rngs=self.rngs)
        
    
    def __call__(self, s:np.array):
        # 确保输入是JAX数组
        s = jnp.array(s)
        # 获取QNN输出
        qnn_output = self.qnn_layer(x=s,params=self.qnn_params)
        # 将QNN输出转换为JAX数组（如果是列表）
        if isinstance(qnn_output, list):
            # 对于批量输入，需要转置以获得正确的形状
            if len(s.shape) > 1:  # 批量输入
                qnn_output = jnp.stack([jnp.array(r) for r in qnn_output]).T
            else:  # 单个输入
                qnn_output = jnp.array(qnn_output)
        # 确保输入是2D数组（批量维度，特征维度）
        if len(qnn_output.shape) == 1:
            qnn_output = qnn_output.reshape(1, -1)
        y = self.Linear(qnn_output)
        return jnp.sum(y, axis=-1)

# 测试模型
if __name__ == "__main__":
    print("创建模型...")
    model = QNNLinear(rngs=nnx.Rngs(params=0), n_qubits=4, n_layer=2)
    
    print("测试单个输入...")
    try:
        result = model(s=[1,0,1,0])
        print(f"单个输入结果: {result}")
    except Exception as e:
        print(f"单个输入错误: {e}")
    
    print("\n测试批量输入...")
    try:
        batch_result = model(s=[[1,0,1,0], [0,1,0,1]])
        print(f"批量输入结果: {batch_result}")
    except Exception as e:
        print(f"批量输入错误: {e}")
    
    print("\n测试完成！")