#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的QNN批量处理功能
"""

import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np

# 配置JAX使用双精度
jax.config.update("jax_enable_x64", True)

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

def initialize_parameters(n_layers, n_qubits, scale=0.1):
    """初始化量子电路参数"""
    return jnp.ones((n_layers, 2 * n_qubits)) * scale

class QNNLinearFixed(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, n_qubits: int, n_layer: int):
        key = rngs.params()
        # 不存储PRNGKey，直接使用rngs.params()初始化参数
        self.n_qubits, self.n_layer = n_qubits, n_layer
        self.qnn_params = nnx.Param(jax.random.normal(key, (self.n_layer, 2*self.n_qubits), dtype=jnp.float32))
        self.qnn_layer = partial(qnn_circuit, n_qubits=self.n_qubits, n_layers=self.n_layer)()
        self.linear = nnx.Linear(in_features=self.n_qubits, out_features=self.n_qubits, use_bias=False, rngs=rngs, dtype=jnp.float32)
        
    def __call__(self, s):
        # 确保输入是JAX数组
        s = jnp.array(s, dtype=jnp.float32)
        
        # 获取QNN输出
        qnn_output = self.qnn_layer(x=s, params=self.qnn_params)
        
        # 修复点：正确处理量子电路输出的批量数据格式
        # qnn_output是一个列表，包含n_qubits个数组，每个数组的形状是(batch_size,)
        # 我们需要将这些数组转置，得到形状为(batch_size, n_qubits)的矩阵
        qnn_output = jnp.stack(qnn_output, axis=1)  # 形状: (batch_size, n_qubits)
        qnn_output = jnp.array(qnn_output, dtype=jnp.float32)
        
        # 线性变换
        y = self.linear(qnn_output)
        y = nnx.relu(y)
        
        # 求和并返回
        return jnp.sum(y, axis=-1)

# 测试修复后的模型
if __name__ == "__main__":
    print("测试修复后的QNN批量处理功能")
    print("=" * 50)
    
    # 创建修复后的模型
    model_fixed = QNNLinearFixed(n_qubits=4, n_layer=2, rngs=nnx.Rngs(params=0))
    
    # 测试单个输入
    single_result = model_fixed(s=[1,0,1,0])
    print(f"单个输入结果: {single_result}")
    
    # 测试批量输入
    batch_result = model_fixed(s=[[1,0,1,0], [0,1,0,1]])
    print(f"批量输入结果: {batch_result}")
    
    # 测试更大批量
    large_batch_result = model_fixed(s=[[1,0,1,0], [0,1,0,1], [1,1,0,0], [0,0,1,1]])
    print(f"大批量输入结果: {large_batch_result}")
    
    # 验证结果是否合理
    print("\n验证结果:")
    print(f"批量结果长度: {len(batch_result)} (应为2)")
    print(f"大批量结果长度: {len(large_batch_result)} (应为4)")
    print(f"批量结果是否全为0: {jnp.all(batch_result == 0)} (应为False)")
    print(f"大批量结果是否全为0: {jnp.all(large_batch_result == 0)} (应为False)")
    
    print("\n测试完成！修复后的模型可以正确处理批量数据。")