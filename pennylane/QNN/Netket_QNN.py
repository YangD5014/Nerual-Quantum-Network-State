"""
量子神经网络波函数 - Netket FLAX版本

本文件实现了基于论文《Quantum-enhanced neural networks for quantum many-body simulations》的量子神经网络，
并将其改编为Netket支持的FLAX波函数，用于变分量子蒙特卡洛模拟。
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import numpy as np
from typing import Optional, Tuple

# 设置随机种子以确保结果可重现
np.random.seed(42)
key = jax.random.PRNGKey(42)


class QuantumNeuralNetworkWaveFunction(nn.Module):
    """量子神经网络波函数
    
    将量子神经网络架构改编为FLAX波函数，用于变分量子蒙特卡洛模拟。
    该模型模拟了量子神经网络的行为，但完全使用经典神经网络实现。
    """
    
    n_qubits: int = 4  # 量子比特数量
    n_layers: int = 2   # 变分层数量
    hidden_dim: int = 16  # 隐藏层维度
    entanglement: str = 'linear'  # 纠缠策略 ('linear' 或 'full')
    
    def setup(self):
        """设置模型参数"""
        # 数据编码层参数
        self.encoding_dense = nn.Dense(features=self.n_qubits)
        
        # 变分层参数
        self.variational_layers = [
            nn.Dense(features=self.hidden_dim) for _ in range(self.n_layers)
        ]
        
        # 输出层参数
        self.output_dense = nn.Dense(features=1)
        
        # 输出权重
        self.output_weights = self.param(
            'output_weights',
            nn.initializers.uniform(scale=1.0),
            (self.n_qubits,)
        )
    
    def data_encoding(self, x):
        """模拟量子数据编码层
        
        Args:
            x (array): 输入数据，形状为 (batch_size, n_sites)
            
        Returns:
            array: 编码后的数据
        """
        # 使用密集层模拟量子态编码
        encoded = self.encoding_dense(x)
        # 应用tanh激活函数模拟量子态归一化
        return nn.tanh(encoded)
    
    def variational_circuit(self, x, layer_idx):
        """模拟量子变分层
        
        Args:
            x (array): 输入数据
            layer_idx (int): 层索引
            
        Returns:
            array: 变分后的数据
        """
        # 应用变分层
        x = self.variational_layers[layer_idx](x)
        # 应用激活函数模拟量子门操作
        x = nn.tanh(x)
        
        # 模拟纠缠操作
        if self.entanglement == 'linear':
            # 线性纠缠：模拟相邻量子比特之间的纠缠
            for i in range(x.shape[-1] - 1):
                # 简单的线性交互模拟CNOT门
                x = x.at[..., i].set(x[..., i] + 0.1 * x[..., i+1])
        elif self.entanglement == 'full':
            # 全纠缠：模拟所有量子比特之间的纠缠
            for i in range(x.shape[-1]):
                for j in range(i+1, x.shape[-1]):
                    x = x.at[..., i].set(x[..., i] + 0.05 * x[..., j])
        
        return x
    
    def __call__(self, x):
        """计算波函数的对数振幅
        
        Args:
            x (array): 输入构型，形状为 (batch_size, n_sites)
            
        Returns:
            array: 波函数的对数振幅，形状为 (batch_size,)
        """
        # 数据编码
        encoded = self.data_encoding(x)
        
        # 应用变分层
        for layer_idx in range(self.n_layers):
            encoded = self.variational_circuit(encoded, layer_idx)
        
        # 测量模拟：计算每个"量子比特"的期望值
        measurements = nn.tanh(encoded)
        
        # 使用输出权重组合测量结果
        output = jnp.sum(measurements * self.output_weights, axis=-1)
        
        # 应用输出层
        log_psi = self.output_dense(output)
        
        # 返回波函数的对数振幅
        return log_psi.squeeze(-1)


class QuantumNeuralNetworkState:
    """量子神经网络态
    
    封装了Netket的变分态，使用量子神经网络波函数作为变分 ansatz。
    """
    
    def __init__(self, hilbert_space, n_qubits=4, n_layers=2, hidden_dim=16, 
                 entanglement='linear', n_samples=1000, sampler=None):
        """初始化量子神经网络态
        
        Args:
            hilbert_space: Netket希尔伯特空间
            n_qubits (int): 量子比特数量
            n_layers (int): 变分层数量
            hidden_dim (int): 隐藏层维度
            entanglement (str): 纠缠策略
            n_samples (int): 采样数量
            sampler: 采样器，如果为None则使用默认采样器
        """
        self.hilbert_space = hilbert_space
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.entanglement = entanglement
        self.n_samples = n_samples
        
        # 创建量子神经网络波函数
        self.model = QuantumNeuralNetworkWaveFunction(
            n_qubits=n_qubits,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            entanglement=entanglement
        )
        
        # 创建采样器
        if sampler is None:
            self.sampler = nk.sampler.MetropolisLocal(hilbert_space)
        else:
            self.sampler = sampler
        
        # 创建变分态
        self.vstate = nk.vqs.MCState(
            sampler=self.sampler,
            model=self.model,
            n_samples=n_samples
        )
    
    def get_vstate(self):
        """获取变分态
        
        Returns:
            nk.vqs.MCState: 变分态
        """
        return self.vstate
    
    def sample(self, n_samples=None):
        """从波函数中采样
        
        Args:
            n_samples (int): 采样数量，如果为None则使用默认值
            
        Returns:
            array: 采样结果
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        return self.vstate.sample(n_samples)
    
    def expect(self, observable):
        """计算可观测量期望值
        
        Args:
            observable: Netket可观测量
            
        Returns:
            float: 期望值
        """
        return self.vstate.expect(observable)
    
    def gradient(self, observable):
        """计算能量梯度
        
        Args:
            observable: Netket可观测量（通常是哈密顿量）
            
        Returns:
            tuple: (期望值, 梯度)
        """
        return self.vstate.expect_and_grad(observable)


def create_qnn_vstate(hilbert_space, **kwargs):
    """创建量子神经网络变分态的便捷函数
    
    Args:
        hilbert_space: Netket希尔伯特空间
        **kwargs: 传递给QuantumNeuralNetworkState的参数
        
    Returns:
        QuantumNeuralNetworkState: 量子神经网络态
    """
    return QuantumNeuralNetworkState(hilbert_space, **kwargs)


# 示例用法
if __name__ == "__main__":
    # 创建一个简单的自旋系统作为示例
    L = 4  # 系统大小
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    
    # 创建量子神经网络波函数
    model = QuantumNeuralNetworkWaveFunction(
        n_qubits=L,
        n_layers=2,
        hidden_dim=16,
        entanglement='linear'
    )
    
    # 创建采样器
    sampler = nk.sampler.MetropolisLocal(hi)
    
    # 创建变分态
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=1000
    )
    
    # 打印模型信息
    print(f"模型参数数量: {vstate.n_parameters}")
    print(f"采样形状: {vstate.sample().shape}")
    
    # 创建一个简单的哈密顿量（横向场伊辛模型）
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
    
    # 计算能量期望值
    energy = vstate.expect(ha)
    print(f"能量期望值: {energy}")
    
    # 创建量子神经网络态的便捷方式
    qnn_state = create_qnn_vstate(
        hilbert_space=hi,
        n_qubits=L,
        n_layers=2,
        hidden_dim=16,
        entanglement='linear',
        n_samples=1000
    )
    
    # 计算能量和梯度
    energy, gradient = qnn_state.gradient(ha)
    print(f"能量期望值: {energy}")
    print(f"梯度形状: {gradient.shape}")