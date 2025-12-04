# H2分子FFNN波函数Ansatz分析

## 概述

本文档分析了在`H2_VQMC 案例_FFNN.ipynb`中使用的FFNN（前馈神经网络）波函数Ansatz的定义、结构、数学表达及其在变分蒙特卡洛(VMC)方法中的应用。

## FFNN模型定义

### 1. 模型结构

FFNN模型是一个简单的单隐藏层前馈神经网络，使用Flax框架实现：

```python
import jax
import jax.numpy as jnp
from flax import nnx

class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        构建一个单隐藏层的前馈神经网络。

        参数:
            N: 输入节点数（链中的自旋数）。
            alpha: 隐藏层的密度。隐藏层将有N*alpha个节点。
            rngs: 随机数生成器种子。
        """
        self.alpha = alpha
        # 定义一个线性（或密集）层，输出节点数为alpha倍的输入节点数
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        # 将线性层应用于输入
        y = self.linear(x)
        # 非线性激活函数是简单的ReLU
        y = nnx.relu(y)
        # 对输出求和
        return jnp.sum(y, axis=-1)
```

### 2. 模型参数

在H2分子计算中，FFNN模型的参数设置为：
- `N = 4`: 输入维度为4，对应H2分子在STO-3G基组下的4个自旋轨道
- `alpha = 1`: 隐藏层节点数为4 (N*alpha)
- `rngs = nnx.Rngs(2)`: 随机数生成器种子为2

## 波函数Ansatz的数学表达

### 1. 波函数形式

在量子力学中，波函数Ψ(x)描述了系统在状态x下的概率幅。在这个实现中，FFNN模型直接输出波函数的对数概率幅：

```
log Ψ(x) = FFN(x)
```

其中x是自旋轨道占据数配置，对于H2分子，x是一个长度为4的向量，每个元素为0或1，表示对应自旋轨道是否被占据。

### 2. 网络前向传播

FFNN的前向传播过程可以表示为：

1. 线性变换：
   ```
   y = Wx + b
   ```
   其中W是权重矩阵，形状为(N*alpha, N)，b是偏置向量，形状为(N*alpha,)

2. ReLU激活：
   ```
   y = max(0, Wx + b)
   ```

3. 输出求和：
   ```
   log Ψ(x) = Σ_i y_i
   ```

### 3. 波函数概率

实际波函数的概率幅为：
```
Ψ(x) = exp(FFN(x))
```

而系统处于状态x的概率为：
```
P(x) = |Ψ(x)|² / Σ_x |Ψ(x)|²
```

## 在VMC中的应用

### 1. 变分蒙特卡洛状态

FFNN模型被封装在NetKet的MCState中：

```python
N = 4
ffnn_model = FFN(N=N, alpha=1, rngs=nnx.Rngs(2))
vs = nk.vqs.MCState(sa, ffnn_model, n_discard_per_chain=10, n_samples=512)
```

其中：
- `sa`是MetropolisFermionHop采样器
- `n_discard_per_chain=10`表示每条链丢弃前10个样本
- `n_samples=512`表示总共采样512个样本

### 2. 优化过程

使用随机梯度下降(SGD)优化器和自然梯度预条件(SR)进行优化：

```python
opt = nk.optimizer.Sgd(learning_rate=0.05)
sr = nk.optimizer.SR(diag_shift=0.01)
gs = nk.driver.VMC(ha, opt, variational_state=vs, preconditioner=sr)
gs.run(300, out="h2_molecule_ffnn")
```

### 3. 能量计算结果

经过300次迭代优化后，FFNN模型得到的H2分子基态能量为：
- VMC能量: -1.11667600 Ha
- FCI能量(参考): -1.13728383 Ha
- 能量误差: 0.02060784 Ha

## FFNN Ansatz的特点与局限性

### 1. 特点

1. **简单性**: FFNN结构简单，只有单隐藏层，易于实现和理解
2. **计算效率**: 由于结构简单，前向传播计算速度快
3. **参数少**: 相比深度网络，参数数量较少，训练相对容易
4. **通用性**: 可以应用于不同大小的量子系统

### 2. 局限性

1. **表达能力有限**: 单隐藏层的表达能力有限，可能无法精确描述复杂的量子关联
2. **缺乏物理先验**: 没有融入量子系统的物理先验知识，如对称性、费米子反对称性等
3. **精度限制**: 从结果看，与FCI能量有约0.02 Ha的误差，精度不如更复杂的Ansatz

## 与其他Ansatz的比较

### 1. 与RBM(受限玻尔兹曼机)比较

RBM是另一种常用的神经网络波函数Ansatz，相比FFNN：
- RBM具有更强的表达能力，可以学习更复杂的量子态
- RBM的采样通常更高效，因为其条件概率分布易于计算
- 但RBM的训练可能更复杂，容易陷入局部最优

### 2. 与Jastrow因子比较

Jastrow因子是一种经典的波函数Ansatz形式：
- Jastrow因子明确包含了粒子间的关联项
- 物理意义更明确，易于解释
- 但灵活性不如神经网络Ansatz

### 3. 与Slater行列式比较

Slater行列式是Hartree-Fock方法的波函数形式：
- Slater行列式满足费米子反对称性
- 是多体波函数的良好近似
- 但无法描述电子关联效应

## 改进方向

1. **增加网络深度**: 使用更深的网络结构可以提高表达能力
2. **融入物理先验**: 在网络设计中融入费米子反对称性等物理约束
3. **混合Ansatz**: 将FFNN与传统量子化学方法结合，如FFNN×Jastrow或FFNN×Slater
4. **优化激活函数**: 尝试其他激活函数，如tanh、sigmoid等，可能提高表达能力
5. **参数优化**: 调整隐藏层大小、学习率等超参数

## 结论

FFNN波函数Ansatz是一种简单而有效的神经网络量子态表示方法。虽然其表达能力有限，对于H2分子这样的小系统，它能够提供一个合理的基态能量近似。对于更复杂的量子系统，可能需要更复杂的网络结构或融入更多物理先验的Ansatz形式。

FFNN的优势在于其简单性和计算效率，使其成为学习和研究神经网络量子态的良好起点。在实际应用中，可以根据具体问题的需求选择或设计更合适的波函数Ansatz。