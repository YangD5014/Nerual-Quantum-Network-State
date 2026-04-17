# Introduction 
变分量子蒙特卡罗(VQMC)中在量子多体系统模拟上得到了突出的应用,特别是近年来,神经网络波函数的引入大大提高了VQMC的精度。神经网络波函数在分子系统VMC模拟中的应用已经被证明在精确描述具有多参考特征的状态和建立强相关性方面是非常成功的。最近，这些方法已经成功地扩展到通过基于惩罚的形式来模拟激发态。虽然在一些应用上的结果已经很有希望，但基于神经网络的激发态模拟方法仍然还在探索中。
本研究将关注使用混合架构的量子-经典混合 Ansatz 来对小分子体系的激发态进行模拟。一方面,有相关研究证明含参数量子线路与经典神经网络的混合架构可以更加

# 神经网络波函数

## 1. 神经网络波函数概述

神经网络波函数是一种基于人工神经网络的波函数表示方法。它将量子系统的波函数表示为一个参数化的函数，其中参数通过神经网络的权重和偏置来学习。神经网络波函数的优势在于它可以学习复杂的波函数表示，并且可以在硬件上进行并行计算，从而提高模拟效率。

对于量子系统，波函数 $\Psi(\mathbf{x})$ 可以表示为：

$$\Psi(\mathbf{x}; \boldsymbol{\theta}) = A(\theta)e^{f(\mathbf{x}; \boldsymbol{\theta})}$$

其中 $\mathbf{x}$ 是系统的配置（如自旋构型或粒子位置），$\boldsymbol{\theta}$ 是神经网络的参数，$f(\mathbf{x}; \boldsymbol{\theta})$ 是神经网络的输出。

## 2. 受限玻尔兹曼机（RBM）

受限玻尔兹曼机是一种两层神经网络，包含可见层和隐藏层。对于实数波函数，RBM 的能量函数为：

$$E(\mathbf{v}, \mathbf{h}) = -\sum_{i=1}^{N_v} a_i v_i - \sum_{j=1}^{N_h} b_j h_j - \sum_{i=1}^{N_v}\sum_{j=1}^{N_h} W_{ij} v_i h_j$$

其中：
- $\mathbf{v} = (v_1, v_2, \ldots, v_{N_v})$ 是可见层神经元的状态
- $\mathbf{h} = (h_1, h_2, \ldots, h_{N_h})$ 是隐藏层神经元的状态
- $a_i$ 是可见层的偏置
- $b_j$ 是隐藏层的偏置
- $W_{ij}$ 是连接权重

波函数的振幅可以表示为：

$$|\Psi(\mathbf{v})| = \sqrt{p(\mathbf{v})} = \frac{1}{\sqrt{Z}} \sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$

其中 $Z$ 是配分函数：

$$Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$

通过对隐藏层求和，可以得到：

$$|\Psi(\mathbf{v})| = \frac{1}{\sqrt{Z}} \prod_{j=1}^{N_h} 2\cosh\left(b_j + \sum_{i=1}^{N_v} W_{ij} v_i\right) \exp\left(\sum_{i=1}^{N_v} a_i v_i\right)$$

## 3. 复数受限玻尔兹曼机（Complex-value RBM）

对于量子系统，波函数通常是复数的。复数受限玻尔兹曼机通过引入复数权重和偏置来表示复数波函数。

# VMC的基本介绍和计算方法

## 1. 变分蒙特卡洛（VMC）基本原理

### 1.1 变分原理

对于量子力学体系，薛定谔方程为：

$$\hat{H}\Psi(\mathbf{R}) = E\Psi(\mathbf{R})$$

其中 $\hat{H}$ 是哈密顿算符，$\Psi(\mathbf{R})$ 是波函数，$E$ 是能量。

变分原理指出：对于任意试探波函数 $\Psi_T(\mathbf{R})$，其期望能量总是大于或等于基态能量：

$$E_0 \leq \frac{\langle \Psi_T|\hat{H}|\Psi_T \rangle}{\langle \Psi_T|\Psi_T \rangle} = \int \frac{|\Psi_T(\mathbf{R})|^2 \frac{\hat{H}\Psi_T(\mathbf{R})}{\Psi_T(\mathbf{R})} d\mathbf{R}}{\int |\Psi_T(\mathbf{R})|^2 d\mathbf{R}}$$

定义局部能量：

$$E_L(\mathbf{R}) = \frac{\hat{H}\Psi_T(\mathbf{R})}{\Psi_T(\mathbf{R})}$$

则期望能量可表示为：

$$E = \int |\Psi_T(\mathbf{R})|^2 E_L(\mathbf{R}) d\mathbf{R}$$

### 1.2 蒙特卡洛积分

利用蒙特卡洛方法，从概率分布 $|\Psi_T(\mathbf{R})|^2$ 中采样 $N$ 个构型 $\{\mathbf{R}_i\}$，则期望能量可近似为：

$$E \approx \frac{1}{N}\sum_{i=1}^{N} E_L(\mathbf{R}_i)$$

## 2. 基态 VMC 方法

### 2.1 Metropolis-Hastings 算法

为了从分布 $|\Psi_T(\mathbf{R})|^2$ 中采样，使用 Metropolis-Hastings 算法：

1. 从当前构型 $\mathbf{R}$ 提出一个新构型 $\mathbf{R}'$
2. 计算接受概率：

$$A(\mathbf{R} \rightarrow \mathbf{R}') = \min\left(1, \frac{|\Psi_T(\mathbf{R}')|^2}{|\Psi_T(\mathbf{R})|^2}\right)$$

3. 以概率 $A$ 接受新构型

### 2.2 能量梯度

对于参数化的试探波函数 $\Psi_T(\mathbf{R}; \boldsymbol{\alpha})$，其中 $\boldsymbol{\alpha} = \{\alpha_1, \alpha_2, \ldots, \alpha_M\}$ 是变分参数，能量关于参数的梯度为：

$$\frac{\partial E}{\partial \alpha_k} = 2\langle E_L(\mathbf{R}) \frac{\partial \ln |\Psi_T(\mathbf{R})|}{\partial \alpha_k} \rangle - 2\langle E_L(\mathbf{R}) \rangle \langle \frac{\partial \ln |\Psi_T(\mathbf{R})|}{\partial \alpha_k} \rangle$$

其中对数导数为：

$$O_k(\mathbf{R}) = \frac{\partial \ln |\Psi_T(\mathbf{R})|}{\partial \alpha_k} = \frac{1}{\Psi_T(\mathbf{R})} \frac{\partial \Psi_T(\mathbf{R})}{\partial \alpha_k}$$

## 3. 激发态 VMC 方法

### 3.1 激发态的变分原理

对于第 $n$ 激发态，如果试探波函数 $\Psi_T^{(n)}(\mathbf{R})$ 与所有低能态正交：

$$\langle \Psi_T^{(n)}|\Psi^{(m)} \rangle = 0, \quad m = 0, 1, \ldots, n-1$$

则其期望能量满足：

$$E_n \leq \frac{\langle \Psi_T^{(n)}|\hat{H}|\Psi_T^{(n)} \rangle}{\langle \Psi_T^{(n)}|\Psi_T^{(n)} \rangle}$$

### 3.2 正交化方法

#### 方法一：显式正交化

已知基态波函数 $\Psi_0(\mathbf{R})$，构造与基态正交的试探波函数：

$$\Psi_T^{(1)}(\mathbf{R}) = \hat{P}_1 \Phi(\mathbf{R}) = \left[1 - |\Psi_0\rangle\langle\Psi_0|\right]\Phi(\mathbf{R})$$

其中 $\Phi(\mathbf{R})$ 是初始试探波函数。

#### 方法二：能量最小化

定义包含正交约束的拉格朗日量：

$$\mathcal{L} = \langle \Psi_T^{(1)}|\hat{H}|\Psi_T^{(1)} \rangle - \lambda \langle \Psi_T^{(1)}|\Psi_T^{(1)} \rangle - \mu \langle \Psi_T^{(1)}|\Psi_0\rangle$$

### 3.3 正交约束下的能量梯度

对于第一激发态，考虑正交约束 $\langle \Psi_T^{(1)}|\Psi_0\rangle = 0$，能量梯度修正为：

$$\frac{\partial E^{(1)}}{\partial \alpha_k} = 2\langle E_L^{(1)}(\mathbf{R}) O_k^{(1)}(\mathbf{R}) \rangle - 2\langle E_L^{(1)}(\mathbf{R}) \rangle \langle O_k^{(1)}(\mathbf{R}) \rangle$$

其中：

$$E_L^{(1)}(\mathbf{R}) = \frac{\hat{H}\Psi_T^{(1)}(\mathbf{R})}{\Psi_T^{(1)}(\mathbf{R})}$$

$$O_k^{(1)}(\mathbf{R}) = \frac{1}{\Psi_T^{(1)}(\mathbf{R})} \frac{\partial \Psi_T^{(1)}(\mathbf{R})}{\partial \alpha_k}$$

## 4. 具体实现方法

### 4.1 神经网络量子态（NQS）

使用神经网络表示波函数：

$$\Psi_T(\mathbf{R}; \boldsymbol{\theta}) = e^{\ln \Psi_{\text{Jastrow}}(\mathbf{R}) + f_{\text{NN}}(\mathbf{R}; \boldsymbol{\theta})}$$

其中 $f_{\text{NN}}(\mathbf{R}; \boldsymbol{\theta})$ 是神经网络输出。

对数导数为：

$$\frac{\partial \ln \Psi_T(\mathbf{R})}{\partial \theta_k} = \frac{\partial f_{\text{NN}}(\mathbf{R}; \boldsymbol{\theta})}{\partial \theta_k}$$

### 4.2 激发态构造

#### 正交约束方法

定义正交化投影算符：

$$\hat{P}_1 = 1 - |\Psi_0\rangle\langle\Psi_0|$$

则第一激发态试探波函数为：

$$\Psi_T^{(1)}(\mathbf{R}) = \hat{P}_1 \Phi(\mathbf{R}) = \Phi(\mathbf{R}) - \Psi_0(\mathbf{R}) \int \Psi_0^*(\mathbf{R}')\Phi(\mathbf{R}') d\mathbf{R}'$$

在蒙特卡洛采样中，正交化系数为：

$$c = \frac{\langle \Psi_0|\Phi \rangle}{\langle \Psi_0|\Psi_0 \rangle} = \int \Psi_0^*(\mathbf{R})\Phi(\mathbf{R}) d\mathbf{R}$$

### 4.3 能量计算

对于正交化后的波函数，局部能量为：

$$E_L^{(1)}(\mathbf{R}) = \frac{\hat{H}\Psi_T^{(1)}(\mathbf{R})}{\Psi_T^{(1)}(\mathbf{R})} = \frac{\hat{H}[\Phi(\mathbf{R}) - c\Psi_0(\mathbf{R})]}{\Phi(\mathbf{R}) - c\Psi_0(\mathbf{R})}$$

## 5. 优化算法

### 5.1 随机梯度下降（SGD）

参数更新规则：

$$\alpha_k^{(t+1)} = \alpha_k^{(t)} - \eta \frac{\partial E}{\partial \alpha_k}$$

其中 $\eta$ 是学习率。

### 5.2 自然梯度

利用费舍尔信息矩阵：

$$F_{kl} = \langle O_k(\mathbf{R}) O_l(\mathbf{R}) \rangle - \langle O_k(\mathbf{R}) \rangle \langle O_l(\mathbf{R}) \rangle$$

自然梯度更新：

$$\boldsymbol{\alpha}^{(t+1)} = \boldsymbol{\alpha}^{(t)} - \eta \mathbf{F}^{-1} \nabla E$$

## 6. 数值稳定性考虑

### 6.1 重正化

为避免数值不稳定，对波函数进行重正化：

$$\Psi_T(\mathbf{R}) \rightarrow \frac{\Psi_T(\mathbf{R})}{\sqrt{\langle \Psi_T|\Psi_T \rangle}}$$

### 6.2 正交约束的数值实现

在蒙特卡洛采样中，正交约束通过以下方式实现：

$$\langle \Psi_T^{(1)}|\Psi_0 \rangle = \frac{1}{N}\sum_{i=1}^{N} \frac{\Psi_T^{(1)}(\mathbf{R}_i)\Psi_0(\mathbf{R}_i)}{|\Psi_T^{(1)}(\mathbf{R}_i)|^2} \approx 0$$

