# 采样算法知识

## 1. 采样算法概述

### 1.1 什么是采样算法

采样算法是从一个概率分布中生成样本的方法。在科学计算、机器学习和统计物理中，采样算法扮演着至关重要的角色，特别是在处理高维空间或复杂分布时。

### 1.2 采样算法的分类

采样算法可以分为几大类：

1. **直接采样法**：适用于简单分布，如均匀分布、正态分布等。
2. **拒绝采样法**：适用于已知概率密度函数但难以直接采样的分布。
3. **重要性采样法**：通过加权样本来估计期望值。
4. **马尔可夫链蒙特卡洛（MCMC）方法**：通过构建马尔可夫链来生成样本。
5. **变分推断方法**：通过优化近似分布来采样。

### 1.3 采样算法在变分量子态中的应用

在变分量子态学习中，采样算法用于：

- 从试探波函数中采样电子构型
- 计算局部能量和期望值
- 优化波函数参数
- 估计物理量的统计误差

## 2. 直接采样法

### 2.1 均匀分布采样

均匀分布是最简单的概率分布之一，可以通过编程语言内置的随机数生成器直接采样。

```python
import numpy as np

# 从[0,1)均匀分布中采样
samples = np.random.uniform(0, 1, size=1000)

# 从[a,b)均匀分布中采样
a, b = 2, 5
samples = np.random.uniform(a, b, size=1000)
```

### 2.2 正态分布采样

正态分布（高斯分布）是统计学和物理学中最重要的分布之一。

```python
# 从标准正态分布N(0,1)中采样
samples = np.random.normal(0, 1, size=1000)

# 从一般正态分布N(μ,σ²)中采样
mu, sigma = 5, 2
samples = np.random.normal(mu, sigma, size=1000)
```

### 2.3 其他常见分布的采样

```python
# 指数分布
scale = 1.0  # 1/λ
samples = np.random.exponential(scale, size=1000)

# 泊松分布
lam = 5  # λ参数
samples = np.random.poisson(lam, size=1000)

# 二项分布
n, p = 10, 0.5  # 试验次数和成功概率
samples = np.random.binomial(n, p, size=1000)
```

### 2.4 逆变换采样法

逆变换采样法是一种通用的采样方法，适用于任何累积分布函数（CDF）可逆的分布。

```python
import numpy as np
import scipy.stats as stats

# 定义目标分布的CDF
def target_cdf(x):
    return 1 - np.exp(-x)  # 指数分布的CDF

# 定义CDF的逆函数
def inverse_cdf(u):
    return -np.log(1 - u)  # 指数分布的CDF逆函数

# 从均匀分布中采样
u = np.random.uniform(0, 1, size=1000)

# 应用逆变换
samples = inverse_cdf(u)
```

## 3. 拒绝采样法

### 3.1 拒绝采样原理

拒绝采样法（Rejection Sampling）适用于已知概率密度函数（PDF）但难以直接采样的分布。其基本思想是：

1. 选择一个易于采样的提议分布（Proposal Distribution）q(x)
2. 确定一个常数M，使得对于所有x，有p(x) ≤ M·q(x)
3. 从q(x)中采样x，并从均匀分布中采样u
4. 如果u ≤ p(x)/(M·q(x))，则接受x；否则拒绝并重复

### 3.2 拒绝采样实现

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布（例如，一个复杂的分布）
def target_pdf(x):
    return 0.5 * np.exp(-np.abs(x-5)) + 0.5 * np.exp(-np.abs(x+5))

# 定义提议分布（例如，正态分布）
def proposal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# 从提议分布中采样
def sample_proposal(size=1):
    return np.random.normal(0, 1, size=size)

# 拒绝采样
def rejection_sampling(size=1000):
    samples = []
    M = 5.0  # 确保M·q(x) ≥ p(x)对所有x成立
    
    while len(samples) < size:
        # 从提议分布中采样
        x = sample_proposal()[0]
        
        # 计算接受概率
        accept_prob = target_pdf(x) / (M * proposal_pdf(x))
        
        # 从均匀分布中采样
        u = np.random.uniform(0, 1)
        
        # 决定是否接受
        if u <= accept_prob:
            samples.append(x)
    
    return np.array(samples)

# 执行拒绝采样
samples = rejection_sampling(size=10000)

# 可视化结果
x = np.linspace(-15, 15, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, target_pdf(x), 'r-', label='目标分布')
plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
plt.legend()
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('拒绝采样结果')
plt.show()
```

### 3.3 拒绝采样的优缺点

**优点**：
- 实现简单
- 不需要知道归一化常数
- 适用于任意形状的分布

**缺点**：
- 效率可能很低，特别是当目标分布与提议分布差异较大时
- 在高维空间中效率急剧下降（维度灾难）
- 难以找到合适的提议分布和常数M

## 4. 重要性采样法

### 4.1 重要性采样原理

重要性采样（Importance Sampling）是一种通过加权样本来估计期望值的方法。其核心思想是：

1. 从一个易于采样的提议分布q(x)中采样
2. 对每个样本x_i，计算重要性权重w_i = p(x_i)/q(x_i)
3. 使用加权样本估计期望值：E[f(X)] ≈ Σ[w_i·f(x_i)] / Σ[w_i]

### 4.2 重要性采样实现

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布（例如，一个复杂的分布）
def target_pdf(x):
    return 0.5 * np.exp(-np.abs(x-5)) + 0.5 * np.exp(-np.abs(x+5))

# 定义提议分布（例如，正态分布）
def proposal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# 从提议分布中采样
def sample_proposal(size=1):
    return np.random.normal(0, 1, size=size)

# 重要性采样
def importance_sampling(size=1000, func=lambda x: x):
    # 从提议分布中采样
    samples = sample_proposal(size=size)
    
    # 计算重要性权重
    weights = target_pdf(samples) / proposal_pdf(samples)
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    # 计算期望值
    expectation = np.sum(weights * func(samples))
    
    return samples, weights, expectation

# 定义一个函数来估计其期望值
def func(x):
    return x**2  # 例如，估计X²的期望值

# 执行重要性采样
samples, weights, expectation = importance_sampling(size=10000, func=func)

print(f"估计的期望值: {expectation}")

# 可视化结果
x = np.linspace(-15, 15, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, target_pdf(x), 'r-', label='目标分布')
plt.plot(x, proposal_pdf(x), 'g-', label='提议分布')
plt.scatter(samples, np.zeros_like(samples), c=weights, cmap='viridis', alpha=0.5, label='采样点')
plt.colorbar(label='重要性权重')
plt.legend()
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('重要性采样结果')
plt.show()
```

### 4.3 重要性采样的优缺点

**优点**：
- 可以估计任意函数的期望值
- 不需要知道归一化常数
- 比拒绝采样更高效

**缺点**：
- 权重可能具有很大的方差，导致估计不稳定
- 在高维空间中仍然面临挑战
- 需要选择合适的提议分布

## 5. 马尔可夫链蒙特卡洛（MCMC）方法

### 5.1 MCMC基本原理

马尔可夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）方法通过构建一个马尔可夫链，使其平稳分布为目标分布，从而生成样本。MCMC方法包括：

1. **Metropolis-Hastings算法**：最通用的MCMC方法
2. **Gibbs采样**：适用于条件分布易于采样的情况
3. **Hamiltonian Monte Carlo (HMC)**：利用物理系统动力学提高采样效率
4. **No-U-Turn Sampler (NUTS)**：HMC的自适应版本

### 5.2 Metropolis-Hastings算法

Metropolis-Hastings算法是MCMC方法中最基础和最通用的算法。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布的对数概率（避免数值下溢）
def log_target_pdf(x):
    return -0.5 * x**2  # 标准正态分布的对数概率

# 定义提议分布（例如，正态分布）
def proposal_sample(x, scale=1.0):
    return x + np.random.normal(0, scale)

# Metropolis-Hastings算法
def metropolis_hastings(size=1000, initial_x=0, scale=1.0):
    samples = []
    current_x = initial_x
    current_log_prob = log_target_pdf(current_x)
    
    for i in range(size):
        # 从提议分布中采样
        proposed_x = proposal_sample(current_x, scale)
        
        # 计算提议状态的对数概率
        proposed_log_prob = log_target_pdf(proposed_x)
        
        # 计算接受概率（对数空间）
        log_accept_prob = proposed_log_prob - current_log_prob
        
        # 决定是否接受
        if np.log(np.random.uniform(0, 1)) < log_accept_prob:
            current_x = proposed_x
            current_log_prob = proposed_log_prob
        
        samples.append(current_x)
    
    return np.array(samples)

# 执行Metropolis-Hastings算法
samples = metropolis_hastings(size=10000, initial_x=0, scale=1.0)

# 可视化结果
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(12, 6))

# 绘制采样结果的时间序列
plt.subplot(1, 2, 1)
plt.plot(samples[:500])
plt.xlabel('迭代次数')
plt.ylabel('x')
plt.title('MCMC采样轨迹')

# 绘制采样结果的直方图
plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
plt.plot(x, np.exp(log_target_pdf(x)), 'r-', label='目标分布')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('MCMC采样分布')
plt.legend()

plt.tight_layout()
plt.show()
```

### 5.3 Gibbs采样

Gibbs采样是一种特殊的MCMC方法，适用于条件分布易于采样的情况。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 定义二维目标分布（例如，二元正态分布）
def target_pdf(x, y):
    mean = np.array([0, 0])
    cov = np.array([[1, 0.8], [0.8, 1]])
    return multivariate_normal.pdf([x, y], mean=mean, cov=cov)

# 定义条件分布
def conditional_x_given_y(y):
    # 对于二元正态分布，条件分布也是正态分布
    mean_x_given_y = 0.8 * y  # μ₁ + ρ(σ₁/σ₂)(y - μ₂)
    var_x_given_y = 1 - 0.8**2  # σ₁²(1 - ρ²)
    return np.random.normal(mean_x_given_y, np.sqrt(var_x_given_y))

def conditional_y_given_x(x):
    # 对于二元正态分布，条件分布也是正态分布
    mean_y_given_x = 0.8 * x  # μ₂ + ρ(σ₂/σ₁)(x - μ₁)
    var_y_given_x = 1 - 0.8**2  # σ₂²(1 - ρ²)
    return np.random.normal(mean_y_given_x, np.sqrt(var_y_given_x))

# Gibbs采样
def gibbs_sampling(size=1000, initial_x=0, initial_y=0):
    samples_x = []
    samples_y = []
    current_x = initial_x
    current_y = initial_y
    
    for i in range(size):
        # 从x的条件分布中采样
        current_x = conditional_x_given_y(current_y)
        
        # 从y的条件分布中采样
        current_y = conditional_y_given_x(current_x)
        
        samples_x.append(current_x)
        samples_y.append(current_y)
    
    return np.array(samples_x), np.array(samples_y)

# 执行Gibbs采样
samples_x, samples_y = gibbs_sampling(size=10000)

# 可视化结果
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = target_pdf(X[j, i], Y[j, i])

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, alpha=0.5)
plt.scatter(samples_x, samples_y, s=5, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gibbs采样结果')
plt.colorbar(label='概率密度')
plt.show()
```

### 5.4 Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo (HMC)是一种利用物理系统动力学提高采样效率的MCMC方法。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布的对数概率及其梯度
def log_target_pdf(x):
    return -0.5 * x**2  # 标准正态分布的对数概率

def grad_log_target_pdf(x):
    return -x  # 标准正态分布的对数概率梯度

# HMC算法
def hmc_sampling(size=1000, initial_x=0, step_size=0.1, num_steps=10):
    samples = []
    current_x = initial_x
    
    for i in range(size):
        # 从正态分布中采样动量
        current_p = np.random.normal(0, 1)
        
        # 计算当前能量
        current_U = -log_target_pdf(current_x)
        current_K = 0.5 * current_p**2
        current_H = current_U + current_K
        
        # 蛙跳法（Leapfrog Method）模拟哈密顿动力学
        x = current_x
        p = current_p
        
        # 半步更新动量
        p = p - 0.5 * step_size * (-grad_log_target_pdf(x))
        
        # 全步更新位置
        for j in range(num_steps):
            x = x + step_size * p
            
            # 除了第一步和最后一步，全步更新动量
            if j < num_steps - 1:
                p = p - step_size * (-grad_log_target_pdf(x))
        
        # 最后半步更新动量
        p = p - 0.5 * step_size * (-grad_log_target_pdf(x))
        
        # 计算提议能量
        proposed_U = -log_target_pdf(x)
        proposed_K = 0.5 * p**2
        proposed_H = proposed_U + proposed_K
        
        # Metropolis接受准则
        if np.log(np.random.uniform(0, 1)) < current_H - proposed_H:
            current_x = x
        
        samples.append(current_x)
    
    return np.array(samples)

# 执行HMC采样
samples = hmc_sampling(size=10000, initial_x=0, step_size=0.1, num_steps=10)

# 可视化结果
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(12, 6))

# 绘制采样结果的时间序列
plt.subplot(1, 2, 1)
plt.plot(samples[:500])
plt.xlabel('迭代次数')
plt.ylabel('x')
plt.title('HMC采样轨迹')

# 绘制采样结果的直方图
plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
plt.plot(x, np.exp(log_target_pdf(x)), 'r-', label='目标分布')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('HMC采样分布')
plt.legend()

plt.tight_layout()
plt.show()
```

## 6. 采样算法在变分量子态中的应用

### 6.1 从试探波函数中采样

在变分量子态中，我们需要从试探波函数|ψ(θ)⟩中采样电子构型。对于玻色子系统，可以直接使用概率分布|⟨x|ψ(θ)⟩|²进行采样；对于费米子系统，由于波函数的反对称性，采样更加复杂。

```python
import numpy as np

# 定义试探波函数（例如，一维谐振子的基态）
def trial_wavefunction(x, alpha=1.0):
    """一维谐振子的基态波函数"""
    return (alpha/np.pi)**0.25 * np.exp(-0.5 * alpha * x**2)

# 定义波函数的概率密度
def probability_density(x, alpha=1.0):
    return trial_wavefunction(x, alpha)**2

# 使用Metropolis-Hastings算法从波函数中采样
def sample_from_wavefunction(size=1000, alpha=1.0, step_size=1.0):
    samples = []
    current_x = np.random.normal(0, 1/np.sqrt(alpha))
    
    for i in range(size):
        # 提议新的位置
        proposed_x = current_x + np.random.normal(0, step_size)
        
        # 计算接受概率
        accept_prob = probability_density(proposed_x, alpha) / probability_density(current_x, alpha)
        
        # 决定是否接受
        if np.random.uniform(0, 1) < accept_prob:
            current_x = proposed_x
        
        samples.append(current_x)
    
    return np.array(samples)

# 执行采样
samples = sample_from_wavefunction(size=10000, alpha=1.0, step_size=1.0)

# 可视化结果
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
plt.plot(x, probability_density(x), 'r-', label='理论概率密度')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('从试探波函数中采样')
plt.legend()
plt.show()
```

### 6.2 计算局部能量

在变分蒙特卡洛方法中，我们需要计算局部能量E_L(x) = Ĥψ(x)/ψ(x)，其中Ĥ是系统的哈密顿量。

```python
import numpy as np

# 定义哈密顿量（例如，一维谐振子）
def hamiltonian(x, wavefunction, alpha=1.0, h=1e-5):
    """计算哈密顿量作用于波函数的结果"""
    # 动能项（使用有限差分近似二阶导数）
    kinetic = -0.5 * (wavefunction(x + h, alpha) - 2 * wavefunction(x, alpha) + wavefunction(x - h, alpha)) / h**2
    
    # 势能项
    potential = 0.5 * x**2 * wavefunction(x, alpha)
    
    return kinetic + potential

# 计算局部能量
def local_energy(x, alpha=1.0, h=1e-5):
    """计算局部能量"""
    psi = trial_wavefunction(x, alpha)
    H_psi = hamiltonian(x, trial_wavefunction, alpha, h)
    return H_psi / psi

# 从波函数中采样
samples = sample_from_wavefunction(size=10000, alpha=1.0, step_size=1.0)

# 计算局部能量
local_energies = np.array([local_energy(x, alpha=1.0) for x in samples])

# 计算平均能量
mean_energy = np.mean(local_energies)
std_energy = np.std(local_energies) / np.sqrt(len(local_energies))

print(f"平均能量: {mean_energy:.6f} ± {std_energy:.6f}")
print(f"理论基态能量: 0.5")

# 可视化局部能量的分布
plt.figure(figsize=(10, 6))
plt.hist(local_energies, bins=50, density=True, alpha=0.5)
plt.axvline(mean_energy, color='r', linestyle='--', label=f'平均值: {mean_energy:.6f}')
plt.axvline(0.5, color='g', linestyle='--', label='理论值: 0.5')
plt.xlabel('局部能量')
plt.ylabel('概率密度')
plt.title('局部能量分布')
plt.legend()
plt.show()
```

### 6.3 优化波函数参数

在变分蒙特卡洛方法中，我们需要优化波函数的参数θ，以最小化能量期望值⟨E(θ)⟩。

```python
import numpy as np

# 定义波函数及其对数导数
def log_trial_wavefunction(x, alpha=1.0):
    """波函数的对数"""
    return 0.25 * np.log(alpha/np.pi) - 0.5 * alpha * x**2

def grad_log_trial_wavefunction(x, alpha=1.0):
    """波函数对数的梯度"""
    return 0.25/alpha - 0.5 * x**2

# 计算能量和梯度
def energy_and_gradient(alpha, size=10000, step_size=1.0):
    """计算能量及其对参数的梯度"""
    # 从波函数中采样
    samples = sample_from_wavefunction(size=size, alpha=alpha, step_size=step_size)
    
    # 计算局部能量
    local_energies = np.array([local_energy(x, alpha) for x in samples])
    
    # 计算波函数对数的梯度
    grad_log_psi = np.array([grad_log_trial_wavefunction(x, alpha) for x in samples])
    
    # 计算平均能量
    mean_energy = np.mean(local_energies)
    
    # 计算能量梯度
    energy_grad = 2 * np.mean((local_energies - mean_energy) * grad_log_psi)
    
    return mean_energy, energy_grad

# 梯度下降优化
def optimize_wavefunction(initial_alpha=1.0, learning_rate=0.01, num_iterations=100):
    """优化波函数参数"""
    alpha = initial_alpha
    energies = []
    alphas = []
    
    for i in range(num_iterations):
        # 计算能量和梯度
        energy, grad = energy_and_gradient(alpha, size=10000, step_size=1.0)
        
        # 更新参数
        alpha -= learning_rate * grad
        
        # 确保参数为正
        alpha = max(alpha, 0.1)
        
        energies.append(energy)
        alphas.append(alpha)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Energy = {energy:.6f}, Alpha = {alpha:.6f}")
    
    return alpha, energies, alphas

# 执行优化
optimal_alpha, energies, alphas = optimize_wavefunction(initial_alpha=2.0, learning_rate=0.01, num_iterations=100)

print(f"最优参数: {optimal_alpha:.6f}")
print(f"最优能量: {energies[-1]:.6f}")

# 可视化优化过程
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(energies)
plt.xlabel('迭代次数')
plt.ylabel('能量')
plt.title('能量优化过程')

plt.subplot(1, 2, 2)
plt.plot(alphas)
plt.xlabel('迭代次数')
plt.ylabel('参数α')
plt.title('参数优化过程')

plt.tight_layout()
plt.show()
```

## 7. 高级采样技术

### 7.1 并行回火（Parallel Tempering）

并行回火是一种通过在不同温度下并行运行多个马尔可夫链，并允许链之间交换状态来提高采样效率的技术。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布的对数概率
def log_target_pdf(x, beta=1.0):
    return -beta * (0.5 * x**2)  # 标准正态分布的对数概率，beta=1/T

# 定义提议分布
def proposal_sample(x, scale=1.0):
    return x + np.random.normal(0, scale)

# 并行回火算法
def parallel_tempering(size=1000, num_chains=5, initial_x=None, scale=1.0):
    if initial_x is None:
        initial_x = np.zeros(num_chains)
    
    # 设置不同温度（逆温度beta）
    betas = np.linspace(0.1, 1.0, num_chains)
    
    # 初始化链
    chains = np.zeros((size, num_chains))
    chains[0, :] = initial_x
    
    for i in range(1, size):
        # 对每个链执行Metropolis步骤
        for j in range(num_chains):
            # 提议新状态
            proposed_x = proposal_sample(chains[i-1, j], scale)
            
            # 计算接受概率
            log_accept_prob = log_target_pdf(proposed_x, betas[j]) - log_target_pdf(chains[i-1, j], betas[j])
            
            # 决定是否接受
            if np.log(np.random.uniform(0, 1)) < log_accept_prob:
                chains[i, j] = proposed_x
            else:
                chains[i, j] = chains[i-1, j]
        
        # 尝试交换相邻链的状态
        for j in range(num_chains - 1):
            # 计算交换接受概率
            log_swap_prob = (log_target_pdf(chains[i, j], betas[j+1]) + log_target_pdf(chains[i, j+1], betas[j]) -
                            log_target_pdf(chains[i, j], betas[j]) - log_target_pdf(chains[i, j+1], betas[j+1]))
            
            # 决定是否交换
            if np.log(np.random.uniform(0, 1)) < log_swap_prob:
                chains[i, j], chains[i, j+1] = chains[i, j+1], chains[i, j]
    
    return chains, betas

# 执行并行回火
chains, betas = parallel_tempering(size=10000, num_chains=5, initial_x=None, scale=1.0)

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制不同温度链的采样结果
for j in range(len(betas)):
    plt.hist(chains[:, j], bins=50, density=True, alpha=0.5, label=f'β={betas[j]:.1f}')

x = np.linspace(-4, 4, 1000)
plt.plot(x, np.exp(log_target_pdf(x, beta=1.0)), 'r-', label='目标分布')

plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('并行回火采样结果')
plt.legend()
plt.show()
```

### 7.2 自适应MCMC

自适应MCMC是一种在采样过程中自动调整提议分布参数的技术，以提高采样效率。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布的对数概率
def log_target_pdf(x):
    return -0.5 * x**2  # 标准正态分布的对数概率

# 自适应Metropolis算法
def adaptive_metropolis(size=1000, initial_x=0, initial_scale=1.0, adapt_start=100, adapt_interval=50):
    samples = []
    current_x = initial_x
    current_log_prob = log_target_pdf(current_x)
    scale = initial_scale
    accept_count = 0
    total_count = 0
    
    for i in range(size):
        # 提议新状态
        proposed_x = current_x + np.random.normal(0, scale)
        
        # 计算接受概率
        proposed_log_prob = log_target_pdf(proposed_x)
        log_accept_prob = proposed_log_prob - current_log_prob
        
        # 决定是否接受
        if np.log(np.random.uniform(0, 1)) < log_accept_prob:
            current_x = proposed_x
            current_log_prob = proposed_log_prob
            accept_count += 1
        
        total_count += 1
        samples.append(current_x)
        
        # 自适应调整提议分布
        if i > adapt_start and i % adapt_interval == 0:
            # 计算接受率
            accept_rate = accept_count / total_count
            
            # 调整尺度以使接受率接近目标值（例如0.234）
            target_rate = 0.234
            if accept_rate > target_rate:
                scale *= 1.1
            else:
                scale /= 1.1
            
            # 重置计数器
            accept_count = 0
            total_count = 0
    
    return np.array(samples), scale

# 执行自适应Metropolis
samples, final_scale = adaptive_metropolis(size=10000, initial_x=0, initial_scale=1.0)

print(f"最终提议分布尺度: {final_scale:.6f}")

# 可视化结果
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(12, 6))

# 绘制采样结果的时间序列
plt.subplot(1, 2, 1)
plt.plot(samples[:500])
plt.xlabel('迭代次数')
plt.ylabel('x')
plt.title('自适应Metropolis采样轨迹')

# 绘制采样结果的直方图
plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
plt.plot(x, np.exp(log_target_pdf(x)), 'r-', label='目标分布')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('自适应Metropolis采样分布')
plt.legend()

plt.tight_layout()
plt.show()
```

## 8. 采样算法的性能评估

### 8.1 有效样本大小

有效样本大小（Effective Sample Size, ESS）是评估MCMC采样效率的重要指标，表示独立样本的等效数量。

```python
import numpy as np

def calculate_ess(samples):
    """计算有效样本大小"""
    # 计算自相关函数
    n = len(samples)
    max_lag = min(n // 10, 1000)  # 最大滞后
    
    # 计算均值和方差
    mean = np.mean(samples)
    var = np.var(samples)
    
    # 计算自相关函数
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # 滞后0的自相关为1
    
    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean)) / var
    
    # 计算积分时间
    # 找到自相关函数首次穿过零的点
    cutoff = np.where(acf < 0)[0]
    if len(cutoff) > 0:
        cutoff = cutoff[0]
    else:
        cutoff = max_lag
    
    # 计算积分时间
    int_time = 1 + 2 * np.sum(acf[1:cutoff])
    
    # 计算有效样本大小
    ess = n / int_time
    
    return ess, int_time

# 从标准正态分布中生成样本
np.random.seed(42)
samples = np.random.normal(0, 1, size=10000)

# 计算有效样本大小
ess, int_time = calculate_ess(samples)
print(f"有效样本大小: {ess:.2f}")
print(f"积分时间: {int_time:.2f}")

# 生成具有高自相关的样本（例如，随机游走）
high_corr_samples = np.cumsum(np.random.normal(0, 0.1, size=10000))

# 计算有效样本大小
ess_high, int_time_high = calculate_ess(high_corr_samples)
print(f"高自相关样本的有效样本大小: {ess_high:.2f}")
print(f"高自相关样本的积分时间: {int_time_high:.2f}")
```

### 8.2 Gelman-Rubin统计量

Gelman-Rubin统计量（R-hat）用于评估多个马尔可夫链的收敛性，值接近1表示收敛。

```python
import numpy as np

def gelman_rubin(chains):
    """计算Gelman-Rubin统计量"""
    # chains的形状为 (num_chains, chain_length)
    num_chains, chain_length = chains.shape
    
    # 计算链内方差
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    
    # 计算链间方差
    overall_mean = np.mean(chain_means)
    between_chain_var = chain_length * np.var(chain_means, ddof=1)
    
    # 计算链内平均方差
    within_chain_var = np.mean(chain_vars)
    
    # 计算估计的边际后验方差
    marginal_posterior_var = ((chain_length - 1) / chain_length) * within_chain_var + (1 / chain_length) * between_chain_var
    
    # 计算R-hat统计量
    r_hat = np.sqrt(marginal_posterior_var / within_chain_var)
    
    return r_hat

# 生成多个收敛的链
num_chains = 4
chain_length = 1000
converged_chains = np.random.normal(0, 1, size=(num_chains, chain_length))

# 计算Gelman-Rubin统计量
r_hat_converged = gelman_rubin(converged_chains)
print(f"收敛链的R-hat: {r_hat_converged:.4f}")

# 生成多个未收敛的链
non_converged_chains = np.zeros((num_chains, chain_length))
for i in range(num_chains):
    non_converged_chains[i, :] = np.random.normal(i, 1, size=chain_length)

# 计算Gelman-Rubin统计量
r_hat_non_converged = gelman_rubin(non_converged_chains)
print(f"未收敛链的R-hat: {r_hat_non_converged:.4f}")
```

## 9. 采样算法的挑战与未来发展

### 9.1 高维采样挑战

在高维空间中，采样算法面临以下挑战：

1. **维度灾难**：随着维度的增加，采样空间呈指数级增长，使得覆盖整个空间变得困难。
2. **模式丢失**：在高维空间中，分布可能包含多个分离的模式，采样算法可能无法探索所有模式。
3. **计算复杂性**：高维空间中的计算成本通常很高，限制了采样效率。

### 9.2 解决方案

针对高维采样的挑战，有以下解决方案：

1. **分层采样**：将高维空间分解为低维子空间，分别进行采样。
2. **自适应采样**：根据采样过程中的信息动态调整采样策略。
3. **并行采样**：利用并行计算资源同时运行多个采样器。
4. **深度学习辅助采样**：使用深度学习模型（如归一化流、变分自编码器等）学习复杂的分布。

### 9.3 未来发展方向

采样算法的未来发展方向包括：

1. **自动化采样**：开发能够自动选择和调整采样策略的算法。
2. **物理信息采样**：结合物理系统的先验知识，提高采样效率。
3. **量子采样**：利用量子计算机的优势，开发量子采样算法。
4. **可解释采样**：提供采样过程的可解释性，帮助理解采样结果。

## 10. 总结与展望

采样算法是变分量子态学习的核心工具，它们使我们能够从复杂的量子态中采样，计算物理量的期望值，并优化波函数参数。随着计算能力的提高和算法的发展，采样算法将在量子物理、机器学习和科学计算中发挥越来越重要的作用。

在未来，我们可以期待更加高效、自动化的采样算法，它们将能够处理更高维、更复杂的分布，为量子态模拟和优化提供更强大的工具。同时，采样算法与其他领域（如深度学习、量子计算）的结合也将带来新的突破和创新。

## 11. 练习与实践

### 11.1 基础练习

1. 实现一个简单的拒绝采样算法，从标准正态分布中采样。
2. 实现一个重要性采样算法，估计一个复杂函数的期望值。
3. 实现一个Metropolis-Hastings算法，从双峰分布中采样。

### 11.2 进阶练习

1. 实现一个Gibbs采样算法，从二元正态分布中采样。
2. 实现一个HMC算法，从高维正态分布中采样。
3. 实现一个并行回火算法，从多峰分布中采样。

### 11.3 应用练习

1. 使用采样算法计算一维谐振子的基态能量。
2. 使用采样算法优化氢原子的变分波函数。
3. 使用采样算法模拟一维Ising模型的相变。

### 11.4 研究项目

1. 研究采样算法在神经网络量子态中的应用。
2. 开发一个新的采样算法，专门用于量子态采样。
3. 研究量子计算如何加速采样算法。

通过这些练习和实践，你将深入理解采样算法的原理和应用，为变分量子态学习打下坚实的基础。