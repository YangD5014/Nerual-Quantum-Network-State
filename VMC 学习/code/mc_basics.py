"""
蒙特卡洛基础示例代码
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 1. 基本蒙特卡洛积分
def monte_carlo_integration(f, a, b, n_samples=10000):
    """
    使用蒙特卡洛方法计算一维积分
    ∫_a^b f(x) dx
    
    参数:
        f: 被积函数
        a, b: 积分区间
        n_samples: 样本数量
        
    返回:
        积分估计值和标准误差
    """
    # 从[a,b]上的均匀分布中采样
    x = np.random.uniform(a, b, n_samples)
    # 计算函数值
    y = f(x)
    # 计算积分估计值
    integral = (b - a) * np.mean(y)
    # 计算标准误差
    std_error = (b - a) * np.std(y) / np.sqrt(n_samples)
    
    return integral, std_error

# 示例：计算 ∫_0^1 x^2 dx
def f1(x):
    return x**2

integral, error = monte_carlo_integration(f1, 0, 1)
print(f"∫_0^1 x^2 dx 的估计值: {integral:.6f} ± {error:.6f}")
print(f"真实值: 1/3 ≈ {1/3:.6f}")

# 2. 高维积分
def monte_carlo_integration_nd(f, bounds, n_samples=10000):
    """
    使用蒙特卡洛方法计算d维积分
    ∫ f(x) dx
    
    参数:
        f: 被积函数
        bounds: 每个维度的积分边界，例如 [(a1,b1), (a2,b2), ...]
        n_samples: 样本数量
        
    返回:
        积分估计值和标准误差
    """
    d = len(bounds)  # 维度
    volume = 1.0
    for a, b in bounds:
        volume *= (b - a)
    
    # 从每个维度的均匀分布中采样
    samples = []
    for a, b in bounds:
        samples.append(np.random.uniform(a, b, n_samples))
    
    # 计算函数值
    y = f(*samples)
    # 计算积分估计值
    integral = volume * np.mean(y)
    # 计算标准误差
    std_error = volume * np.std(y) / np.sqrt(n_samples)
    
    return integral, std_error

# 示例：计算 ∫_0^1 ∫_0^1 (x+y) dx dy
def f2(x, y):
    return x + y

integral, error = monte_carlo_integration_nd(f2, [(0,1), (0,1)])
print(f"∫_0^1 ∫_0^1 (x+y) dx dy 的估计值: {integral:.6f} ± {error:.6f}")
print(f"真实值: 1 ≈ {1:.6f}")

# 3. 重要性采样
def importance_sampling(f, p, p_sampler, n_samples=10000):
    """
    使用重要性采样计算积分
    ∫ f(x) dx = ∫ [f(x)/p(x)] p(x) dx
    
    参数:
        f: 被积函数
        p: 重要性分布的概率密度函数
        p_sampler: 从重要性分布中采样的函数
        n_samples: 样本数量
        
    返回:
        积分估计值和标准误差
    """
    # 从重要性分布中采样
    x = p_sampler(n_samples)
    # 计算权重
    weights = f(x) / p(x)
    # 计算积分估计值
    integral = np.mean(weights)
    # 计算标准误差
    std_error = np.std(weights) / np.sqrt(n_samples)
    
    return integral, std_error

# 示例：计算 ∫_0^∞ x^2 exp(-x) dx
# 使用指数分布作为重要性分布
def f3(x):
    return x**2 * np.exp(-x)

def p_exp(x):
    return np.exp(-x)  # 指数分布的PDF

def sample_exp(n_samples):
    return np.random.exponential(1, n_samples)  # 从指数分布中采样

integral, error = importance_sampling(f3, p_exp, sample_exp)
print(f"∫_0^∞ x^2 exp(-x) dx 的估计值: {integral:.6f} ± {error:.6f}")
print(f"真实值: 2! = 2 ≈ {2:.6f}")

# 4. Metropolis-Hastings算法
def metropolis_hastings(log_target, proposal_sampler, n_samples=10000, burn_in=1000):
    """
    Metropolis-Hastings算法
    
    参数:
        log_target: 目标分布的对数概率密度函数
        proposal_sampler: 提议分布的采样函数，返回 (x_new, log_q_ratio)
                        其中 log_q_ratio = log(q(x_old|x_new)) - log(q(x_new|x_old))
        n_samples: 样本数量
        burn_in: 预热期步数
        
    返回:
        样本数组
    """
    # 初始化
    x_current = 0.0  # 初始状态
    samples = []
    
    for i in range(n_samples + burn_in):
        # 从提议分布中采样
        x_new, log_q_ratio = proposal_sampler(x_current)
        
        # 计算接受概率的对数
        log_alpha = log_target(x_new) - log_target(x_current) + log_q_ratio
        
        # 决定是否接受
        if np.log(np.random.uniform(0, 1)) < log_alpha:
            x_current = x_new
        
        # 记录样本（跳过预热期）
        if i >= burn_in:
            samples.append(x_current)
    
    return np.array(samples)

# 示例：从标准正态分布中采样
def log_normal(x):
    return -0.5 * x**2  # 标准正态分布的对数PDF（忽略常数）

def normal_proposal(x_current):
    # 使用正态分布作为提议分布
    x_new = x_current + np.random.normal(0, 1)
    # 对称提议分布，log_q_ratio = 0
    return x_new, 0.0

samples = metropolis_hastings(log_normal, normal_proposal, n_samples=10000)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label='MCMC样本')
x = np.linspace(-4, 4, 1000)
plt.plot(x, np.exp(log_normal(x)) / np.sqrt(2*np.pi), 'r-', label='真实PDF')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('Metropolis-Hastings算法从标准正态分布中采样')
plt.legend()
plt.grid(True)
plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/metropolis_hastings_normal.png')
plt.close()

# 5. Gibbs采样
def gibbs_sampling(conditionals, n_samples=10000, burn_in=1000):
    """
    Gibbs采样算法
    
    参数:
        conditionals: 条件分布采样函数的列表，每个函数接受当前状态并返回该维度的采样
        n_samples: 样本数量
        burn_in: 预热期步数
        
    返回:
        样本数组，形状为 (n_samples, d)
    """
    d = len(conditionals)  # 维度
    # 初始化
    x_current = np.zeros(d)
    samples = []
    
    for i in range(n_samples + burn_in):
        # 对每个维度进行采样
        for j in range(d):
            x_current[j] = conditionals[j](x_current)
        
        # 记录样本（跳过预热期）
        if i >= burn_in:
            samples.append(x_current.copy())
    
    return np.array(samples)

# 示例：从二元正态分布中采样
# 假设我们想从 N(μ, Σ) 中采样，其中 μ = [0, 0]，Σ = [[1, 0.8], [0.8, 1]]
mu = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])
cov_inv = np.linalg.inv(cov)

# 条件分布：p(x1|x2) 和 p(x2|x1)
def conditional_x1(x):
    # p(x1|x2) ~ N(μ1 + Σ12*Σ22^{-1}*(x2-μ2), Σ11 - Σ12*Σ22^{-1}*Σ21)
    mean = mu[0] + cov[0,1]/cov[1,1] * (x[1] - mu[1])
    var = cov[0,0] - cov[0,1]**2/cov[1,1]
    return np.random.normal(mean, np.sqrt(var))

def conditional_x2(x):
    # p(x2|x1) ~ N(μ2 + Σ21*Σ11^{-1}*(x1-μ1), Σ22 - Σ21*Σ11^{-1}*Σ12)
    mean = mu[1] + cov[1,0]/cov[0,0] * (x[0] - mu[0])
    var = cov[1,1] - cov[1,0]**2/cov[0,0]
    return np.random.normal(mean, np.sqrt(var))

samples = gibbs_sampling([conditional_x1, conditional_x2], n_samples=10000)

# 绘制结果
plt.figure(figsize=(10, 8))
plt.scatter(samples[:,0], samples[:,1], alpha=0.5, s=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gibbs采样从二元正态分布中采样')
plt.grid(True)

# 绘制等高线
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
rv = stats.multivariate_normal(mu, cov)
plt.contour(x, y, rv.pdf(pos), levels=5, colors='red')

plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/gibbs_bivariate_normal.png')
plt.close()

# 6. Ising模型模拟
def ising_model_metropolis(size, temperature, n_steps=10000):
    """
    使用Metropolis算法模拟Ising模型
    
    参数:
        size: 格点大小 (size x size)
        temperature: 温度
        n_steps: 蒙特卡洛步数
        
    返回:
        最终自旋构型和能量演化
    """
    # 初始化随机自旋构型
    spins = np.random.choice([-1, 1], size=(size, size))
    
    # 计算初始能量
    def calculate_energy(spins):
        energy = 0
        for i in range(size):
            for j in range(size):
                # 周期性边界条件
                right = (i + 1) % size
                down = (j + 1) % size
                energy -= spins[i, j] * (spins[right, j] + spins[i, down])
        return energy
    
    energy = calculate_energy(spins)
    energies = [energy]
    
    # 蒙特卡洛模拟
    for step in range(n_steps):
        # 随机选择一个格点
        i, j = np.random.randint(0, size, 2)
        
        # 计算翻转自旋的能量变化
        right = (i + 1) % size
        left = (i - 1) % size
        down = (j + 1) % size
        up = (j - 1) % size
        
        delta_E = 2 * spins[i, j] * (spins[right, j] + spins[left, j] + spins[i, down] + spins[i, up])
        
        # Metropolis准则
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / temperature):
            spins[i, j] *= -1
            energy += delta_E
        
        energies.append(energy)
    
    return spins, np.array(energies)

# 模拟不同温度下的Ising模型
temperatures = [1.0, 2.27, 4.0]  # 2.27是临界温度的近似值
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, T in enumerate(temperatures):
    spins, energies = ising_model_metropolis(32, T, n_steps=10000)
    
    # 绘制最终自旋构型
    axes[idx].imshow(spins, cmap='binary')
    axes[idx].set_title(f'T = {T}')
    axes[idx].axis('off')

plt.suptitle('Ising模型在不同温度下的自旋构型')
plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/ising_model_temperatures.png')
plt.close()

# 绘制能量演化
plt.figure(figsize=(10, 6))
for T in temperatures:
    spins, energies = ising_model_metropolis(32, T, n_steps=10000)
    plt.plot(energies, label=f'T = {T}')

plt.xlabel('蒙特卡洛步数')
plt.ylabel('能量')
plt.title('Ising模型的能量演化')
plt.legend()
plt.grid(True)
plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/ising_energy_evolution.png')
plt.close()

# 7. 方差减少技术：重要性采样示例
def variance_reduction_example():
    """
    比较普通蒙特卡洛和重要性采样的方差
    """
    # 计算积分 ∫_0^1 exp(x) dx
    def f(x):
        return np.exp(x)
    
    # 普通蒙特卡洛
    n_samples = 1000
    x_uniform = np.random.uniform(0, 1, n_samples)
    mc_estimates = (1/n_samples) * np.sum(f(x_uniform))
    mc_variance = np.var(f(x_uniform)) / n_samples
    
    # 重要性采样：使用线性分布 p(x) = 2x
    def p_linear(x):
        return 2 * x
    
    def sample_linear(n_samples):
        # 从p(x) = 2x中采样，使用逆变换法
        u = np.random.uniform(0, 1, n_samples)
        return np.sqrt(u)
    
    x_linear = sample_linear(n_samples)
    weights = f(x_linear) / p_linear(x_linear)
    is_estimates = np.mean(weights)
    is_variance = np.var(weights) / n_samples
    
    print(f"普通蒙特卡洛: 估计值 = {mc_estimates:.6f}, 方差 = {mc_variance:.6f}")
    print(f"重要性采样: 估计值 = {is_estimates:.6f}, 方差 = {is_variance:.6f}")
    print(f"真实值: e-1 ≈ {np.e-1:.6f}")
    print(f"方差减少: {mc_variance/is_variance:.2f}倍")

variance_reduction_example()

# 8. 收敛性分析
def convergence_analysis():
    """
    分析蒙特卡洛积分的收敛性
    """
    # 计算积分 ∫_0^1 x^2 dx
    def f(x):
        return x**2
    
    # 不同样本数量
    sample_sizes = np.logspace(1, 5, 20).astype(int)
    estimates = []
    errors = []
    
    for n in sample_sizes:
        integral, error = monte_carlo_integration(f, 0, 1, n_samples=n)
        estimates.append(integral)
        errors.append(error)
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    # 估计值随样本数量的变化
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, estimates, 'o-')
    plt.axhline(y=1/3, color='r', linestyle='--', label='真实值')
    plt.xscale('log')
    plt.xlabel('样本数量')
    plt.ylabel('积分估计值')
    plt.title('蒙特卡洛积分估计值随样本数量的变化')
    plt.legend()
    plt.grid(True)
    
    # 误差随样本数量的变化
    plt.subplot(1, 2, 2)
    plt.plot(sample_sizes, errors, 'o-', label='实际误差')
    plt.plot(sample_sizes, 1/np.sqrt(sample_sizes), 'r--', label='1/√n')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('样本数量')
    plt.ylabel('误差')
    plt.title('蒙特卡洛积分误差随样本数量的变化')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/monte_carlo_convergence.png')
    plt.close()

convergence_analysis()

print("蒙特卡洛基础示例代码执行完成！")