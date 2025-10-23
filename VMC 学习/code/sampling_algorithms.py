"""
采样算法知识的Python代码示例
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import multivariate_normal
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重复
np.random.seed(42)

# ==============================
# 1. 直接采样法
# ==============================

def uniform_sampling_example():
    """
    均匀分布采样示例
    """
    print("=== 均匀分布采样示例 ===")
    
    # 从[0,1)均匀分布中采样
    samples = np.random.uniform(0, 1, size=1000)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.7, label='采样结果')
    plt.plot([0, 1], [1, 1], 'r-', label='理论概率密度')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('均匀分布采样')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 从[a,b)均匀分布中采样
    a, b = 2, 5
    samples_ab = np.random.uniform(a, b, size=1000)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.hist(samples_ab, bins=30, density=True, alpha=0.7, label='采样结果')
    plt.plot([a, b], [1/(b-a), 1/(b-a)], 'r-', label='理论概率密度')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title(f'[{a},{b})均匀分布采样')
    plt.legend()
    plt.grid(True)
    plt.show()

def normal_sampling_example():
    """
    正态分布采样示例
    """
    print("\n=== 正态分布采样示例 ===")
    
    # 从标准正态分布N(0,1)中采样
    samples = np.random.normal(0, 1, size=10000)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='采样结果')
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', label='理论概率密度')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('标准正态分布采样')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 从一般正态分布N(μ,σ²)中采样
    mu, sigma = 5, 2
    samples_general = np.random.normal(mu, sigma, size=10000)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.hist(samples_general, bins=50, density=True, alpha=0.7, label='采样结果')
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='理论概率密度')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title(f'正态分布N({mu},{sigma}²)采样')
    plt.legend()
    plt.grid(True)
    plt.show()

def other_distributions_sampling_example():
    """
    其他常见分布的采样示例
    """
    print("\n=== 其他常见分布的采样示例 ===")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 指数分布
    scale = 1.0  # 1/λ
    samples_exp = np.random.exponential(scale, size=10000)
    axes[0, 0].hist(samples_exp, bins=50, density=True, alpha=0.7, label='采样结果')
    x = np.linspace(0, 10, 1000)
    axes[0, 0].plot(x, stats.expon.pdf(x, scale=scale), 'r-', label='理论概率密度')
    axes[0, 0].set_title('指数分布')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('概率密度')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 泊松分布
    lam = 5  # λ参数
    samples_poisson = np.random.poisson(lam, size=10000)
    axes[0, 1].hist(samples_poisson, bins=np.arange(0, 20) - 0.5, density=True, alpha=0.7, label='采样结果')
    x = np.arange(0, 20)
    axes[0, 1].plot(x, stats.poisson.pmf(x, lam), 'ro-', label='理论概率质量')
    axes[0, 1].set_title('泊松分布')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_ylabel('概率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 二项分布
    n, p = 10, 0.5  # 试验次数和成功概率
    samples_binomial = np.random.binomial(n, p, size=10000)
    axes[1, 0].hist(samples_binomial, bins=np.arange(0, n+2) - 0.5, density=True, alpha=0.7, label='采样结果')
    x = np.arange(0, n+1)
    axes[1, 0].plot(x, stats.binom.pmf(x, n, p), 'ro-', label='理论概率质量')
    axes[1, 0].set_title('二项分布')
    axes[1, 0].set_xlabel('k')
    axes[1, 0].set_ylabel('概率')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Beta分布
    a, b = 2, 5  # α和β参数
    samples_beta = np.random.beta(a, b, size=10000)
    axes[1, 1].hist(samples_beta, bins=50, density=True, alpha=0.7, label='采样结果')
    x = np.linspace(0, 1, 1000)
    axes[1, 1].plot(x, stats.beta.pdf(x, a, b), 'r-', label='理论概率密度')
    axes[1, 1].set_title('Beta分布')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('概率密度')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def inverse_transform_sampling_example():
    """
    逆变换采样法示例
    """
    print("\n=== 逆变换采样法示例 ===")
    
    # 定义目标分布的CDF
    def target_cdf(x):
        return 1 - np.exp(-x)  # 指数分布的CDF
    
    # 定义CDF的逆函数
    def inverse_cdf(u):
        return -np.log(1 - u)  # 指数分布的CDF逆函数
    
    # 从均匀分布中采样
    u = np.random.uniform(0, 1, size=10000)
    
    # 应用逆变换
    samples = inverse_cdf(u)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='采样结果')
    x = np.linspace(0, 10, 1000)
    plt.plot(x, stats.expon.pdf(x), 'r-', label='理论概率密度')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('逆变换采样法')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================
# 2. 拒绝采样法
# ==============================

def rejection_sampling_example():
    """
    拒绝采样法示例
    """
    print("\n=== 拒绝采样法示例 ===")
    
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
    
    # 计算接受率
    acceptance_rate = len(samples) / (len(samples) * M)  # 近似计算
    
    print(f"接受率: {acceptance_rate:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    x = np.linspace(-15, 15, 1000)
    plt.plot(x, target_pdf(x), 'r-', label='目标分布')
    plt.plot(x, M * proposal_pdf(x), 'g--', label='M·提议分布')
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('拒绝采样结果')
    plt.grid(True)
    plt.show()

# ==============================
# 3. 重要性采样法
# ==============================

def importance_sampling_example():
    """
    重要性采样法示例
    """
    print("\n=== 重要性采样法示例 ===")
    
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
    
    print(f"估计的期望值: {expectation:.4f}")
    
    # 计算有效样本大小
    effective_sample_size = 1 / np.sum(weights**2)
    print(f"有效样本大小: {effective_sample_size:.2f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    x = np.linspace(-15, 15, 1000)
    plt.plot(x, target_pdf(x), 'r-', label='目标分布')
    plt.plot(x, proposal_pdf(x), 'g-', label='提议分布')
    scatter = plt.scatter(samples, np.zeros_like(samples), c=weights, cmap='viridis', alpha=0.5, label='采样点')
    plt.colorbar(scatter, label='重要性权重')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('重要性采样结果')
    plt.grid(True)
    plt.show()

# ==============================
# 4. 马尔可夫链蒙特卡洛（MCMC）方法
# ==============================

def metropolis_hastings_example():
    """
    Metropolis-Hastings算法示例
    """
    print("\n=== Metropolis-Hastings算法示例 ===")
    
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
        accepted = 0
        
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
                accepted += 1
            
            samples.append(current_x)
        
        acceptance_rate = accepted / size
        return np.array(samples), acceptance_rate
    
    # 执行Metropolis-Hastings算法
    samples, acceptance_rate = metropolis_hastings(size=10000, initial_x=0, scale=1.0)
    
    print(f"接受率: {acceptance_rate:.4f}")
    
    # 可视化结果
    x = np.linspace(-4, 4, 1000)
    plt.figure(figsize=(12, 6))
    
    # 绘制采样结果的时间序列
    plt.subplot(1, 2, 1)
    plt.plot(samples[:500])
    plt.xlabel('迭代次数')
    plt.ylabel('x')
    plt.title('MCMC采样轨迹')
    plt.grid(True)
    
    # 绘制采样结果的直方图
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
    plt.plot(x, np.exp(log_target_pdf(x)), 'r-', label='目标分布')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('MCMC采样分布')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def gibbs_sampling_example():
    """
    Gibbs采样示例
    """
    print("\n=== Gibbs采样示例 ===")
    
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
    plt.grid(True)
    plt.show()

def hamiltonian_monte_carlo_example():
    """
    Hamiltonian Monte Carlo (HMC)示例
    """
    print("\n=== Hamiltonian Monte Carlo (HMC)示例 ===")
    
    # 定义目标分布的对数概率及其梯度
    def log_target_pdf(x):
        return -0.5 * x**2  # 标准正态分布的对数概率
    
    def grad_log_target_pdf(x):
        return -x  # 标准正态分布的对数概率梯度
    
    # HMC算法
    def hmc_sampling(size=1000, initial_x=0, step_size=0.1, num_steps=10):
        samples = []
        current_x = initial_x
        accepted = 0
        
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
                accepted += 1
            
            samples.append(current_x)
        
        acceptance_rate = accepted / size
        return np.array(samples), acceptance_rate
    
    # 执行HMC采样
    samples, acceptance_rate = hmc_sampling(size=10000, initial_x=0, step_size=0.1, num_steps=10)
    
    print(f"接受率: {acceptance_rate:.4f}")
    
    # 可视化结果
    x = np.linspace(-4, 4, 1000)
    plt.figure(figsize=(12, 6))
    
    # 绘制采样结果的时间序列
    plt.subplot(1, 2, 1)
    plt.plot(samples[:500])
    plt.xlabel('迭代次数')
    plt.ylabel('x')
    plt.title('HMC采样轨迹')
    plt.grid(True)
    
    # 绘制采样结果的直方图
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
    plt.plot(x, np.exp(log_target_pdf(x)), 'r-', label='目标分布')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('HMC采样分布')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================
# 5. 采样算法在变分量子态中的应用
# ==============================

def sample_from_wavefunction_example():
    """
    从试探波函数中采样示例
    """
    print("\n=== 从试探波函数中采样示例 ===")
    
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
        accepted = 0
        
        for i in range(size):
            # 提议新的位置
            proposed_x = current_x + np.random.normal(0, step_size)
            
            # 计算接受概率
            accept_prob = probability_density(proposed_x, alpha) / probability_density(current_x, alpha)
            
            # 决定是否接受
            if np.random.uniform(0, 1) < accept_prob:
                current_x = proposed_x
                accepted += 1
            
            samples.append(current_x)
        
        acceptance_rate = accepted / size
        return np.array(samples), acceptance_rate
    
    # 执行采样
    samples, acceptance_rate = sample_from_wavefunction(size=10000, alpha=1.0, step_size=1.0)
    
    print(f"接受率: {acceptance_rate:.4f}")
    
    # 可视化结果
    x = np.linspace(-4, 4, 1000)
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
    plt.plot(x, probability_density(x), 'r-', label='理论概率密度')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('从试探波函数中采样')
    plt.legend()
    plt.grid(True)
    plt.show()

def local_energy_example():
    """
    计算局部能量示例
    """
    print("\n=== 计算局部能量示例 ===")
    
    # 定义试探波函数（例如，一维谐振子的基态）
    def trial_wavefunction(x, alpha=1.0):
        """一维谐振子的基态波函数"""
        return (alpha/np.pi)**0.25 * np.exp(-0.5 * alpha * x**2)
    
    # 定义波函数的概率密度
    def probability_density(x, alpha=1.0):
        return trial_wavefunction(x, alpha)**2
    
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
    plt.grid(True)
    plt.show()

def optimize_wavefunction_example():
    """
    优化波函数参数示例
    """
    print("\n=== 优化波函数参数示例 ===")
    
    # 定义波函数及其对数导数
    def log_trial_wavefunction(x, alpha=1.0):
        """波函数的对数"""
        return 0.25 * np.log(alpha/np.pi) - 0.5 * alpha * x**2
    
    def grad_log_trial_wavefunction(x, alpha=1.0):
        """波函数对数的梯度"""
        return 0.25/alpha - 0.5 * x**2
    
    # 定义波函数的概率密度
    def probability_density(x, alpha=1.0):
        return np.exp(2 * log_trial_wavefunction(x, alpha))
    
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
        psi = np.exp(log_trial_wavefunction(x, alpha))
        H_psi = hamiltonian(x, lambda x, a: np.exp(log_trial_wavefunction(x, a)), alpha, h)
        return H_psi / psi
    
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
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(alphas)
    plt.xlabel('迭代次数')
    plt.ylabel('参数α')
    plt.title('参数优化过程')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================
# 6. 高级采样技术
# ==============================

def parallel_tempering_example():
    """
    并行回火示例
    """
    print("\n=== 并行回火示例 ===")
    
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
    plt.grid(True)
    plt.show()

def adaptive_metropolis_example():
    """
    自适应Metropolis示例
    """
    print("\n=== 自适应Metropolis示例 ===")
    
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
        scales = []
        
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
            scales.append(scale)
            
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
        
        return np.array(samples), np.array(scales)
    
    # 执行自适应Metropolis
    samples, scales = adaptive_metropolis(size=10000, initial_x=0, initial_scale=1.0)
    
    print(f"最终提议分布尺度: {scales[-1]:.6f}")
    
    # 可视化结果
    x = np.linspace(-4, 4, 1000)
    plt.figure(figsize=(12, 6))
    
    # 绘制采样结果的时间序列
    plt.subplot(1, 2, 1)
    plt.plot(samples[:500])
    plt.xlabel('迭代次数')
    plt.ylabel('x')
    plt.title('自适应Metropolis采样轨迹')
    plt.grid(True)
    
    # 绘制采样结果的直方图
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='采样结果')
    plt.plot(x, np.exp(log_target_pdf(x)), 'r-', label='目标分布')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('自适应Metropolis采样分布')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 可视化尺度调整过程
    plt.figure(figsize=(10, 6))
    plt.plot(scales)
    plt.xlabel('迭代次数')
    plt.ylabel('提议分布尺度')
    plt.title('自适应Metropolis尺度调整过程')
    plt.grid(True)
    plt.show()

# ==============================
# 7. 采样算法的性能评估
# ==============================

def effective_sample_size_example():
    """
    有效样本大小示例
    """
    print("\n=== 有效样本大小示例 ===")
    
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
        
        return ess, int_time, acf
    
    # 从标准正态分布中生成样本
    np.random.seed(42)
    samples = np.random.normal(0, 1, size=10000)
    
    # 计算有效样本大小
    ess, int_time, acf = calculate_ess(samples)
    print(f"独立样本的有效样本大小: {ess:.2f}")
    print(f"积分时间: {int_time:.2f}")
    
    # 生成具有高自相关的样本（例如，随机游走）
    high_corr_samples = np.cumsum(np.random.normal(0, 0.1, size=10000))
    
    # 计算有效样本大小
    ess_high, int_time_high, acf_high = calculate_ess(high_corr_samples)
    print(f"高自相关样本的有效样本大小: {ess_high:.2f}")
    print(f"高自相关样本的积分时间: {int_time_high:.2f}")
    
    # 可视化自相关函数
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(acf[:100])
    plt.xlabel('滞后')
    plt.ylabel('自相关函数')
    plt.title('独立样本的自相关函数')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(acf_high[:100])
    plt.xlabel('滞后')
    plt.ylabel('自相关函数')
    plt.title('高自相关样本的自相关函数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def gelman_rubin_example():
    """
    Gelman-Rubin统计量示例
    """
    print("\n=== Gelman-Rubin统计量示例 ===")
    
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
    
    # 可视化链的轨迹
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i in range(num_chains):
        plt.plot(converged_chains[i, :200], alpha=0.7, label=f'链 {i+1}')
    plt.xlabel('迭代次数')
    plt.ylabel('x')
    plt.title(f'收敛链 (R-hat = {r_hat_converged:.4f})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i in range(num_chains):
        plt.plot(non_converged_chains[i, :200], alpha=0.7, label=f'链 {i+1}')
    plt.xlabel('迭代次数')
    plt.ylabel('x')
    plt.title(f'未收敛链 (R-hat = {r_hat_non_converged:.4f})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================
# 8. 主程序
# ==============================

def main():
    """
    主程序，运行所有示例
    """
    print("采样算法知识示例程序")
    print("=" * 50)
    
    # 直接采样法示例
    uniform_sampling_example()
    normal_sampling_example()
    other_distributions_sampling_example()
    inverse_transform_sampling_example()
    
    # 拒绝采样法示例
    rejection_sampling_example()
    
    # 重要性采样法示例
    importance_sampling_example()
    
    # 马尔可夫链蒙特卡洛方法示例
    metropolis_hastings_example()
    gibbs_sampling_example()
    hamiltonian_monte_carlo_example()
    
    # 采样算法在变分量子态中的应用示例
    sample_from_wavefunction_example()
    local_energy_example()
    optimize_wavefunction_example()
    
    # 高级采样技术示例
    parallel_tempering_example()
    adaptive_metropolis_example()
    
    # 采样算法的性能评估示例
    effective_sample_size_example()
    gelman_rubin_example()
    
    print("\n所有示例运行完成！")

if __name__ == "__main__":
    main()