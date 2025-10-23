"""
变分蒙特卡洛方法示例代码
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 1. 一维谐振子的变分蒙特卡洛
class HarmonicOscillatorVMC:
    """
    一维谐振子的变分蒙特卡洛模拟
    """
    def __init__(self, mass=1.0, omega=1.0, hbar=1.0):
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
        
    def wavefunction(self, x, alpha):
        """
        试探波函数: psi(x) = exp(-alpha * x^2)
        """
        return np.exp(-alpha * x**2)
    
    def local_energy(self, x, alpha):
        """
        局部能量计算
        E_L(x) = - (hbar^2 / 2m) * (d^2/dx^2 psi) / psi + (1/2) * m * omega^2 * x^2
        """
        # 对于 psi(x) = exp(-alpha * x^2)
        # d^2/dx^2 psi / psi = -2*alpha + 4*alpha^2*x^2
        kinetic = -self.hbar**2 / (2 * self.mass) * (-2*alpha + 4*alpha**2 * x**2)
        potential = 0.5 * self.mass * self.omega**2 * x**2
        return kinetic + potential
    
    def metropolis_sample(self, alpha, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        使用Metropolis算法从|psi(x)|^2中采样
        """
        x = 0.0  # 初始位置
        samples = []
        
        for i in range(n_samples + burn_in):
            # 提议新位置
            x_new = x + step_size * np.random.normal()
            
            # 计算接受概率
            psi_ratio = self.wavefunction(x_new, alpha) / self.wavefunction(x, alpha)
            acceptance_prob = psi_ratio**2
            
            # 决定是否接受
            if np.random.random() < acceptance_prob:
                x = x_new
            
            # 记录样本（跳过预热期）
            if i >= burn_in:
                samples.append(x)
        
        return np.array(samples)
    
    def energy_estimate(self, alpha, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        计算能量期望值
        """
        samples = self.metropolis_sample(alpha, n_samples, step_size, burn_in)
        local_energies = self.local_energy(samples, alpha)
        return np.mean(local_energies), np.std(local_energies) / np.sqrt(len(samples))
    
    def optimize_alpha(self, alphas=None):
        """
        优化alpha参数以最小化能量
        """
        if alphas is None:
            alphas = np.linspace(0.1, 2.0, 20)
        
        energies = []
        errors = []
        
        for alpha in alphas:
            energy, error = self.energy_estimate(alpha)
            energies.append(energy)
            errors.append(error)
            print(f"alpha = {alpha:.3f}, Energy = {energy:.6f} ± {error:.6f}")
        
        # 找到最优alpha
        min_idx = np.argmin(energies)
        optimal_alpha = alphas[min_idx]
        min_energy = energies[min_idx]
        
        # 绘制能量随alpha的变化
        plt.figure(figsize=(10, 6))
        plt.errorbar(alphas, energies, yerr=errors, fmt='o-', capsize=5)
        plt.axvline(x=optimal_alpha, color='r', linestyle='--', label=f'最优alpha = {optimal_alpha:.3f}')
        plt.axhline(y=0.5*self.hbar*self.omega, color='g', linestyle='--', label='真实基态能量')
        plt.xlabel('alpha')
        plt.ylabel('能量')
        plt.title('一维谐振子的变分能量随参数alpha的变化')
        plt.legend()
        plt.grid(True)
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/harmonic_oscillator_energy.png')
        plt.close()
        
        return optimal_alpha, min_energy

# 运行一维谐振子的VMC模拟
print("=== 一维谐振子的变分蒙特卡洛模拟 ===")
ho_vmc = HarmonicOscillatorVMC()
optimal_alpha, min_energy = ho_vmc.optimize_alpha()
print(f"最优alpha: {optimal_alpha:.6f}")
print(f"最小能量: {min_energy:.6f}")
print(f"真实基态能量: {0.5:.6f}")
print(f"解析解alpha: {ho_vmc.mass * ho_vmc.omega / ho_vmc.hbar:.6f}")

# 2. 氢原子的变分蒙特卡洛
class HydrogenAtomVMC:
    """
    氢原子的变分蒙特卡洛模拟
    """
    def __init__(self, hbar=1.0, mass=1.0, e=1.0):
        self.hbar = hbar
        self.mass = mass
        self.e = e
        
    def wavefunction(self, r, alpha):
        """
        试探波函数: psi(r) = exp(-alpha * r)
        """
        return np.exp(-alpha * r)
    
    def local_energy(self, r, alpha):
        """
        局部能量计算
        E_L(r) = - (hbar^2 / 2m) * (d^2/dr^2 psi) / psi - e^2 / r
        """
        # 对于 psi(r) = exp(-alpha * r)
        # d^2/dr^2 psi / psi = alpha^2 - 2*alpha/r
        kinetic = -self.hbar**2 / (2 * self.mass) * (alpha**2 - 2*alpha/r)
        potential = -self.e**2 / r
        return kinetic + potential
    
    def metropolis_sample(self, alpha, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        使用Metropolis算法从|psi(r)|^2中采样
        """
        r = 1.0  # 初始位置
        samples = []
        
        for i in range(n_samples + burn_in):
            # 提议新位置（确保r > 0）
            r_new = max(0.01, r + step_size * np.random.normal())
            
            # 计算接受概率（考虑三维体积元）
            psi_ratio = self.wavefunction(r_new, alpha) / self.wavefunction(r, alpha)
            acceptance_prob = psi_ratio**2 * (r_new / r)**2
            
            # 决定是否接受
            if np.random.random() < acceptance_prob:
                r = r_new
            
            # 记录样本（跳过预热期）
            if i >= burn_in:
                samples.append(r)
        
        return np.array(samples)
    
    def energy_estimate(self, alpha, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        计算能量期望值
        """
        samples = self.metropolis_sample(alpha, n_samples, step_size, burn_in)
        local_energies = self.local_energy(samples, alpha)
        return np.mean(local_energies), np.std(local_energies) / np.sqrt(len(samples))
    
    def optimize_alpha(self, alphas=None):
        """
        优化alpha参数以最小化能量
        """
        if alphas is None:
            alphas = np.linspace(0.5, 1.5, 20)
        
        energies = []
        errors = []
        
        for alpha in alphas:
            energy, error = self.energy_estimate(alpha)
            energies.append(energy)
            errors.append(error)
            print(f"alpha = {alpha:.3f}, Energy = {energy:.6f} ± {error:.6f}")
        
        # 找到最优alpha
        min_idx = np.argmin(energies)
        optimal_alpha = alphas[min_idx]
        min_energy = energies[min_idx]
        
        # 绘制能量随alpha的变化
        plt.figure(figsize=(10, 6))
        plt.errorbar(alphas, energies, yerr=errors, fmt='o-', capsize=5)
        plt.axvline(x=optimal_alpha, color='r', linestyle='--', label=f'最优alpha = {optimal_alpha:.3f}')
        plt.axhline(y=-0.5*self.mass*self.e**4/self.hbar**2, color='g', linestyle='--', label='真实基态能量')
        plt.xlabel('alpha')
        plt.ylabel('能量')
        plt.title('氢原子的变分能量随参数alpha的变化')
        plt.legend()
        plt.grid(True)
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/hydrogen_atom_energy.png')
        plt.close()
        
        return optimal_alpha, min_energy

# 运行氢原子的VMC模拟
print("\n=== 氢原子的变分蒙特卡洛模拟 ===")
h_vmc = HydrogenAtomVMC()
optimal_alpha, min_energy = h_vmc.optimize_alpha()
print(f"最优alpha: {optimal_alpha:.6f}")
print(f"最小能量: {min_energy:.6f}")
print(f"真实基态能量: {-0.5:.6f}")
print(f"解析解alpha: {h_vmc.mass * h_vmc.e**2 / h_vmc.hbar**2:.6f}")

# 3. 一维量子谐振子链的变分蒙特卡洛
class HarmonicChainVMC:
    """
    一维量子谐振子链的变分蒙特卡洛模拟
    """
    def __init__(self, n_sites=10, mass=1.0, omega=1.0, coupling=0.1, hbar=1.0):
        self.n_sites = n_sites
        self.mass = mass
        self.omega = omega
        self.coupling = coupling
        self.hbar = hbar
        
    def wavefunction(self, x, alpha, beta):
        """
        试探波函数: psi(x) = exp(-alpha * sum_i x_i^2 - beta * sum_i x_i * x_{i+1})
        """
        onsite = alpha * np.sum(x**2)
        neighbor = beta * np.sum(x[:-1] * x[1:])
        return np.exp(-onsite - neighbor)
    
    def local_energy(self, x, alpha, beta):
        """
        局部能量计算
        """
        # 计算动能项
        # 对于 psi(x) = exp(-alpha * sum_i x_i^2 - beta * sum_i x_i * x_{i+1})
        # d/dx_i ln psi = -2*alpha*x_i - beta*(x_{i-1} + x_{i+1})
        # d^2/dx_i^2 ln psi = -2*alpha
        # (d/dx_i ln psi)^2 = [2*alpha*x_i + beta*(x_{i-1} + x_{i+1})]^2
        
        kinetic = 0
        for i in range(self.n_sites):
            # 周期性边界条件
            i_prev = (i - 1) % self.n_sites
            i_next = (i + 1) % self.n_sites
            
            d_ln_psi = -2*alpha*x[i] - beta*(x[i_prev] + x[i_next])
            d2_ln_psi = -2*alpha
            
            kinetic += -self.hbar**2 / (2 * self.mass) * (d2_ln_psi + d_ln_psi**2)
        
        # 计算势能项
        potential = 0
        for i in range(self.n_sites):
            # 在位势能
            potential += 0.5 * self.mass * self.omega**2 * x[i]**2
            
            # 邻位耦合
            i_next = (i + 1) % self.n_sites
            potential += 0.5 * self.coupling * (x[i] - x[i_next])**2
        
        return kinetic + potential
    
    def metropolis_sample(self, alpha, beta, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        使用Metropolis算法从|psi(x)|^2中采样
        """
        x = np.random.normal(0, 1, self.n_sites)  # 初始位置
        samples = []
        
        for i in range(n_samples + burn_in):
            # 提议新位置
            site = np.random.randint(0, self.n_sites)
            x_new = x.copy()
            x_new[site] += step_size * np.random.normal()
            
            # 计算接受概率
            psi_ratio = self.wavefunction(x_new, alpha, beta) / self.wavefunction(x, alpha, beta)
            acceptance_prob = psi_ratio**2
            
            # 决定是否接受
            if np.random.random() < acceptance_prob:
                x = x_new
            
            # 记录样本（跳过预热期）
            if i >= burn_in:
                samples.append(x.copy())
        
        return np.array(samples)
    
    def energy_estimate(self, alpha, beta, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        计算能量期望值
        """
        samples = self.metropolis_sample(alpha, beta, n_samples, step_size, burn_in)
        local_energies = np.array([self.local_energy(x, alpha, beta) for x in samples])
        return np.mean(local_energies), np.std(local_energies) / np.sqrt(len(samples))
    
    def optimize_parameters(self, alphas=None, betas=None):
        """
        优化alpha和beta参数以最小化能量
        """
        if alphas is None:
            alphas = np.linspace(0.1, 2.0, 10)
        if betas is None:
            betas = np.linspace(-0.5, 0.5, 10)
        
        # 创建参数网格
        alpha_grid, beta_grid = np.meshgrid(alphas, betas)
        energy_grid = np.zeros_like(alpha_grid)
        
        # 计算每个参数组合的能量
        for i in range(len(betas)):
            for j in range(len(alphas)):
                energy, _ = self.energy_estimate(alpha_grid[i,j], beta_grid[i,j])
                energy_grid[i,j] = energy
                print(f"alpha = {alpha_grid[i,j]:.3f}, beta = {beta_grid[i,j]:.3f}, Energy = {energy:.6f}")
        
        # 找到最优参数
        min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
        optimal_alpha = alpha_grid[min_idx]
        optimal_beta = beta_grid[min_idx]
        min_energy = energy_grid[min_idx]
        
        # 绘制能量随参数的变化
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(alpha_grid, beta_grid, energy_grid, 20, cmap='viridis')
        plt.colorbar(contour, label='能量')
        plt.plot(optimal_alpha, optimal_beta, 'r*', markersize=15, label=f'最优参数')
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.title('一维量子谐振子链的变分能量随参数的变化')
        plt.legend()
        plt.grid(True)
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/harmonic_chain_energy.png')
        plt.close()
        
        return optimal_alpha, optimal_beta, min_energy

# 运行一维量子谐振子链的VMC模拟
print("\n=== 一维量子谐振子链的变分蒙特卡洛模拟 ===")
hc_vmc = HarmonicChainVMC(n_sites=6, coupling=0.1)
optimal_alpha, optimal_beta, min_energy = hc_vmc.optimize_parameters()
print(f"最优alpha: {optimal_alpha:.6f}")
print(f"最优beta: {optimal_beta:.6f}")
print(f"最小能量: {min_energy:.6f}")

# 4. 梯度优化示例
class GradientOptimizationVMC:
    """
    使用梯度优化的变分蒙特卡洛示例
    """
    def __init__(self, mass=1.0, omega=1.0, hbar=1.0):
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
        
    def wavefunction(self, x, alpha):
        """
        试探波函数: psi(x) = exp(-alpha * x^2)
        """
        return np.exp(-alpha * x**2)
    
    def wavefunction_log(self, x, alpha):
        """
        波函数的对数
        """
        return -alpha * x**2
    
    def wavefunction_log_gradient(self, x, alpha):
        """
        波函数对数关于alpha的梯度
        """
        return -x**2
    
    def local_energy(self, x, alpha):
        """
        局部能量计算
        """
        kinetic = -self.hbar**2 / (2 * self.mass) * (-2*alpha + 4*alpha**2 * x**2)
        potential = 0.5 * self.mass * self.omega**2 * x**2
        return kinetic + potential
    
    def metropolis_sample(self, alpha, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        使用Metropolis算法从|psi(x)|^2中采样
        """
        x = 0.0  # 初始位置
        samples = []
        
        for i in range(n_samples + burn_in):
            # 提议新位置
            x_new = x + step_size * np.random.normal()
            
            # 计算接受概率
            psi_ratio = self.wavefunction(x_new, alpha) / self.wavefunction(x, alpha)
            acceptance_prob = psi_ratio**2
            
            # 决定是否接受
            if np.random.random() < acceptance_prob:
                x = x_new
            
            # 记录样本（跳过预热期）
            if i >= burn_in:
                samples.append(x)
        
        return np.array(samples)
    
    def energy_gradient(self, alpha, n_samples=10000, step_size=1.0, burn_in=1000):
        """
        计算能量关于alpha的梯度
        dE/dalpha = 2 * <(E_L - <E_L>) * d(ln psi)/dalpha>
        """
        samples = self.metropolis_sample(alpha, n_samples, step_size, burn_in)
        local_energies = self.local_energy(samples, alpha)
        log_gradients = self.wavefunction_log_gradient(samples, alpha)
        
        energy_mean = np.mean(local_energies)
        gradient = 2 * np.mean((local_energies - energy_mean) * log_gradients)
        
        return gradient
    
    def gradient_descent(self, initial_alpha=0.5, learning_rate=0.01, n_iterations=100):
        """
        使用梯度下降优化alpha
        """
        alpha = initial_alpha
        alphas = [alpha]
        energies = []
        
        for i in range(n_iterations):
            # 计算能量
            energy, _ = self.energy_estimate(alpha)
            energies.append(energy)
            
            # 计算梯度
            gradient = self.energy_gradient(alpha)
            
            # 更新alpha
            alpha -= learning_rate * gradient
            alphas.append(alpha)
            
            if i % 10 == 0:
                print(f"Iteration {i}: alpha = {alpha:.6f}, Energy = {energy:.6f}, Gradient = {gradient:.6f}")
        
        # 绘制优化过程
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(alphas[:-1], energies, 'o-')
        plt.xlabel('alpha')
        plt.ylabel('能量')
        plt.title('能量随alpha的变化')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(n_iterations), alphas[:-1], 'o-')
        plt.axhline(y=self.mass*self.omega/self.hbar, color='r', linestyle='--', label='解析解')
        plt.xlabel('迭代次数')
        plt.ylabel('alpha')
        plt.title('alpha的优化过程')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/gradient_optimization.png')
        plt.close()
        
        return alpha, energy

# 运行梯度优化示例
print("\n=== 梯度优化示例 ===")
grad_vmc = GradientOptimizationVMC()
optimal_alpha, final_energy = grad_vmc.gradient_descent(initial_alpha=0.5, learning_rate=0.01, n_iterations=100)
print(f"最优alpha: {optimal_alpha:.6f}")
print(f"最终能量: {final_energy:.6f}")
print(f"解析解alpha: {grad_vmc.mass * grad_vmc.omega / grad_vmc.hbar:.6f}")

# 5. 统计误差分析
def statistical_error_analysis():
    """
    统计误差分析示例
    """
    # 创建VMC实例
    vmc = HarmonicOscillatorVMC()
    
    # 固定alpha值
    alpha = 1.0
    
    # 不同样本数量
    sample_sizes = np.logspace(2, 5, 10).astype(int)
    energies = []
    errors = []
    
    for n_samples in sample_sizes:
        energy, error = vmc.energy_estimate(alpha, n_samples=n_samples)
        energies.append(energy)
        errors.append(error)
        print(f"样本数量: {n_samples}, 能量: {energy:.6f} ± {error:.6f}")
    
    # 绘制误差随样本数量的变化
    plt.figure(figsize=(10, 6))
    plt.loglog(sample_sizes, errors, 'o-', label='实际误差')
    plt.loglog(sample_sizes, 1/np.sqrt(sample_sizes), 'r--', label='1/√n')
    plt.xlabel('样本数量')
    plt.ylabel('误差')
    plt.title('统计误差随样本数量的变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/statistical_error.png')
    plt.close()
    
    # 自举法示例
    n_bootstrap = 100
    bootstrap_energies = []
    
    # 生成大样本
    samples = vmc.metropolis_sample(alpha, n_samples=10000)
    local_energies = vmc.local_energy(samples, alpha)
    
    # 自举重采样
    for _ in range(n_bootstrap):
        bootstrap_samples = np.random.choice(local_energies, size=len(local_energies), replace=True)
        bootstrap_energies.append(np.mean(bootstrap_samples))
    
    bootstrap_mean = np.mean(bootstrap_energies)
    bootstrap_error = np.std(bootstrap_energies)
    
    print(f"\n自举法结果:")
    print(f"能量: {bootstrap_mean:.6f} ± {bootstrap_error:.6f}")
    
    # 绘制自举分布
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_energies, bins=20, density=True, alpha=0.7)
    plt.axvline(x=bootstrap_mean, color='r', linestyle='--', label=f'均值 = {bootstrap_mean:.6f}')
    plt.axvline(x=bootstrap_mean + bootstrap_error, color='g', linestyle='--', label=f'±1σ = {bootstrap_error:.6f}')
    plt.axvline(x=bootstrap_mean - bootstrap_error, color='g', linestyle='--')
    plt.xlabel('能量')
    plt.ylabel('概率密度')
    plt.title('自采样的能量分布')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/code/bootstrap_distribution.png')
    plt.close()

# 运行统计误差分析
print("\n=== 统计误差分析 ===")
statistical_error_analysis()

print("\n变分蒙特卡洛方法示例代码执行完成！")