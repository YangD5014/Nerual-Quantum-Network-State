"""
量子力学基础示例代码
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.constants import hbar, m_e, electron_volt
import matplotlib.animation as animation

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class QuantumMechanicsBasics:
    """量子力学基础示例类"""
    
    def __init__(self):
        """初始化参数"""
        self.hbar = hbar  # 约化普朗克常数
        self.m = m_e      # 电子质量
        self.eV = electron_volt  # 电子伏特
        
    def wave_function_normalization(self, psi, x):
        """
        波函数归一化
        
        参数:
            psi: 未归一化的波函数
            x: 位置坐标数组
            
        返回:
            归一化的波函数
        """
        # 计算归一化常数
        norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
        return psi / norm
    
    def infinite_square_well(self, x, n, L):
        """
        一维无限深势阱的波函数
        
        参数:
            x: 位置坐标数组
            n: 量子数
            L: 势阱宽度
            
        返回:
            波函数
        """
        # 只在势阱内部有非零值
        psi = np.zeros_like(x)
        mask = (x >= 0) & (x <= L)
        psi[mask] = np.sqrt(2/L) * np.sin(n * np.pi * x[mask] / L)
        return psi
    
    def infinite_square_well_energy(self, n, L):
        """
        一维无限深势阱的能量本征值
        
        参数:
            n: 量子数
            L: 势阱宽度
            
        返回:
            能量本征值（电子伏特）
        """
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.m * L**2) / self.eV
    
    def harmonic_oscillator(self, x, n, m, omega):
        """
        一维谐振子的波函数
        
        参数:
            x: 位置坐标数组
            n: 量子数
            m: 质量
            omega: 角频率
            
        返回:
            波函数
        """
        # 计算特征长度
        x0 = np.sqrt(self.hbar / (m * omega))
        
        # 计算无量纲坐标
        xi = x / x0
        
        # 计算归一化常数
        norm = 1.0 / np.sqrt(2**n * np.math.factorial(n) * x0)
        
        # 计算厄米多项式
        H_n = hermite(n)
        
        # 计算波函数
        psi = norm * H_n(xi) * np.exp(-xi**2 / 2)
        
        return psi
    
    def harmonic_oscillator_energy(self, n, omega):
        """
        一维谐振子的能量本征值
        
        参数:
            n: 量子数
            omega: 角频率
            
        返回:
            能量本征值（电子伏特）
        """
        return (n + 0.5) * self.hbar * omega / self.eV
    
    def probability_density(self, psi):
        """
        计算概率密度
        
        参数:
            psi: 波函数
            
        返回:
            概率密度
        """
        return np.abs(psi)**2
    
    def expectation_value(self, operator, psi, x):
        """
        计算期望值
        
        参数:
            operator: 算符函数
            psi: 波函数
            x: 位置坐标数组
            
        返回:
            期望值
        """
        # 应用算符
        op_psi = operator(psi, x)
        
        # 计算内积
        numerator = np.trapz(np.conj(psi) * op_psi, x)
        denominator = np.trapz(np.conj(psi) * psi, x)
        
        return numerator / denominator
    
    def uncertainty_principle(self, psi, x):
        """
        验证不确定性原理
        
        参数:
            psi: 波函数
            x: 位置坐标数组
            
        返回:
            (Δx, Δp, ΔxΔp)
        """
        # 定义位置算符
        def position_operator(psi, x):
            return x * psi
        
        # 定义动量算符（使用有限差分）
        def momentum_operator(psi, x):
            dx = x[1] - x[0]
            dpsi_dx = np.gradient(psi, dx)
            return -1j * self.hbar * dpsi_dx
        
        # 计算位置期望值
        x_exp = self.expectation_value(position_operator, psi, x)
        
        # 计算动量期望值
        p_exp = self.expectation_value(momentum_operator, psi, x)
        
        # 计算位置平方的期望值
        def x_squared_operator(psi, x):
            return x**2 * psi
        x2_exp = self.expectation_value(x_squared_operator, psi, x)
        
        # 计算动量平方的期望值
        def p_squared_operator(psi, x):
            dx = x[1] - x[0]
            dpsi_dx = np.gradient(psi, dx)
            d2psi_dx2 = np.gradient(dpsi_dx, dx)
            return -self.hbar**2 * d2psi_dx2
        p2_exp = self.expectation_value(p_squared_operator, psi, x)
        
        # 计算不确定度
        delta_x = np.sqrt(x2_exp - x_exp**2)
        delta_p = np.sqrt(p2_exp - p_exp**2)
        
        return delta_x, delta_p, delta_x * delta_p
    
    def plot_wave_functions(self):
        """绘制不同量子系统的波函数"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 设置位置坐标
        x = np.linspace(-5, 5, 1000)
        
        # 1. 无限深势阱
        L = 2.0
        x_well = np.linspace(0, L, 1000)
        
        for n in range(1, 4):
            psi = self.infinite_square_well(x_well, n, L)
            prob = self.probability_density(psi)
            energy = self.infinite_square_well_energy(n, L)
            
            axes[0, 0].plot(x_well, psi + energy*10, label=f'n={n}, E={energy:.2f} eV')
        
        axes[0, 0].set_title('一维无限深势阱波函数')
        axes[0, 0].set_xlabel('位置 (m)')
        axes[0, 0].set_ylabel('波函数')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 谐振子
        m = self.m
        omega = 1e15  # 角频率
        
        for n in range(0, 4):
            psi = self.harmonic_oscillator(x, n, m, omega)
            prob = self.probability_density(psi)
            energy = self.harmonic_oscillator_energy(n, omega)
            
            axes[0, 1].plot(x, psi + energy*10, label=f'n={n}, E={energy:.2f} eV')
        
        axes[0, 1].set_title('一维谐振子波函数')
        axes[0, 1].set_xlabel('位置 (m)')
        axes[0, 1].set_ylabel('波函数')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 概率密度
        for n in range(1, 4):
            psi = self.infinite_square_well(x_well, n, L)
            prob = self.probability_density(psi)
            
            axes[1, 0].plot(x_well, prob, label=f'n={n}')
        
        axes[1, 0].set_title('一维无限深势阱概率密度')
        axes[1, 0].set_xlabel('位置 (m)')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 不确定性原理验证
        # 使用高斯波包
        sigma = 0.5
        x0 = 0.0
        k0 = 5.0
        psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
        psi = self.wave_function_normalization(psi, x)
        
        delta_x, delta_p, delta_x_delta_p = self.uncertainty_principle(psi, x)
        
        axes[1, 1].plot(x, np.abs(psi)**2, label='|ψ|²')
        axes[1, 1].set_title(f'不确定性原理验证\nΔxΔp = {delta_x_delta_p:.2e}ℏ ≥ ℏ/2')
        axes[1, 1].set_xlabel('位置 (m)')
        axes[1, 1].set_ylabel('概率密度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/quantum_basics_plots.png')
        plt.show()
    
    def superposition_state(self, x, c1, c2, psi1, psi2):
        """
        创建叠加态
        
        参数:
            x: 位置坐标数组
            c1, c2: 叠加系数
            psi1, psi2: 基态波函数
            
        返回:
            叠加态波函数
        """
        # 归一化系数
        norm = np.sqrt(np.abs(c1)**2 + np.abs(c2)**2)
        c1, c2 = c1/norm, c2/norm
        
        return c1 * psi1 + c2 * psi2
    
    def time_evolution(self, psi, E, t):
        """
        波函数的时间演化
        
        参数:
            psi: 初始波函数
            E: 能量
            t: 时间
            
        返回:
            时间演化后的波函数
        """
        return psi * np.exp(-1j * E * t / self.hbar)
    
    def plot_superposition_time_evolution(self):
        """绘制叠加态的时间演化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 设置位置坐标
        x = np.linspace(-5, 5, 1000)
        
        # 创建两个基态
        m = self.m
        omega = 1e15
        psi1 = self.harmonic_oscillator(x, 0, m, omega)
        psi2 = self.harmonic_oscillator(x, 1, m, omega)
        
        # 计算能量
        E1 = self.harmonic_oscillator_energy(0, omega)
        E2 = self.harmonic_oscillator_energy(1, omega)
        
        # 创建叠加态
        c1, c2 = 1.0, 1.0
        psi_super = self.superposition_state(x, c1, c2, psi1, psi2)
        
        # 时间演化
        times = np.linspace(0, 2*np.pi/(E2-E1)*self.hbar/self.eV, 4)
        
        for i, t in enumerate(times):
            psi1_t = self.time_evolution(psi1, E1, t)
            psi2_t = self.time_evolution(psi2, E2, t)
            psi_super_t = self.superposition_state(x, c1, c2, psi1_t, psi2_t)
            
            prob = self.probability_density(psi_super_t)
            
            row = i // 2
            col = i % 2
            axes[row, col].plot(x, prob, label=f't = {t:.2e} s')
            axes[row, col].set_title(f'叠加态概率密度 (t = {t:.2e} s)')
            axes[row, col].set_xlabel('位置 (m)')
            axes[row, col].set_ylabel('概率密度')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/superposition_time_evolution.png')
        plt.show()

# 示例使用
if __name__ == "__main__":
    qm = QuantumMechanicsBasics()
    
    # 绘制波函数
    qm.plot_wave_functions()
    
    # 绘制叠加态的时间演化
    qm.plot_superposition_time_evolution()
    
    # 验证不确定性原理
    x = np.linspace(-5, 5, 1000)
    sigma = 0.5
    x0 = 0.0
    k0 = 5.0
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi = qm.wave_function_normalization(psi, x)
    
    delta_x, delta_p, delta_x_delta_p = qm.uncertainty_principle(psi, x)
    print(f"位置不确定度: {delta_x:.2e} m")
    print(f"动量不确定度: {delta_p:.2e} kg·m/s")
    print(f"乘积: {delta_x_delta_p:.2e} J·s")
    print(f"ℏ/2: {qm.hbar/2:.2e} J·s")
    print(f"不确定性原理验证: {delta_x_delta_p >= qm.hbar/2}")