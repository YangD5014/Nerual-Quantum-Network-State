"""
统计力学基础示例代码
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import zeta
from scipy.constants import k, hbar, m_e, electron_volt
import matplotlib.animation as animation

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class StatisticalMechanicsBasics:
    """统计力学基础示例类"""
    
    def __init__(self):
        """初始化参数"""
        self.k = k  # 玻尔兹曼常数
        self.hbar = hbar  # 约化普朗克常数
        self.m = m_e  # 电子质量
        self.eV = electron_volt  # 电子伏特
        
    def boltzmann_distribution(self, E, T):
        """
        玻尔兹曼分布
        
        参数:
            E: 能量
            T: 温度
            
        返回:
            玻尔兹曼分布概率
        """
        beta = 1 / (self.k * T)
        return np.exp(-beta * E)
    
    def partition_function(self, energies, T):
        """
        计算配分函数
        
        参数:
            energies: 能级数组
            T: 温度
            
        返回:
            配分函数
        """
        beta = 1 / (self.k * T)
        return np.sum(np.exp(-beta * energies))
    
    def average_energy(self, energies, T):
        """
        计算平均能量
        
        参数:
            energies: 能级数组
            T: 温度
            
        返回:
            平均能量
        """
        Z = self.partition_function(energies, T)
        beta = 1 / (self.k * T)
        
        # 使用数值微分计算平均能量
        # U = -d(lnZ)/dβ
        d_beta = 1e-6
        Z_plus = np.sum(np.exp(-(beta + d_beta) * energies))
        Z_minus = np.sum(np.exp(-(beta - d_beta) * energies))
        
        d_ln_Z_d_beta = (np.log(Z_plus) - np.log(Z_minus)) / (2 * d_beta)
        
        return -d_ln_Z_d_beta
    
    def entropy(self, energies, T):
        """
        计算熵
        
        参数:
            energies: 能级数组
            T: 温度
            
        返回:
            熵
        """
        Z = self.partition_function(energies, T)
        U = self.average_energy(energies, T)
        beta = 1 / (self.k * T)
        
        return self.k * (np.log(Z) + beta * U)
    
    def free_energy(self, energies, T):
        """
        计算自由能
        
        参数:
            energies: 能级数组
            T: 温度
            
        返回:
            自由能
        """
        Z = self.partition_function(energies, T)
        return -self.k * T * np.log(Z)
    
    def fermi_dirac_distribution(self, E, mu, T):
        """
        费米-狄拉克分布
        
        参数:
            E: 能量
            mu: 化学势
            T: 温度
            
        返回:
            费米-狄拉克分布概率
        """
        if T == 0:
            return 1.0 if E < mu else 0.0
        
        beta = 1 / (self.k * T)
        return 1.0 / (np.exp(beta * (E - mu)) + 1)
    
    def bose_einstein_distribution(self, E, mu, T):
        """
        玻色-爱因斯坦分布
        
        参数:
            E: 能量
            mu: 化学势
            T: 温度
            
        返回:
            玻色-爱因斯坦分布概率
        """
        if T == 0:
            return np.inf if E == mu else 0.0
        
        beta = 1 / (self.k * T)
        return 1.0 / (np.exp(beta * (E - mu)) - 1)
    
    def fermi_energy(self, n):
        """
        计算费米能量
        
        参数:
            n: 粒子数密度
            
        返回:
            费米能量
        """
        return (self.hbar**2 / (2 * self.m)) * (3 * np.pi**2 * n)**(2/3)
    
    def bose_einstein_condensation_temperature(self, n):
        """
        计算玻色-爱因斯坦凝聚临界温度
        
        参数:
            n: 粒子数密度
            
        返回:
            临界温度
        """
        return (2 * np.pi * self.hbar**2 / (self.m * self.k)) * (n / zeta(3/2))**(2/3)
    
    def density_matrix_pure_state(self, psi):
        """
        纯态的密度矩阵
        
        参数:
            psi: 波函数
            
        返回:
            密度矩阵
        """
        return np.outer(psi, np.conj(psi))
    
    def density_matrix_mixed_state(self, states, probabilities):
        """
        混合态的密度矩阵
        
        参数:
            states: 态列表
            probabilities: 对应的概率列表
            
        返回:
            密度矩阵
        """
        rho = np.zeros((len(states[0]), len(states[0])), dtype=complex)
        for psi, p in zip(states, probabilities):
            rho += p * self.density_matrix_pure_state(psi)
        return rho
    
    def expectation_value(self, operator, rho):
        """
        计算期望值
        
        参数:
            operator: 算符矩阵
            rho: 密度矩阵
            
        返回:
            期望值
        """
        return np.trace(np.dot(operator, rho)).real
    
    def von_neumann_entropy(self, rho):
        """
        计算冯·诺依曼熵
        
        参数:
            rho: 密度矩阵
            
        返回:
            冯·诺依曼熵
        """
        eigenvalues = np.linalg.eigvals(rho)
        # 只考虑非零特征值
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log(eigenvalues))
    
    def plot_distributions(self):
        """绘制各种统计分布"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 设置能量范围
        E = np.linspace(-2, 2, 1000)
        mu = 0.0  # 化学势
        T = 300  # 温度 (K)
        
        # 1. 玻尔兹曼分布
        P_boltzmann = self.boltzmann_distribution(E, T)
        axes[0, 0].plot(E, P_boltzmann, label='玻尔兹曼分布')
        axes[0, 0].set_title('玻尔兹曼分布')
        axes[0, 0].set_xlabel('能量 (eV)')
        axes[0, 0].set_ylabel('概率')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 费米-狄拉克分布
        P_fermi = self.fermi_dirac_distribution(E, mu, T)
        axes[0, 1].plot(E, P_fermi, label='费米-狄拉克分布')
        axes[0, 1].set_title('费米-狄拉克分布')
        axes[0, 1].set_xlabel('能量 (eV)')
        axes[0, 1].set_ylabel('占据概率')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 玻色-爱因斯坦分布
        # 注意：对于玻色-爱因斯坦分布，化学势必须小于最低能量
        mu = -0.1
        E_positive = E[E > mu]
        P_bose = self.bose_einstein_distribution(E_positive, mu, T)
        axes[1, 0].plot(E_positive, P_bose, label='玻色-爱因斯坦分布')
        axes[1, 0].set_title('玻色-爱因斯坦分布')
        axes[1, 0].set_xlabel('能量 (eV)')
        axes[1, 0].set_ylabel('占据概率')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 三种分布的比较
        mu = 0.0
        T_high = 3000  # 高温
        T_low = 300    # 低温
        
        P_boltzmann_high = self.boltzmann_distribution(E, T_high)
        P_fermi_high = self.fermi_dirac_distribution(E, mu, T_high)
        P_bose_high = self.bose_einstein_distribution(E[E > mu], mu, T_high)
        
        P_boltzmann_low = self.boltzmann_distribution(E, T_low)
        P_fermi_low = self.fermi_dirac_distribution(E, mu, T_low)
        P_bose_low = self.bose_einstein_distribution(E[E > mu], mu, T_low)
        
        axes[1, 1].plot(E, P_boltzmann_high, 'r--', label='玻尔兹曼 (高温)')
        axes[1, 1].plot(E, P_fermi_high, 'g--', label='费米-狄拉克 (高温)')
        axes[1, 1].plot(E[E > mu], P_bose_high, 'b--', label='玻色-爱因斯坦 (高温)')
        
        axes[1, 1].plot(E, P_boltzmann_low, 'r-', label='玻尔兹曼 (低温)')
        axes[1, 1].plot(E, P_fermi_low, 'g-', label='费米-狄拉克 (低温)')
        axes[1, 1].plot(E[E > mu], P_bose_low, 'b-', label='玻色-爱因斯坦 (低温)')
        
        axes[1, 1].set_title('不同温度下的统计分布比较')
        axes[1, 1].set_xlabel('能量 (eV)')
        axes[1, 1].set_ylabel('概率/占据概率')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/statistical_distributions.png')
        plt.show()
    
    def plot_thermodynamic_quantities(self):
        """绘制热力学量随温度的变化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 创建一个简单的能级系统（谐振子）
        n_levels = 10
        energies = np.array([i * 0.1 for i in range(n_levels)])  # 能级间隔为0.1 eV
        
        # 温度范围
        T = np.linspace(10, 1000, 100)
        
        # 计算各种热力学量
        Z = np.array([self.partition_function(energies, t) for t in T])
        U = np.array([self.average_energy(energies, t) for t in T])
        S = np.array([self.entropy(energies, t) for t in T])
        F = np.array([self.free_energy(energies, t) for t in T])
        
        # 1. 配分函数
        axes[0, 0].plot(T, Z)
        axes[0, 0].set_title('配分函数随温度的变化')
        axes[0, 0].set_xlabel('温度 (K)')
        axes[0, 0].set_ylabel('配分函数')
        axes[0, 0].grid(True)
        
        # 2. 平均能量
        axes[0, 1].plot(T, U)
        axes[0, 1].set_title('平均能量随温度的变化')
        axes[0, 1].set_xlabel('温度 (K)')
        axes[0, 1].set_ylabel('平均能量 (eV)')
        axes[0, 1].grid(True)
        
        # 3. 熵
        axes[1, 0].plot(T, S)
        axes[1, 0].set_title('熵随温度的变化')
        axes[1, 0].set_xlabel('温度 (K)')
        axes[1, 0].set_ylabel('熵 (eV/K)')
        axes[1, 0].grid(True)
        
        # 4. 自由能
        axes[1, 1].plot(T, F)
        axes[1, 1].set_title('自由能随温度的变化')
        axes[1, 1].set_xlabel('温度 (K)')
        axes[1, 1].set_ylabel('自由能 (eV)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/thermodynamic_quantities.png')
        plt.show()
    
    def plot_density_matrix(self):
        """绘制密度矩阵示例"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 创建一个简单的二维系统
        # 纯态
        psi1 = np.array([1, 0], dtype=complex)  # |0⟩
        psi2 = np.array([0, 1], dtype=complex)  # |1⟩
        psi_super = (psi1 + psi2) / np.sqrt(2)  # (|0⟩ + |1⟩)/√2
        
        rho1 = self.density_matrix_pure_state(psi1)
        rho2 = self.density_matrix_pure_state(psi2)
        rho_super = self.density_matrix_pure_state(psi_super)
        
        # 混合态
        rho_mixed = 0.5 * rho1 + 0.5 * rho2
        
        # 绘制密度矩阵
        im1 = axes[0].imshow(np.abs(rho1), cmap='viridis')
        axes[0].set_title('纯态 |0⟩')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(np.abs(rho_super), cmap='viridis')
        axes[1].set_title('纯态 (|0⟩ + |1⟩)/√2')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(np.abs(rho_mixed), cmap='viridis')
        axes[2].set_title('混合态 0.5|0⟩⟨0| + 0.5|1⟩⟨1|')
        plt.colorbar(im3, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('态')
            ax.set_ylabel('态')
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/density_matrix.png')
        plt.show()
        
        # 计算并打印冯·诺依曼熵
        print("纯态 |0⟩ 的冯·诺依曼熵:", self.von_neumann_entropy(rho1))
        print("纯态 (|0⟩ + |1⟩)/√2 的冯·诺依曼熵:", self.von_neumann_entropy(rho_super))
        print("混合态 0.5|0⟩⟨0| + 0.5|1⟩⟨1| 的冯·诺依曼熵:", self.von_neumann_entropy(rho_mixed))
    
    def plot_quantum_statistics(self):
        """绘制量子统计示例"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 费米能量与密度的关系
        n = np.logspace(20, 30, 100)  # 粒子数密度范围
        E_F = self.fermi_energy(n)
        
        axes[0, 0].loglog(n, E_F)
        axes[0, 0].set_title('费米能量与粒子数密度的关系')
        axes[0, 0].set_xlabel('粒子数密度 (m^-3)')
        axes[0, 0].set_ylabel('费米能量 (J)')
        axes[0, 0].grid(True)
        
        # 费米-狄拉克分布随温度的变化
        E = np.linspace(-1, 1, 1000)
        mu = 0.0
        T_values = [10, 100, 300, 1000]  # 不同温度
        
        for T in T_values:
            P_fermi = self.fermi_dirac_distribution(E, mu, T)
            axes[0, 1].plot(E, P_fermi, label=f'T = {T} K')
        
        axes[0, 1].set_title('费米-狄拉克分布随温度的变化')
        axes[0, 1].set_xlabel('能量 (eV)')
        axes[0, 1].set_ylabel('占据概率')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 玻色-爱因斯坦凝聚临界温度与密度的关系
        T_c = self.bose_einstein_condensation_temperature(n)
        
        axes[1, 0].loglog(n, T_c)
        axes[1, 0].set_title('玻色-爱因斯坦凝聚临界温度与粒子数密度的关系')
        axes[1, 0].set_xlabel('粒子数密度 (m^-3)')
        axes[1, 0].set_ylabel('临界温度 (K)')
        axes[1, 0].grid(True)
        
        # 玻色-爱因斯坦分布
        E_positive = np.linspace(0.1, 2, 1000)
        mu = 0.0
        T_values = [10, 100, 300, 1000]  # 不同温度
        
        for T in T_values:
            P_bose = self.bose_einstein_distribution(E_positive, mu, T)
            axes[1, 1].plot(E_positive, P_bose, label=f'T = {T} K')
        
        axes[1, 1].set_title('玻色-爱因斯坦分布随温度的变化')
        axes[1, 1].set_xlabel('能量 (eV)')
        axes[1, 1].set_ylabel('占据概率')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/quantum_statistics.png')
        plt.show()

# 示例使用
if __name__ == "__main__":
    sm = StatisticalMechanicsBasics()
    
    # 绘制统计分布
    sm.plot_distributions()
    
    # 绘制热力学量
    sm.plot_thermodynamic_quantities()
    
    # 绘制密度矩阵
    sm.plot_density_matrix()
    
    # 绘制量子统计
    sm.plot_quantum_statistics()
    
    # 计算费米能量
    n = 1e28  # 电子密度 (m^-3)
    E_F = sm.fermi_energy(n)
    print(f"电子密度为 {n:.1e} m^-3 时的费米能量: {E_F/sm.eV:.2f} eV")
    
    # 计算玻色-爱因斯坦凝聚临界温度
    T_c = sm.bose_einstein_condensation_temperature(n)
    print(f"粒子密度为 {n:.1e} m^-3 时的玻色-爱因斯坦凝聚临界温度: {T_c:.2f} K")