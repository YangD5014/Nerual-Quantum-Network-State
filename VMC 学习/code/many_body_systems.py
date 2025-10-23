"""
多体量子系统示例代码
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, det
from scipy.special import factorial
from scipy.constants import hbar, m_e, electron_volt
import matplotlib.animation as animation
from itertools import product

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ManyBodyQuantumSystems:
    """多体量子系统示例类"""
    
    def __init__(self):
        """初始化参数"""
        self.hbar = hbar  # 约化普朗克常数
        self.m = m_e      # 电子质量
        self.eV = electron_volt  # 电子伏特
        
    def slater_determinant(self, orbitals, positions):
        """
        构造斯莱特行列式
        
        参数:
            orbitals: 单粒子轨道函数列表
            positions: 粒子位置数组
            
        返回:
            斯莱特行列式波函数值
        """
        n_particles = len(positions)
        n_orbitals = len(orbitals)
        
        if n_particles > n_orbitals:
            raise ValueError("粒子数不能超过轨道数")
        
        # 构造矩阵
        matrix = np.zeros((n_particles, n_particles), dtype=complex)
        for i in range(n_particles):
            for j in range(n_particles):
                matrix[i, j] = orbitals[j](positions[i])
        
        # 计算行列式
        return det(matrix) / np.sqrt(factorial(n_particles))
    
    def hartree_fock_energy(self, orbitals, h_core, eri):
        """
        计算Hartree-Fock能量
        
        参数:
            orbitals: 轨道系数矩阵
            h_core: 单电子积分矩阵
            eri: 双电子积分张量
            
        返回:
            Hartree-Fock能量
        """
        n_orbitals = orbitals.shape[1]
        
        # 计算密度矩阵
        density = np.dot(orbitals[:, :n_orbitals//2], orbitals[:, :n_orbitals//2].T)
        
        # 计算单电子能量
        E_one_electron = np.sum(density * h_core)
        
        # 计算双电子能量
        E_two_electron = 0.5 * np.sum(density * eri * density)
        
        return E_one_electron + E_two_electron
    
    def hubbard_hamiltonian(self, n_sites, t, U, periodic=True):
        """
        构造Hubbard模型哈密顿量
        
        参数:
            n_sites: 格点数
            t: 跃迁强度
            U: 在位相互作用强度
            periodic: 是否使用周期性边界条件
            
        返回:
            Hubbard模型哈密顿量矩阵
        """
        # 计算希尔伯特空间维度
        dim = 4 ** n_sites
        
        # 初始化哈密顿量矩阵
        H = np.zeros((dim, dim))
        
        # 构造所有可能的占据数态
        states = list(product([0, 1], repeat=2*n_sites))
        
        # 构造哈密顿量
        for i, state in enumerate(states):
            # 在位相互作用项
            for site in range(n_sites):
                if state[2*site] == 1 and state[2*site+1] == 1:
                    H[i, i] += U
            
            # 跃迁项
            for site in range(n_sites):
                next_site = (site + 1) % n_sites if periodic else site + 1
                if next_site >= n_sites:
                    continue
                
                # 自旋向上
                if state[2*site] == 1 and state[2*next_site] == 0:
                    new_state = list(state)
                    new_state[2*site] = 0
                    new_state[2*next_site] = 1
                    j = states.index(tuple(new_state))
                    H[i, j] -= t
                    H[j, i] -= t
                
                # 自旋向下
                if state[2*site+1] == 1 and state[2*next_site+1] == 0:
                    new_state = list(state)
                    new_state[2*site+1] = 0
                    new_state[2*next_site+1] = 1
                    j = states.index(tuple(new_state))
                    H[i, j] -= t
                    H[j, i] -= t
        
        return H
    
    def heisenberg_hamiltonian(self, n_sites, J, periodic=True):
        """
        构造Heisenberg模型哈密顿量
        
        参数:
            n_sites: 格点数
            J: 交换耦合强度
            periodic: 是否使用周期性边界条件
            
        返回:
            Heisenberg模型哈密顿量矩阵
        """
        # 计算希尔伯特空间维度
        dim = 2 ** n_sites
        
        # 初始化哈密顿量矩阵
        H = np.zeros((dim, dim))
        
        # 构造所有可能的自旋态
        states = list(product([0, 1], repeat=n_sites))
        
        # Pauli矩阵
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        # 构造哈密顿量
        for i, state in enumerate(states):
            for site in range(n_sites):
                next_site = (site + 1) % n_sites if periodic else site + 1
                if next_site >= n_sites:
                    continue
                
                # 计算自旋算符的期望值
                # S_i · S_j = S_i^x S_j^x + S_i^y S_j^y + S_i^z S_j^z
                
                # S_i^z S_j^z 项
                if state[site] == state[next_site]:
                    H[i, i] += J/4
                else:
                    H[i, i] -= J/4
                
                # S_i^x S_j^x + S_i^y S_j^y 项
                if state[site] != state[next_site]:
                    new_state = list(state)
                    new_state[site] = 1 - new_state[site]
                    new_state[next_site] = 1 - new_state[next_site]
                    j = states.index(tuple(new_state))
                    H[i, j] += J/2
                    H[j, i] += J/2
        
        return H
    
    def bose_hubbard_hamiltonian(self, n_sites, t, U, n_max, periodic=True):
        """
        构造Bose-Hubbard模型哈密顿量
        
        参数:
            n_sites: 格点数
            t: 跃迁强度
            U: 在位相互作用强度
            n_max: 每个格点上的最大粒子数
            periodic: 是否使用周期性边界条件
            
        返回:
            Bose-Hubbard模型哈密顿量矩阵
        """
        # 计算希尔伯特空间维度
        dim = (n_max + 1) ** n_sites
        
        # 初始化哈密顿量矩阵
        H = np.zeros((dim, dim))
        
        # 构造所有可能的占据数态
        states = list(product(range(n_max + 1), repeat=n_sites))
        
        # 构造哈密顿量
        for i, state in enumerate(states):
            # 在位相互作用项
            for site in range(n_sites):
                H[i, i] += U/2 * state[site] * (state[site] - 1)
            
            # 跃迁项
            for site in range(n_sites):
                next_site = (site + 1) % n_sites if periodic else site + 1
                if next_site >= n_sites:
                    continue
                
                if state[site] > 0 and state[next_site] < n_max:
                    new_state = list(state)
                    new_state[site] -= 1
                    new_state[next_site] += 1
                    j = states.index(tuple(new_state))
                    H[i, j] -= t * np.sqrt(state[site] * (state[next_site] + 1))
                    H[j, i] -= t * np.sqrt(state[site] * (state[next_site] + 1))
        
        return H
    
    def molecular_hamiltonian(self, positions, charges, n_electrons):
        """
        构造分子哈密顿量
        
        参数:
            positions: 原子核位置数组
            charges: 原子核电荷数组
            n_electrons: 电子数
            
        返回:
            分子哈密顿量矩阵
        """
        n_nuclei = len(positions)
        
        # 计算原子核-原子核排斥能
        V_nn = 0
        for i in range(n_nuclei):
            for j in range(i+1, n_nuclei):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                V_nn += charges[i] * charges[j] / r_ij
        
        # 构造单电子积分
        # 这里简化处理，实际需要计算动能积分和核吸引积分
        n_basis = n_nuclei  # 假设每个原子核对应一个基函数
        h_core = np.zeros((n_basis, n_basis))
        
        # 构造双电子积分
        # 这里简化处理，实际需要计算四中心积分
        eri = np.zeros((n_basis, n_basis, n_basis, n_basis))
        
        return h_core, eri, V_nn
    
    def plot_slater_determinant(self):
        """绘制斯莱特行列式波函数"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 定义单粒子轨道
        def orbital_1(x):
            return np.exp(-x**2)
        
        def orbital_2(x):
            return x * np.exp(-x**2)
        
        orbitals = [orbital_1, orbital_2]
        
        # 1. 单粒子轨道
        x = np.linspace(-3, 3, 1000)
        for i, orbital in enumerate(orbitals):
            axes[0, 0].plot(x, orbital(x), label=f'轨道 {i+1}')
        
        axes[0, 0].set_title('单粒子轨道')
        axes[0, 0].set_xlabel('位置')
        axes[0, 0].set_ylabel('波函数')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 两粒子斯莱特行列式
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
        X1, X2 = np.meshgrid(x1, x2)
        
        psi = np.zeros_like(X1)
        for i in range(len(x1)):
            for j in range(len(x2)):
                positions = [x1[i], x2[j]]
                psi[j, i] = self.slater_determinant(orbitals, positions)
        
        im = axes[0, 1].contourf(X1, X2, np.abs(psi)**2, levels=20, cmap='viridis')
        axes[0, 1].set_title('两粒子斯莱特行列式概率密度')
        axes[0, 1].set_xlabel('粒子1位置')
        axes[0, 1].set_ylabel('粒子2位置')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. 交换效应
        # 对称波函数（玻色子）
        psi_sym = np.zeros_like(X1)
        for i in range(len(x1)):
            for j in range(len(x2)):
                positions = [x1[i], x2[j]]
                psi_sym[j, i] = (orbital_1(x1[i])*orbital_2(x2[j]) + orbital_1(x2[j])*orbital_2(x1[i])) / np.sqrt(2)
        
        # 反对称波函数（费米子）
        psi_antisym = np.zeros_like(X1)
        for i in range(len(x1)):
            for j in range(len(x2)):
                positions = [x1[i], x2[j]]
                psi_antisym[j, i] = (orbital_1(x1[i])*orbital_2(x2[j]) - orbital_1(x2[j])*orbital_2(x1[i])) / np.sqrt(2)
        
        # 对角线上的概率密度
        diag_idx = np.diag_indices_from(X1)
        axes[1, 0].plot(x1, np.abs(psi_sym[diag_idx])**2, label='对称波函数 (玻色子)')
        axes[1, 0].plot(x1, np.abs(psi_antisym[diag_idx])**2, label='反对称波函数 (费米子)')
        axes[1, 0].set_title('对角线上的概率密度')
        axes[1, 0].set_xlabel('位置')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 泡利不相容原理
        # 两个费米子不能处于同一位置
        axes[1, 1].plot(x1, np.abs(psi_antisym[diag_idx])**2, label='反对称波函数')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', label='零线')
        axes[1, 1].set_title('泡利不相容原理')
        axes[1, 1].set_xlabel('位置')
        axes[1, 1].set_ylabel('概率密度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/slater_determinant.png')
        plt.show()
    
    def plot_lattice_models(self):
        """绘制格点模型"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Hubbard模型能谱
        n_sites = 2
        t = 1.0
        U_values = np.linspace(0, 10, 100)
        
        energies = np.zeros((len(U_values), 4))
        for i, U in enumerate(U_values):
            H = self.hubbard_hamiltonian(n_sites, t, U)
            eigvals, _ = eigh(H)
            energies[i] = eigvals[:4]
        
        for i in range(4):
            axes[0, 0].plot(U_values, energies[:, i], label=f'能级 {i+1}')
        
        axes[0, 0].set_title('Hubbard模型能谱 (2格点)')
        axes[0, 0].set_xlabel('U/t')
        axes[0, 0].set_ylabel('能量')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Heisenberg模型能谱
        n_sites = 2
        J_values = np.linspace(-2, 2, 100)
        
        energies = np.zeros((len(J_values), 4))
        for i, J in enumerate(J_values):
            H = self.heisenberg_hamiltonian(n_sites, J)
            eigvals, _ = eigh(H)
            energies[i] = eigvals[:4]
        
        for i in range(4):
            axes[0, 1].plot(J_values, energies[:, i], label=f'能级 {i+1}')
        
        axes[0, 1].set_title('Heisenberg模型能谱 (2格点)')
        axes[0, 1].set_xlabel('J')
        axes[0, 1].set_ylabel('能量')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Bose-Hubbard模型能谱
        n_sites = 2
        t = 1.0
        U_values = np.linspace(0, 10, 100)
        n_max = 2
        
        energies = np.zeros((len(U_values), 9))
        for i, U in enumerate(U_values):
            H = self.bose_hubbard_hamiltonian(n_sites, t, U, n_max)
            eigvals, _ = eigh(H)
            energies[i] = eigvals[:9]
        
        for i in range(9):
            axes[1, 0].plot(U_values, energies[:, i], label=f'能级 {i+1}')
        
        axes[1, 0].set_title('Bose-Hubbard模型能谱 (2格点, n_max=2)')
        axes[1, 0].set_xlabel('U/t')
        axes[1, 0].set_ylabel('能量')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 格点可视化
        n_sites = 4
        positions = np.array([[i, 0] for i in range(n_sites)])
        
        # Hubbard模型
        axes[1, 1].scatter(positions[:, 0], positions[:, 1], s=200, c='blue', label='格点')
        for i in range(n_sites):
            axes[1, 1].text(positions[i, 0], positions[i, 1]+0.2, f'站点 {i}', ha='center')
        
        # 绘制跃迁
        for i in range(n_sites-1):
            axes[1, 1].plot([positions[i, 0], positions[i+1, 0]], 
                           [positions[i, 1], positions[i+1, 1]], 
                           'k--', alpha=0.5)
        
        axes[1, 1].set_title('一维格点模型')
        axes[1, 1].set_xlabel('位置')
        axes[1, 1].set_ylabel('位置')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/lattice_models.png')
        plt.show()
    
    def plot_molecular_system(self):
        """绘制分子系统"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. H2分子势能曲线
        R = np.linspace(0.5, 5.0, 100)  # 键长
        
        # 简化的H2分子势能曲线
        E = -1.0 / R + 0.5 * np.exp(-2*R)  # 吸引项 + 排斥项
        
        axes[0, 0].plot(R, E)
        axes[0, 0].set_title('H2分子势能曲线')
        axes[0, 0].set_xlabel('键长 (Å)')
        axes[0, 0].set_ylabel('能量 (Hartree)')
        axes[0, 0].grid(True)
        
        # 2. 电子密度
        x = np.linspace(-3, 3, 1000)
        
        # 简化的H2分子电子密度
        R = 1.4  # 平衡键长
        rho = np.exp(-(x+R/2)**2) + np.exp(-(x-R/2)**2)
        
        axes[0, 1].plot(x, rho)
        axes[0, 1].set_title('H2分子电子密度')
        axes[0, 1].set_xlabel('位置 (Å)')
        axes[0, 1].set_ylabel('电子密度')
        axes[0, 1].grid(True)
        
        # 3. 分子轨道
        # 成键轨道
        psi_bonding = np.exp(-(x+R/2)**2) + np.exp(-(x-R/2)**2)
        # 反键轨道
        psi_antibonding = np.exp(-(x+R/2)**2) - np.exp(-(x-R/2)**2)
        
        axes[1, 0].plot(x, psi_bonding, label='成键轨道')
        axes[1, 0].plot(x, psi_antibonding, label='反键轨道')
        axes[1, 0].set_title('H2分子轨道')
        axes[1, 0].set_xlabel('位置 (Å)')
        axes[1, 0].set_ylabel('波函数')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 分子结构可视化
        positions = np.array([[-R/2, 0], [R/2, 0]])
        
        axes[1, 1].scatter(positions[:, 0], positions[:, 1], s=200, c='red', label='H原子')
        for i in range(2):
            axes[1, 1].text(positions[i, 0], positions[i, 1]+0.2, 'H', ha='center')
        
        # 绘制化学键
        axes[1, 1].plot([positions[0, 0], positions[1, 0]], 
                       [positions[0, 1], positions[1, 1]], 
                       'k-', linewidth=2)
        
        axes[1, 1].set_title('H2分子结构')
        axes[1, 1].set_xlabel('位置 (Å)')
        axes[1, 1].set_ylabel('位置 (Å)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('/Users/yangjianfei/mac_vscode/神经网络量子态/VMC 学习/molecular_system.png')
        plt.show()

# 示例使用
if __name__ == "__main__":
    mbqs = ManyBodyQuantumSystems()
    
    # 绘制斯莱特行列式
    mbqs.plot_slater_determinant()
    
    # 绘制格点模型
    mbqs.plot_lattice_models()
    
    # 绘制分子系统
    mbqs.plot_molecular_system()
    
    # 计算Hubbard模型能谱
    n_sites = 2
    t = 1.0
    U = 2.0
    H = mbqs.hubbard_hamiltonian(n_sites, t, U)
    eigvals, eigvecs = eigh(H)
    print("Hubbard模型能谱 (2格点, t=1.0, U=2.0):")
    for i, E in enumerate(eigvals):
        print(f"能级 {i+1}: {E:.4f}")
    
    # 计算Heisenberg模型能谱
    n_sites = 2
    J = 1.0
    H = mbqs.heisenberg_hamiltonian(n_sites, J)
    eigvals, eigvecs = eigh(H)
    print("\nHeisenberg模型能谱 (2格点, J=1.0):")
    for i, E in enumerate(eigvals):
        print(f"能级 {i+1}: {E:.4f}")
    
    # 计算Bose-Hubbard模型能谱
    n_sites = 2
    t = 1.0
    U = 2.0
    n_max = 2
    H = mbqs.bose_hubbard_hamiltonian(n_sites, t, U, n_max)
    eigvals, eigvecs = eigh(H)
    print("\nBose-Hubbard模型能谱 (2格点, t=1.0, U=2.0, n_max=2):")
    for i, E in enumerate(eigvals):
        print(f"能级 {i+1}: {E:.4f}")