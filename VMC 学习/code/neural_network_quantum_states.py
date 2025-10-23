"""
神经网络量子态的Python代码示例
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import det
from typing import Callable, Tuple, List, Optional
import time

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)

# ==============================
# 1. 基本神经网络结构
# ==============================

class RBM(nn.Module):
    """
    受限玻尔兹曼机 (Restricted Boltzmann Machine)
    用于表示量子态的神经网络结构
    """
    def __init__(self, num_visible: int, num_hidden: int):
        """
        初始化RBM
        
        参数:
            num_visible: 可见单元数量 (通常对应自旋数量)
            num_hidden: 隐藏单元数量
        """
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # 初始化权重和偏置
        self.weights = nn.Parameter(torch.randn(num_hidden, num_visible) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(num_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (可见单元状态)
            
        返回:
            波函数的对数振幅
        """
        # 计算隐藏单元的激活
        hidden_activation = torch.matmul(x, self.weights.t()) + self.hidden_bias
        
        # 计算波函数的对数振幅
        log_psi = torch.sum(torch.log(torch.cosh(hidden_activation)), dim=1)
        log_psi += torch.matmul(x, self.visible_bias)
        
        return log_psi
    
    def sample(self, num_samples: int, num_steps: int = 100) -> torch.Tensor:
        """
        使用Gibbs采样从RBM中采样
        
        参数:
            num_samples: 采样数量
            num_steps: Gibbs采样步数
            
        返回:
            采样得到的可见单元状态
        """
        # 初始化可见单元状态
        visible = torch.randint(0, 2, (num_samples, self.num_visible)) * 2 - 1
        
        for _ in range(num_steps):
            # 计算隐藏单元的概率
            hidden_prob = torch.sigmoid(torch.matmul(visible, self.weights.t()) + self.hidden_bias)
            hidden = (torch.rand_like(hidden_prob) < hidden_prob) * 2 - 1
            
            # 计算可见单元的概率
            visible_prob = torch.sigmoid(torch.matmul(hidden, self.weights) + self.visible_bias)
            visible = (torch.rand_like(visible_prob) < visible_prob) * 2 - 1
        
        return visible

class FeedForwardNN(nn.Module):
    """
    前馈神经网络 (Feedforward Neural Network)
    用于表示量子态的神经网络结构
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 1):
        """
        初始化前馈神经网络
        
        参数:
            input_size: 输入层大小
            hidden_sizes: 隐藏层大小列表
            output_size: 输出层大小 (默认为1)
        """
        super(FeedForwardNN, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())  # 使用tanh激活函数
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            波函数的对数振幅
        """
        return self.network(x).squeeze(-1)

# ==============================
# 2. 费米子神经网络
# ==============================

class FermiNet(nn.Module):
    """
    费米子神经网络 (Fermionic Neural Network)
    用于表示费米子系统的波函数
    """
    def __init__(self, num_electrons: int, num_orbitals: int, hidden_sizes: List[int]):
        """
        初始化费米子神经网络
        
        参数:
            num_electrons: 电子数量
            num_orbitals: 轨道数量
            hidden_sizes: 隐藏层大小列表
        """
        super(FermiNet, self).__init__()
        self.num_electrons = num_electrons
        self.num_orbitals = num_orbitals
        
        # 电子流网络
        self.electron_stream = nn.Sequential(
            nn.Linear(3, hidden_sizes[0]),  # 3D坐标
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh()
        )
        
        # 轨道流网络
        self.orbital_stream = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[2], 1)
            ) for _ in range(num_orbitals)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 电子坐标张量，形状为 (batch_size, num_electrons, 3)
            
        返回:
            波函数的对数振幅
        """
        batch_size = x.shape[0]
        
        # 计算电子流特征
        electron_features = self.electron_stream(x.view(-1, 3))
        electron_features = electron_features.view(batch_size, self.num_electrons, -1)
        
        # 计算轨道矩阵
        orbital_matrix = torch.zeros(batch_size, self.num_electrons, self.num_orbitals)
        
        for i in range(self.num_orbitals):
            orbital_matrix[:, :, i] = self.orbital_stream[i](electron_features).squeeze(-1)
        
        # 计算Slater行列式
        log_psi = torch.zeros(batch_size)
        for b in range(batch_size):
            try:
                slater_det = det(orbital_matrix[b].detach().numpy())
                log_psi[b] = torch.log(torch.abs(torch.tensor(slater_det)))
            except:
                log_psi[b] = torch.tensor(-float('inf'))
        
        return log_psi

# ==============================
# 3. 变分蒙特卡洛方法
# ==============================

class VariationalMonteCarlo:
    """
    变分蒙特卡洛 (Variational Monte Carlo) 类
    用于训练神经网络量子态
    """
    def __init__(self, model: nn.Module, hamiltonian: Callable, 
                 learning_rate: float = 0.01, num_samples: int = 1000):
        """
        初始化变分蒙特卡洛
        
        参数:
            model: 神经网络模型
            hamiltonian: 哈密顿量函数
            learning_rate: 学习率
            num_samples: 采样数量
        """
        self.model = model
        self.hamiltonian = hamiltonian
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.num_samples = num_samples
        
        # 用于记录训练历史
        self.energy_history = []
        self.variance_history = []
    
    def local_energy(self, samples: torch.Tensor) -> torch.Tensor:
        """
        计算局部能量
        
        参数:
            samples: 采样得到的构型
            
        返回:
            局部能量
        """
        # 计算波函数的对数振幅
        log_psi = self.model(samples)
        
        # 计算局部能量
        local_energies = torch.zeros_like(log_psi)
        
        for i in range(len(samples)):
            # 这里简化处理，实际应用中需要根据具体的哈密顿量计算
            local_energies[i] = self.hamiltonian(samples[i])
        
        return local_energies
    
    def metropolis_hastings_step(self, current_sample: torch.Tensor) -> torch.Tensor:
        """
        Metropolis-Hastings采样步骤
        
        参数:
            current_sample: 当前构型
            
        返回:
            新构型
        """
        # 生成候选构型
        candidate_sample = current_sample.clone()
        
        # 随机翻转一个自旋
        flip_idx = np.random.randint(0, len(current_sample))
        candidate_sample[flip_idx] *= -1
        
        # 计算接受概率
        log_psi_current = self.model(current_sample.unsqueeze(0))
        log_psi_candidate = self.model(candidate_sample.unsqueeze(0))
        
        # 对于实数波函数，接受概率为 |ψ(candidate)|² / |ψ(current)|²
        accept_prob = torch.exp(2 * (log_psi_candidate - log_psi_current)).item()
        
        # 决定是否接受候选构型
        if np.random.random() < accept_prob:
            return candidate_sample
        else:
            return current_sample
    
    def sample(self) -> torch.Tensor:
        """
        从当前波函数中采样
        
        返回:
            采样得到的构型
        """
        # 初始化随机构型
        samples = torch.randint(0, 2, (self.num_samples, self.model.num_visible)) * 2 - 1
        
        # 执行Metropolis-Hastings采样
        for i in range(self.num_samples):
            samples[i] = self.metropolis_hastings_step(samples[i])
        
        return samples
    
    def train_step(self) -> Tuple[float, float]:
        """
        执行一步训练
        
        返回:
            平均能量和能量方差
        """
        # 采样
        samples = self.sample()
        
        # 计算局部能量
        local_energies = self.local_energy(samples)
        
        # 计算能量和方差
        energy = torch.mean(local_energies)
        variance = torch.var(local_energies)
        
        # 计算梯度
        log_psi = self.model(samples)
        
        # 计算能量关于参数的梯度
        self.optimizer.zero_grad()
        
        # 使用重新参数化技巧计算梯度
        loss = torch.mean((local_energies - energy.detach()) * log_psi)
        loss.backward()
        
        # 更新参数
        self.optimizer.step()
        
        # 记录历史
        self.energy_history.append(energy.item())
        self.variance_history.append(variance.item())
        
        return energy.item(), variance.item()
    
    def train(self, num_epochs: int = 100, print_every: int = 10) -> None:
        """
        训练模型
        
        参数:
            num_epochs: 训练轮数
            print_every: 每隔多少轮打印一次信息
        """
        print(f"开始训练，共 {num_epochs} 轮...")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            energy, variance = self.train_step()
            
            if (epoch + 1) % print_every == 0:
                elapsed_time = time.time() - start_time
                print(f"轮次 {epoch+1}/{num_epochs} - 能量: {energy:.6f}, 方差: {variance:.6f}, 用时: {elapsed_time:.2f}s")
                start_time = time.time()
    
    def plot_training_history(self) -> None:
        """
        绘制训练历史
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制能量历史
        ax1.plot(self.energy_history)
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('能量')
        ax1.set_title('能量随训练轮次的变化')
        ax1.grid(True)
        
        # 绘制方差历史
        ax2.plot(self.variance_history)
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('能量方差')
        ax2.set_title('能量方差随训练轮次的变化')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# ==============================
# 4. 示例应用
# ==============================

def ising_hamiltonian(spins: torch.Tensor, J: float = 1.0, h: float = 0.1) -> float:
    """
    一维Ising模型的哈密顿量
    
    参数:
        spins: 自旋构型
        J: 耦合常数
        h: 外场
        
    返回:
        能量
    """
    energy = 0.0
    N = len(spins)
    
    # 最近邻相互作用
    for i in range(N):
        energy -= J * spins[i] * spins[(i + 1) % N]
    
    # 外场
    for i in range(N):
        energy -= h * spins[i]
    
    return energy

def example_ising_model():
    """
    使用神经网络量子态求解一维Ising模型的示例
    """
    print("=== 一维Ising模型示例 ===")
    
    # 设置参数
    N = 20  # 自旋数量
    num_hidden = 10  # 隐藏单元数量
    learning_rate = 0.01
    num_samples = 1000
    num_epochs = 100
    
    # 创建RBM模型
    model = RBM(num_visible=N, num_hidden=num_hidden)
    
    # 创建哈密顿量函数
    hamiltonian = lambda spins: ising_hamiltonian(spins, J=1.0, h=0.1)
    
    # 创建变分蒙特卡洛对象
    vmc = VariationalMonteCarlo(
        model=model,
        hamiltonian=hamiltonian,
        learning_rate=learning_rate,
        num_samples=num_samples
    )
    
    # 训练模型
    vmc.train(num_epochs=num_epochs, print_every=10)
    
    # 绘制训练历史
    vmc.plot_training_history()
    
    # 计算最终能量
    final_energy = vmc.energy_history[-1]
    print(f"最终能量: {final_energy:.6f}")
    
    return model, vmc

def heisenberg_hamiltonian(spins: torch.Tensor, J: float = 1.0) -> float:
    """
    一维海森堡模型的哈密顿量
    
    参数:
        spins: 自旋构型
        J: 耦合常数
        
    返回:
        能量
    """
    energy = 0.0
    N = len(spins)
    
    # 最近邻相互作用
    for i in range(N):
        # 对于自旋1/2系统，S_i · S_j = (1/4) σ_i · σ_j
        # 其中σ是Pauli矩阵
        # 在计算基中，σ_z σ_z = 1，σ_x σ_x 和 σ_y σ_y 会翻转自旋
        energy -= J * 0.25 * spins[i] * spins[(i + 1) % N]
    
    return energy

def example_heisenberg_model():
    """
    使用神经网络量子态求解一维海森堡模型的示例
    """
    print("\n=== 一维海森堡模型示例 ===")
    
    # 设置参数
    N = 16  # 自旋数量
    num_hidden = 16  # 隐藏单元数量
    learning_rate = 0.005
    num_samples = 2000
    num_epochs = 200
    
    # 创建RBM模型
    model = RBM(num_visible=N, num_hidden=num_hidden)
    
    # 创建哈密顿量函数
    hamiltonian = lambda spins: heisenberg_hamiltonian(spins, J=1.0)
    
    # 创建变分蒙特卡洛对象
    vmc = VariationalMonteCarlo(
        model=model,
        hamiltonian=hamiltonian,
        learning_rate=learning_rate,
        num_samples=num_samples
    )
    
    # 训练模型
    vmc.train(num_epochs=num_epochs, print_every=20)
    
    # 绘制训练历史
    vmc.plot_training_history()
    
    # 计算最终能量
    final_energy = vmc.energy_history[-1]
    print(f"最终能量: {final_energy:.6f}")
    
    # 对于一维海森堡模型，精确基态能量约为 -N * ln(2) + 1/4
    exact_energy = -N * np.log(2) + 0.25
    print(f"精确能量: {exact_energy:.6f}")
    print(f"相对误差: {abs(final_energy - exact_energy) / abs(exact_energy) * 100:.2f}%")
    
    return model, vmc

def hydrogen_molecule_hamiltonian(electron_coords: torch.Tensor) -> float:
    """
    氢分子的哈密顿量 (简化版)
    
    参数:
        electron_coords: 电子坐标
        
    返回:
        能量
    """
    # 这里简化处理，实际应用中需要考虑完整的哈密顿量
    # 包括动能项、电子-核吸引项、电子-电子排斥项、核-核排斥项
    
    # 简化版：只考虑电子-电子排斥和电子-核吸引
    energy = 0.0
    
    # 假设两个氢核位于 (0, 0, 0) 和 (1, 0, 0)
    nuclei_positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    # 电子-核吸引项
    for i in range(len(electron_coords)):
        for j in range(len(nuclei_positions)):
            r = torch.norm(electron_coords[i] - nuclei_positions[j])
            energy -= 1.0 / r  # 假设单位电荷
    
    # 电子-电子排斥项
    for i in range(len(electron_coords)):
        for j in range(i+1, len(electron_coords)):
            r = torch.norm(electron_coords[i] - electron_coords[j])
            energy += 1.0 / r  # 假设单位电荷
    
    # 核-核排斥项
    r_nn = torch.norm(nuclei_positions[0] - nuclei_positions[1])
    energy += 1.0 / r_nn  # 假设单位电荷
    
    return energy

def example_hydrogen_molecule():
    """
    使用费米子神经网络求解氢分子基态的示例
    """
    print("\n=== 氢分子示例 ===")
    
    # 设置参数
    num_electrons = 2
    num_orbitals = 2
    hidden_sizes = [16, 32, 16]
    learning_rate = 0.01
    num_samples = 1000
    num_epochs = 50
    
    # 创建费米子神经网络模型
    model = FermiNet(num_electrons=num_electrons, num_orbitals=num_orbitals, hidden_sizes=hidden_sizes)
    
    # 创建哈密顿量函数
    hamiltonian = hydrogen_molecule_hamiltonian
    
    # 创建变分蒙特卡洛对象
    vmc = VariationalMonteCarlo(
        model=model,
        hamiltonian=hamiltonian,
        learning_rate=learning_rate,
        num_samples=num_samples
    )
    
    # 训练模型
    vmc.train(num_epochs=num_epochs, print_every=5)
    
    # 绘制训练历史
    vmc.plot_training_history()
    
    # 计算最终能量
    final_energy = vmc.energy_history[-1]
    print(f"最终能量: {final_energy:.6f} Hartree")
    
    # 氢分子的精确基态能量约为 -1.174 Hartree (在平衡距离下)
    exact_energy = -1.174
    print(f"精确能量: {exact_energy:.6f} Hartree")
    print(f"相对误差: {abs(final_energy - exact_energy) / abs(exact_energy) * 100:.2f}%")
    
    return model, vmc

# ==============================
# 5. 主程序
# ==============================

def main():
    """
    主程序，运行所有示例
    """
    print("神经网络量子态示例程序")
    print("=" * 50)
    
    # 运行Ising模型示例
    ising_model, ising_vmc = example_ising_model()
    
    # 运行海森堡模型示例
    heisenberg_model, heisenberg_vmc = example_heisenberg_model()
    
    # 运行氢分子示例
    hydrogen_model, hydrogen_vmc = example_hydrogen_molecule()
    
    print("\n所有示例运行完成！")

if __name__ == "__main__":
    main()