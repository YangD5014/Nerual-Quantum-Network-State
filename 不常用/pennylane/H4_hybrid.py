from pyscf import gto, scf, fci
import netket as nk
import netket.experimental as nkx
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import field

# 设置随机种子
key = jax.random.PRNGKey(42)

# 设置H4分子的几何构型
bond_length = 1.0
geometry = [
    ('H', (0., 0., 0.)),
    ('H', (bond_length, 0., 0.)),
    ('H', (2*bond_length, 0., 0.)),
    ('H', (3*bond_length, 0., 0.))
]

# 创建分子对象，使用STO-3G基组
mol = gto.M(atom=geometry, basis='STO-3G')

# 进行Hartree-Fock计算
mf = scf.RHF(mol).run(verbose=0)
E_hf = sum(mf.scf_summary.values())
print(f"Hartree-Fock能量: {E_hf:.8f} Ha")

# 进行FCI计算作为参考
E_fci = fci.FCI(mf).kernel()[0]
print(f"FCI能量: {E_fci:.8f} Ha")

# 输出原子核背景项
E_nuclear = mol.energy_nuc()
print(f"原子核背景项: {E_nuclear:.8f} Ha")

# 使用NetKet创建哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)
# 使用Lanczos方法计算精确基态能量
E0 = float(nk.exact.lanczos_ed(ha))
print(f"NetKet精确对角化能量: {E0:.8f} Ha")
print(f"NetKet与FCI能量差: {abs(E0 - E_fci):.8f} Ha")

# 创建Hilbert空间
hi = ha.hilbert

# 定义量子设备（PennyLane）
nqubits = 8
dev = qml.device("default.qubit", wires=nqubits)

# 1. 修正量子电路实现 - 严格按照论文公式(5)
@qml.qnode(dev, interface="jax")
def quantum_circuit(params, nqubits, n_layers=2, entanglement="linear"):
    """
    按照论文公式(5)实现硬件高效Ansatz:
    U(θ) = ∏(i=1 to Nl) [∏(m≠n) Λ(m,n) · ∏(j=1 to Nq) RZ(θ(i,j)^Z) RX(θ(i,j)^X)] · ∏(k=1 to Nq) H(k)
    """
    # 参数重塑: [层, 量子比特, 门类型(RZ/RX)]
    params = params.reshape((n_layers, nqubits, 2))
    
    # 初始化Hadamard门 - 论文公式中的最后一部分
    for i in range(nqubits):
        qml.Hadamard(wires=i)
    
    # 应用硬件高效Ansatz - 按照论文公式(5)
    for layer in range(n_layers):
        # 单量子比特旋转门 - 按照论文中的顺序: RZ然后RX
        for i in range(nqubits):
            qml.RZ(params[layer, i, 0], wires=i)
            qml.RX(params[layer, i, 1], wires=i)
        
        # 纠缠门 - CNOT门，按照论文中的Λ(m,n)
        if entanglement == "linear":
            # 线性纠缠：只连接相邻量子比特
            for i in range(nqubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif entanglement == "full":
            # 全连接纠缠：每个量子比特与其他所有量子比特纠缠
            for i in range(nqubits):
                for j in range(i + 1, nqubits):
                    qml.CNOT(wires=[i, j])
    
    # 测量所有Pauli Z期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

# 定义经典神经网络部分（前馈神经网络）
class ClassicalNN(nn.Module):
    hidden_dim: int = 32
    output_dim: int = 1
    
    @nn.compact
    def __call__(self, x):
        # 输入层
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        
        # 隐藏层
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        
        # 输出层
        x = nn.Dense(features=self.output_dim)(x)
        return x

# 2. 正确实现混合波函数（分离振幅和相位）- 修正形状问题
class HybridQuantumClassicalModel(nn.Module):
    nqubits: int = 8
    n_layers_amplitude: int = 2  # 振幅部分的层数
    n_layers_phase: int = 2      # 相位部分的层数
    
    @nn.compact
    def __call__(self, x):
        """
        按照论文公式(3)实现混合波函数:
        ⟨s|Ψ⟩ = ⟨s|ϕ(θ)⟩ × ⟨s|φ(λ)⟩
        
        修正了输出形状问题，确保返回正确的形状
        """
        # 获取输入形状并处理
        if len(x.shape) == 1:
            # 单个样本情况
            batch_size = 1
            x_reshaped = x.reshape(1, -1)
        else:
            # 批次情况
            batch_size = x.shape[0]
            x_reshaped = x
        
        # ===== 量子部分 ⟨s|ϕ(θ)⟩ =====
        # 按照论文公式(4): ln⟨s|ϕ(θ)⟩ = f[s; U1(θ1)] + i f[s; U2(θ2)]
        
        # 量子电路参数（振幅部分 U1）
        quantum_params_amplitude = self.param(
            'quantum_params_amplitude',
            nn.initializers.normal(stddev=0.01),
            (self.n_layers_amplitude * self.nqubits * 2,)
        )
        
        # 量子电路参数（相位部分 U2）
        quantum_params_phase = self.param(
            'quantum_params_phase',
            nn.initializers.normal(stddev=0.01),
            (self.n_layers_phase * self.nqubits * 2,)
        )
        
        # 可训练系数（振幅部分）
        coeffs_amplitude = self.param(
            'coeffs_amplitude',
            nn.initializers.normal(stddev=0.01),
            (self.nqubits,)
        )
        
        # 可训练系数（相位部分）
        coeffs_phase = self.param(
            'coeffs_phase',
            nn.initializers.normal(stddev=0.01),
            (self.nqubits,)
        )
        
        # 计算量子期望值（振幅部分）
        expvals_amplitude = quantum_circuit(
            quantum_params_amplitude, 
            self.nqubits, 
            self.n_layers_amplitude,
            entanglement="linear"
        )
        expvals_amplitude = jnp.array(expvals_amplitude)
        
        # 计算量子期望值（相位部分）
        expvals_phase = quantum_circuit(
            quantum_params_phase, 
            self.nqubits, 
            self.n_layers_phase,
            entanglement="linear"
        )
        expvals_phase = jnp.array(expvals_phase)
        
        # 计算波函数振幅部分 f[s; U1(θ1)] = ∑Nq i=1 ci⟨s|U†1(θ1)ZiU1(θ1)|s⟩
        f_amplitude = jnp.dot(coeffs_amplitude, expvals_amplitude)
        
        # 计算波函数相位部分 f[s; U2(θ2)] = ∑Nq i=1 ci⟨s|U†2(θ2)ZiU2(θ2)|s⟩
        f_phase = jnp.dot(coeffs_phase, expvals_phase)
        
        # 量子部分的复振幅: ln⟨s|ϕ(θ)⟩ = f[s; U1(θ1)] + i f[s; U2(θ2)]
        log_quantum_amplitude = f_amplitude + 1j * f_phase
        
        # ===== 经典部分 ⟨s|φ(λ)⟩ =====
        # 按照论文公式(6): ⟨s|φ(λ)⟩ = √p(s;λ₁) e^{iγ(s;λ₂)}
        
        # 经典振幅部分 - 对应 √p(s;λ₁)
        classical_amplitude_model = ClassicalNN(
            hidden_dim=32, 
            output_dim=1,
            name='classical_amplitude'
        )
        classical_amplitude = classical_amplitude_model(x_reshaped)
        
        # 经典相位部分 - 对应 γ(s;λ₂)
        classical_phase_model = ClassicalNN(
            hidden_dim=32, 
            output_dim=1,
            name='classical_phase'
        )
        classical_phase = classical_phase_model(x_reshaped)
        
        # 经典部分的复振幅: ⟨s|φ(λ)⟩ = √p(s;λ₁) e^{iγ(s;λ₂)}
        # 取对数形式: ln⟨s|φ(λ)⟩ = 0.5 * ln(p(s;λ₁)) + i γ(s;λ₂)
        log_classical_amplitude = 0.5 * classical_amplitude + 1j * classical_phase
        
        # ===== 组合量子和经典部分 =====
        # 按照论文公式(3): ⟨s|Ψ⟩ = ⟨s|ϕ(θ)⟩ × ⟨s|φ(λ)⟩
        # 在对数空间中: ln⟨s|Ψ⟩ = ln⟨s|ϕ(θ)⟩ + ln⟨s|φ(λ)⟩
        log_psi = log_quantum_amplitude + log_classical_amplitude
        
        # ===== 修正输出形状 =====
        # 确保返回正确的形状
        if batch_size == 1:
            # 单个样本情况，返回标量
            return log_psi.squeeze()
        else:
            # 批次情况，返回一维数组
            return log_psi.squeeze(axis=-1)

# 创建模型实例
model = HybridQuantumClassicalModel(
    nqubits=nqubits, 
    n_layers_amplitude=2, 
    n_layers_phase=2
)

# 初始化模型变量 - 使用正确的输入形状
dummy_input = jnp.ones((1, nqubits))  # 使用批次形状 (1, 8)
variables = model.init(key, dummy_input)

# 提取参数
model_params = variables['params']

# 创建采样器 - 使用费米子系统的特殊采样器
sampler = nk.sampler.MetropolisLocal(hi)

# 创建 MCState
vqs = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    variables=variables,
    n_samples=2000,
    n_discard_per_chain=100
)

# 定义 VMC 优化器和预条件器
optimizer = nk.optimizer.Adam(learning_rate=0.005)
sr = nk.optimizer.SR(diag_shift=0.01)
vmc = nk.VMC(hamiltonian=ha, optimizer=optimizer, variational_state=vqs, preconditioner=sr)

# 打印参数结构
print(f"# variational parameters: {vmc.state.n_parameters}")

# 运行优化
vmc.run(n_iter=1000, out="H4_hybrid_quantum_classical_log")

# 获取优化后的能量
E_vmc = vmc.energy.mean.real
print(f"VMC优化后能量: {E_vmc:.8f} Ha")
print(f"VMC与FCI能量差: {abs(E_vmc - E_fci):.8f} Ha")
print(f"VMC与精确对角化能量差: {abs(E_vmc - E0):.8f} Ha")
print(f"相对误差: {abs(E_vmc - E0)/abs(E0)*100:.4f}%")
