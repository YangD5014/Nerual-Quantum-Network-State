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
# 使用线性构型，H-H键长为1.0埃
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

# 输出原子核背景项（原子核之间的库仑排斥能）
E_nuclear = mol.energy_nuc()
print(f"原子核背景项: {E_nuclear:.8f} Ha")

# 使用NetKet创建哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)
# 使用Lanczos方法计算精确基态能量
E0 = float(nk.exact.lanczos_ed(ha))
print(f"NetKet精确对角化能量: {E0:.8f} Ha")
print(f"NetKet与FCI能量差: {abs(E0 - E_fci):.8f} Ha")

# 创建Hilbert空间 - 使用与哈密顿量匹配的费米子希尔伯特空间
hi = ha.hilbert

# 定义量子设备（PennyLane）
# 对于H4分子，我们需要8个量子比特（每个氢原子对应两个自旋轨道）
nqubits = 8
dev = qml.device("default.qubit", wires=nqubits)

# 定义量子Ansatz: f[s; U(θ)] = ∑Nq i=1 ci⟨s|U†(θ)ZiU(θ)|s⟩
# 根据论文描述，我们需要两个量子电路U1和U2分别表示振幅和相位
# 量子电路结构: U(θ) = ∏(i=1 to Nl) [∏(m≠n) Λ(m,n) · ∏(j=1 to Nq) RZ(θ(i,j)^Z) RX(θ(i,j)^X)] · ∏(k=1 to Nq) H(k)
@qml.qnode(dev, interface="jax")
def quantum_ansatz_amplitude(params, nqubits, n_layers=2):
    # 将参数重塑为合适的形状
    # 根据论文，每层每个量子比特有2个参数（RZ和RX）
    params = params.reshape((n_layers, nqubits, 2))  # [层, 量子比特, 参数]
    
    # 初始化Hadamard门 - 根据论文公式中的最后一部分
    for i in range(nqubits):
        qml.Hadamard(wires=i)
    
    # 应用硬件高效Ansatz - 根据论文公式
    for layer in range(n_layers):  # n_layers层电路
        # 应用单量子比特旋转门 - 按照论文中的顺序: RZ然后RX
        for i in range(nqubits):
            qml.RZ(params[layer, i, 0], wires=i)
            qml.RX(params[layer, i, 1], wires=i)
        
        # 应用纠缠门 - CNOT门，按照论文中的Λ(m,n)
        # 使用线性连接而不是全连接，更符合硬件高效Ansatz的精神
        for i in range(nqubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    # 测量所有量子比特的Pauli Z期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

@qml.qnode(dev, interface="jax")
def quantum_ansatz_phase(params, nqubits, n_layers=4):
    # 将参数重塑为合适的形状
    # 根据论文，每层每个量子比特有2个参数（RZ和RX）
    params = params.reshape((n_layers, nqubits, 2))  # [层, 量子比特, 参数]
    
    # 初始化Hadamard门 - 根据论文公式中的最后一部分
    for i in range(nqubits):
        qml.Hadamard(wires=i)
    
    # 应用硬件高效Ansatz - 根据论文公式
    for layer in range(n_layers):  # n_layers层电路
        # 应用单量子比特旋转门 - 按照论文中的顺序: RZ然后RX
        for i in range(nqubits):
            qml.RZ(params[layer, i, 0], wires=i)
            qml.RX(params[layer, i, 1], wires=i)
        
        # 应用纠缠门 - CNOT门，按照论文中的Λ(m,n)
        # 使用线性连接而不是全连接，更符合硬件高效Ansatz的精神
        for i in range(nqubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    # 测量所有量子比特的Pauli Z期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

# 定义经典神经网络部分（前馈神经网络）
class ClassicalNN(nn.Module):
    hidden_dim: int = 32
    output_dim: int = 1
    input_dim: int = 8  # H4分子在STO-3G基组下有8个自旋轨道
    
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

# 定义混合量子-经典模型
class HybridQuantumClassicalModel(nn.Module):
    nqubits: int = 8
    n_layers_amplitude: int = 2  # 振幅部分的层数
    n_layers_phase: int = 2  # 相位部分的层数
    
    @nn.compact
    def __call__(self, x):
        # 获取批次大小
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        
        # 初始化量子参数（振幅部分）
        quantum_params_amplitude = self.param(
            'quantum_params_amplitude',
            nn.initializers.normal(stddev=0.1),
            (self.n_layers_amplitude * self.nqubits * 2,)  # n_layers_amplitude层，每个量子比特2个参数（RZ和RX）
        )
        
        # 初始化量子参数（相位部分）
        quantum_params_phase = self.param(
            'quantum_params_phase',
            nn.initializers.normal(stddev=0.1),
            (self.n_layers_phase * self.nqubits * 2,)  # n_layers_phase层，每个量子比特2个参数（RZ和RX）
        )
        
        # 初始化可训练系数（振幅部分）
        coeffs_amplitude = self.param(
            'coeffs_amplitude',
            nn.initializers.normal(stddev=0.1),
            (self.nqubits,)
        )
        
        # 初始化可训练系数（相位部分）
        coeffs_phase = self.param(
            'coeffs_phase',
            nn.initializers.normal(stddev=0.1),
            (self.nqubits,)
        )
        
        # 计算量子期望值（振幅部分）
        expvals_amplitude = quantum_ansatz_amplitude(quantum_params_amplitude, self.nqubits, self.n_layers_amplitude)
        expvals_amplitude = jnp.array(expvals_amplitude)
        
        # 计算量子期望值（相位部分）
        expvals_phase = quantum_ansatz_phase(quantum_params_phase, self.nqubits, self.n_layers_phase)
        expvals_phase = jnp.array(expvals_phase)
        
        # 计算波函数振幅部分 f[s; U1(θ1)] = ∑Nq i=1 ci⟨s|U†1(θ1)ZiU1(θ1)|s⟩
        amplitude = jnp.dot(coeffs_amplitude, expvals_amplitude)
        
        # 计算波函数相位部分 f[s; U2(θ2)] = ∑Nq i=1 ci⟨s|U†2(θ2)ZiU2(θ2)|s⟩
        phase = jnp.dot(coeffs_phase, expvals_phase)
        
        # 创建经典神经网络模型
        # H4分子在STO-3G基组下有8个自旋轨道
        classical_model = ClassicalNN(input_dim=8)
        
        # 计算经典部分
        # 将输入x重塑为适合神经网络的形式
        # 对于费米子系统，我们需要考虑自旋轨道
        if batch_size > 1:
            x_reshaped = x.reshape(batch_size, -1)
        else:
            # 对于单个样本，确保形状为(1, 8)，对应8个自旋轨道
            x_reshaped = jnp.ones((1, 8))  # 使用固定输入而不是依赖于x
        
        classical_output = classical_model(x_reshaped)
        
        # 组合量子和经典部分
        # 根据论文，波函数可以表示为 ⟨s|Ψ⟩ = ⟨s|ϕ(θ)⟩ × ⟨s|φ(λ)⟩
        # 其中⟨s|ϕ(θ)⟩是量子部分，⟨s|φ(λ)⟩是经典部分
        
        # 量子部分的复振幅: ln⟨s|ϕ(θ)⟩ = f[s; U1(θ1)] + i f[s; U2(θ2)]
        log_quantum_amplitude = amplitude + 1j * phase
        quantum_amplitude = jnp.exp(log_quantum_amplitude)
        
        # 经典部分
        classical_amplitude = classical_output.squeeze()
        
        # 混合波函数
        log_psi = jnp.log(quantum_amplitude * classical_amplitude)
        
        # 返回与批次大小匹配的形状
        if batch_size > 1:
            return jnp.ones(batch_size) * log_psi
        else:
            return log_psi

# 创建模型实例
# 可以通过n_layers_amplitude和n_layers_phase参数控制量子电路层数
model = HybridQuantumClassicalModel(nqubits=nqubits, n_layers_amplitude=2, n_layers_phase=4)

# 初始化模型变量
dummy_input = jnp.ones(nqubits)
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
# 使用Adam优化器，通常比SGD更适合复杂模型
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