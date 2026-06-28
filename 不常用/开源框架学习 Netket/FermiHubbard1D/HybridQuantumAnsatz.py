# ========= 导入必要的库 ========= #
import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import netket.experimental as nkx
from typing import Any
import numpy as np
import warnings

# MindQuantum (如果未安装，请运行 pip install mindquantum-gpu)
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, X, RX, RZ, RY
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz

# 忽略 MindQuantum 可能产生的一些警告
warnings.filterwarnings("ignore", category=UserWarning, module='mindquantum')

# ========= MindQuantum 电路 (与之前相同) ========= #

def initial_state(n_qubit: int, configuration: np.ndarray) -> Circuit:
    """根据给定的配置(0/1序列)创建初始态"""
    circ = Circuit()
    config_01 = ((configuration + 1) / 2).astype(int)
    for i in range(n_qubit):
        if config_01[i] == 1:
            circ += X(i)
    return circ

def generate_ansatz_pair(n_qubit: int, n_layer: int = 1):
    """生成振幅和相位的PQC ansatz"""
    amp_ansatz = Circuit()
    amp_ansatz += UN(H, range(n_qubit))
    amp_ansatz += HardwareEfficientAnsatz(n_qubits=n_qubit, single_rot_gate_seq=[RX, RZ], depth=n_layer).circuit

    phase_ansatz = Circuit()
    phase_ansatz += UN(H, range(n_qubit))
    phase_ansatz += HardwareEfficientAnsatz(n_qubits=n_qubit, single_rot_gate_seq=[RY, RZ], depth=n_layer).circuit
    return amp_ansatz, phase_ansatz

# ========= Forward + Backward with MindQuantum (与之前相同) ========= #

def objective_fun_real(q_ansatz: Circuit, initial_circ: Circuit, theta: np.ndarray, coefficients: np.ndarray):
    """计算单个样本的量子部分输出和梯度"""
    simulator = Simulator('mqvector', n_qubits=q_ansatz.n_qubits)
    full_circ = initial_circ + q_ansatz
    
    hamiltonians_with_co = [Hamiltonian(QubitOperator(f'Z{i}', coefficients[i])) for i in range(q_ansatz.n_qubits)]
    hamiltonians_without_co = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(q_ansatz.n_qubits)]
    
    grad_ops_with = simulator.get_expectation_with_grad(hams=hamiltonians_with_co, circ_right=full_circ)
    grad_ops_base = simulator.get_expectation_with_grad(hams=hamiltonians_without_co, circ_right=full_circ)

    f_with, g_with = grad_ops_with(theta)
    f_base, _ = grad_ops_base(theta)

    energy = np.sum(f_with)
    grad_theta = np.sum(g_with[0], axis=0)
    grad_coeffs = f_base[0].flatten()
    return energy, grad_theta, grad_coeffs

# ========= 自定义 JAX 可微接口 (核心修改) ========= #

def _mq_forward_sample(params_all, x, amp_ansatz, phase_ansatz):
    """
    前向传播，处理单个样本 x。
    严格按照论文公式：log(ψ) = f_amp + i * f_phase
    """
    params_amp, params_phase, coeffs_amp, coeffs_phase = params_all
    init_circ = initial_state(amp_ansatz.n_qubits, x)

    # f_amp
    f_amp, _, _ = objective_fun_real(amp_ansatz, init_circ, np.array(params_amp), np.array(coeffs_amp))
    # f_phase
    f_phase, _, _ = objective_fun_real(phase_ansatz, init_circ, np.array(params_phase), np.array(coeffs_phase))
    
    # 返回复数形式的对数波函数
    return f_amp + 1j * f_phase

def _mq_backward_sample(params_all, x, grad_output, amp_ansatz, phase_ansatz):
    """
    反向传播，处理单个样本 x 的梯度。
    根据 Wirtinger calculus, dL/dθ = Re[ (dL/d(logψ))^* * d(logψ)/dθ ]
    """
    params_amp, params_phase, coeffs_amp, coeffs_phase = params_all
    init_circ = initial_state(amp_ansatz.n_qubits, x)

    # 计算 ∂f/∂θ 和 ∂f/∂c
    _, g_theta_amp, g_c_amp = objective_fun_real(amp_ansatz, init_circ, np.array(params_amp), np.array(coeffs_amp))
    _, g_theta_phase, g_c_phase = objective_fun_real(phase_ansatz, init_circ, np.array(params_phase), np.array(coeffs_phase))

    # grad_output 是 dL/d(logψ)
    go = grad_output
    
    # d(logψ)/d(param_amp) = d(f_amp)/d(param_amp) = g_theta_amp
    # d(logψ)/d(param_phase) = i * d(f_phase)/d(param_phase) = i * g_theta_phase
    g_amp = go * g_theta_amp
    g_phase = go * 1j * g_theta_phase
    g_coeffs_amp = go * g_c_amp
    g_coeffs_phase = go * 1j * g_c_phase
    
    return g_amp, g_phase, g_coeffs_amp, g_coeffs_phase

def build_quantum_log_ansatz(amp_ansatz: Circuit, phase_ansatz: Circuit):
    @jax.custom_vjp
    def quantum_log_ansatz(params_all, x):
        return jax.vmap(lambda s: _mq_forward_sample(params_all, s, amp_ansatz, phase_ansatz))(x)

    def fwd(params_all, x):
        y = quantum_log_ansatz(params_all, x)
        return y, (params_all, x)

    def bwd(res, grad_y):
        params_all, x = res
        
        g_amp_v, g_phase_v, g_c_amp_v, g_c_phase_v = jax.vmap(
            lambda s, g: _mq_backward_sample(params_all, s, g, amp_ansatz, phase_ansatz)
        )(x, grad_y)
        
        # 参数是实数，所以总梯度是各个梯度贡献的实部之和
        g_amp = g_amp_v.sum(axis=0).real
        g_phase = g_phase_v.sum(axis=0).real
        g_coeffs_amp = g_c_amp_v.sum(axis=0).real
        g_coeffs_phase = g_c_phase_v.sum(axis=0).real
        
        return ((g_amp, g_phase, g_coeffs_amp, g_coeffs_phase), jnp.zeros_like(x))

    quantum_log_ansatz.defvjp(fwd, bwd)
    return quantum_log_ansatz