from mindquantum.core.circuit import Circuit,UN
from mindquantum.core.gates import RY, RX, RZ,H,X,BarrierGate
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian
import numpy as np


def initial_state(n_qubit: int, configuration: str):
    '''
    生成初始态
    :param n_qubit: 量子比特数
    :param configuration: 配置字符串，例如 '010101010'，长度必须等于 n_qubit
    :return: 初始态电路
    '''
    # 检查配置字符串长度是否匹配量子比特数
    if len(configuration) != n_qubit:
        raise ValueError(f"配置字符串长度({len(configuration)})与量子比特数({n_qubit})不匹配")
    
    initial_state = Circuit()
    for i in range(n_qubit):
        if configuration[i] == '1':
            initial_state += X(i)
    initial_state += BarrierGate().on(range(n_qubit))
    return initial_state



def generate_ansatz(n_qubit:int,initial_state:Circuit=None,n_layer:int=1):
    if initial_state is not None:
        ansatz = initial_state
        ansatz +=UN(H,range(n_qubit))
    else:
        ansatz = UN(H,range(n_qubit))
    ansatz += HardwareEfficientAnsatz(n_qubits=n_qubit,single_rot_gate_seq=[RX,RZ],depth=n_layer).circuit
    return ansatz
    

# def objective_fun_ops(q_ansatz:Circuit):
#     """
#     计算目标函数
#     :param q_ansatz: 量子电路 需要包含 configuration_state
#     :param coefficients: 系数数组
#     :return: 目标函数值
#     """
#     simulator = Simulator('mqvector', n_qubits=q_ansatz.n_qubits)
#     # if len(q_ansatz.params_name) != len(initial_params):
#     #     raise ValueError(f"参数名称长度({len(q_ansatz.params_name)})与初始参数长度({len(initial_params)})不匹配")
#     hamiltonians = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(q_ansatz.n_qubits)]
#     grad_ops = simulator.get_expectation_with_grad(hams=hamiltonians,circ_right=q_ansatz)
#     return grad_ops

def objective_fun_ops(q_ansatz:Circuit,coefficients:np.ndarray):
    """
    计算目标函数
    :param q_ansatz: 量子电路 需要包含 configuration_state
    :param coefficients: 系数数组
    :return: 目标函数值
    """
    simulator = Simulator('mqvector', n_qubits=q_ansatz.n_qubits)
    # if len(q_ansatz.params_name) != len(initial_params):
    #     raise ValueError(f"参数名称长度({len(q_ansatz.params_name)})与初始参数长度({len(initial_params)})不匹配")
    hamiltonians_with_co = [Hamiltonian(QubitOperator(f'Z{i}',coefficient=coefficients[i])) for i in range(q_ansatz.n_qubits)]
    hamiltonians_without_co = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(q_ansatz.n_qubits)]
    grad_ops_with_co = simulator.get_expectation_with_grad(hams=hamiltonians_with_co,circ_right=q_ansatz)
    grad_ops_without_co = simulator.get_expectation_with_grad(hams=hamiltonians_without_co,circ_right=q_ansatz)
    return grad_ops_with_co,grad_ops_without_co
    
def objective_fun_real(q_ansatz:Circuit,theta:np.array,coefficient:np.array):
    if theta.shape[0] != len(q_ansatz.params_name):
        raise ValueError(f"参数长度({theta.shape[0]})与电路参数名称长度({len(q_ansatz.params_name)})不匹配")
        
    grad_ops_with_co,grad_ops_without_co = objective_fun_ops(q_ansatz,coefficient)
    f_with_co,g_with_co = grad_ops_with_co(theta) # f.shape = (1,N-terms,1)
    f_without_co,g_without_co = grad_ops_without_co(theta) # f.shape = (1,N-terms,1)
    fun = np.real(np.sum(f_with_co[0])) # f.shape = (1,1)
    gradient_theta = np.array([np.sum(g_with_co[0][:,i]) for i in range(len(q_ansatz.params_name))])
    gradient_coefficients = f_without_co[0] # f.shape = (N-terms,1)
    return fun,gradient_theta,gradient_coefficients


    
    
    
    
    
    
    
    