import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pyscf import gto, scf, fci
import netket.experimental as nkx
import expect_grad_ex
import vmc_ex
import jax
from functools import reduce


def setup_h2_system(bond_length=0.74):
    """
    设置H2分子系统
    
    Args:
        bond_length: H2分子的键长（埃）
    
    Returns:
        mol: PySCF分子对象
        ha: NetKet哈密顿量
        hi: Hilbert空间
        sampler: 采样器
        e_fci_all: FCI所有能级
        E_fci: FCI基态能量
    """
    geometry = [
        ('H', (0., 0., 0.)),
        ('H', (bond_length, 0., 0.)),
    ]
    
    mol = gto.M(atom=geometry, basis='STO-3G')
    
    mf = scf.RHF(mol).run(verbose=0)
    E_hf = mf.e_tot
    print(f"Hartree-Fock能量: {E_hf:.8f} Ha")
    
    cisolver = fci.FCI(mf)
    cisolver.nroots = 4
    E_fcis, fcivec = cisolver.kernel()
    E_fci = E_fcis[0]
    print(f"FCI基态能量: {E_fci:.8f} Ha")
    
    e_fci_all = E_fcis
    print(f"\nFCI所有能级:")
    for i, e in enumerate(e_fci_all[:4]):
        if i == 0:
            print(f"  E{i} (基态) = {e:.8f} Ha")
        else:
            print(f"  E{i} (第{i}激发态) = {e:.8f} Ha")
    
    ha = nkx.operator.from_pyscf_molecule(mol)
    
    hi = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2,
        s=1/2,
        n_fermions_per_spin=(1, 1)
    )
    
    print(f"\nHilbert空间维度: {hi.n_states}")
    print(f"空间轨道数: {hi.n_orbitals}")
    #print(f"自旋轨道数: {hi.n_spinorbitals}")
    print(f"电子数: {hi.n_fermions}")
    
    g = nk.graph.Graph(edges=[(0,1),(2,3)])
    sampler = nk.sampler.MetropolisFermionHop(
        hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
    )
    
    return mol, ha, hi, sampler, e_fci_all, E_fci


def compute_ground_state(ha, sampler, 
                       n_iter=1000, 
                       model=None,
                       learning_rate=0.1,
                       diag_shift=0.01,
                       n_discard_per_chain=100,
                       n_samples=1024,
                       output_path='h2_ground_state'):
    """
    计算H2分子的基态
    
    Args:
        ha: NetKet哈密顿量
        sampler: 采样器
        n_iter: 迭代次数
        model: 神经网络模型，如果为None则使用默认的RBM(alpha=2)
        learning_rate: 学习率
        diag_shift: SR的对角位移
        n_discard_per_chain: 每条链丢弃的样本数
        n_samples: 样本数
        output_path: 输出文件路径
    
    Returns:
        vs: 变分态对象
        final_energy_gs: 基态能量
        gs_params: 基态参数
    """
    if model is None:
        ma = nk.models.RBM(alpha=2, param_dtype=complex, use_visible_bias=False)
    else:
        ma = model
    vs = nk.vqs.MCState(sampler, ma, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples)
    
    opt = nk.optimizer.Sgd(learning_rate=learning_rate)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    
    gs = nk.driver.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    
    print(f"\n开始计算基态...")
    gs.run(out=output_path, n_iter=n_iter)
    
    data = json.load(open(output_path + '.log'))
    energy_gs = data["Energy"]["Mean"]["real"]
    final_energy_gs = reduce(lambda x, y: x if y is None else y, energy_gs)
    
    gs_params = vs.parameters
    
    print(f"基态能量: {final_energy_gs:.8f} Ha")
    
    return vs, final_energy_gs, gs_params


def compute_1st_excited_state(ha, sampler, gs_params, e_fci,
                             n_iter=2000,
                             model=None,
                             learning_rate=0.03,
                             diag_shift=0.01,
                             n_discard_per_chain=100,
                             n_samples=1024,
                             shift=0.3,
                             output_path='h2_excited_state_1'):
    """
    计算H2分子的第一激发态
    
    Args:
        ha: NetKet哈密顿量
        sampler: 采样器
        gs_params: 基态参数
        e_fci: FCI基态能量
        n_iter: 迭代次数
        model: 神经网络模型，如果为None则使用默认的RBM(alpha=2)
        learning_rate: 学习率
        diag_shift: SR的对角位移
        n_discard_per_chain: 每条链丢弃的样本数
        n_samples: 样本数
        shift: 正交化约束的位移参数（推荐0.3）
        output_path: 输出文件路径
    
    Returns:
        vs_ex1: 第一激发态变分态对象
        final_energy_ex1: 第一激发态能量
        ex1_params: 第一激发态参数
        vs_gs: 基态变分态对象
    """
    if model is None:
        ma = nk.models.RBM(alpha=2, param_dtype=complex, use_visible_bias=False)
    else:
        ma = model
    
    vs_gs = nk.vqs.MCState(sampler, ma, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples)
    vs_gs.init_parameters(jax.nn.initializers.normal(stddev=0.25))
    vs_gs.parameters = gs_params
    
    vs_ex1 = nk.vqs.MCState(sampler, ma, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples)
    
    opt_ex1 = nk.optimizer.Sgd(learning_rate=learning_rate)
    sr_ex1 = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    
    shift_list = [shift]
    state_list = [vs_gs]
    
    gs_ex1 = vmc_ex.VMC_ex(
        hamiltonian=ha,
        optimizer=opt_ex1,
        variational_state=vs_ex1,
        preconditioner=sr_ex1,
        state_list=state_list,
        shift_list=shift_list
    )
    
    print(f"\n开始计算第一激发态 (shift={shift})...")
    gs_ex1.run(out=output_path, n_iter=n_iter)
    
    data_ex1 = json.load(open(output_path + '.log'))
    energy_ex1 = data_ex1["Energy"]["Mean"]["real"]
    final_energy_ex1 = reduce(lambda x, y: x if y is None else y, energy_ex1)
    
    ex1_params = vs_ex1.parameters
    
    print(f"第一激发态能量: {final_energy_ex1:.8f} Ha")
    print(f"激发能: {final_energy_ex1 - e_fci:.8f} Ha")
    
    return vs_ex1, final_energy_ex1, ex1_params, vs_gs


def compute_2nd_excited_state(ha, sampler, gs_params, ex1_params, e_fci,
                             n_iter=2000,
                             model=None,
                             learning_rate=0.03,
                             diag_shift=0.01,
                             n_discard_per_chain=100,
                             n_samples=1024,
                             shift=0.3,
                             output_path='h2_excited_state_2'):
    """
    计算H2分子的第二激发态
    
    Args:
        ha: NetKet哈密顿量
        sampler: 采样器
        gs_params: 基态参数
        ex1_params: 第一激发态参数
        e_fci: FCI基态能量
        n_iter: 迭代次数
        model: 神经网络模型，如果为None则使用默认的RBM(alpha=2)
        learning_rate: 学习率
        diag_shift: SR的对角位移
        n_discard_per_chain: 每条链丢弃的样本数
        n_samples: 样本数
        shift: 正交化约束的位移参数（推荐0.3）
        output_path: 输出文件路径
    
    Returns:
        vs_ex2: 第二激发态变分态对象
        final_energy_ex2: 第二激发态能量
        ex2_params: 第二激发态参数
        vs_gs: 基态变分态对象
        vs_ex1_loaded: 第一激发态变分态对象
    """
    if model is None:
        ma = nk.models.RBM(alpha=2, param_dtype=complex, use_visible_bias=False)
    else:
        ma = model
    
    vs_gs = nk.vqs.MCState(sampler, ma, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples)
    vs_gs.init_parameters(jax.nn.initializers.normal(stddev=0.25))
    vs_gs.parameters = gs_params
    
    vs_ex1_loaded = nk.vqs.MCState(sampler, ma, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples)
    vs_ex1_loaded.init_parameters(jax.nn.initializers.normal(stddev=0.25))
    vs_ex1_loaded.parameters = ex1_params
    
    vs_ex2 = nk.vqs.MCState(sampler, ma, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples)
    
    opt_ex2 = nk.optimizer.Sgd(learning_rate=learning_rate)
    sr_ex2 = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    
    shift_list = [shift, shift]
    state_list = [vs_gs, vs_ex1_loaded]
    
    gs_ex2 = vmc_ex.VMC_ex(
        hamiltonian=ha,
        optimizer=opt_ex2,
        variational_state=vs_ex2,
        preconditioner=sr_ex2,
        state_list=state_list,
        shift_list=shift_list
    )
    
    print(f"\n开始计算第二激发态 (shift={shift})...")
    gs_ex2.run(out=output_path, n_iter=n_iter)
    
    data_ex2 = json.load(open(output_path + '.log'))
    energy_ex2 = data_ex2["Energy"]["Mean"]["real"]
    final_energy_ex2 = reduce(lambda x, y: x if y is None else y, energy_ex2)
    
    ex2_params = vs_ex2.parameters
    
    print(f"第二激发态能量: {final_energy_ex2:.8f} Ha")
    print(f"激发能: {final_energy_ex2 - e_fci:.8f} Ha")
    
    return vs_ex2, final_energy_ex2, ex2_params, vs_gs, vs_ex1_loaded


def compute_all_excited_states(bond_length=1.4,
                             n_iter=2000,
                             model=None,
                             learning_rate=0.03,
                             diag_shift=0.01,
                             n_discard_per_chain=100,
                             n_samples=1024,
                             shift=0.3,
                             save_params=True,
                             output_dir='Data'):
    """
    计算H2分子的基态、第一激发态和第二激发态
    
    Args:
        bond_length: H2分子的键长（埃）
        n_iter: 迭代次数
        model: 神经网络模型，如果为None则使用默认的RBM(alpha=2)
        learning_rate: 学习率
        diag_shift: SR的对角位移
        n_discard_per_chain: 每条链丢弃的样本数
        n_samples: 样本数
        shift: 正交化约束的位移参数（推荐0.3）
        save_params: 是否保存参数
        output_dir: 输出目录
    
    Returns:
        results: 包含所有结果的字典
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    mol, ha, hi, sampler, e_fci_all, E_fci = setup_h2_system(bond_length)
    
    print("\n" + "="*60)
    print("开始计算H2分子的所有能级")
    print("="*60)
    
    results = {
        'bond_length': bond_length,
        'e_fci_all': e_fci_all,
        'E_fci': E_fci
    }
    
    vs_gs, final_energy_gs, gs_params = compute_ground_state(
        ha, sampler,
        n_iter=n_iter,
        model=model,
        learning_rate=learning_rate,
        diag_shift=diag_shift,
        n_discard_per_chain=n_discard_per_chain,
        n_samples=n_samples,
        output_path=f'{output_dir}/h2_ground_state'
    )
    
    results['final_energy_gs'] = final_energy_gs
    results['gs_params'] = gs_params
    
    if save_params:
        with open(f'{output_dir}/v0.json', 'wb') as f:
            pickle.dump(gs_params, f)
        print(f"基态参数已保存到 {output_dir}/v0.json")
    
    vs_ex1, final_energy_ex1, ex1_params, vs_gs = compute_1st_excited_state(
        ha, sampler, gs_params, E_fci,
        n_iter=n_iter,
        model=model,
        learning_rate=learning_rate,
        diag_shift=diag_shift,
        n_discard_per_chain=n_discard_per_chain,
        n_samples=n_samples,
        shift=shift,
        output_path=f'{output_dir}/h2_excited_state_1'
    )
    
    results['final_energy_ex1'] = final_energy_ex1
    results['ex1_params'] = ex1_params
    
    if save_params:
        with open(f'{output_dir}/v1.json', 'wb') as f:
            pickle.dump(ex1_params, f)
        print(f"第一激发态参数已保存到 {output_dir}/v1.json")
    
    vs_ex2, final_energy_ex2, ex2_params, vs_gs, vs_ex1_loaded = compute_2nd_excited_state(
        ha, sampler, gs_params, ex1_params, E_fci,
        n_iter=n_iter,
        model=model,
        learning_rate=learning_rate,
        diag_shift=diag_shift,
        n_discard_per_chain=n_discard_per_chain,
        n_samples=n_samples,
        shift=shift,
        output_path=f'{output_dir}/h2_excited_state_2'
    )
    
    results['final_energy_ex2'] = final_energy_ex2
    results['ex2_params'] = ex2_params
    
    if save_params:
        with open(f'{output_dir}/v2.json', 'wb') as f:
            pickle.dump(ex2_params, f)
        print(f"第二激发态参数已保存到 {output_dir}/v2.json")
    
    print("\n" + "="*60)
    print("计算完成！结果总结：")
    print("="*60)
    print(f"基态能量 (E0): {final_energy_gs:.8f} Ha")
    print(f"第一激发态能量 (E1): {final_energy_ex1:.8f} Ha")
    print(f"第二激发态能量 (E2): {final_energy_ex2:.8f} Ha")
    print(f"\n第一激发能 (E1-E0): {final_energy_ex1 - final_energy_gs:.8f} Ha")
    print(f"第二激发能 (E2-E0): {final_energy_ex2 - final_energy_gs:.8f} Ha")
    print(f"\nFCI基态能量: {E_fci:.8f} Ha")
    print(f"FCI第一激发态能量: {e_fci_all[1]:.8f} Ha")
    print(f"FCI第二激发态能量: {e_fci_all[2]:.8f} Ha")
    print(f"\n基态误差: {abs(final_energy_gs - E_fci):.8f} Ha")
    print(f"第一激发态误差: {abs(final_energy_ex1 - e_fci_all[1]):.8f} Ha")
    print(f"第二激发态误差: {abs(final_energy_ex2 - e_fci_all[2]):.8f} Ha")
    
    return results


def plot_convergence(output_paths, e_fci_all, E_fci, labels=None):
    """
    绘制所有状态的收敛曲线
    
    Args:
        output_paths: 输出文件路径列表
        e_fci_all: FCI所有能级
        E_fci: FCI基态能量
        labels: 状态标签列表
    """
    if labels is None:
        labels = ['Ground State', '1st Excited State', '2nd Excited State']
    
    plt.figure(figsize=(12, 6))
    
    for i, (output_path, label) in enumerate(zip(output_paths, labels)):
        data = json.load(open(output_path + '.log'))
        iters = data["Energy"]["iters"]
        energy = data["Energy"]["Mean"]["real"]
        plt.plot(iters, energy, label=label, linewidth=2)
    
    for i, E_fci in enumerate(e_fci_all[:3]):
        plt.axhline(y=E_fci, color=f'C{i}', linestyle='--', alpha=0.5, label=f'FCI E{i}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Ha)')
    plt.title('H2 Energy Convergence for Different States')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_energy_levels(energies, e_fci_all, labels=None):
    """
    绘制能级图
    
    Args:
        energies: 计算得到的能级列表
        e_fci_all: FCI所有能级
        labels: 状态标签列表
    """
    if labels is None:
        labels = ['Ground State', '1st Excited', '2nd Excited']
    
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    for i, (E, label, color) in enumerate(zip(energies, labels, colors)):
        plt.hlines(E, 0, 1, colors=color, linewidth=3, label=label)
        plt.text(1.05, E, f'{E:.6f} Ha', va='center', fontsize=10)
    
    for i, E_fci in enumerate(e_fci_all[:3]):
        plt.hlines(E_fci, 1.5, 2.5, colors='gray', linestyle='--', alpha=0.5)
        plt.text(2.55, E_fci, f'FCI: {E_fci:.6f} Ha', va='center', fontsize=8, color='gray')
    
    plt.xlim(-0.1, 3)
    plt.ylabel('Energy (Ha)')
    plt.title('H2 Molecular Energy Levels')
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    results = compute_all_excited_states(
        bond_length=1.4,
        n_iter=2000,
        learning_rate=0.03,
        diag_shift=0.01,
        n_discard_per_chain=100,
        n_samples=1024,
        shift=0.3,
        save_params=True,
        output_dir='Data'
    )
    
    energies = [results['final_energy_gs'], results['final_energy_ex1'], results['final_energy_ex2']]
    output_paths = [
        'Data/h2_ground_state',
        'Data/h2_excited_state_1',
        'Data/h2_excited_state_2'
    ]
    
    plot_convergence(output_paths, results['e_fci_all'], results['E_fci'])
    plot_energy_levels(energies, results['e_fci_all'])



import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from functools import partial
import numpy as np
from QNN_jax import initialize_parameters
import jax
jax.config.update("jax_enable_x64", True)
# -------------------------- 第一步：改造量子电路，原生支持Batch输入 --------------------------
def quantum_neural_network(x, params, n_qubits, n_layers):
    """
    兼容Batch的量子电路核心逻辑：
    - x: 输入张量，形状为 (batch_size, n_qubits)（Batch维度在前）
    - params: 量子电路参数，形状为 (n_layers, 2 * n_qubits)
    - 所有量子门操作自动沿Batch维度向量化，无需手动循环
    """
    # 1. 强制将输入转为JAX张量（兼容np.array/其他格式），并确保是二维（batch, n_qubits）
    x = jnp.atleast_2d(x)
    # 校验特征维度（Batch维度不校验，由NNX自动兼容）
    if x.shape[-1] != n_qubits:
        raise ValueError(f"输入特征维度需为{n_qubits}，当前为{x.shape[-1]}")
    
    # 2. 数据编码：向量化RX门（自动兼容Batch）
    # qml.RX支持批量角度输入，会自动为每个Batch样本应用对应角度的门
    for i in range(n_qubits):
        qml.RX(x[:, i] * jnp.pi, wires=i)  # x[:, i] 取所有Batch样本的第i个特征
    
    # 3. 变分层：向量化旋转/纠缠门（Batch维度自动兼容）
    for layer in range(n_layers):
        # 纠缠层（CNOT无参数，Batch不影响）
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.Barrier(wires=range(n_qubits))
        # 旋转层：参数向量化，自动适配Batch
        for i in range(n_qubits):
            qml.RX(params[layer, 2*i], wires=i)  # RX参数
            qml.RZ(params[layer, 2*i+1], wires=i)  # RZ参数
    # 4. 测量：返回每个量子比特的期望值

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

