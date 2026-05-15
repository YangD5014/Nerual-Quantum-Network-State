"""
NES-VMC 激发态能量求解 - 探索与测试文件
目标：找到能稳定收敛到FCI基准的NES-VMC实现方法
"""

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx
import optax
from tqdm import tqdm
from functools import partial
from jax import flatten_util
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("NES-VMC 激发态能量求解 - 探索测试")
print("="*70)

# ==============================================================================
# 1. H₂ 分子定义 & FCI 基准
# ==============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print("\n" + "="*70)
print("H₂ FCI 基准能量")
print("="*70)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")

# ==============================================================================
# 2. NetKet 哈密顿量和采样器
# ==============================================================================
ha = nkx.operator.from_pyscf_molecule(mol)
ha_jax = ha.to_jax_operator()

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

K = 2
hi_ext = hi ** K
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)

try:
    single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
except AttributeError:
    from netket.experimental.sampler.rules.fermion_2nd import ParticleExchangeRule
    single_rule = ParticleExchangeRule(hilbert=hi, graph=g)

tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)

# ==============================================================================
# 3. 模型定义
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


class NESTotalAnsatz(nnx.Module):
    """NES-VMC 总波函数 Ansatz"""
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=rngs)
            for _ in range(self.K)
        ]

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            M = []
            for i in range(self.K):
                row = []
                for j in range(self.K):
                    val = self.single_ansatz_list[j](x_single[i])
                    row.append(val)
                M.append(jnp.stack(row))
            M = jnp.stack(M)
            log_det = jnp.linalg.det(M)
            return log_det, M

        if len(x.shape) == 2 and x.shape[0] == self.K and x.shape[1] == self.n_spin:
            log_psi, log_M = _forward_single(x)
        else:
            x = x.reshape(-1, K, self.n_spin)
            log_psi, log_M = jax.vmap(_forward_single)(x)
        return log_psi, log_M


def create_machine(model: NESTotalAnsatz):
    """创建 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_psi_total

    return machine, graphdef, state


# ==============================================================================
# 4. 局域能量矩阵计算 - 多种版本
# ==============================================================================
def apply_hamiltonian_on_M_v1(model, hamiltonian_jax, M, sigma, K):
    """标准版：H_M = H_ext · M"""
    def apply_hamiltonian_to_sample(args):
        M_i, sigma_i = args
        
        def apply_hamiltonian_to_row(sigma_row):
            eta_i, H_mat_i = hamiltonian_jax.get_conn_padded(sigma_row[None, :])
            eta_i = eta_i[0]
            H_mat_i = H_mat_i[0]
            
            H_M_row = []
            for j in range(K):
                psi_j = model.single_ansatz_list[j]
                psi_j_eta = jax.vmap(psi_j)(eta_i)
                H_psi_j = jnp.sum(H_mat_i * psi_j_eta)
                H_M_row.append(H_psi_j)
            return jnp.stack(H_M_row)
        
        H_M_i = jax.vmap(apply_hamiltonian_to_row)(sigma_i)
        return H_M_i
    
    H_M = jax.vmap(apply_hamiltonian_to_sample)((M, sigma))
    return H_M


def compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K, eps=1e-4):
    """标准版：E_L = M⁻¹ · H · M"""
    model = nnx.merge(model_graphdef, params)
    log_psi, M = model(sigma)
    M += eps * jnp.eye(K)
    
    H_M = apply_hamiltonian_on_M_v1(model, hamiltonian_jax, M, sigma, K)
    E_L = jax.vmap(lambda m, hm: jnp.linalg.solve(m, hm))(M, H_M)
    
    return E_L, log_psi, M


def compute_local_energy_matrix_v2(model_graphdef, params, sigma, hamiltonian_jax, K, eps=1e-4):
    """改进版：添加数值稳定性保护"""
    model = nnx.merge(model_graphdef, params)
    log_psi, M = model(sigma)
    
    det_M = jnp.linalg.det(M)
    
    M_reg = M + eps * jnp.eye(K)
    
    H_M = apply_hamiltonian_on_M_v1(model, hamiltonian_jax, M, sigma, K)
    
    def safe_solve(A, B):
        det = jnp.linalg.det(A)
        if jnp.abs(det) < eps:
            return jnp.linalg.solve(A + eps * jnp.eye(K), B)
        return jnp.linalg.solve(A, B)
    
    E_L = jax.vmap(safe_solve)(M_reg, H_M)
    
    return E_L, log_psi, M, det_M


# ==============================================================================
# 5. 损失函数 - 多种定义
# ==============================================================================
def compute_loss_v1(model_graphdef, params, sigma, hamiltonian_jax, K):
    """版本1：直接最小化迹"""
    E_L, _, _ = compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K)
    return jnp.mean(jnp.trace(E_L, axis1=-2, axis2=-1))


def compute_loss_v2(model_graphdef, params, sigma, hamiltonian_jax, K):
    """版本2：迹的平均"""
    E_L, _, _ = compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K)
    tr_el = jnp.trace(E_L, axis1=-2, axis2=-1)
    return jnp.mean(tr_el)


def compute_loss_v3(model_graphdef, params, sigma, hamiltonian_jax, K):
    """版本3：带方差惩罚的迹"""
    E_L, _, _ = compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K)
    tr_el = jnp.trace(E_L, axis1=-2, axis2=-1)
    mean = jnp.mean(tr_el)
    var = jnp.var(tr_el)
    return mean + 0.1 * var


def compute_loss_v4(model_graphdef, params, sigma, hamiltonian_jax, K):
    """版本4：所有矩阵元的平方和（更稳定）"""
    E_L, _, _ = compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K)
    return jnp.mean(E_L)


def compute_loss_and_grad_v1(model_graphdef, params, sigma, hamiltonian_jax, K):
    """标准损失函数和梯度"""
    E_L, log_psi, M = compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K)
    tr_el = jnp.trace(E_L, axis1=-2, axis2=-1)
    loss = jnp.mean(tr_el)
    
    E_L_mean = jnp.mean(E_L, axis=0)
    E_L_centered = E_L - E_L_mean
    
    def total_loss(p):
        model = nnx.merge(model_graphdef, p)
        logp, _ = model(sigma)
        el, _, _ = compute_local_energy_matrix_v1(model_graphdef, p, sigma, hamiltonian_jax, K)
        return jnp.real(jnp.mean(jnp.trace(el, axis1=-2, axis2=-1)))
    
    grads = grad(total_loss)(params)
    
    return loss, grads, E_L_mean


def compute_loss_and_grad_v2(model_graphdef, params, sigma, hamiltonian_jax, K):
    """改进损失函数：使用伪逆"""
    E_L, log_psi, M = compute_local_energy_matrix_v1(model_graphdef, params, sigma, hamiltonian_jax, K)
    
    tr_el = jnp.trace(E_L, axis1=-2, axis2=-1)
    loss = jnp.mean(tr_el)
    
    E_L_mean = jnp.mean(E_L, axis=0)
    
    def total_loss(p):
        model = nnx.merge(model_graphdef, p)
        logp, _ = model(sigma)
        el, _, _ = compute_local_energy_matrix_v1(model_graphdef, p, sigma, hamiltonian_jax, K)
        return jnp.real(jnp.mean(jnp.trace(el, axis1=-2, axis2=-1)))
    
    grads = grad(total_loss)(params)
    
    return loss, grads, E_L_mean


# ==============================================================================
# 6. QGT 和自然梯度
# ==============================================================================
@partial(jax.jit, static_argnames=("model_graphdef",))
def compute_QGT(model_graphdef, params, sigma):
    """量子几何张量"""
    def log_psi(p, x):
        model = nnx.merge(model_graphdef, p)
        log_p, _ = model(x)
        return jnp.real(log_p)
    
    grad_log = grad(log_psi, argnums=0)
    batch_g = vmap(grad_log, (None, 0))(params, sigma)
    
    mean_g = jax.tree.map(lambda g: jnp.nanmean(g, axis=0), batch_g)
    
    def qgt_cov(g, mg):
        eg = jnp.nanmean(g * jnp.conj(g), axis=0)
        return eg - mg * jnp.conj(mg)
    
    S = jax.tree.map(qgt_cov, batch_g, mean_g)
    return S


@jax.jit
def apply_natural_gradient(grads, S, eps=1e-4):
    return jax.tree.map(lambda g, s: g / (s + eps), grads, S)


# ==============================================================================
# 7. 训练函数
# ==============================================================================
def train_nes(loss_version='v1', use_natural_grad=False, n_iter=200, lr=0.01, hidden_dim=16, seed=21):
    """训练 NES-VMC"""
    
    rngs = nnx.Rngs(seed)
    total_ansatz = NESTotalAnsatz(4, K, hidden_dim, rngs=rngs)
    machine, graphdef, params = create_machine(total_ansatz)
    
    sampler_state = sampler.init_state(machine, params, seed=seed)
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    history = {
        'loss': [],
        'energies': [],
        'errors': []
    }
    
    loss_func = compute_loss_and_grad_v1 if loss_version == 'v1' else compute_loss_and_grad_v2
    
    print(f"\n训练配置:")
    print(f"  - 损失函数版本: {loss_version}")
    print(f"  - 自然梯度: {use_natural_grad}")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print(f"  - 学习率: {lr}")
    print(f"  - 迭代次数: {n_iter}")
    print(f"  - 随机种子: {seed}")
    print("-" * 70)
    
    for step in range(n_iter):
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, chain_length=20
        )
        samples = samples.reshape(-1, K, 4)
        
        loss, grads, E_L_avg = loss_func(graphdef, params, samples, ha_jax, K)
        
        if use_natural_grad:
            S = compute_QGT(graphdef, params, samples)
            grads = apply_natural_gradient(grads, S)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        eig_vals = jnp.linalg.eigvalsh(E_L_avg)
        
        history['loss'].append(float(loss.real) if np.isfinite(loss.real) else np.nan)
        history['energies'].append([float(e) if np.isfinite(e) else np.nan for e in eig_vals])
        
        if np.isfinite(eig_vals[0]):
            err = abs(eig_vals[0] - E_fcis[0])
            history['errors'].append(float(err))
        else:
            history['errors'].append(np.nan)
        
        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss.real:.6f} | "
                  f"E0: {eig_vals[0]:.6f} (FCI: {E_fcis[0]:.6f}) | "
                  f"E1: {eig_vals[1]:.6f} (FCI: {E_fcis[1]:.6f})")
    
    final_energies = history['energies'][-1]
    print("-" * 70)
    print(f"最终结果:")
    print(f"  E0 = {final_energies[0]:.8f} (FCI: {E_fcis[0]:.8f}, 误差: {abs(final_energies[0]-E_fcis[0]):.6f})")
    print(f"  E1 = {final_energies[1]:.8f} (FCI: {E_fcis[1]:.8f}, 误差: {abs(final_energies[1]-E_fcis[1]):.6f})")
    
    return history, final_energies


# ==============================================================================
# 8. 主测试流程
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("开始 NES-VMC 探索实验")
    print("="*70)
    
    results = {}
    
    # 实验1: 不同损失函数版本
    print("\n" + "="*70)
    print("实验1: 测试不同损失函数版本")
    print("="*70)
    
    print("\n>>> 测试损失函数版本1（直接最小化迹）")
    history1, final1 = train_nes(loss_version='v1', n_iter=150, lr=0.005, hidden_dim=16, seed=21)
    results['loss_v1'] = {'history': history1, 'final': final1}
    
    print("\n>>> 测试损失函数版本2（迹的平均）")
    history2, final2 = train_nes(loss_version='v2', n_iter=150, lr=0.005, hidden_dim=16, seed=21)
    results['loss_v2'] = {'history': history2, 'final': final2}
    
    # 实验2: 不同超参数
    print("\n" + "="*70)
    print("实验2: 测试不同超参数")
    print("="*70)
    
    print("\n>>> 测试更小的学习率")
    history3, final3 = train_nes(loss_version='v1', n_iter=200, lr=0.001, hidden_dim=16, seed=21)
    results['small_lr'] = {'history': history3, 'final': final3}
    
    print("\n>>> 测试更大的隐藏层")
    history4, final4 = train_nes(loss_version='v1', n_iter=200, lr=0.005, hidden_dim=32, seed=21)
    results['large_hidden'] = {'history': history4, 'final': final4}
    
    # 实验3: 使用自然梯度
    print("\n" + "="*70)
    print("实验3: 测试自然梯度优化")
    print("="*70)
    
    print("\n>>> 测试使用自然梯度")
    history5, final5 = train_nes(loss_version='v1', use_natural_grad=True, n_iter=200, lr=0.01, hidden_dim=16, seed=21)
    results['natural_grad'] = {'history': history5, 'final': final5}
    
    # 实验4: 不同随机种子
    print("\n" + "="*70)
    print("实验4: 测试不同随机初始化")
    print("="*70)
    
    print("\n>>> 测试随机种子42")
    history6, final6 = train_nes(loss_version='v1', n_iter=200, lr=0.005, hidden_dim=16, seed=42)
    results['seed_42'] = {'history': history6, 'final': final6}
    
    print("\n>>> 测试随机种子123")
    history7, final7 = train_nes(loss_version='v1', n_iter=200, lr=0.005, hidden_dim=16, seed=123)
    results['seed_123'] = {'history': history7, 'final': final7}
    
    # ==============================================================================
    # 结果汇总
    # ==============================================================================
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    print(f"\nFCI 基准:")
    print(f"  E0 = {E_fcis[0]:.8f} Ha")
    print(f"  E1 = {E_fcis[1]:.8f} Ha")
    print("\n实验结果:")
    print("-"*70)
    
    all_results = [
        ("损失函数v1", results['loss_v1']['final']),
        ("损失函数v2", results['loss_v2']['final']),
        ("小学习率", results['small_lr']['final']),
        ("大隐藏层", results['large_hidden']['final']),
        ("自然梯度", results['natural_grad']['final']),
        ("种子42", results['seed_42']['final']),
        ("种子123", results['seed_123']['final']),
    ]
    
    best_e0_err = float('inf')
    best_e1_err = float('inf')
    best_config_e0 = None
    best_config_e1 = None
    
    for name, final in all_results:
        if np.isfinite(final[0]) and np.isfinite(final[1]):
            err0 = abs(final[0] - E_fcis[0])
            err1 = abs(final[1] - E_fcis[1])
            print(f"{name:20s}: E0={final[0]:.6f} (err={err0:.4f}), E1={final[1]:.6f} (err={err1:.4f})")
            
            if err0 < best_e0_err:
                best_e0_err = err0
                best_config_e0 = name
            if err1 < best_e1_err:
                best_e1_err = err1
                best_config_e1 = name
        else:
            print(f"{name:20s}: NaN 或 Inf")
    
    print("-"*70)
    print(f"\n最佳基态能量配置: {best_config_e0} (误差: {best_e0_err:.6f})")
    print(f"最佳第一激发态配置: {best_config_e1} (误差: {best_e1_err:.6f})")
    
    # ==============================================================================
    # 可视化
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for name, result in results.items():
        history = result['history']
        energies = history['energies']
        valid_indices = [i for i, e in enumerate(energies) if np.isfinite(e[0])]
        if len(valid_indices) > 0:
            e0s = [energies[i][0] for i in valid_indices]
            axes[0, 0].plot(valid_indices, e0s, label=name, alpha=0.7)
    
    axes[0, 0].axhline(y=E_fcis[0], color='r', linestyle='--', label='FCI E0', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('E0 (Ha)')
    axes[0, 0].set_title('基态能量收敛对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for name, result in results.items():
        history = result['history']
        energies = history['energies']
        valid_indices = [i for i, e in enumerate(energies) if np.isfinite(e[1])]
        if len(valid_indices) > 0:
            e1s = [energies[i][1] for i in valid_indices]
            axes[0, 1].plot(valid_indices, e1s, label=name, alpha=0.7)
    
    axes[0, 1].axhline(y=E_fcis[1], color='r', linestyle='--', label='FCI E1', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('E1 (Ha)')
    axes[0, 1].set_title('第一激发态能量收敛对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for name, result in results.items():
        history = result['history']
        errors = history['errors']
        valid_errors = [(i, e) for i, e in enumerate(errors) if np.isfinite(e)]
        if len(valid_errors) > 0:
            indices, vals = zip(*valid_errors)
            axes[1, 0].semilogy(indices, vals, label=name, alpha=0.7)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Error (Ha)')
    axes[1, 0].set_title('基态能量误差对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    config_names = [name for name, _ in all_results]
    final_e0_errors = [abs(final[0] - E_fcis[0]) if np.isfinite(final[0]) else np.nan 
                      for _, final in all_results]
    final_e1_errors = [abs(final[1] - E_fcis[1]) if np.isfinite(final[1]) else np.nan 
                      for _, final in all_results]
    
    x = np.arange(len(config_names))
    width = 0.35
    axes[1, 1].bar(x - width/2, final_e0_errors, width, label='E0 Error', alpha=0.7)
    axes[1, 1].bar(x + width/2, final_e1_errors, width, label='E1 Error', alpha=0.7)
    axes[1, 1].set_ylabel('Error (Ha)')
    axes[1, 1].set_title('最终误差对比')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(config_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('nes_vmc_exploration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("探索完成！结果图表已保存为 'nes_vmc_exploration.png'")
    print("="*70)
