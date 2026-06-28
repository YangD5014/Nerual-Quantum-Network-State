"""
NES-VMC 测试脚本

测试NES-VMC实现是否能正确运行
"""

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from pyscf import gto, scf, fci
import netket.experimental as nkx
import sys
sys.path.insert(0, '/Users/yangjianfei/mac_vscode/神经网络量子态/组会汇报 0205')

print("="*60)
print("NES-VMC 测试")
print("="*60)

print(f"\nNetKet版本: {nk.__version__}")
print(f"JAX版本: {jax.__version__}")

print("\n1. 设置H2分子系统...")
bond_length = 1.4
geometry = [
    ('H', (0., 0., 0.)),
    ('H', (bond_length, 0., 0.)),
]

mol = gto.M(atom=geometry, basis='STO-3G')
mf = scf.RHF(mol).run(verbose=0)
E_hf = mf.e_tot
print(f"   Hartree-Fock能量: {E_hf:.8f} Ha")

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print(f"   FCI基态能量: {E_fcis[0]:.8f} Ha")

ha = nkx.operator.from_pyscf_molecule(mol)

print("\n2. 创建Hilbert空间和采样器...")
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1)
)
print(f"   Hilbert空间维度: {hi.n_states}")

g = nk.graph.Graph(edges=[(0,1),(2,3)])
sa = nk.sampler.MetropolisFermionHop(
    hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
)

print("\n3. 定义神经网络模型...")
class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int = 2, *, rngs: nnx.Rngs):
        self.alpha = alpha
        self.linear1 = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(in_features=alpha * N, out_features=alpha * N, rngs=rngs, param_dtype=complex)
        self.linear_out = nnx.Linear(in_features=alpha * N, out_features=1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array):
        y = self.linear1(x)
        y = nnx.tanh(y)
        y = self.linear2(y)
        y = nnx.tanh(y)
        y = self.linear_out(y)
        return jnp.squeeze(y, axis=-1)

N = 4
print(f"   输入维度: {N}")

print("\n4. 创建NES-VMC驱动器...")
try:
    from nes_vmc_driver import NESVMC
    
    n_states = 2
    vstate_list = []
    for i in range(n_states):
        model_i = FFN(N=N, alpha=2, rngs=nnx.Rngs(42 + i * 100))
        vs = nk.vqs.MCState(
            sa, 
            model_i, 
            n_discard_per_chain=10, 
            n_samples=256
        )
        vstate_list.append(vs)
    
    print(f"   创建了 {len(vstate_list)} 个变分态")
    
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.01)
    
    nes_driver = NESVMC(
        hamiltonian=ha,
        optimizer=opt,
        variational_states=vstate_list,
        preconditioner=sr,
        n_states=n_states
    )
    
    print("   NES-VMC驱动器创建成功！")
    
    print("\n5. 运行短测试...")
    log_data = nes_driver.run(n_iter=10, out='test_nes_vmc')
    
    print("\n6. 结果:")
    final_energies = nes_driver.get_state_energies()
    print(f"   计算得到的能量: {final_energies}")
    print(f"   FCI参考能量: {[E_fcis[i] for i in range(n_states)]}")
    
    print("\n" + "="*60)
    print("测试完成！NES-VMC实现运行正常。")
    print("="*60)
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
