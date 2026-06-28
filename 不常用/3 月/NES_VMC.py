import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx
import jax
import jax.numpy as jnp
from flax import nnx
import jax.tree_util as jtu
import sys



class NES_VMC():
    def __init__(self,bond_length:float=1.4,n_states:int=3):
        geometry = [
            ('H', (0., 0., 0.)),
            ('H', (bond_length, 0., 0.)),
        ]
        self.n_states = n_states

        # 创建分子对象
        mol = gto.M(atom=geometry, basis='STO-3G')
        # Hartree-Fock计算
        mf = scf.RHF(mol).run(verbose=0)
        self.E_hf = mf.e_tot

        # FCI计算（参考值）
        cisolver = fci.FCI(mf)
        cisolver.nroots = 4
        self.E_fcis, fcivec = cisolver.kernel()


        # 创建NetKet哈密顿量
        self.ha = nkx.operator.from_pyscf_molecule(mol)
        # 创建Hilbert空间
        self.hi = nk.hilbert.SpinOrbitalFermions(
            n_orbitals=2,
            s=1/2,
            n_fermions_per_spin=(1, 1)
        )
        # 创建采样器
        self.graph = nk.graph.Graph(edges=[(0,1),(2,3)])
        self.sampler = nk.sampler.MetropolisFermionHop(
            self.hi, graph=self.graph, n_chains=self.n_states, spin_symmetric=True, sweep_size=64
        )
        
        
    def setup_sampler(self):
        self.model = SingleStateAnsatz(n_spin_orbitals=4, hidden_dim=4*3, rngs=nnx.Rngs(1))
        self.model_state= nnx.split(self.model)
        sampler_state = self.sampler.init_state(NES_VMC.forward, self.model_state, seed=1)
        sampler_state = self.sampler.reset(NES_VMC.forward, self.model_state, sampler_state)
        # Generate samples
        samples, sampler_state = self.sampler.sample(
            NES_VMC.forward, self.model_state, state=sampler_state,chain_length=100)
        return samples
                    
                    
    @staticmethod
    def forward(state,x):
        y, ffnn_state = nnx.call(state)(x)
        return y

class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        # 复数值线性层（费米子波函数需相位描述）
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        输入：x = 单组自旋轨道占据态（形状 [n_spin_orbitals,]，如 H₂ 的 [0,1,0,1]）
        输出：复数值波函数值 ψ(x)
        """
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)  # 输出形状 []（标量复数）


class NESTotalAnsatz(nnx.Module):
    """NES-VMC 总 Ansatz：K 个单态 Ansatz 的行列式"""
    def __init__(self, n_spin_orbitals: int, n_states: int = 3, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states  # K：要计算的激发态数量（如 H₂ 案例中 K=4）
        self.n_spin_orbitals = n_spin_orbitals
        
        # 初始化 K 个独立的单态 Ansatz（避免参数共享，保证态独立性）
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(42+i))
            for i in range(n_states)
        ]

    def __call__(self, x_batch: jax.Array) -> jax.Array:
        """
        输入：x_batch = K 组自旋轨道占据态（形状 [n_states, n_spin_orbitals]）
        输出：总波函数值 Ψ = det([ψ_i(x^j)])（标量复数）
        """
        M = []
        for i in range(self.n_states):
            for j in range(self.n_states):
                # 对第 i 个单态 Ansatz，批量计算 K 组状态的输出（形状 [n_states,]）
                psi_i_xj = self.single_ansatz_list[j](x_batch[i])
                #print(f'第{i}行 第{j}列的元素={psi_i_xj}')
                M.append(psi_i_xj)
        M = jnp.stack(M, axis=0)  # 转为 JAX 数组，形状 [n_states, n_states]（K×K）
        M = M.reshape(self.n_states, self.n_states)
        #print(M)
        # 计算行列式（对应原文总 Ansatz 定义）
        psi_total = jnp.linalg.det(M)
        return psi_totals