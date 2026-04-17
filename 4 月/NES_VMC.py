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
sys.path.insert(0, '.')



def Ham_psi(ha:nk.operator.DiscreteOperator,model,x:jnp.array):
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels*psi_values)
    return H_psi_x



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
    def __init__(self, n_spin_orbitals: int, n_states: int = 3, hidden_dim: int = 12, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states  # K
        self.n_spin_orbitals = n_spin_orbitals
        
        # 初始化 K 个独立的单态 Ansatz
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(42+i))
            for i in range(n_states)
        ]

    def __call__(self, x_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        """x_batch.shape = [K, n_spin_orbitals]
        
        输出：
            psi_total : 行列式标量（给 NetKet 采样）
            M_matrix  : K×K 矩阵（给 Eq.17 局部能量计算）
        """
        if x_batch.shape[0] != self.n_states:
            raise ValueError(f"x_batch.shape[0] != {self.n_states}")
        K = self.n_states
        M = []
        for i in range(K):
            for j in range(K):
                # M[i,j] = ψ_j(x_i)
                psi_i_xj = self.single_ansatz_list[j](x_batch[i])
                M.append(psi_i_xj)
        
        # 构建 K×K 矩阵
        M = jnp.stack(M, axis=0)
        M_matrix = M.reshape(K, K)  # 这就是论文里的 Ψ 矩阵！
        # 计算行列式
        psi_total = jnp.linalg.det(M_matrix)
        return psi_total, M_matrix
    
    
def Ham_Psi(ha:nk.operator.DiscreteOperator,total_ansatz:NESTotalAnsatz,x:jnp.array):
    k = total_ansatz.n_states
    if x.shape[0] != k:
        raise ValueError(f"Input array must have shape ({k},) but got shape {x.shape}")
    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha,model=total_ansatz.single_ansatz_list[j],x=x[i])
            tmp.append(ele)
            
    
        H_psi_x_i.append(tmp)
    
    HPsi = jnp.array(H_psi_x_i).reshape(k,k)    
    return HPsi            