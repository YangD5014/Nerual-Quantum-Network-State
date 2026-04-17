import netket as nk
import numpy as np
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx
import jax
import jax.numpy as jnp
from flax import nnx
import itertools


class FFN_Amplitude(nnx.Module):
    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        self.alpha = alpha
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.linear(x)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)


class FFN_Phase(nnx.Module):
    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        self.alpha = alpha
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.linear(x)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)


class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int, rngs_amplitude: nnx.Rngs, rngs_phase: nnx.Rngs) -> None:
        self.ffn_amplitude = FFN_Amplitude(N=N, alpha=alpha, rngs=rngs_amplitude)
        self.ffn_phase = FFN_Phase(N=N, alpha=alpha, rngs=rngs_phase)

    def __call__(self, x: jax.Array):
        y = self.ffn_amplitude(x) + 1j * self.ffn_phase(x)
        return y


def run_h4_vmc(bond_distance: float, n_iterations: int = 300, exp_name: str = "h4_molecule_ffnn_layer1"):
    """
    运行 H4 分子的 VMC 计算，返回能量收敛数据

    Parameters:
    -----------
    bond_distance : float
        H-H 键长（埃）
    n_iterations : int, optional
        VMC 优化迭代次数，默认为 300
    exp_name : str, optional
        实验名称，用于保存日志文件，默认为 "h4_molecule_ffnn_layer1"

    Returns:
    --------
    dict
        包含以下键的字典：
        - 'energy_mean': 能量均值数组 (data["Energy"]["Mean"]['real'])
        - 'iters': 迭代次数数组
        - 'E_hf': Hartree-Fock 能量
        - 'E_fci': FCI 能量
        - 'final_vmc_energy': 最终 VMC 能量
    """
    
    geometry = [
        ('H', (0., 0., 0.)),
        ('H', (bond_distance * 1, 0., 0.)),
        ('H', (bond_distance * 2., 0., 0.)),
        ('H', (bond_distance * 3, 0., 0.)),
    ]

    mol = gto.M(atom=geometry, basis='STO-3G')

    mf = scf.RHF(mol).run(verbose=0)
    E_hf = mf.e_tot

    cisolver = fci.FCI(mf)
    E_fci, fcivec = cisolver.kernel()

    ha = nkx.operator.from_pyscf_molecule(mol)

    letters_alpha = [0, 1, 2, 3]
    letters_beta = [4, 5, 6, 7]
    combinations_alpha = itertools.combinations(letters_alpha, 2)
    combinations_beta = itertools.combinations(letters_beta, 2)
    clusters = list(combinations_alpha) + list(combinations_beta)

    hi = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=4,
        s=1/2,
        n_fermions_per_spin=(2, 2)
    )

    g = nk.graph.Graph(edges=clusters)
    sa = nk.sampler.MetropolisFermionHop(
        hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
    )

    N = 8
    ffnn_model = FFN(N=N, alpha=1, rngs_amplitude=nnx.Rngs(2), rngs_phase=nnx.Rngs(3))
    vs = nk.vqs.MCState(sa, ffnn_model, n_discard_per_chain=10, n_samples=512)

    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.01)

    gs = nk.driver.VMC(ha, opt, variational_state=vs, preconditioner=sr)

    gs.run(n_iterations, out=exp_name)

    with open(f"{exp_name}.log") as f:
        data = json.load(f)

    result = {
        'energy_mean': data["Energy"]["Mean"]['real'],
        'iters': data["Energy"]["iters"],
        'E_hf': E_hf,
        'E_fci': E_fci,
        'final_vmc_energy': data["Energy"]["Mean"]['real'][-1]
    }

    return result


if __name__ == "__main__":
    bond_distance = 0.74
    result = run_h4_vmc(bond_distance)
    
    print(f"最终VMC能量: {result['final_vmc_energy']:.8f} Ha")
    print(f"FCI能量: {result['E_fci']:.8f} Ha")
    print(f"Hartree-Fock能量: {result['E_hf']:.8f} Ha")
    print(f"与FCI能量误差: {abs(result['final_vmc_energy'] - result['E_fci']):.8f} Ha")
