import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx
from tqdm import tqdm
from functools import partial
from jax import flatten_util
import jax
import jax.numpy as jnp
from functools import partial



# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

ha = nkx.operator.from_pyscf_molecule(mol)
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

# ==============================================================================
# 2. 神经网络 Ansatz 
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)
    

def make_get_all_next_states(edges):
    # 把 edges 变成 静态 Python 元组！核心！
    edges = tuple(tuple(e) for e in edges)

    @jax.jit
    def get_all_next_states_jit(S: jnp.ndarray):
        s_arr = S
        next_states = []
        masks = []

        # 现在 edges 是 Python 静态元组 → 完全安全！
        for (i, j) in edges:
            occ_i = s_arr[i]
            occ_j = s_arr[j]
            valid = (occ_i == 1) & (occ_j == 0) | (occ_i == 0) & (occ_j == 1)
            new_state = s_arr.at[i].set(occ_j).at[j].set(occ_i)
            next_states.append(new_state)
            masks.append(valid)

        return jnp.stack(next_states), jnp.stack(masks)
    
    return get_all_next_states_jit

def make_metropolis_hastings_step(edges, log_pdf):
    get_all_next = make_get_all_next_states(edges)
    
    @jax.jit
    def metropolis_hastings_step_jit(S: jnp.ndarray, key: jax.Array):
        # 1. 候选
        candidates, valid_mask = get_all_next(S)
        key, subk = jax.random.split(key)
        idx = jax.random.choice(subk, candidates.shape[0])
        S_cand = candidates[idx]
        is_valid = valid_mask[idx]

        # ==============================
        # ✅ 严格正确：log_pdf = ln(ψ)
        # ==============================
        log_accept_ratio = 2 * jnp.real(log_pdf(S_cand) -log_pdf(S))

        # 接受条件（无exp，超快）
        key, subk = jax.random.split(key)
        u = jax.random.uniform(subk)
        accept = is_valid & (log_accept_ratio > jnp.log(u))

        S_new = jnp.where(accept, S_cand, S)
        return S_new, accept, key

    return metropolis_hastings_step_jit

# 🔥 正确静态参数
@partial(jax.jit, static_argnums=(0, 1, 3, 4,5))
def mcmc_sampler(
    n_samples: int,
    n_warmup: int,
    initial_state: jnp.ndarray,
    edges: tuple[tuple[int, int]],
    log_pdf: callable,
    seed: int = 42
):
    key = jax.random.PRNGKey(seed)
    mh_step = make_metropolis_hastings_step(edges, log_pdf)

    # 预烧
    def warmup_loop(carry, _):
        state, rng = carry
        state, _, rng = mh_step(state, rng)
        return (state, rng), None

    (current_state, key), _ = jax.lax.scan(
        warmup_loop,
        (initial_state, key),
        xs=None,
        length=n_warmup
    )

    # 采样
    def sample_loop(carry, _):
        state, rng = carry
        state, accepted, rng = mh_step(state, rng)
        return (state, rng), state

    (_, _), samples = jax.lax.scan(
        sample_loop,
        (current_state, key),
        xs=None,
        length=n_samples
    )

    return samples