# NES-VMC Algorithm Reproduction Using NetKet Official API: Objectives and Progress

## 1. Research Objectives

**Goal**: Implement the **NES-VMC (Natural Excited State Variational Monte Carlo) algorithm** using NetKet framework and Flax.nnx APIs to compute the first $K$ excited state energies of quantum many-body systems (such as H₂ molecule).

**Requirements**:

- Use NetKet's built-in extended Hilbert space `hi ** K`
- After training, diagonalize the averaged local energy matrix to obtain ground state and excited state energies

## 2. Core Concepts of NES-VMC Algorithm

### 2.1 Problem Background

In quantum mechanics, we typically need to solve the eigenvalue problem of Hamiltonian operator $\hat{H}$, finding the lowest $K$ eigenfunctions. For quantum many-body systems, direct diagonalization of the Hamiltonian matrix is usually infeasible because the Hilbert space dimension grows exponentially with particle number.

NES-VMC **equivalently transforms** the problem of finding the first $K$ excited states of the original system into a **ground state problem of an "extended system"**.

Below is the description for solving the first K excited states of the H₂ molecule. Do not modify unless necessary.

```python
"""
NES-VMC (Natural Excited State Variational Monte Carlo) Algorithm Implementation

This file implements the NES-VMC algorithm based on native JAX and partial NetKet,
for computing excited state energies of quantum many-body systems.
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
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import jax
import time
from functools import partial

# ==============================================================================
# 1. Global Parameters & H₂ Molecule Definition
# ==============================================================================
# ===================== H₂ Molecule Definition & FCI Benchmark =====================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI Exact Benchmark
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI Benchmark Energies")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  Excitation energy: {exc:.4f} eV")
# ===================== NetKet Hamiltonian and Sampler =====================
ha = nkx.operator.from_pyscf_molecule(mol)

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K=2
hi_ext = hi**K
edges = [(0, 1), (2, 3),(4, 5),(6,7)]

```

### 2.2 Extended Hilbert Space

Let $\mathbf{X} = (x_1, \dots, x_N)$ represent a set of $N$ particles (particle set), where $x_i$ denotes the state of the $i$-th particle. The extended Hilbert space consists of $K$ copies of the original system via tensor product, where each configuration corresponds to $K$ configurations $\mathbf{x} = (x^1, \dots, x^K)$.


### 2.4 SingleStateAnsatz Structure

$\psi(\mathbf{x})$ corresponds to the Ansatz in ordinary VMC algorithm. Note that in this case, $\mathbf{x}$ corresponds to the 4 legal configurations of the H₂ molecule under particle number conservation, spin conservation, and STO-3G, with spin order $[ \alpha_1, \alpha_2,\beta_1,\beta_2]$. The 4 legal configurations are $[1,0,1,0],[0,1,0,1],[1,0,1,1],[1,0,0,1]$.

The code for SingleStateAnsatz is:

```python
class SingleStateAnsatz(nnx.Module):
    """Single-state Ansatz: Complex-valued FFNN for fermionic systems"""

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
def create_single_machine(model: SingleStateAnsatz):
    """Wrap Flax NNX model as NetKet-style machine function"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state
```

Note that in this case, the model parameters are complex-valued by default, and the output is $\ln{\psi(x)}$.

### 2.3 TotalAnsatz Structure

Let $\psi_i$ denote the $i$-th $N$-particle wave function (possibly unnormalized). Then the **TotalAnsatz** is defined as the determinant of matrix $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$:

$$
\Psi(\mathbf{x}) \equiv \det\begin{pmatrix}
\psi_1(x^1) & \psi_2(x^1) & \cdots & \psi_K(x^1) \\
\psi_1(x^2) & \psi_2(x^2) & \cdots & \psi_K(x^2) \\
\vdots & \vdots & \ddots & \vdots \\
\psi_1(x^K) & \psi_2(x^K) & \cdots & \psi_K(x^K)
\end{pmatrix}

$$

Where:

- $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$: Matrix combining all electron sets with all wave functions
- $\psi_i(x^j)$: Value of the $i$-th single-state Ansatz on the $j$-th particle set
- $\Psi(\mathbf{x}) = \det(\Psi(\mathbf{x}))$: Total Ansatz, can be viewed as an unnormalized Slater determinant composed of $N$-particle wave functions

**Key Property**: By expressing the total Ansatz as a determinant of single-state Ansatze, different Ansatze can be prevented from collapsing to the same state without explicitly requiring them to be orthogonal.

```python
class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            
            ansatz = SingleStateAnsatz(
                n_spin_orbitals, 
                hidden_dim, 
                rngs=sub_rngs
            )
            self.single_ansatz_list.append(ansatz)
    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            # Shape: [K, n_spin]
            #print(f'x_single.shape: {x_single.shape}')
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            L_stable = L - L.max()
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi_stable, L_stable , L.max()
        
        # Safe batch processing
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            # Direct processing of single sample
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin*self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            # Direct processing of batch samples
            return jax.vmap(_forward_single)(x)
        
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim ==1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f'Unsupported input shape: {x.shape}')
            
            
    
def create_machine(model: NESTotalAnsatz):
    """Wrap Flax NNX model as NetKet-style machine function"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix,_ = m(sigma)
        return log_psi_total

    return machine, graphdef, state

def create_machine_matrix(model: NESTotalAnsatz):
    """Wrap Flax NNX model as NetKet-style machine function"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix,_ = m(sigma)
        return log_M_matrix

    return machine, graphdef, state

def create_machine_max(model: NESTotalAnsatz):
    """Wrap Flax NNX model as NetKet-style machine function"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix,L_max = m(sigma)
        return L_max

    return machine, graphdef, state

```

The output of NESTotalAnsatz is:

$$
\ln{\Psi(\mathbf{X})} = \ln{\det{\mathbf{M} \triangleq \ln{\det{     \begin{pmatrix}
\psi_1(\mathbf{x}^1) & \dots & \psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\psi_1(\mathbf{x}^K) & \dots & \psi_K(\mathbf{x}^K)
\end{pmatrix}}}}}
$$

To enable stable computation, according to Section S8 of the paper, log-domain transformation should be performed.


### 2.4 Extended Hamiltonian

Define the extended Hamiltonian $\tilde{H} = \hat{H}_1 \oplus \hat{H}_2 \oplus \cdots \oplus \hat{H}_K$, where $\hat{H}_i$ is the Hamiltonian acting only on the $i$-th particle set. The ground state energy of $\tilde{H}$ equals the sum of the lowest $K$ energies of the original system $\hat{H}$, and its ground state wave function is precisely the determinant form $\Psi^\star$ described above.

Since NetKet does not seem to support Hamiltonian direct sum in this form, we use an indirect approach:

```python
hi.all_states() 
>>Array([[0, 1, 0, 1],
       [0, 1, 1, 0],
       [1, 0, 0, 1],
       [1, 0, 1, 0]], dtype=int8)

K=2
hi_ext = hi**K
hi_ext.all_states()
>>Array([[0, 1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 0, 1, 1, 0],
       [0, 1, 0, 1, 1, 0, 0, 1],
       [0, 1, 0, 1, 1, 0, 1, 0],
       [0, 1, 1, 0, 0, 1, 0, 1],
       [0, 1, 1, 0, 0, 1, 1, 0],
       [0, 1, 1, 0, 1, 0, 0, 1],
       [0, 1, 1, 0, 1, 0, 1, 0],
       [1, 0, 0, 1, 0, 1, 0, 1],
       [1, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 0, 1, 1, 0, 0, 1],
       [1, 0, 0, 1, 1, 0, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1],
       [1, 0, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 1, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 1, 0]], dtype=int8)
```

Later, we will mention that although the loss function is defined as: $\Psi(\mathbf{x})^{-1}\hat{\mathcal{H}}\Psi(\mathbf{x})$, where $\mathcal{H}$ refers to the extended Hamiltonian $\tilde{H}$.

This can be equivalently viewed as:

$$ \begin{align*}
\Psi(\mathbf{x})^{-1}\hat{\mathcal{H}}\Psi(\mathbf{x})
&= \mathrm{Tr}\left[ \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x}) \right]
\end{align*} $$

Where $\hat{H}\Psi(\mathbf{x})$ is given by the `Ham_psi` and `Ham_Psi` functions below.

### 2.5 Ham_psi and Ham_Psi Functions

The NES-VMC loss function (corresponding to Eq.29 in the original paper):

TotalAnsatz value: $\Psi(\mathbf{X})$ or $\ln{\Psi(\mathbf{x})}$

The difference between X and x is as explained above.

SingleStateAnsatz output: $\ln(\psi(\mathbf{x}))$

$$
\hat{H}\Psi(\mathbf{x}) \triangleq 
\begin{pmatrix}
\hat{H}\psi_1(\mathbf{x}^1) & \dots & \hat{H}\psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\hat{H}\psi_1(\mathbf{x}^K) & \dots & \hat{H}\psi_K(\mathbf{x}^K)
\end{pmatrix}
$$

```python
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    🔥 Supports both:
    - Single state x: (n_spin,)
    - Batch states x: (batch_size, n_spin)
    """
    # ======================
    # Core: Automatically add batch dimension for single sample
    # ======================
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    # ======================
    # Vectorized computation (batch processing)
    # ======================
    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    # Batch processing
    H_psi_batch = jax.vmap(_single_hpsi)(x)

    # ======================
    # If single input, squeeze back to single output
    # ======================
    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
    
def Ham_Psi(ha, single_machine_list, total_params, x):
    K = len(single_machine_list)
    # ======================
    # Core: Auto compatibility for single sample and batch processing
    # ======================
    if x.ndim == 2:
        # Input shape: (K, n_spin) → single extended state → return (K,K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]  # single state: (4,)
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        return _single_HamPsi(x)

    elif x.ndim == 3:
        # Input shape: (batch, K, n_spin) → batch → return (batch, K, K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        # Auto batch processing!
        return jax.vmap(_single_HamPsi)(x)

    else:
        raise ValueError(f'Unsupported input shape: {x.shape}')
```

Where `Ham_psi` is used to compute $\hat{H}\psi_1(\mathbf{x}^1)$

`Ham_Psi` is used to compute $\hat{H}\Psi(\mathbf{x})$

### 2.6 Sampler Configuration

In the NES-VMC algorithm, the sampler operates on the **extended Hilbert space** $\mathbf{x} = (x^1, x^2, \ldots, x^K)$, sampling $K$ configurations at once, where each $x^k$ belongs to the Hilbert space $\hat{H}$ of the original system.

I constructed the NES-VMC sampler based on NetKet's Sampler + custom Rule.

```python
# Single system Hilbert space
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2  # NES expansion copies
hi_ext = hi ** K  # Extended Hilbert space
SINGLE_SIZE = hi.size  # Single subsystem dimension = 4
single_edges = ((0, 1), (2, 3))  # Fermion transition edges
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)

total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)

N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =50
SINGLE_SIZE = hi.size  # Single subsystem dimension = 4

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # Convert to jax array (key fix)
print(ext_edges)

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20
)

sampler_state = nes_sampler.init_state(total_machine, total_params, seed=1)
samples_raw, sampler_state = nes_sampler.sample(
    total_machine, total_params, state=sampler_state, chain_length=40
)
samples_raw.shape

```

#### Core Constraint: No Duplicate Configurations

**Key Constraint**: Extended states must satisfy $x^i \neq x^j$ (when $i \neq j$). This is because the determinant structure of the total Ansatz requires that configurations of each copy must be distinct; otherwise, the matrix $\Psi(\mathbf{x})$ will have identical rows/columns, leading to a zero determinant.

For $K=2$, the number of valid configurations for extended states is $N_s^2 - N_s = 4^2 - 4 = 12$ (where $N_s=4$ is the Hilbert space dimension of the H₂ molecule single system), rather than simply $4^2 = 16$.

#### Approach 1: NetKet Built-in Sampler

The code uses NetKet's `TensorRule` to build the sampler for the extended Hilbert space:

```python
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)
```

Where:
- `FermionHopRule`: Performs fermion transitions on the single-system Hilbert space (satisfying particle number conservation)
- `TensorRule`: Replicates the single-system sampling rule $K$ times, applying it to the extended Hilbert space

Full code:

```python
@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    # [Only keep edges: JAX only allows jax arrays, completely remove hi_ext!]
    edges: jnp.ndarray

    def _check_duplicate(self, sigma_ext):
        """NES constraint: Sub-configurations must not repeat (use global K and SINGLE_SIZE, completely safe)"""
        sub = sigma_ext.reshape((*sigma_ext.shape[:-1], K, SINGLE_SIZE))
        return jnp.any(jnp.all(sub[...,1:,:] == sub[...,0:1,:], axis=-1), axis=-1)

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """Transition rule (no modification)"""
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:,0], sel_e[:,1]

        sigma_cand = sigma.at[jnp.arange(batch_size),i].set(sigma[jnp.arange(batch_size),j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size),j].set(sigma[jnp.arange(batch_size),i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        """[Core fix] Use sampler.hilbert instead of custom hi_ext (NetKet standard写法, never fails)"""
        sigma_shape = state.σ.shape
        # Directly get Hilbert space from sampler (official standard usage, 100% JAX compatible)
        hilbert = sampler.hilbert

        def gen_single(key):
            max_tries = 100  # Prevent infinite loop
            def cond(c): 
                return (c[0] < max_tries) & c[2]
            
            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)  # Update RNG each time, prevent infinite loop
                s = hilbert.random_state(k_new)
                is_dup = self._check_duplicate(s)
                return (tries + 1, k, is_dup, s)
            
            init_c = (0, key, True, hilbert.random_state(key))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            return jax.lax.cond(is_dup, lambda: hilbert.random_state(key), lambda: s)
        
        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)
```


## 3. Loss Function

### 3.1 Objective Function (Rayleigh Quotient)

The NES-VMC objective function is the Rayleigh quotient of the extended Hamiltonian with respect to the total Ansatz:

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\langle\Psi|\Psi\rangle}

$$

Using the matrix determinant lemma, it can be rewritten in trace form:

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\det(S)} = \mathrm{Tr}\left(S^{-1}\hat{H}\right) = \mathrm{Tr}\left(\Psi^{-1}\tilde{H}\Psi\right)

$$

Where $\Psi^{-1}H\Psi$ is computed using the following function:

```python
def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # Add regularization term to prevent matrix singularity
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. Batch local energy matrix
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    #print(f'E_L_batch.shape={E_L_batch.shape}')
    
    E_L_centered = E_L_batch - E_L_mean
    
    tr_centered =  jnp.trace(E_L_centered, axis1=-2, axis2=-1) 

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. Compute ∇logΨs
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 5. Core weighted average
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean

```

### 3.2 Local Energy Matrix

Through Monte Carlo sampling, the loss function can be written in expectation form:

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})\right)\right]

$$

Define the **local energy matrix** as:

$$
E\_L(\mathbf{x}) \equiv \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})

$$

This is a $K \times K$ matrix, whose trace is the scalar local energy. When $K = 1$, this reduces to the local energy in standard VMC.

## 4. Gradient Formula

### 4.1 Standard VMC Gradient Review

For ground state VMC, the gradient of energy with respect to variational parameters $\theta$ is:

$$
\nabla_\theta \frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} = 2\mathbb{E}_{x \sim \psi^2}\left[\left(E_L(x) - \mathbb{E}_{x' \sim \psi^2}[E_L(x')]\right)\nabla_\theta \log|\psi(x)|\right]

$$

### 4.2 NES-VMC Gradient

For the total Ansatz, the gradient computation is similar. The loss function is in trace form $\mathcal{L} = \mathrm{Tr}(E_L(\mathbf{x}))$, define the log amplitude:

$$
\log|\Psi(\mathbf{x})| = \log\det(\Psi(\mathbf{x})) = \mathrm{Tr}\left(\log(\Psi(\mathbf{x}))\right)

$$

The gradient formula is:

$$
\nabla_\theta \mathcal{L} = 2\mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\left(E_L(\mathbf{x}) - \bar{E}_L\right)\nabla_\theta \log\Psi(\mathbf{x})\right)\right]

$$

Where:
- $E_L(\mathbf{x}) = \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})$ is the local energy matrix
- $\bar{E}_L = \mathbb{E}_{\mathbf{x}' \sim \Psi^2}[E_L(\mathbf{x}')]$ is the expectation of the local energy matrix
- $\nabla_\theta \log\Psi(\mathbf{x})$ is the log gradient of the wave function matrix with respect to parameters $\theta$

When $K = 1$, the above reduces to the standard VMC gradient formula.

```python
def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # Add regularization term to prevent matrix singularity
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. Batch local energy matrix
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    #print(f'E_L_batch.shape={E_L_batch.shape}')
    
    tr_batch = loss_batch
    tr_mean = tr_batch.mean()
    tr_centered = tr_batch - tr_mean  # ✅ Correct weights

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. Compute ∇logΨs
    dlogPsi_batch = vmap_grad_logPsia(total_params, x_batch)

    # 5. Core weighted average
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean
```

### 4.3 Gradient Estimation with Batch Walkers

Similar to standard VMC, unbiased gradient estimates can be obtained using independent walkers in the same batch:

$$
\nabla_\theta \mathcal{L} = \frac{N-1}{2N}\mathbb{E}_{x\_1,\dots,x\_N}\left[\frac{1}{N}\sum_{i=1}^N\left(E_L(x_i) - \frac{1}{N}\sum\_{j=1}^N E_L(x_j)\right)\nabla_\theta \log|\Psi(x_i)|\right]

$$

## 5. Excited State Energy Extraction

### 5.1 Diagonalization of Energy Matrix

After training, accumulate local energy matrices through extensive sampling:

$$
\bar{E}_L = \mathbb{E}_{\mathbf{x} \sim \Psi^2}[E_L(\mathbf{x})]

$$

Then diagonalize $\bar{E}\_L$:

$$
\bar{E}_L = U\Lambda U^{-1}

$$

Where $\Lambda = \mathrm{diag}(E\_1, E\_2, \dots, E\_K)$ contains eigenvalues sorted by energy.

### 5.2 Physical Interpretation

When the single-state Ansatz is a linear combination of eigenfunctions $\psi\_i = \sum\_j a\_{ij}\psi\_j^\star$:

$$

\Psi^{-1}\hat{H}\Psi = A^{-1}\Lambda A

$$

Where $A$ is the coefficient matrix. Therefore, diagonalization directly yields the energies of each excited state $E\_1, E\_2, \dots, E\_K$.

The following code should not be modified. Note that edges = [α1,α2,β1,β2] is the order.


## 6. Test Case

```python

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx
import sys
sys.path.append('..')
from NES_VMC import NESTotalAnsatz, create_machine,init_sampler_state,\
    generate_random_initial_states,ha,SingleStateAnsatz,create_single_machine,\
        create_machine_matrix,Ham_psi,Ham_Psi,NES_loss_energy,nes_vmc_gradient,hi,E_fcis,mcmc_sampler_multichain,\
            compute_qgt
import optax
from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
import time
from collections import Counter
import numpy as np
K=2
hi_ext = hi**K

N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =200
SINGLE_SIZE = hi.size  # Single subsystem dimension = 4


total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
    

optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # Convert to jax array (key fix)

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20
)


# Sampler state initialization (replaces original init_sampler_state)
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ==================== Training Loop (only sampler part replaced) ====================
print("\n" + "="*60)
print("Starting Multi-chain NES-VMC Training (NetKet Custom Sampler + Vanilla Gradient Descent)")
print("="*60)
print(f"Ground state energy={E_fcis[0]:.8f} Ha| First excited state energy={E_fcis[1]:.8f} Ha| Second excited state energy={E_fcis[2]:.8f} Ha")

history = {
    'step': [],
    'energy_0st': [],
    'energy_1st': [],
    'energy_std': [],
    'loss': [],
    'params': [],
    'E_Lmatrix':[],
    'natural_grad':[],
    'grad_flat':[],
    'samples':[],
    'log_Psi':[],
    'log_M':[],
    'log_Psi_mean':[],
    'log_Psi_min':[],
    'log_Psi_max':[],
    'grad_norm':[],
}

start_time = time.time()
for step in range(N_ITER):
    # 2. Formal sampling
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine, parameters=total_params, 
        state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN
    )
        # 3. Reshape dimensions, adapt to gradient function input
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, 4)
    # 3. Compute energy and natural gradient (logic consistent with original code)
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))
    #grad = jax.tree_util.tree_map(lambda x: x * 2, grad)
    
    grad_flat , grad_unravel_fn = ravel_pytree(grad)
    # qgt_reg, unravel_fn = compute_qgt(total_machine, total_params, samples.reshape(-1,2,4), diag_shift=0.1)
    
    # # # Natural gradient solution
    # natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    # natural_grad = grad_unravel_fn(natural_grad_flat)
    # grad = natural_grad
        
    # 4. Update parameters
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
    
    
    log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
    eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
    grad_norm = jnp.linalg.norm(grad_flat)
    
    
    history['step'].append(step)
    history['E_Lmatrix'].append(E_L_mean)
    history['samples'].append(samples)
    history['loss'].append(loss_mean)
    history['log_Psi_mean'].append(log_Psi_batch.mean())
    history['log_Psi_min'].append(log_Psi_batch.min())
    history['log_Psi_max'].append(log_Psi_batch.max())
    history['grad_norm'].append(grad_norm)
    history['energy_0st'].append(eig_vals[0])
    history['energy_1st'].append(eig_vals[1])
    history['params'].append(total_params)
    # 5. Record history
    if step % 50 == 0 or step == N_ITER - 1:
        # --------------------- [NES-VMC Monitoring Template] Use directly ---------------------
        # 1. Monitor log_Psi
        #log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
        print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")

        # 2. Monitor gradient norm
        
        print(f"grad norm = {grad_norm:.4f}")

        # 5. Local energy matrix
        #print(f"E_L mean =\n{E_L_mean}")
    
        #eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
        # #history['natural_grad'].append(natural_grad)
        # history['grad_flat'].append(grad_flat)
        # history['log_Psi'].append(log_Psi)
        # history['log_M'].append(log_M)
        
        print(f"Step {step:3d} | Loss: {loss_mean}|0st energy={eig_vals[0]:.8f} Ha| 1st energy={eig_vals[1]:.8f} Ha")
        # print(f'grad={grad_flat[30:31]}')
        print('#-----------------------------------------#')


end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")
# Final results
print("\n" + "="*60)
print(f"Training complete!")
# print(f"Final energy: {final_energy.real:.8f} ± {final_std:.6f} Ha")
# print(f"FCI benchmark: {E_fcis[0]:.8f} Ha")
# print(f"Absolute error: {final_error:.6f} Ha")
# print(f"Relative error: {final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print("="*60)

```

Output:

```python

============================================================
Starting Multi-chain NES-VMC Training (NetKet Custom Sampler + Vanilla Gradient Descent)
============================================================
Ground state energy=-1.01546825 Ha| First excited state energy=-0.87542794 Ha| Second excited state energy=-0.42938376 Ha
log_Psi: mean=1.338+0.070j | min=0.024-3.083j | max=1.513+0.075j
grad norm = 0.8250
Step   0 | Loss: -1.2576295690685688|0st energy=-0.99557002 Ha| 1st energy=-0.26205955 Ha
#-----------------------------------------#
log_Psi: mean=11.451-0.414j | min=9.999-2.433j | max=11.630+1.492j
grad norm = 0.5640
Step  50 | Loss: -1.547645795833384|0st energy=-56.85370946 Ha| 1st energy=55.30606367 Ha
#-----------------------------------------#
log_Psi: mean=16.215-0.251j | min=16.215-1.625j | max=16.215+1.516j
grad norm = 0.0000
Step 100 | Loss: -1.5937140959638543|0st energy=-0.94145467 Ha| 1st energy=-0.65225943 Ha
#-----------------------------------------#
log_Psi: mean=16.215-0.251j | min=16.215-1.625j | max=16.215+1.516j
grad norm = 0.0000
Step 150 | Loss: -1.5937140959638543|0st energy=-0.94145467 Ha| 1st energy=-0.65225943 Ha
#-----------------------------------------#
log_Psi: mean=10.994+0.302j | min=9.058-2.065j | max=11.009+2.080j
grad norm = 0.1187
Step 199 | Loss: -1.5891390172196438|0st energy=-1.60155252 Ha| 1st energy=0.01241350 Ha
#-----------------------------------------#
...

============================================================
Training complete!
============================================================
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

```
