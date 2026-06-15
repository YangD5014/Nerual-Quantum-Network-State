# NES-VMC Logarithmic Domain Transformation Code Interpretation

## 1. Research Background

In the NES-VMC (Natural Excited State Variational Monte Carlo) algorithm, when neural network training progresses to later stages, the wavefunction matrix $\Psi(\mathbf{x})$ elements may become very large or very small. This leads to **numerical overflow** issues when computing the loss function.

This document provides a comparative analysis of the core differences between the **before** (original version) and **after** (logarithmic domain transformation version).

---

## 2. Core Modification Comparison

### 2.1 NESTotalAnsatz Model Output

#### Before Modification (Original Version)

```python
# File: NES_new_vserison/NES_VMC.py
# Lines: 91-104

def _forward_single(x_single):
    x_single = x_single.reshape(self.K, self.n_spin)
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_single[i])
            )
    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
    log_Psi = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi, L  # Returns L directly, not stabilized
```

#### After Modification (Logarithmic Domain Version)

```python
# File: Logarithmic Domain NES/NES_VMC.py
# Lines: 91-104

def _forward_single(x_single):
    x_single = x_single.reshape(self.K, self.n_spin)
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_single[i])
            )
    L_stable = L - L.max()  # Core modification: subtract row maximum
    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
    log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi_stable, L_stable, L.max()  # Returns stabilized L_stable and L.max()
```

#### Key Differences

| Item | Before | After |
|------|--------|-------|
| `L` matrix handling | Not stabilized | Stabilized: `L_stable = L - L.max()` |
| Return values | `(log_Psi, L)` | `(log_Psi_stable, L_stable, L.max())` |
| Numerical stability | Risk of overflow | Numerically stable |

---

### 2.2 Loss Function Computation

#### Before Modification (Original Version)

```python
# File: NES_new_vserison/NES_VMC.py
# Lines: 248-255

def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)  # Get original L matrix
    Psi_Matrix = jnp.exp(log_M)  # Direct exponentiation, may overflow
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### After Modification (Logarithmic Domain Version)

```python
# File: Logarithmic Domain NES/NES_VMC.py
# Lines: 85-97

def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                          single_machine_list, total_params, x):
    L_stable = total_matrix_machine(total_params, x)  # Get stabilized L_stable
    Psi_Matrix_stable = jnp.exp(L_stable)  # Stabilized exponentiation

    M = jnp.log(Ham_Psi(ha, single_machine_list, total_params, x))  # Take log of H_psi
    M_stable = M - total_max_machine(total_params, x).reshape(-1,1,1)  # Stabilization
    HPsi_stable = jnp.exp(M_stable)  # Stabilized exponentiation

    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### Key Differences

| Step | Before | After |
|------|--------|-------|
| `Psi_Matrix` | Direct `exp(L)` | Stabilized `exp(L_stable)` |
| `H_psi_x` | Use raw values directly | Take `log(H_psi)` first, then subtract `L.max()` |
| `HPsi` | Direct `H_psi_x` | Stabilized `exp(M_stable)` |

---

### 2.3 New `create_machine_max` Function

#### Before Modification

Only two functions existed:
- `create_machine`: returns `log_psi_total`
- `create_machine_matrix`: returns `log_M_matrix`

#### After Modification

New `create_machine_max` function added:

```python
# File: Logarithmic Domain NES/NES_VMC.py

def create_machine_max(model: NESTotalAnsatz):
    """Wraps Flax NNX model into NetKet-style machine function"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix, L_max = m(sigma)  # Returns L.max()
        return L_max

    return machine, graphdef, state
```

---

## 3. Mathematical Stability Analysis

### 3.1 Numerical Overflow Problem

When `L` matrix elements are large (e.g., late training), `jnp.exp(L)` causes numerical overflow:

```python
# Assuming L elements are 1000
jnp.exp(1000)  # → inf (numerical overflow)

# After modification: L_stable = L - L.max() = L - 1000 ≈ [0, -10, ...]
jnp.exp(L_stable)  # → [1, 4.5e-5, ...] (numerically stable)
```

### 3.2 Physical Consistency of Logarithmic Domain Transformation

The post-modification computation is equivalent to:

$$
\Psi^{-1} \tilde{H} \Psi = \exp\left(L_{\text{stable}}\right)^{-1} \cdot \exp\left(M_{\text{stable}}\right)
$$

Where:
- $L_{\text{stable}} = \log(\Psi) - L_{\max}$
- $M_{\text{stable}} = \log(\tilde{H}\Psi) - L_{\max}$

This is consistent with the numerical stabilization scheme in Paper Section S8.

---

## 4. Reasonableness Analysis

### ✅ Reasonable Aspects

1. **Stability improvement**: Logarithmic domain transformation effectively avoids numerical overflow in `exp()` operations
2. **Mathematical equivalence**: After removing the common factor $e^{L_{\max}}$, the relative relationships in matrix inversion and determinant calculations remain unchanged
3. **Architecture compatibility**: The modification does not change the overall structure and sampler of `NESTotalAnsatz`

### ⚠️ Aspects to Note

1. **New `create_machine_max` introduced**: Requires additional computation and passing of `L.max()`, increasing code complexity
2. **Batch processing consistency**: `L.max()` needs correct reshaping to `(batch, 1, 1)` to match the shape of `M`
3. **Training monitoring**: The value range of log_Psi will change (due to stabilization), requiring attention during monitoring

---

## 5. Summary

| Modification | Content | Purpose |
|-------------|---------|---------|
| `L_stable = L - L.max()` | Matrix stabilization | Prevent exp() overflow |
| `M_stable = M - L.max()` | Hamiltonian matrix stabilization | Ensure numerical consistency with Psi_Matrix |
| New `create_machine_max` | Returns `L.max()` | Provide common factor for stabilization |
| `NES_loss_energy_stable` | New loss function | Perform matrix operations in logarithmic domain |

The modification improves numerical stability, which is particularly important for deep networks or long training scenarios, while maintaining mathematical equivalence with the original algorithm.

---

## 6. Reference Files

- Original version: `/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/NES_new_vserison/NES_VMC.py`
- Modified version: `/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC.py`
- Modification test: `/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC对数域改造 K2.ipynb`
