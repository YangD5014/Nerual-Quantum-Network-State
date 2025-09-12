# pqc_ansatz_jax.py
from typing import Literal, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray

def _ket_from_bits(bits01: Array) -> Array:
    """
    bits01: [B, N] in {0,1}
    returns |psi0>: [B, 2^N] one-hot computational basis states matching bits01.
    """
    B, N = bits01.shape
    # basis index from bitstring (little-endian by default). We choose big-endian: b0 is MSB
    # idx = sum_{k=0}^{N-1} bits[k] * 2^{N-1-k}
    powers = (2 ** jnp.arange(N-1, -1, -1))[None, :]  # [1,N]
    idx = (bits01 * powers).sum(axis=1)  # [B]
    dim = 1 << N
    # one-hot
    return jax.nn.one_hot(idx, dim, dtype=jnp.complex64)

def _Z_expectation_from_state(psi: Array, N: int) -> Array:
    """
    psi: [B, 2^N] normalized or unnormalized amplitudes
    returns <Z_i>: [B, N]
    """
    B = psi.shape[0]
    probs = jnp.abs(psi) ** 2  # [B, 2^N]
    # For each basis index, compute z-bit values (+1 for 0, -1 for 1) for all qubits
    # Build a table zvals: [2^N, N] with +1/-1
    dim = 1 << N
    idxs = jnp.arange(dim, dtype=jnp.uint32)[:, None]  # [2^N,1]
    # extract bits big-endian
    bit_positions = jnp.arange(N-1, -1, -1, dtype=jnp.uint32)[None, :]  # [1,N]
    bits = ((idxs >> (N-1 - bit_positions)) & 1).astype(jnp.int8)  # [2^N,N]
    zvals = (1 - 2*bits).astype(jnp.float32)  # 0 -> +1, 1 -> -1
    # <Z_i> = sum_k p_k * z_i(k)
    Ez = (probs @ zvals).astype(jnp.float32)  # [B,N]
    return Ez

def _apply_single_qubit_unitary(psi: Array, U: Array, q: int, N: int) -> Array:
    """
    psi: [B, 2^N], U: [2,2] complex, apply on qubit q (0..N-1, big-endian)
    """
    dim = 1 << N
    # reshape to [B, 2^(q), 2, 2^(N-q-1)]
    left = 1 << q
    right = 1 << (N - q - 1)
    psi_reshaped = psi.reshape((-1, left, 2, right))
    # contract U on the '2' axis
    out = jnp.einsum("uv,blvr->blur", U, psi_reshaped) # [B,left,2,right]
    return out.reshape((-1, dim))

def _apply_cnot(psi: Array, control: int, target: int, N: int) -> Array:
    """
    Apply CNOT(control->target) on psi: [B, 2^N] (big-endian qubit numbering)
    """
    # Implement via conditional X on target when control bit == 1
    # Build bit masks for control/target in big-endian indexing
    ctrl_mask = jnp.uint32(1 << (N-1-control))
    targ_mask = jnp.uint32(1 << (N-1-target))

    def flip_if_ctrl(idx):
        # if control bit is 1, flip target bit
        return jnp.where((idx & ctrl_mask) != 0, idx ^ targ_mask, idx)

    dim = psi.shape[1]
    idxs = jnp.arange(dim, dtype=jnp.uint32)
    perm = flip_if_ctrl(idxs)
    # permute amplitudes per batch
    return psi[:, perm]

def _rot_x(theta: Array) -> Array:
    c = jnp.cos(theta/2.0)
    s = -1j*jnp.sin(theta/2.0)
    return jnp.stack([jnp.stack([c, s], -1),
                      jnp.stack([s, c], -1)], -2)  # [2,2]

def _rot_z(theta: Array) -> Array:
    return jnp.array([[jnp.exp(-1j*theta/2.0), 0.0j],
                      [0.0j, jnp.exp(1j*theta/2.0)]], dtype=jnp.complex64)

_H = (1/jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0],
                                    [1.0, -1.0]], dtype=jnp.complex64)

class HardwareEfficientU(nn.Module):
    n_qubits: int
    n_layers: int
    entanglement: Literal["linear","full"] = "linear"

    @nn.compact
    def __call__(self, psi0: Array, thetas_x: Array, thetas_z: Array) -> Array:
        """
        psi0: [B, 2^N] initial amplitudes (|s>)
        thetas_x/z: [L, N] rotation angles for Rx/Rz
        returns psi = U(theta) |s>
        """
        B = psi0.shape[0]
        N = self.n_qubits
        psi = psi0

        # Initial layer of Hadamards on all qubits
        for q in range(N):
            psi = _apply_single_qubit_unitary(psi, _H, q, N)

        # L layers
        for l in range(self.n_layers):
            # single-qubit rotations RZ then RX on all qubits
            for q in range(N):
                psi = _apply_single_qubit_unitary(psi, _rot_z(thetas_z[l, q]), q, N)
            for q in range(N):
                psi = _apply_single_qubit_unitary(psi, _rot_x(thetas_x[l, q]), q, N)

            # entanglers
            if self.entanglement == "linear":
                for q in range(N-1):
                    psi = _apply_cnot(psi, control=q, target=q+1, N=N)
            else:  # full
                for c in range(N):
                    for t in range(N):
                        if c != t:
                            psi = _apply_cnot(psi, control=c, target=t, N=N)

        return psi

class PQCPart(nn.Module):
    """
    One PQC that maps s -> f[s; U(theta)] = sum_i c_i <Z_i>.
    """
    n_qubits: int
    n_layers: int
    entanglement: Literal["linear","full"] = "linear"
    coef_scale: float = 1.0  # optional, can be used like 'a tanh' trick in the paper

    def setup(self):
        N, L = self.n_qubits, self.n_layers
        self.coef = self.param("coef", nn.initializers.normal(stddev=0.02), (N,))
        # small init around 0 as suggested to keep weights tame
        self.thetas_x = self.param("thetas_x", nn.initializers.normal(stddev=1e-3), (L, N))
        self.thetas_z = self.param("thetas_z", nn.initializers.normal(stddev=1e-3), (L, N))
        self.circuit = HardwareEfficientU(self.n_qubits, self.n_layers, self.entanglement)

    def __call__(self, s_bits01: Array) -> Array:
        """
        s_bits01: [B, N] in {0,1} or {-1,+1}
        returns f: [B]
        """
        # Normalize input to {0,1}
        if s_bits01.dtype != jnp.int8 and s_bits01.dtype != jnp.int32:
            s_bits01 = s_bits01.astype(jnp.int32)
        # map {-1,+1} -> {0,1}
        s01 = jnp.where(s_bits01 < 0, 0, s_bits01)
        s01 = jnp.where(s01 > 1, 1, s01)

        psi0 = _ket_from_bits(s01)  # [B, 2^N]
        psi = self.circuit(psi0, self.thetas_x, self.thetas_z)  # [B, 2^N]

        # Optionally normalize (not strictly necessary for expectation of Z)
        # psi = psi / jnp.linalg.norm(psi, axis=1, keepdims=True)

        Ez = _Z_expectation_from_state(psi, self.n_qubits)  # [B,N]
        f = (Ez * (self.coef * self.coef_scale)).sum(axis=1)  # [B]
        return f

class QuantumCircuitAnsatz(nn.Module):
    """
    Full PQC part giving complex log-psi contribution:
    log φ(s) = f_amp(s) + i * f_phase(s)
    """
    n_qubits: int
    n_layers: int
    entanglement: Literal["linear","full"] = "linear"
    coef_scale: float = 1.0

    def setup(self):
        self.amp = PQCPart(self.n_qubits, self.n_layers, self.entanglement, self.coef_scale)
        self.phase = PQCPart(self.n_qubits, self.n_layers, self.entanglement, self.coef_scale)

    def __call__(self, s: Array) -> Array:
        """
        s: [B, N] spin configurations (0/1 or ±1)
        returns complex log-amplitude: [B]
        """
        f1 = self.amp(s)         # real
        f2 = self.phase(s)       # real
        return f1 + 1j * f2
