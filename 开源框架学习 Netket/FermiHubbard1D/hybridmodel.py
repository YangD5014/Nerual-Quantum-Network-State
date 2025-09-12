import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import netket.experimental as nkx
from typing import Any
import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, X, RX, RZ, RY, BarrierGate
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz

# ========= MindQuantum 电路（用于 Fermi-Hubbard） ========= #

def initial_state(n_qubit: int, configuration: str) -> Circuit:
    if len(configuration) != n_qubit:
        raise ValueError(f"配置字符串长度({len(configuration)})与量子比特数({n_qubit})不匹配")
    circ = Circuit()
    for i in range(n_qubit):
        if configuration[i] == '1':
            circ += X(i)
    circ += BarrierGate().on(range(n_qubit))
    return circ

def generate_ansatz_pair(n_qubit: int, init_state: Circuit, n_layer: int = 1):
    amp_circ = init_state + UN(H, range(n_qubit))
    amp_circ += HardwareEfficientAnsatz(n_qubits=n_qubit, single_rot_gate_seq=[RX, RZ], depth=n_layer).circuit

    phase_circ = UN(H, range(n_qubit))
    phase_circ += HardwareEfficientAnsatz(n_qubits=n_qubit, single_rot_gate_seq=[RY, RZ], depth=n_layer).circuit

    return amp_circ, phase_circ

# ========= 接口封装：量子部分 ========= #

def _mq_forward_dual(params_amp, params_phase, coeffs_amp, coeffs_phase, x, amp_ansatz, phase_ansatz):
    n_qubit = amp_ansatz.n_qubits
    sim_amp = Simulator("mqvector", n_qubit)
    sim_phase = Simulator("mqvector", n_qubit)

    hams_amp = [Hamiltonian(QubitOperator(f"Z{i}", float(coeffs_amp[i]))) for i in range(n_qubit)]
    hams_phase = [Hamiltonian(QubitOperator(f"Z{i}", float(coeffs_phase[i]))) for i in range(n_qubit)]

    grad_amp = sim_amp.get_expectation_with_grad(hams=hams_amp, circ_right=amp_ansatz)
    grad_phase = sim_phase.get_expectation_with_grad(hams=hams_phase, circ_right=phase_ansatz)

    out = []
    for _ in range(x.shape[0]):
        f_amp, _ = grad_amp(np.array(params_amp))
        f_phase, _ = grad_phase(np.array(params_phase))
        energy_amp = np.sum(f_amp[0])
        energy_phase = np.sum(f_phase[0])
        log_amp = -jnp.abs(energy_amp)
        log_phase = 1j * jnp.real(energy_phase)
        out.append(log_amp + log_phase)
    return jnp.array(out, dtype=jnp.complex64)

def build_quantum_log_amp_dual(amp_ansatz: Circuit, phase_ansatz: Circuit):
    @jax.custom_vjp
    def quantum_log_amp_dual(params_all, x):
        params_amp, params_phase, coeffs_amp, coeffs_phase = params_all
        return _mq_forward_dual(params_amp, params_phase, coeffs_amp, coeffs_phase, x, amp_ansatz, phase_ansatz)

    def fwd(params_all, x):
        params_amp, params_phase, coeffs_amp, coeffs_phase = params_all
        y = _mq_forward_dual(params_amp, params_phase, coeffs_amp, coeffs_phase, x, amp_ansatz, phase_ansatz)
        return y, (params_amp, params_phase, coeffs_amp, coeffs_phase, x)

    def bwd(aux, grad_y):
        params_amp, params_phase, coeffs_amp, coeffs_phase, x = aux
        g_amp = jnp.zeros_like(params_amp)
        g_phase = jnp.zeros_like(params_phase)
        g_coeffs_amp = jnp.zeros_like(coeffs_amp)
        g_coeffs_phase = jnp.zeros_like(coeffs_phase)
        return (g_amp, g_phase, g_coeffs_amp, g_coeffs_phase), jnp.zeros_like(x)

    quantum_log_amp_dual.defvjp(fwd, bwd)
    return quantum_log_amp_dual

# ========= Flax 网络 ========= #

class ClassicalFFNN(nn.Module):
    hidden_dims: list[int]

    @nn.compact
    def __call__(self, x):
        h = x.astype(jnp.float32)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.tanh(h)
        out = nn.Dense(1)(h)
        return jnp.squeeze(out, axis=-1)

class HybridWavefunction(nn.Module):
    hidden_dims: list[int]
    amp_ansatz: Any
    phase_ansatz: Any
    n_params_amp: int
    n_params_phase: int
    n_coeffs: int

    def setup(self):
        self.classical = ClassicalFFNN(self.hidden_dims)
        self.params_amp = self.param("params_amp", nn.initializers.normal(), (self.n_params_amp,))
        self.params_phase = self.param("params_phase", nn.initializers.normal(), (self.n_params_phase,))
        self.coeffs_amp = self.param("coeffs_amp", nn.initializers.normal(), (self.n_coeffs,))
        self.coeffs_phase = self.param("coeffs_phase", nn.initializers.normal(), (self.n_coeffs,))
        self.qlogamp = build_quantum_log_amp_dual(self.amp_ansatz, self.phase_ansatz)

    def __call__(self, x):
        log_cl = self.classical(x)
        log_q = self.qlogamp((self.params_amp, self.params_phase, self.coeffs_amp, self.coeffs_phase), x)
        return log_cl + log_q

# ========= 应用示例：Fermi-Hubbard on 1x4 ========= #

L = 4
D = 1
t = 1.0
U = 0.1

# NetKet graph & Hilbert
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=False)
n_sites = g.n_nodes
hilbert = nk.hilbert.SpinOrbitalFermions(n_sites, s=1/2, n_fermions_per_spin=(2, 2))
ham = nkx.operator.FermiHubbardJax(hilbert, t=t, U=U, graph=g)

# Sampler
sampler = nk.sampler.MetropolisFermionHop(hilbert, graph=g, n_chains=8, sweep_size=32)

# MindQuantum ansatz
conf = '1010'  # example occupation config
init = initial_state(n_qubit=L, configuration=conf)
amp_circ, phase_circ = generate_ansatz_pair(n_qubit=L, init_state=init, n_layer=1)

# Hybrid model
n_params_amp = len(amp_circ.params_name)
n_params_phase = len(phase_circ.params_name)
n_coeffs = L

model = HybridWavefunction(
    hidden_dims=[32],
    amp_ansatz=amp_circ,
    phase_ansatz=phase_circ,
    n_params_amp=n_params_amp,
    n_params_phase=n_params_phase,
    n_coeffs=n_coeffs
)

def init_fn(key, x): return model.init(key, x)
def apply_fn(params, x): return model.apply(params, x)

# Variational state
vstate = nk.vqs.MCState(sampler, apply_fn=apply_fn, init_fn=init_fn, n_samples=1024)

# Optimizer and SR
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.1)

# Driver
driver = nk.driver.VMC(
    hamiltonian=ham,
    optimizer=optimizer,
    variational_state=vstate,
    preconditioner=sr
)

# Training
output_name = "fermi_hubbard_hybrid0717"
driver.run(300, out=output_name)
