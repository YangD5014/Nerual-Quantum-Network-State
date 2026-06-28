import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode the classical data
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
    
    # Apply parameterized quantum gates
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    
    # Entangle qubits
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    # Measure expectation value of Z on the first qubit
    return qml.expval(qml.PauliZ(0))


class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits):
        super(HybridModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Classical layers
        self.pre_quantum = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits)
        )
        
        # Quantum circuit weights
        self.quantum_weights = nn.Parameter(torch.randn(n_qubits))
        
        # Post-quantum layer
        self.post_quantum = nn.Linear(1, 2)  # Binary classification
    
    def forward(self, x):
        # Classical pre-processing
        pre_processed = self.pre_quantum(x)
        pre_processed = torch.tanh(pre_processed)  # Scale for quantum input
        
        # Quantum processing
        q_out = torch.zeros(x.shape[0], 1)
        for i in range(x.shape[0]):
            q_out[i] = quantum_circuit(pre_processed[i].detach().numpy(), self.quantum_weights.detach().numpy())
        
        # Classical post-processing
        out = self.post_quantum(q_out)
        return out
