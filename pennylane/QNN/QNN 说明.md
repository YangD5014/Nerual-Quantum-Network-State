我想要求解 VMC 的多体物理模拟的问题，使用 Netket.
一般来说 Netket 是使用神经网络 Ansatz来进行求解 但是我想尝试使用变分量子线路(PQC)进行求解。
以下是我的基于 PennyLane 的 PQC 定义:
```python
from flax.nnx.bridge import nnx_in_bridge_mdl
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image
from jax import numpy as jnp

def quantum_neural_network(x, n_qubits, params, n_layers):
    """完全JAX兼容的量子神经网络实现"""
    wires = range(n_qubits)
    
    # 数据编码 - 使用JAX友好的操作
    # 是指对输入的构型S 要编译到量子线路中 其实就是HF方法
    for i in range(n_qubits):
        qml.RX(jnp.pi * x[i], wires=i)
    qml.Barrier(wires=wires)
    
    # 变分层
    for layer in range(n_layers):
        # 单量子比特旋转
        for i in range(n_qubits):
            qml.RX(params[layer, 2*i], wires=i)
            qml.RZ(params[layer, 2*i+1], wires=i)
        
        # 纠缠门
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.Barrier(wires=wires)
    
    # 测量所有量子比特的PauliZ期望值
    return [qml.expval(qml.PauliZ(i)) for i in wires]


def initialize_parameters(n_layers, n_qubits):
    """初始化量子神经网络参数
    
    Args:
        n_layers (int): 变分层数量
        n_qubits (int): 量子比特数量
        
    Returns:
        array: 初始化的参数
    """
    # 每个变分层有3*n_qubits个参数（每个量子比特有RX, RY, RZ三个旋转门）
    params_shape = (n_layers, 2 * n_qubits)
    return np.random.uniform(0, 2*np.pi, size=params_shape)

```

我尝试在 Netket 中使用 PQC 进行求解 但是遇到了一些问题。

```python

dev = qml.device("default.qubit", wires=4)
qnode = qml.QNode(func=quantum_neural_network, device=dev,interface='jax')
qnode(x=[1,0,1,0],params=initialize_parameters(2,4),n_layers=2,n_qubits=4)

```
以下是一个模拟的案例:
```python
import os
import netket as nk
from scipy.sparse.linalg import eigsh
from netket.operator.spin import sigmax, sigmaz

N = 6
hi = nk.hilbert.Spin(s=1 / 2, N=N)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
hi.random_state(jax.random.key(0), 3)
Gamma = -1
H = sum([Gamma * sigmax(hi, i) for i in range(N)])
V = -1
H += sum([V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) for i in range(N)])
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
print("eigenvalues with scipy sparse:", eig_vals)
E_gs = eig_vals[0]

# Create the local sampler on the hilbert space
sampler = nk.sampler.MetropolisLocal(hi)

class FFN(nnx.Module):

    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        Construct a Feed-Forward Neural Network with a single hidden layer.

        Args:
            N: The number of input nodes (number of spins in the chain).
            alpha: The density of the hidden layer. The hidden layer will have
                N*alpha nodes.
            rngs: The random number generator seed.
        """
        self.alpha = alpha

        # We define a linear (or dense) layer with `alpha` times the number of input nodes
        # as output nodes.
        # We must pass forward the rngs object to the dense layer.
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):
        print(f'input X={x}')
        # we apply the linear layer to the input
        y = self.linear(x)

        # the non-linearity is a simple ReLu
        y = nnx.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)


model = FFN(N=N, alpha=1, rngs=nnx.Rngs(2))

vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

optimizer = nk.optimizer.Sgd(learning_rate=0.1)

# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

ffn_energy = vstate.expect(H)
error = abs((ffn_energy.mean - eig_vals[0]) / eig_vals[0])
print("Optimized energy and relative error: ", ffn_energy, error)

from matplotlib import pyplot as plt
plt.errorbar(
    log.data["Energy"].iters[50:],
    log.data["Energy"].Mean[50:],
    yerr=log.data["Energy"].Sigma[50:],
    label="SymmModel",
)

plt.axhline(
    y=eig_vals[0],
    xmin=0,
    xmax=log.data["Energy"].iters[-1],
    linewidth=2,
    color="k",
    label="Exact",
)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.legend(frameon=False)

```

你来帮我写一个使用 PQC 来代替案例中的 FNN的版本