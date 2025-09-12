Pennylane 的 Hardware Efficient Ansatz 实现：

`python
import pennylane as qml
from math import pi

n_wires = 3
dev = qml.device('default.qubit', wires=n_wires)

@qml.qnode(dev)
def circuit(weights):
    qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
    return [qml.expval(qml.Z(i)) for i in range(n_wires)]
`
