# NES-VMC 的能量重建问题
NES-VMC 在训练完成后(达到化学精度或收敛条件),此时总波函数$Psi(X)$内的所有变分参数可以视为训练完成，进行保存。然而总波函数可以视为是$\Psi(X) = det(M)$,其中M是KxK 的矩阵，每一列由$\psi_i(x)$构成.且$\psi_i(x)$的变分参数也完全知晓。现在我想要基于某一列$\psi_i(x)$来重建其对应的能级能量。
这可能对应的是一个广义特征值的问题。

请你安排证明实验:
基于 $H_2$分子的 NES-VMC 实验。
```python
from H2_631G import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock,hi,E_fcis
>>>
============================================================
H2 分子基本信息
============================================================
HF energy = -0.99749729 Ha
Total electrons = (1, 1)
Total basis functions = 4
============================================================
H₂ FCI 基准能量
============================================================
E0 = -1.05434745 Ha  |  激发能: 0.0000 eV
E1 = -0.95790573 Ha  |  激发能: 2.6243 eV
E2 = -0.66895227 Ha  |  激发能: 10.4871 eV
E3 = -0.55140192 Ha  |  激发能: 13.6859 eV

HF reference state: [0 0 0 1 0 0 0 1]
single_edges: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]

total_ansatz = NESTotalAnsatz(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
) # 总波函数拟设
single_ansatz = SingleStateAnsatz(
    n_spin_orbitals=SINGLE_SIZE,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
) # 单态波函数拟设
```
当训练好之后,载入训练好的数据:

```python
HISTORY_FILE = r'./data/26-09-08-18-09_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)

history['Energy_levels'][200]
>>Array([-1.0463837 +0.00378942j, -0.9575794 +0.00010092j,
       -0.66382189+0.00164811j, -0.54445152+0.0001313j ],      dtype=complex128)
```
会发现其实与精确解已经很接近了 我现在需要你使用训练好的参数开重建能级能量。
```python
finnal_params = history['params'][-1]
single_graphdef,single_params = nnx.split(total_ansatz.single_ansatz_list[0])
fine_single_ansatz = nnx.merge(single_graphdef,finnal_params['single_ansatz_list'][0])
```