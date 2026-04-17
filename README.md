# Nerual Quantum Network State(NQS)  

本项目是用于探究神经网络量子态在量子动态模拟中的应用。  

1. [H2分子求基态能量案例1-RBM](./开源框架学习%20Netket/H2H4/H2_VQMC%20案例_RBM.ipynb)
2. [H2分子求基态能量案例2-均值场](./开源框架学习%20Netket/H2H4/H2_VQMC%20案例_MeanFiled.ipynb)
3. [H2分子求基态能量案例3-Jastow](./开源框架学习%20Netket/H2H4/H2_VQMC%20案例_Jastow.ipynb)
4. [H2分子求基态能量案例4-FFNN](./开源框架学习%20Netket/H2H4/H2_VQMC%20案例_FFNN.ipynb)
5. [H4分子求基态能量案例-RBM](./开源框架学习%20Netket/H2H4/H4_VQMC%20案例_RBM.ipynb)
6. [H4分子求基态能量案例-FFNN](./开源框架学习%20Netket/H2H4/H4_VQMC%20案例_FFNN.ipynb)  

7. [H2分子-FFNN 分析](./开源框架学习%20Netket/H2H4/H2_FFNN波函数Ansatz分析.md)
8. [H2分子-FFNN相位+振幅](./开源框架学习%20Netket/H2H4/H2_VQMC%20案例_FFNN_幅度相位.ipynb)


## QNN 作为 ansatz 的一系列尝试:
1. [QNN的定义](./pennylane/QNN/QNN_ansatz完整定义.ipynb)  
2. [QNN作为ansatz的尝试](./pennylane/QNN/QNN_ansatz%E5%AE%8C%E6%95%B4%E5%AE%9A%E4%B9%89-2.ipynb)
3. [QNN作为ansatz的成功尝试-H2](./pennylane/QNN/QNN_ansatz完整定义-H2正式版.ipynb)
4. []()

## Fermi-Hubbard Model 案例
1.[Fermi-Hubbard model 1D](./开源框架学习%20Netket/FermiHubbard1D/前馈神经网络_FH.ipynb)


## JAX NNX相关教程总结：
1. [认识 Pytree](./FLAX%20学习/认识pytree.md)

## 从头拆解 MCMC 算法
1. [MCMC算法案例](./蒙特卡洛算法学习/Metropolis_Hastings_Algorithm.ipynb)


## 激发态能量计算



## 目前的进展：
1. 对于小分子$H_2$的模拟显示, 只有RBM模型可以得到正确的基态结果，可以逼近 FCI,其余模型只能逼近 Hartree-Fock 结果，说明根本没有找到关联性。需要调查出来 RBM 的不同。 
- 要把 param_type 设为 complex, 才能得到正确的结果。  

2. 对于 H2 分子，FFNN 模型可以得到正确的基态结果，说明 FFNN 模型是可以用来模拟 H2 分子的。
3. QNN 作为 ansatz 时，完全成功!


## 统计参数

```python
import jax.tree_util as jtu

# 统计参数量
param_leaves = jtu.tree_leaves(nnx.state(ffnn_model, nnx.Param))
total_params = sum(leaf.size for leaf in param_leaves)

print(f"模型参数量: {total_params:,}")
```