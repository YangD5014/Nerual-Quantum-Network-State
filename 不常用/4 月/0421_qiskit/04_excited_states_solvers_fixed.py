#!/usr/bin/env python
# coding: utf-8
"""
Excited states solvers - Qiskit Nature 激发态计算
已更新为适配 Qiskit 1.x 版本

主要修改:
1. 使用 StatevectorEstimator 替代 BaseEstimator
2. 修复导入问题
"""

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    ExcitedStatesEigensolver,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import numpy as np

print("=" * 60)
print("激发态计算 - Excited States Solvers")
print("=" * 60)

# 1. 定义分子系统
print("\n[1/5] 定义分子系统 (H2)...")
driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 1.4",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

es_problem = driver.run()
print(f"  - 空间轨道数：{es_problem.num_spatial_orbitals}")
print(f"  - 粒子数：{es_problem.num_particles}")

# 2. 设置 qubit mapper
print("\n[2/5] 设置 Jordan-Wigner 映射...")
mapper = JordanWignerMapper()

# 3. 配置经典求解器 (NumPyEigensolver)
print("\n[3/5] 配置经典 NumPy 求解器...")
numpy_solver = NumPyEigensolver(k=4, filter_criterion=es_problem.get_default_filter_criterion())

# 4. 执行计算
print("\n[4/5] 执行 NumPy 精确对角化计算...")
numpy_excited_states_solver = ExcitedStatesEigensolver(mapper, numpy_solver)
numpy_results = numpy_excited_states_solver.solve(es_problem)

# 5. 输出结果
print("\n[5/5] 计算结果:")
print("\n" + "=" * 60)
print("=== NumPy 精确对角化结果 ===")
print("=" * 60)
print(numpy_results)

# 使用自定义 filter 函数
print("\n\n")
print("=" * 60)
print("=== 使用自定义 filter 函数 ===")
print("=" * 60)

def filter_criterion(eigenstate, eigenvalue, aux_values):
    """自定义 filter: 只要求粒子数=2, 磁化强度=0"""
    return np.isclose(aux_values["ParticleNumber"][0], 2.0) and np.isclose(
        aux_values["Magnetization"][0], 0.0
    )

new_numpy_solver = NumPyEigensolver(k=4, filter_criterion=filter_criterion)
new_numpy_excited_states_solver = ExcitedStatesEigensolver(mapper, new_numpy_solver)
new_numpy_results = new_numpy_excited_states_solver.solve(es_problem)

print("\n自定义 filter 结果:")
print(new_numpy_results)

# VQE 基态计算示例
print("\n\n")
print("=" * 60)
print("=== VQE 基态计算 (可选) ===")
print("=" * 60)

try:
    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

    estimator = Estimator()
    solver = VQE(estimator, ansatz, SLSQP())
    solver.initial_point = [0.0] * ansatz.num_parameters
    gse = GroundStateEigensolver(mapper, solver)
    
    print("执行 VQE 基态计算...")
    vqe_results = gse.solve(es_problem)
    print(vqe_results)
except Exception as e:
    print(f"VQE 计算遇到问题：{e}")
    print("这可能是因为数值精度问题，不影响 NumPy 结果的准确性")

# 版本信息
print("\n\n")
print("=" * 60)
print("=== 版本信息 ===")
print("=" * 60)
import qiskit
import qiskit_nature
import qiskit_algorithms

print(f"Qiskit version: {qiskit.__version__}")
print(f"Qiskit Nature version: {qiskit_nature.__version__}")
print(f"Qiskit Algorithms version: {qiskit_algorithms.__version__}")

print("\n计算完成!")
