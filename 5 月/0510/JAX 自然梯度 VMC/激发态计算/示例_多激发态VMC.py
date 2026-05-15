#!/usr/bin/env python3
"""
多激发态VMC使用示例

这个脚本演示如何使用 MultiStateVMC.py 来训练基态和多个激发态
"""

import sys
sys.path.insert(0, '/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/JAX 自然梯度 VMC/激发态计算')

from MultiStateVMC import *

print("="*70)
print("H₂ 分子多激发态VMC示例")
print("="*70)
print("\nFCI基准能量:")
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"  E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

print("\n" + "="*70)
print("示例1: 使用便捷函数训练基态 + 第1激发态")
print("="*70)

trainer1 = train_excited_states(
    n_excited=1,
    n_iter_ground=200,
    n_iter_excited=200,
    n_samples=504,
    lambda_penalty=5.0,
    verbose=True
)

print("\n" + "="*70)
print("示例2: 使用便捷函数训练基态 + 第1、第2激发态")
print("="*70)

trainer2 = train_excited_states(
    n_excited=2,
    n_iter_ground=200,
    n_iter_excited=200,
    n_samples=504,
    lambda_penalty=5.0,
    verbose=False
)

trainer2.print_summary()

print("\n" + "="*70)
print("示例3: 使用MultiStateVMCTrainer类的高级用法")
print("="*70)

trainer3 = MultiStateVMCTrainer(
    n_spin_orbitals=4,
    n_excited_states=2,
    hidden_dim=12,
    learning_rate=0.01,
    lambda_penalty=5.0,
    seed=123
)

print("\n训练配置:")
print(f"  激发态数量: {trainer3.n_excited_states}")
print(f"  隐藏层维度: {trainer3.hidden_dim}")
print(f"  惩罚系数: {trainer3.lambda_penalty}")

trainer3.train_all_states(
    n_iter_ground=150,
    n_iter_excited=150,
    n_samples=504,
    verbose=True
)

results = trainer3.get_final_results()
print("\n最终结果:")
for result in results:
    print(f"  {result['state_name']}: {result['energy']:.8f} Ha", end='')
    if result['error'] is not None:
        print(f" (误差: {result['error']:.6f} Ha)", end='')
    if 'total_overlap_sq' in result:
        print(f" [重叠²: {result['total_overlap_sq']:.6f}]", end='')
    print()

print("\n" + "="*70)
print("所有示例运行完成!")
print("="*70)
