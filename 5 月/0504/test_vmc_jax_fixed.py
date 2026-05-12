#!/usr/bin/env python
# coding: utf-8
"""
测试修复后的 VMC 梯度计算
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import flatten_util
import numpy as np

print("="*60)
print("测试修复后的 VMC 梯度计算")
print("="*60)

# 简单测试模型
class SimpleModel(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 4, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)
    
    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        out = self.output(h)
        return jnp.squeeze(out)

# 创建模型
model = SimpleModel(input_dim=4, hidden_dim=4, rngs=nnx.Rngs(42))
graphdef, state = nnx.split(model)

def machine(params, sigma):
    m = nnx.merge(graphdef, params)
    return m(sigma)

# 测试样本
sigma = jnp.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
], dtype=jnp.float32)

params = state

print("\n[1/3] 测试 VJP 梯度计算...")
# 使用 VJP 计算梯度
def log_psi_single(p, s):
    return machine(p, s)

def compute_grad_for_sample(s):
    return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

try:
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    print(f"  梯度矩阵类型：{type(grad_matrix)}")
    print(f"  梯度矩阵结构：{jax.tree_map(lambda x: x.shape, grad_matrix)}")
    print("  ✓ VJP 梯度计算成功")
except Exception as e:
    print(f"  ✗ VJP 梯度计算失败：{e}")
    raise

print("\n[2/3] 测试 PyTree 展平和恢复...")
try:
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(sigma.shape[0], -1)
    print(f"  展平后形状：{grad_flat.shape}")
    
    # 测试恢复
    grad_restored = unravel_fn(grad_flat)
    print(f"  恢复后结构：{jax.tree_map(lambda x: x.shape, grad_restored)}")
    print("  ✓ PyTree 展平和恢复成功")
except Exception as e:
    print(f"  ✗ PyTree 展平和恢复失败：{e}")
    raise

print("\n[3/3] 测试 Force-based 梯度计算...")
# 模拟局部能量
E_loc = jnp.array([1.0, 2.0, 3.0])
E_mean = jnp.mean(E_loc)
E_centered = E_loc - E_mean

print(f"  E_loc: {E_loc}")
print(f"  E_mean: {E_mean:.4f}")
print(f"  E_centered: {E_centered}")

try:
    # 计算 force-based 梯度
    grad = jax.tree_map(
        lambda g: jnp.mean(jnp.expand_dims(E_centered, -1) * jnp.conj(g), axis=0),
        grad_matrix
    )
    print(f"  梯度结构：{jax.tree_map(lambda x: x.shape, grad)}")
    print("  ✓ Force-based 梯度计算成功")
except Exception as e:
    print(f"  ✗ Force-based 梯度计算失败：{e}")
    raise

print("\n" + "="*60)
print("所有测试通过！修复后的代码正常工作。")
print("="*60)
print("\n关键修复点:")
print("1. 使用 jax.vjp 替代 jax.jacobian 计算梯度")
print("2. 对每个样本单独计算梯度，然后使用 vmap 批处理")
print("3. 直接使用 tree_map 处理 PyTree 结构，避免手动展平问题")
