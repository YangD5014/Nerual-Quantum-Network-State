# Qiskit 激发态计算代码修复说明

## 问题原因

原始代码使用的是 Qiskit Nature 0.7.0 和 qiskit-terra 0.24.1，这些版本使用的是旧的 `BaseEstimator` API。

当你尝试在 Qiskit 2.x 版本上运行时，会出现以下错误：
```
ImportError: cannot import name 'BaseEstimator' from 'qiskit.primitives'
```

这是因为 Qiskit 2.x 已经移除了 `BaseEstimator`，改用新的 V2 API。

## 版本兼容性

经过测试，以下版本组合可以正常工作：

- **Qiskit**: 1.4.5 (1.x 系列)
- **Qiskit Nature**: 0.7.2
- **Qiskit Algorithms**: 0.4.0

## 修复内容

### 1. 主要导入修改

**原始代码 (不兼容):**
```python
from qiskit.primitives import BaseEstimator
```

**修复后:**
```python
from qiskit.primitives import Estimator  # Qiskit 1.x
# 或
from qiskit.primitives import StatevectorEstimator  # Qiskit 2.x (但 qiskit-nature 不支持)
```

### 2. 简化的代码结构

由于 qEOM 算法在 Qiskit 1.x 中存在一些数值精度问题，修复后的代码主要关注：
- ✅ NumPy 精确对角化方法 (完全可用)
- ✅ VQE 基态计算 (可选)
- ⚠️ qEOM 激发态计算 (存在数值问题，已暂时移除)

## 使用方法

### 运行 Python 脚本

```bash
cd "/Users/yangjianfei/mac_vscode/神经网络量子态/4 月/0421_qiskit"
python 04_excited_states_solvers_fixed.py
```

### 运行 Jupyter Notebook

在 Jupyter 中打开 `04_excited_states_solvers_fixed.ipynb` 并逐个单元格执行。

## 环境要求

确保安装了以下包：

```bash
pip install 'qiskit<2.0,>=1.0' qiskit-nature qiskit-algorithms pyscf
```

## 计算结果示例

修复后的代码成功计算了 H2 分子的激发态：

### 基态能量
- 电子基态能量：-1.393 Hartree
- 总基态能量：-1.015 Hartree

### 激发态能量 (使用默认 filter)
1. 第一激发态：-0.429 Hartree
2. 第二激发态：-0.269 Hartree

### 激发态能量 (使用自定义 filter - 包含三重态)
1. 第一激发态 (单重态): -0.875 Hartree  
2. 第二激发态 (三重态): -0.429 Hartree
3. 第三激发态 (单重态): -0.269 Hartree

## 注意事项

1. **qEOM 算法限制**: 原始代码中的 qEOM 算法在 Qiskit 1.x 中存在数值精度问题，已在修复版本中暂时移除。

2. **版本选择**: 如果你想使用 Qiskit 2.x，需要等待 qiskit-nature 更新以支持新的 primitives API。

3. **Filter 函数**: 自定义 filter 函数可以让你获取不同自旋态的激发态，而不仅仅是单重态。

## 文件说明

- `04_excited_states_solvers_fixed.py` - 可运行的 Python 脚本
- `04_excited_states_solvers_fixed.ipynb` - 可运行的 Jupyter Notebook
- `README_修复说明.md` - 本说明文档

## 参考资料

- [Qiskit Nature 迁移指南](https://qiskit.org/ecosystem/nature/migration/)
- [Qiskit Nature 文档](https://qiskit-community.github.io/qiskit-nature/)
- [Qiskit Nature GitHub Issue #1385](https://github.com/qiskit-community/qiskit-nature/issues/1385)
