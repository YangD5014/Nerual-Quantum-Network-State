# H₂ 分子 Fock 空间组态可视化

使用 **py3Dmol** 创建的 H₂ 分子四种合法组态的交互式 3D 可视化

## 📁 文件说明

### 主要文件
- **`h2_configurations_py3dmol.html`** - 独立 HTML 文件，可在浏览器中直接打开查看交互式可视化
- **`H2_configurations_py3dmol.ipynb`** - Jupyter Notebook 版本，可在 notebook 环境中运行

### Python 脚本
- **`visualize_h2_py3dmol.py`** - 生成 HTML 文件的 Python 脚本
- **`visualize_h2_py3dmol_notebook.py`** - Jupyter Notebook 中使用的脚本版本

## 🧪 系统信息

- **分子**: H₂ (氢分子)
- **键长**: 1.4 Bohr (0.74 Å)
- **基组**: STO-3G
- **空间轨道**: 2 (每个氢原子 1 个)
- **自旋轨道**: 4 (2 空间 × 2 自旋)
- **电子**: 2 个费米子 (1↑, 1↓)

## 📊 四种合法组态

1. **|1100⟩** - 两个电子都在空间轨道 1 (↑₁↓₁)
2. **|1010⟩** - 上自旋在轨道 1，上自旋在轨道 2 (↑₁↑₂)
3. **|0110⟩** - 下自旋在轨道 1，上自旋在轨道 2 (↓₁↑₂)
4. **|0101⟩** - 两个电子都在空间轨道 2 (↑₂↓₂)

## 🖥️ 使用方法

### 方法 1: 直接打开 HTML 文件
```bash
# 在浏览器中打开
open h2_configurations_py3dmol.html
```

### 方法 2: 在 Jupyter Notebook 中运行
```python
# 在 Jupyter Notebook 中运行
%run visualize_h2_py3dmol_notebook.py
```

或者直接使用 notebook 文件：
```bash
jupyter notebook H2_configurations_py3dmol.ipynb
```

### 方法 3: 运行 Python 脚本生成 HTML
```bash
python visualize_h2_py3dmol.py
```

## 🎮 交互控制

- **旋转**: 左键点击并拖动
- **平移**: 右键点击并拖动
- **缩放**: 滚动鼠标滚轮

## 📋 图例

- <span style="color: blue; font-size: 18px; font-weight: bold;">↑</span> **上自旋** (α 电子) - 蓝色
- <span style="color: red; font-size: 18px; font-weight: bold;">↓</span> **下自旋** (β 电子) - 红色

## 💡 说明

这四种组态是 Fock 空间中满足以下约束条件的所有合法组态：
- **粒子数守恒**: 总电子数 = 2
- **自旋守恒**: 1 个上自旋 + 1 个下自旋

这些组态构成了 H₂ 分子在 STO-3G 基组下的完整 Fock 空间基底。

## 🔗 依赖

- [py3Dmol](https://3dmol.org/) - 基于 WebGL 的分子可视化库
- 现代 Web 浏览器 (Chrome, Firefox, Safari, Edge 等)
