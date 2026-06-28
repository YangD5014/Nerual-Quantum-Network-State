#!/usr/bin/env python3
"""
H₂ 分子四种合法组态的可视化
系统：4 个 spin-orbital, 2 个费米子 (1↑, 1↓)
使用 matplotlib 绘制
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.patches import Rectangle

def setup_figure():
    """创建图形"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax

def draw_orbital(ax, x, y, label, alpha=0.3):
    """绘制轨道（半透明圆圈）"""
    # 轨道背景
    circle = Circle((x, y), 0.6, color='lightgray', alpha=alpha, linewidth=2, edgecolor='black')
    ax.add_patch(circle)
    
    # 标签
    ax.text(x, y - 1.0, label, ha='center', va='top', fontsize=14, fontweight='bold')

def draw_electron(ax, x, y, spin):
    """绘制电子（实心圆）"""
    if spin == 'up':
        color = 'blue'
        label = '↑'
    else:
        color = 'red'
        label = '↓'
    
    circle = Circle((x, y), 0.4, color=color, alpha=0.8, linewidth=2, edgecolor='white', zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=16, color='white', fontweight='bold', zorder=6)

def visualize_configuration(ax, config_index, occupation, x_offset):
    """
    可视化一个组态
    
    Parameters:
    -----------
    config_index : int
        组态编号
    occupation : list
        占据数列表 [n0, n1, n2, n3]
    x_offset : float
        x 方向偏移
    """
    # 轨道信息：(轨道索引，标签，自旋)
    orbitals = [
        (0, '↑₁', 'up'),    # 轨道 0: 空间轨道 1，上自旋
        (1, '↓₁', 'down'),  # 轨道 1: 空间轨道 1，下自旋
        (2, '↑₂', 'up'),    # 轨道 2: 空间轨道 2，上自旋
        (3, '↓₂', 'down'),  # 轨道 3: 空间轨道 2，下自旋
    ]
    
    # 绘制轨道
    for i, (orb_idx, label, spin) in enumerate(orbitals):
        x = x_offset + i * 1.5
        y = 0
        draw_orbital(ax, x, y, label)
        
        # 如果该轨道被占据，绘制电子
        if occupation[orb_idx] == 1:
            draw_electron(ax, x, y + 0.2, spin)
    
    # 添加组态标签
    config_label = f"|{''.join(map(str, occupation))}⟩"
    ax.text(x_offset + 2.25, -2.0, config_label, ha='center', va='center', 
            fontsize=18, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def add_legend(ax):
    """添加图例"""
    # 上自旋图例
    up_circle = Circle((0.3, -2.5), 0.2, color='blue', alpha=0.8)
    ax.add_patch(up_circle)
    ax.text(0.8, -2.5, '↑ (up spin)', ha='left', va='center', fontsize=12, color='blue', fontweight='bold')
    
    # 下自旋图例
    down_circle = Circle((2.3, -2.5), 0.2, color='red', alpha=0.8)
    ax.add_patch(down_circle)
    ax.text(2.8, -2.5, '↓ (down spin)', ha='left', va='center', fontsize=12, color='red', fontweight='bold')

def main():
    """主函数：可视化四种合法组态"""
    fig, ax = setup_figure()
    
    # 四种合法组态 (4 个 spin-orbital, 2 个费米子)
    configurations = [
        ([1, 1, 0, 0], "Both in orbital 1"),   # 组态 1: 两个电子都在空间轨道 1
        ([1, 0, 1, 0], "↑₁, ↑₂"),              # 组态 2: 上自旋在轨道 1，上自旋在轨道 2
        ([0, 1, 1, 0], "↓₁, ↑₂"),              # 组态 3: 下自旋在轨道 1，上自旋在轨道 2
        ([0, 1, 0, 1], "Both in orbital 2"),   # 组态 4: 两个电子都在空间轨道 2
    ]
    
    # 可视化每个组态，水平排列
    for i, (occupation, description) in enumerate(configurations):
        x_offset = i * 3.5  # 每个组态间隔
        visualize_configuration(ax, i + 1, occupation, x_offset)
    
    # 添加图例
    add_legend(ax)
    
    # 添加标题
    ax.text(7, 2.3, 'H₂ Molecule: 4 Spin-Orbitals, 2 Fermions (1↑, 1↓)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加说明
    ax.text(7, 1.7, 'Two H atoms with STO-3G basis → 2 spatial orbitals × 2 spins = 4 spin-orbitals',
            ha='center', va='center', fontsize=10, style='italic', alpha=0.7)
    
    # 保存图片
    plt.tight_layout()
    output_file = '/Users/yangjianfei/mac_vscode/神经网络量子态/4 月/0421/h2_configurations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # 显示
    plt.show()
    
    print("\nFour valid configurations:")
    print("  1. |1100⟩ - Both electrons in spatial orbital 1 (↑₁↓₁)")
    print("  2. |1010⟩ - ↑ in orbital 1, ↑ in orbital 2")
    print("  3. |0110⟩ - ↓ in orbital 1, ↑ in orbital 2")
    print("  4. |0101⟩ - Both electrons in spatial orbital 2 (↑₂↓₂)")
    print("\nNote: Configurations must conserve particle number and spin")

if __name__ == '__main__':
    main()
