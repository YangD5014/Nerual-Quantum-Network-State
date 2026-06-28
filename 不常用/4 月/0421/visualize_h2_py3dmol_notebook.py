#!/usr/bin/env python3
"""
H₂ 分子四种合法组态的 py3Dmol 可视化 (Jupyter Notebook 版本)
系统：4 个 spin-orbital, 2 个费米子 (1↑, 1↓)

使用方法：
    在 Jupyter notebook 中运行:
    %run visualize_h2_py3dmol_notebook.py
"""

import py3Dmol
from IPython.display import display, HTML, Markdown

def create_single_config(occupation, config_name, description):
    """
    创建单个组态的 py3Dmol 可视化
    
    Parameters:
    -----------
    occupation : list
        占据数列表 [n0, n1, n2, n3]
    config_name : str
        组态名称，如 "|1100⟩"
    description : str
        组态描述
    """
    # 创建视图
    view = py3Dmol.view(width=320, height=280)
    
    # 添加 H2 分子 (键长 0.74 Å = 1.4 Bohr)
    h2_xyz = f"""2
H2 molecule - {config_name}
H    0.000000    0.000000    0.000000
H    0.740000    0.000000    0.000000
"""
    
    view.addModel(h2_xyz, 'xyz')
    
    # 设置样式 - 球棍模型
    view.setStyle({'stick': {'radius': 0.12, 'color': 'gray'}, 
                   'sphere': {'scale': 0.25, 'color': 'white'}})
    
    # 添加轨道框（用半透明盒子的方式表示）
    # H1 原子轨道
    view.addShape('ellipsoid', {
        'center': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'radii': {'x': 0.8, 'y': 0.8, 'z': 0.8},
        'color': 'lightgray',
        'alpha': 0.15,
        'hidden': False
    })
    
    # H2 原子轨道
    view.addShape('ellipsoid', {
        'center': {'x': 0.74, 'y': 0.0, 'z': 0.0},
        'radii': {'x': 0.8, 'y': 0.8, 'z': 0.8},
        'color': 'lightgray',
        'alpha': 0.15,
        'hidden': False
    })
    
    # 根据占据情况添加自旋标签
    # 轨道 0: ↑₁ (H1 上自旋), 轨道 1: ↓₁ (H1 下自旋)
    # 轨道 2: ↑₂ (H2 上自旋), 轨道 3: ↓₂ (H2 下自旋)
    
    # H1 原子的电子标签
    y_offset_h1 = 0.0
    if occupation[0] == 1 and occupation[1] == 1:
        # 两个电子都在 H1
        view.addLabel('↑', {'position': {'x': -0.4, 'y': 0.5, 'z': 0}, 
                           'color': 'blue', 'fontSize': 24, 'font': 'Arial', 'bold': True})
        view.addLabel('↓', {'position': {'x': -0.4, 'y': -0.5, 'z': 0}, 
                           'color': 'red', 'fontSize': 24, 'font': 'Arial', 'bold': True})
    elif occupation[0] == 1:
        view.addLabel('↑', {'position': {'x': -0.4, 'y': 0.0, 'z': 0}, 
                           'color': 'blue', 'fontSize': 24, 'font': 'Arial', 'bold': True})
    elif occupation[1] == 1:
        view.addLabel('↓', {'position': {'x': -0.4, 'y': 0.0, 'z': 0}, 
                           'color': 'red', 'fontSize': 24, 'font': 'Arial', 'bold': True})
    
    # H2 原子的电子标签
    if occupation[2] == 1 and occupation[3] == 1:
        # 两个电子都在 H2
        view.addLabel('↑', {'position': {'x': 1.14, 'y': 0.5, 'z': 0}, 
                           'color': 'blue', 'fontSize': 24, 'font': 'Arial', 'bold': True})
        view.addLabel('↓', {'position': {'x': 1.14, 'y': -0.5, 'z': 0}, 
                           'color': 'red', 'fontSize': 24, 'font': 'Arial', 'bold': True})
    elif occupation[2] == 1:
        view.addLabel('↑', {'position': {'x': 1.14, 'y': 0.0, 'z': 0}, 
                           'color': 'blue', 'fontSize': 24, 'font': 'Arial', 'bold': True})
    elif occupation[3] == 1:
        view.addLabel('↓', {'position': {'x': 1.14, 'y': 0.0, 'z': 0}, 
                           'color': 'red', 'fontSize': 24, 'font': 'Arial', 'bold': True})
    
    # 添加组态标签（顶部）
    view.addLabel(f'<b>{config_name}</b>', {
        'position': {'x': 0.37, 'y': 0, 'z': 1.2},
        'color': 'black',
        'fontSize': 18,
        'font': 'Arial',
        'backgroundColor': 'wheat',
        'showBorder': True,
        'borderColor': 'black',
        'borderWidth': 1
    })
    
    # 设置视角和背景
    view.zoomTo()
    view.setBackgroundColor('#ffffff')
    view.rotate('y', 0.3)
    
    return view

def display_all_configurations():
    """显示所有四种组态"""
    
    # 四种合法组态
    configurations = [
        ([1, 1, 0, 0], "|1100⟩", "Both e⁻ in orbital 1\n(↑₁↓₁)"),
        ([1, 0, 1, 0], "|1010⟩", "↑ in orbital 1\n↑ in orbital 2"),
        ([0, 1, 1, 0], "|0110⟩", "↓ in orbital 1\n↑ in orbital 2"),
        ([0, 1, 0, 1], "|0101⟩", "Both e⁻ in orbital 2\n(↑₂↓₂)"),
    ]
    
    # 创建 HTML 布局
    html = """
    <div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f9f9f9;">
        <h1 style="text-align: center; color: #333;">H₂ Molecule: Fock Space Configurations</h1>
        <p style="text-align: center; color: #666; font-size: 14px;">
            4 Spin-Orbitals (2 spatial × 2 spin), 2 Fermions (1↑, 1↓)<br>
            STO-3G basis, bond length = 1.4 Bohr
        </p>
        
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 30px;">
    """
    
    # 为每个组态创建占位符
    for i, (occupation, config_name, desc) in enumerate(configurations):
        html += f"""
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div id="config-{i}" style="width:320px;height:280px;"></div>
            <p style="text-align: center; color: #555; font-size: 12px; margin-top: 10px;">{desc.replace(chr(10), '<br>')}</p>
        </div>
        """
    
    html += """
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding: 15px; background: white; 
                    border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #333; margin-bottom: 10px;">Legend</h3>
            <span style="color: blue; font-size: 20px; font-weight: bold; margin: 0 20px;">↑</span> Up Spin (α)
            <span style="color: red; font-size: 20px; font-weight: bold; margin: 0 20px;">↓</span> Down Spin (β)
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <script>
    """
    
    # 添加 JavaScript 代码
    for i, (occupation, config_name, desc) in enumerate(configurations):
        h2_xyz = f"""2
H2 molecule
H    0.000000    0.000000    0.000000
H    0.740000    0.000000    0.000000
"""
        
        # 构建自旋标签
        labels_js = []
        
        if occupation[0] == 1 and occupation[1] == 1:
            labels_js.append("{'text': '↑', 'pos': {'x': -0.4, 'y': 0.5, 'z': 0}, 'color': 'blue', 'fontSize': 24}")
            labels_js.append("{'text': '↓', 'pos': {'x': -0.4, 'y': -0.5, 'z': 0}, 'color': 'red', 'fontSize': 24}")
        elif occupation[0] == 1:
            labels_js.append("{'text': '↑', 'pos': {'x': -0.4, 'y': 0.0, 'z': 0}, 'color': 'blue', 'fontSize': 24}")
        elif occupation[1] == 1:
            labels_js.append("{'text': '↓', 'pos': {'x': -0.4, 'y': 0.0, 'z': 0}, 'color': 'red', 'fontSize': 24}")
        
        if occupation[2] == 1 and occupation[3] == 1:
            labels_js.append("{'text': '↑', 'pos': {'x': 1.14, 'y': 0.5, 'z': 0}, 'color': 'blue', 'fontSize': 24}")
            labels_js.append("{'text': '↓', 'pos': {'x': 1.14, 'y': -0.5, 'z': 0}, 'color': 'red', 'fontSize': 24}")
        elif occupation[2] == 1:
            labels_js.append("{'text': '↑', 'pos': {'x': 1.14, 'y': 0.0, 'z': 0}, 'color': 'blue', 'fontSize': 24}")
        elif occupation[3] == 1:
            labels_js.append("{'text': '↓', 'pos': {'x': 1.14, 'y': 0.0, 'z': 0}, 'color': 'red', 'fontSize': 24}")
        
        labels_str = '[' + ', '.join(labels_js) + ']'
        
        html += f"""
        (function() {{
            let element = document.getElementById('config-{i}');
            let config = {{backgroundColor: 'white', width: 320, height: 280}};
            let viewer = $3Dmol.createViewer(element, config);
            
            let xyz = `{h2_xyz}`;
            viewer.addModel(xyz, 'xyz');
            viewer.setStyle({{}}, {{stick: {{radius: 0.12, color: 'gray'}}, sphere: {{scale: 0.25, color: 'white'}}}});
            
            // Add orbital ellipsoids
            viewer.addShape('ellipsoid', {{
                center: {{x: 0.0, y: 0.0, z: 0.0}},
                radii: {{x: 0.8, y: 0.8, z: 0.8}},
                color: 'lightgray',
                alpha: 0.15
            }});
            
            viewer.addShape('ellipsoid', {{
                center: {{x: 0.74, y: 0.0, z: 0.0}},
                radii: {{x: 0.8, y: 0.8, z: 0.8}},
                color: 'lightgray',
                alpha: 0.15
            }});
            
            // Add spin labels
            let labels = {labels_str};
            labels.forEach(function(label) {{
                viewer.addLabel(label.text, {{
                    position: label.pos,
                    color: label.color,
                    fontSize: label.fontSize,
                    font: 'Arial',
                    bold: true
                }});
            }});
            
            // Add configuration label
            viewer.addLabel('{config_name}', {{
                position: {{x: 0.37, y: 0, z: 1.2}},
                color: 'black',
                fontSize: 18,
                font: 'Arial',
                backgroundColor: 'wheat',
                showBorder: true,
                borderColor: 'black'
            }});
            
            viewer.zoomTo();
            viewer.rotate('y', 0.3);
            viewer.render();
        }})();
        """
    
    html += """
    </script>
    """
    
    # 显示 HTML
    display(HTML(html))

def main():
    """主函数"""
    print("=" * 70)
    print("H₂ Molecule: Four Valid Fock Space Configurations")
    print("=" * 70)
    print("\nSystem: 4 spin-orbitals, 2 fermions (1↑, 1↓)")
    print("Basis: STO-3G, Bond length: 1.4 Bohr (0.74 Å)")
    print("\nFour configurations:")
    print("  1. |1100⟩ - Both electrons in spatial orbital 1 (↑₁↓₁)")
    print("  2. |1010⟩ - ↑ in orbital 1, ↑ in orbital 2")
    print("  3. |0110⟩ - ↓ in orbital 1, ↑ in orbital 2")
    print("  4. |0101⟩ - Both electrons in spatial orbital 2 (↑₂↓₂)")
    print("\n" + "=" * 70)
    print("\nRendering interactive visualization with py3Dmol...\n")
    
    # 显示可视化
    display_all_configurations()
    
    print("\n✓ Interactive visualization displayed above!")
    print("  You can rotate, zoom, and pan the molecules with your mouse.")

if __name__ == '__main__':
    main()
