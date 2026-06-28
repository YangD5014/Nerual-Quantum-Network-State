#!/usr/bin/env python3
"""
H₂ 分子四种合法组态的 py3Dmol 可视化 (独立 HTML 版本)
系统：4 个 spin-orbital, 2 个费米子 (1↑, 1↓)

生成的 HTML 文件可以在任何现代浏览器中打开，无需 Python 环境
"""

def create_h2_dashboard():
    """创建包含四种组态的交互式 HTML 仪表板"""
    
    # 四种合法组态
    configurations = [
        ([1, 1, 0, 0], "|1100⟩", "Both e⁻ in orbital 1 (↑₁↓₁)"),
        ([1, 0, 1, 0], "|1010⟩", "↑ in orbital 1, ↑ in orbital 2"),
        ([0, 1, 1, 0], "|0110⟩", "↓ in orbital 1, ↑ in orbital 2"),
        ([0, 1, 0, 1], "|0101⟩", "Both e⁻ in orbital 2 (↑₂↓₂)"),
    ]
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>H₂ Molecule - Fock Space Configurations</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 36px;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 16px;
            opacity: 0.9;
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .config-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .config-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.3);
        }
        
        .config-viewer {
            width: 100%;
            height: 300px;
            border-radius: 10px;
            overflow: hidden;
            background: #f8f8f8;
        }
        
        .config-info {
            text-align: center;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #f0f0f0;
        }
        
        .config-label {
            display: inline-block;
            background: linear-gradient(135deg, #f5af19 0%, #f12711 100%);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .config-description {
            color: #555;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .legend {
            background: white;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }
        
        .legend h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 20px;
        }
        
        .legend-items {
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 16px;
            color: #555;
        }
        
        .spin-symbol {
            font-size: 28px;
            font-weight: bold;
            width: 35px;
            text-align: center;
        }
        
        .spin-up {
            color: #0066ff;
        }
        
        .spin-down {
            color: #ff3333;
        }
        
        .info-box {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-top: 25px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }
        
        .info-box h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 18px;
        }
        
        .info-box p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 8px;
        }
        
        .controls {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
        }
        
        .controls p {
            color: #666;
            font-size: 14px;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 28px;
            }
            
            .config-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 H₂ Molecule: Fock Space Configurations</h1>
            <p>4 Spin-Orbitals (2 spatial × 2 spin) | 2 Fermions (1↑, 1↓) | STO-3G Basis</p>
        </div>
        
        <div class="config-grid">
"""
    
    # 为每个组态生成卡片
    for i, (occupation, config_name, description) in enumerate(configurations):
        html += f"""
            <div class="config-card">
                <div id="config-{i}" class="config-viewer"></div>
                <div class="config-info">
                    <div class="config-label">{config_name}</div>
                    <div class="config-description">{description}</div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="legend">
            <h3>📋 Legend</h3>
            <div class="legend-items">
                <div class="legend-item">
                    <span class="spin-symbol spin-up">↑</span>
                    <span>Up Spin (α electron)</span>
                </div>
                <div class="legend-item">
                    <span class="spin-symbol spin-down">↓</span>
                    <span>Down Spin (β electron)</span>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <p><strong>🖱️ Interactive Controls:</strong> Left-click to rotate | Right-click to pan | Scroll to zoom</p>
        </div>
        
        <div class="info-box">
            <h3>ℹ️ System Information</h3>
            <p><strong>Molecule:</strong> H₂ (Hydrogen dimer)</p>
            <p><strong>Bond Length:</strong> 1.4 Bohr (0.74 Å)</p>
            <p><strong>Basis Set:</strong> STO-3G (minimal basis)</p>
            <p><strong>Spatial Orbitals:</strong> 2 (one per hydrogen atom)</p>
            <p><strong>Spin-Orbitals:</strong> 4 (2 spatial × 2 spin states)</p>
            <p><strong>Electrons:</strong> 2 fermions with conserved particle number and spin (1↑, 1↓)</p>
        </div>
    </div>
    
    <script>
"""
    
    # 为每个组态添加 JavaScript 代码
    for i, (occupation, config_name, description) in enumerate(configurations):
        h2_xyz = """2
H2 molecule
H    0.000000    0.000000    0.000000
H    0.740000    0.000000    0.000000"""
        
        # 构建自旋标签
        labels_js = []
        
        # H1 原子的电子
        if occupation[0] == 1 and occupation[1] == 1:
            labels_js.append("{text: '↑', pos: {x: -0.5, y: 0.6, z: 0}, color: 'blue', fontSize: 28}")
            labels_js.append("{text: '↓', pos: {x: -0.5, y: -0.6, z: 0}, color: 'red', fontSize: 28}")
        elif occupation[0] == 1:
            labels_js.append("{text: '↑', pos: {x: -0.5, y: 0, z: 0}, color: 'blue', fontSize: 28}")
        elif occupation[1] == 1:
            labels_js.append("{text: '↓', pos: {x: -0.5, y: 0, z: 0}, color: 'red', fontSize: 28}")
        
        # H2 原子的电子
        if occupation[2] == 1 and occupation[3] == 1:
            labels_js.append("{text: '↑', pos: {x: 1.24, y: 0.6, z: 0}, color: 'blue', fontSize: 28}")
            labels_js.append("{text: '↓', pos: {x: 1.24, y: -0.6, z: 0}, color: 'red', fontSize: 28}")
        elif occupation[2] == 1:
            labels_js.append("{text: '↑', pos: {x: 1.24, y: 0, z: 0}, color: 'blue', fontSize: 28}")
        elif occupation[3] == 1:
            labels_js.append("{text: '↓', pos: {x: 1.24, y: 0, z: 0}, color: 'red', fontSize: 28}")
        
        labels_str = '[' + ', '.join(labels_js) + ']'
        
        html += f"""
        (function() {{
            let element = document.getElementById('config-{i}');
            let config = {{backgroundColor: '#f8f8f8', width: '100%', height: 300}};
            let viewer = $3Dmol.createViewer(element, config);
            
            let xyz = `{h2_xyz}`;
            viewer.addModel(xyz, 'xyz');
            
            // Style: stick and sphere model
            viewer.setStyle({{}}, {{
                stick: {{radius: 0.15, color: '#888888'}},
                sphere: {{scale: 0.3, color: '#ffffff'}}
            }});
            
            // Add orbital ellipsoids (translucent)
            viewer.addShape('ellipsoid', {{
                center: {{x: 0.0, y: 0.0, z: 0.0}},
                radii: {{x: 0.9, y: 0.9, z: 0.9}},
                color: '#cccccc',
                alpha: 0.2
            }});
            
            viewer.addShape('ellipsoid', {{
                center: {{x: 0.74, y: 0.0, z: 0.0}},
                radii: {{x: 0.9, y: 0.9, z: 0.9}},
                color: '#cccccc',
                alpha: 0.2
            }});
            
            // Add spin labels
            let labels = {labels_str};
            labels.forEach(function(label) {{
                viewer.addLabel(label.text, {{
                    position: label.pos,
                    color: label.color,
                    fontSize: label.fontSize,
                    font: 'Arial',
                    bold: true,
                    showBorder: false
                }});
            }});
            
            // Add configuration label at top
            viewer.addLabel('{config_name}', {{
                position: {{x: 0.37, y: 0, z: 1.5}},
                color: 'white',
                fontSize: 22,
                font: 'Arial',
                backgroundColor: '#f5af19',
                showBorder: true,
                borderColor: '#f12711',
                borderWidth: 2,
                borderRadius: 15
            }});
            
            viewer.zoomTo();
            viewer.rotate('y', 0.2);
            viewer.render();
        }})();
"""
    
    html += """
    </script>
</body>
</html>
"""
    
    return html

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
    
    # 创建 HTML
    html_content = create_h2_dashboard()
    
    # 保存 HTML 文件
    output_file = '/Users/yangjianfei/mac_vscode/神经网络量子态/4 月/0421/h2_configurations_py3dmol.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Interactive visualization saved to:")
    print(f"   {output_file}")
    print(f"\n🌐 Open this HTML file in any modern web browser to view the 3D visualization.")
    print(f"   You can rotate, zoom, and pan the molecules interactively!")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
