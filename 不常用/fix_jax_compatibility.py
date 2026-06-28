#!/usr/bin/env python3
"""
修复JAX版本兼容性问题的脚本
将jax.tree_map替换为jax.tree_util.tree_map
"""

import os
import re
import sys

def fix_jax_compatibility(file_path):
    """
修复文件中的JAX兼容性问题
    将 jax.tree_map 替换为 jax.tree_util.tree_map
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换 jax.tree_map 为 jax.tree_util.tree_map
        pattern = r'jax\.tree_map\('
        replacement = 'jax.tree_util.tree_map('
        modified_content = re.sub(pattern, replacement, content)
        
        if content != modified_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"已修复文件: {file_path}")
            return True
        else:
            print(f"文件无需修复: {file_path}")
            return False
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {str(e)}")
        return False

def find_and_fix_deepqmc_files():
    """
查找并修复deepqmc库中的相关文件
    """
    # 尝试查找deepqmc安装路径
    import deepqmc
    deepqmc_path = os.path.dirname(deepqmc.__file__)
    print(f"找到deepqmc安装路径: {deepqmc_path}")
    
    # 需要修复的文件列表
    files_to_fix = [
        'ewm.py',
        'fit.py',
        'train.py'
    ]
    
    fixed_files = []
    
    for filename in files_to_fix:
        file_path = os.path.join(deepqmc_path, filename)
        if os.path.exists(file_path):
            if fix_jax_compatibility(file_path):
                fixed_files.append(file_path)
        else:
            print(f"文件不存在: {file_path}")
    
    return fixed_files

if __name__ == "__main__":
    print("开始修复JAX版本兼容性问题...")
    fixed_files = find_and_fix_deepqmc_files()
    
    if fixed_files:
        print(f"\n成功修复 {len(fixed_files)} 个文件:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
        print("\n修复完成！现在可以重新运行LiH.ipynb了。")
    else:
        print("\n没有找到需要修复的文件。")