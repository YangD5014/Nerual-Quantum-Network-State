#!/bin/bash
# 用法：在仓库根目录直接运行 bash yang.sh
# 行为：清空暂存区 → 只添加 <20MB 的普通文件（排除 .git / __pycache__ / .DS_Store /
#       子模块内部路径）→ 提交 → 推送 origin main。不改动任何工作区文件内容。
set -u

timestamp=$(date "+%Y-%m-%d %H:%M:%S")

# 1. 先清空暂存区，防止之前残留
git reset -q

# 2. 收集子模块路径：git 不允许把子模块内部的文件当普通路径 add
sub_paths=()
while read -r _key sub_path; do
    [ -n "$sub_path" ] && sub_paths+=("./$sub_path")
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null)

# 3. 构造 find 条件：<20MB、排除 .git、__pycache__、.DS_Store、子模块目录
find_expr=(. -type f -size -20M
           -not -path "./.git/*"
           -not -path "*/__pycache__/*"
           -not -name ".DS_Store")
for p in "${sub_paths[@]}"; do
    find_expr+=(-not -path "${p}/*")
done

# 4. 逐个文件 add：用 -print0/-0 处理含空格、中文的文件名；单个失败不影响其它文件
added=0
failed=0
while IFS= read -r -d '' f; do
    if git add -- "$f"; then
        added=$((added + 1))
    else
        failed=$((failed + 1))
        echo "跳过（无法添加）: $f" >&2
    fi
done < <(find "${find_expr[@]}" -print0)

echo "已暂存 $added 个文件，失败 $failed 个"

# 5. 提交（无改动则直接退出，不产生空提交）
if ! git commit -m "杨建飞 $timestamp 更新"; then
    echo "没有需要提交的改动，脚本结束。"
    exit 0
fi

# 6. 推送
git push origin main
