#!/bin/bash

# 获取当前时间戳（格式：YYYY-MM-DD HH:MM:SS）
timestamp=$(date "+%Y-%m-%d %H:%M:%S")

# 执行Git操作
git add .
git commit -m "杨建飞 $timestamp 更新"
git push origin YangJianfei_Branch