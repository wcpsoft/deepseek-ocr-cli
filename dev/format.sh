#!/bin/bash

# 代码格式化脚本
# 使用black、isort和ruff对代码进行格式化

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 代码格式化脚本"
echo "========================================="

# 检查项目根目录
if [ ! -f "pyproject.toml" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查uv是否已安装
if ! command -v uv &> /dev/null; then
    echo "警告: 未找到uv命令，将使用系统Python"
    USE_UV=false
else
    USE_UV=true
fi

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "已激活虚拟环境"
fi

# 检查并安装格式化工具
if [ "$USE_UV" = true ]; then
    echo "使用uv安装格式化工具..."
    uv pip install black isort ruff
else
    echo "使用pip安装格式化工具..."
    pip install black isort ruff
fi

# 运行isort（导入排序）
echo "运行isort进行导入排序..."
python -m isort src/ cli/ tests/ dev/

# 运行black（代码格式化）
echo "运行black进行代码格式化..."
python -m black src/ cli/ tests/ dev/

# 运行ruff（代码检查和修复）
echo "运行ruff进行代码检查..."
python -m ruff check src/ cli/ tests/ dev/ --fix

echo "========================================="
echo "  代码格式化完成"
echo "========================================="