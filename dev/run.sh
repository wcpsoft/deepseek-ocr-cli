#!/bin/bash

# DeepSeek OCR CLI 运行脚本
# 用于快速运行OCR处理任务

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 运行脚本"
echo "========================================="

# 检查是否在项目根目录
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
    echo "警告: 未找到uv命令，将使用pip安装"
    python3 -m pip install uv
fi

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    uv venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装项目依赖
echo "安装项目依赖..."
uv pip install -e .

# 检查是否提供了输入文件
if [ $# -eq 0 ]; then
    echo "使用方法: ./dev/run.sh <输入文件> [输出目录]"
    echo "示例: ./dev/run.sh samples/1.pdf output"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-output}"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 运行OCR处理
echo "处理文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"

python3 -m cli.main "$INPUT_FILE" -o "$OUTPUT_DIR"

echo "========================================="
echo "  处理完成"
echo "  结果保存在: $OUTPUT_DIR"
echo "========================================="