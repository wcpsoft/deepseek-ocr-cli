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

# 自动检测GPU类型并安装相应依赖
echo "检测GPU环境..."
GPU_TYPE=$(python3 dev/detect_gpu.py | grep "GPU_TYPE=" | cut -d'=' -f2)
EXTRA_SUFFIX=$(python3 dev/detect_gpu.py | grep "EXTRA_SUFFIX=" | cut -d'=' -f2)

echo "安装项目依赖..."
if [ -n "$EXTRA_SUFFIX" ]; then
    echo "检测到 $GPU_TYPE GPU，安装相应版本的PyTorch..."
    # 使用更精确的依赖安装方式，避免跨平台依赖冲突
    case $GPU_TYPE in
        "nvidia")
            echo "执行命令: uv pip install -e .[nvidia]"
            uv pip install -e .[nvidia]
            ;;
        "amd")
            echo "执行命令: uv pip install -e .[amd]"
            uv pip install -e .[amd]
            ;;
        "mps")
            echo "执行命令: uv pip install -e .[mps]"
            uv pip install -e .[mps]
            ;;
        "dcu")
            echo "执行命令: uv pip install -e .[dcu]"
            uv pip install -e .[dcu]
            ;;
        "cpu")
            echo "执行命令: uv pip install -e ."
            uv pip install -e .
            ;;
        *)
            echo "未知GPU类型，使用默认CPU配置安装..."
            uv pip install -e .
            ;;
    esac
else
    echo "未检测到专用GPU，安装CPU版本的PyTorch..."
    uv pip install -e .
fi

# 检查模型是否存在，如果不存在则下载默认模型
echo "检查模型..."
if [ ! -d "models/deepseek-ocr" ]; then
    echo "默认模型不存在，正在下载..."
    python3 -m cli.download_models -m deepseek-ocr
else
    echo "默认模型已存在"
fi

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