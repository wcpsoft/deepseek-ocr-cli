#!/bin/bash

# DeepSeek OCR CLI 一键安装脚本
# 用于快速设置开发环境

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 一键安装脚本"
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
    echo "安装uv包管理器..."
    python3 -m pip install uv
fi

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    uv venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装基础依赖
echo "安装基础依赖..."
uv pip install -e .

# 安装开发依赖
echo "安装开发依赖..."
uv pip install -e '.[dev]' || echo "警告: 开发依赖安装失败"

# 自动检测GPU类型并安装相应依赖
echo "检测GPU环境..."
GPU_TYPE=$(python3 dev/detect_gpu.py | grep "GPU_TYPE=" | cut -d'=' -f2)

echo "安装硬件特定依赖..."
case $GPU_TYPE in
    "nvidia")
        echo "检测到NVIDIA GPU，安装相应依赖..."
        uv pip install -r requirements/requirements-nvidia.txt
        ;;
    "amd")
        echo "检测到AMD GPU，安装相应依赖..."
        uv pip install -r requirements/requirements-amd.txt
        ;;
    "mps")
        echo "检测到Apple Silicon MPS，安装相应依赖..."
        uv pip install -r requirements/requirements-mps.txt
        ;;
    "dcu")
        echo "检测到DCU，安装相应依赖..."
        uv pip install -r requirements/requirements-dcu.txt
        ;;
    "cpu"|*)
        echo "使用CPU模式，安装相应依赖..."
        uv pip install -r requirements/requirements-cpu.txt
        ;;
esac

echo "========================================="
echo "  安装完成！"
echo "  可以运行以下命令开始使用："
echo "  - 运行测试: ./dev/run_tests.sh"
echo "  - 处理文件: ./dev/run.sh <输入文件> [输出目录]"
echo "  - 调试OCR: ./dev/debug_ocr.py"
echo "========================================="