#!/bin/bash

# DeepSeek OCR CLI 测试脚本
# 用于运行单元测试和端到端测试

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 测试脚本"
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

echo "安装开发依赖..."
if [ "$GPU_TYPE" = "nvidia" ]; then
    echo "检测到NVIDIA GPU，安装CUDA版本的PyTorch..."
    uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    uv pip install -e .[dev]
elif [ "$GPU_TYPE" = "amd" ]; then
    echo "检测到AMD GPU，安装ROCm版本的PyTorch..."
    uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1
    uv pip install -e .[dev]
elif [ "$GPU_TYPE" = "mps" ]; then
    echo "检测到Apple Silicon，安装CPU版本的PyTorch..."
    uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    uv pip install -e .[dev]
elif [ "$GPU_TYPE" = "dcu" ]; then
    echo "检测到DCU，安装CPU版本的PyTorch..."
    uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    uv pip install -e .[dev]
else
    echo "未检测到专用GPU，安装CPU版本的PyTorch..."
    uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    uv pip install -e .[dev]
fi

# 运行单元测试
echo "运行单元测试..."
python3 -m pytest tests/test_cli.py -v

# 运行端到端测试
echo "运行端到端测试..."
python3 -m pytest tests/test_e2e.py -v

echo "========================================="
echo "  所有测试完成"
echo "========================================="