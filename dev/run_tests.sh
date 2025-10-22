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
EXTRA_SUFFIX=$(python3 dev/detect_gpu.py | grep "EXTRA_SUFFIX=" | cut -d'=' -f2)

# 添加调试信息
echo "DEBUG: GPU_TYPE='$GPU_TYPE'"
echo "DEBUG: EXTRA_SUFFIX='$EXTRA_SUFFIX'"

echo "安装开发依赖..."
if [ -n "$EXTRA_SUFFIX" ]; then
    echo "检测到 $GPU_TYPE GPU，安装相应版本的PyTorch..."
    # 使用更精确的依赖安装方式，避免跨平台依赖冲突
    # 首先安装基础依赖和开发依赖
    uv pip install -e .[dev]
    
    # 然后安装硬件特定依赖
    case $GPU_TYPE in
        "nvidia")
            echo "安装NVIDIA GPU特定依赖..."
            uv pip install -r requirements/requirements-nvidia.txt
            ;;
        "amd")
            echo "安装AMD GPU特定依赖..."
            uv pip install -r requirements/requirements-amd.txt
            ;;
        "mps")
            echo "安装Apple Silicon MPS特定依赖..."
            uv pip install -r requirements/requirements-mps.txt
            ;;
        "dcu")
            echo "安装DCU特定依赖..."
            uv pip install -r requirements/requirements-dcu.txt
            ;;
        "cpu")
            echo "安装CPU特定依赖..."
            uv pip install -r requirements/requirements-cpu.txt
            ;;
        *)
            echo "未知GPU类型，安装CPU特定依赖..."
            uv pip install -r requirements/requirements-cpu.txt
            ;;
    esac
else
    echo "未检测到专用GPU，安装CPU版本的PyTorch..."
    uv pip install -e .[dev]
fi

# 运行单元测试
echo "运行单元测试..."
.venv/bin/python -m pytest tests/unit/ -v

# 运行集成测试
echo "运行集成测试..."
.venv/bin/python -m pytest tests/integration/ -v

# 运行端到端测试
echo "运行端到端测试..."

# 检查模型是否存在且完整
if [ -d "models" ]; then
    # 检查默认模型目录是否存在
    if [ -d "models/deepseek-ocr" ]; then
        # 检查模型必要文件是否存在
        if [ -f "models/deepseek-ocr/config.json" ]; then
            # 检查是否存在模型权重文件
            if ls models/deepseek-ocr/pytorch_model*.bin 1> /dev/null 2>&1 || ls models/deepseek-ocr/*.safetensors 1> /dev/null 2>&1 || [ -f "models/deepseek-ocr/model.safetensors.index.json" ]; then
                echo "模型已存在，开始运行端到端测试..."
                if [ -d "samples" ]; then
                    .venv/bin/python -m pytest tests/e2e/ -v
                else
                    echo "跳过端到端测试：未找到samples目录"
                    echo "所有单元测试和集成测试已通过！"
                fi
            else
                echo "注意: 端到端测试需要下载模型文件才能正常运行"
                echo "请先运行 'deepseek-ocr --download-models' 下载模型"
            fi
        else
            echo "注意: 端到端测试需要下载模型文件才能正常运行"
            echo "请先运行 'deepseek-ocr --download-models' 下载模型"
        fi
    else
        echo "注意: 端到端测试需要下载模型文件才能正常运行"
        echo "请先运行 'deepseek-ocr --download-models' 下载模型"
    fi
else
    echo "注意: 端到端测试需要下载模型文件才能正常运行"
    echo "请先运行 'deepseek-ocr --download-models' 下载模型"
fi

echo "========================================="
echo "  所有测试完成"
echo "========================================="
echo ""
echo "提示: 如果端到端测试因模型依赖问题失败，请确保已安装正确的Transformers版本"
echo "      并下载了DeepSeek-OCR模型文件到models目录"