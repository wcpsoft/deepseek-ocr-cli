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
    case $GPU_TYPE in
        "nvidia")
            echo "执行命令: uv pip install -e .[dev,nvidia]"
            uv pip install -e .[dev,nvidia]
            ;;
        "amd")
            echo "执行命令: uv pip install -e .[dev,amd]"
            uv pip install -e .[dev,amd]
            ;;
        "mps")
            echo "执行命令: uv pip install -e .[dev,mps]"
            uv pip install -e .[dev,mps]
            ;;
        "dcu")
            echo "执行命令: uv pip install -e .[dev,dcu]"
            uv pip install -e .[dev,dcu]
            ;;
        "cpu")
            echo "执行命令: uv pip install -e .[dev]"
            uv pip install -e .[dev]
            ;;
        *)
            echo "未知GPU类型，使用默认CPU配置安装..."
            uv pip install -e .[dev]
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
echo "注意: 端到端测试需要下载模型文件才能正常运行"
echo "请先运行 'deepseek-ocr --download-models' 下载模型"
if [ -d "samples" ] && [ -d "models" ]; then
    .venv/bin/python -m pytest tests/e2e/ -v
else
    echo "跳过端到端测试：未找到samples目录或models目录"
    echo "所有单元测试和集成测试已通过！"
fi

echo "========================================="
echo "  所有测试完成"
echo "========================================="
echo ""
echo "提示: 如果端到端测试因模型依赖问题失败，请确保已安装正确的Transformers版本"
echo "      并下载了DeepSeek-OCR模型文件到models目录"