#!/bin/bash

# DeepSeek OCR CLI 运行脚本
# 用于快速运行OCR处理任务

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 运行脚本"
echo "========================================="

# 解析命令行参数
DOWNLOAD_MODELS=false
INPUT_FILE=""
OUTPUT_DIR="output"
MODE="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        --download-models)
            DOWNLOAD_MODELS=true
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        -*)
            echo "未知选项: $1"
            echo "使用方法: ./dev/run.sh [--download-models] [--mode MODE] [输入文件] [输出目录]"
            exit 1
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            else
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

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
RECOMMENDED_MODE=$(python3 dev/detect_gpu.py | grep "推荐的推理模式:" | cut -d' ' -f3)

echo "安装项目依赖..."
if [ -n "$EXTRA_SUFFIX" ]; then
    echo "检测到 $GPU_TYPE GPU，安装相应版本的PyTorch..."
    # 使用更精确的依赖安装方式，避免跨平台依赖冲突
    # 首先安装基础依赖
    uv pip install -e .
    
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
    uv pip install -e .
fi

# 检查是否需要下载模型
if [ "$DOWNLOAD_MODELS" = true ]; then
    echo "下载模型..."
    python3 -m cli.download_models -m deepseek-ocr --force
    exit 0
fi

# 检查模型是否存在，如果不存在则下载默认模型
echo "检查模型..."
if [ ! -d "models/deepseek-ocr" ]; then
    echo "默认模型不存在，正在下载..."
    python3 -m cli.download_models -m deepseek-ocr
else
    echo "默认模型已存在"
fi

# 如果没有提供输入文件，则退出
if [ -z "$INPUT_FILE" ]; then
    echo "使用方法: ./dev/run.sh [--download-models] [--mode MODE] <输入文件> [输出目录]"
    echo "示例: ./dev/run.sh samples/1.pdf output"
    # 显示推荐的推理模式（如果有的话）
    if [ -n "$RECOMMENDED_MODE" ] && [ "$RECOMMENDED_MODE" != "auto" ]; then
        echo "提示: 检测到您的系统推荐使用 $RECOMMENDED_MODE 模式"
        echo "     可以使用 --mode $RECOMMENDED_MODE 参数指定推理模式"
    fi
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 处理模式参数
FINAL_MODE="$MODE"
if [ -n "$RECOMMENDED_MODE" ] && [ "$RECOMMENDED_MODE" != "auto" ]; then
    # 如果有推荐模式且用户没有明确指定模式，则使用推荐模式
    if [ "$MODE" = "auto" ]; then
        FINAL_MODE="$RECOMMENDED_MODE"
        echo "使用推荐的推理模式: $FINAL_MODE"
    elif [ "$MODE" != "$RECOMMENDED_MODE" ]; then
        echo "警告: 当前模式 ($MODE) 与系统推荐模式 ($RECOMMENDED_MODE) 不一致"
    fi
else
    FINAL_MODE="$MODE"
fi

# 运行OCR处理
echo "处理文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "推理模式: $FINAL_MODE"

python3 -m cli.main "$INPUT_FILE" -o "$OUTPUT_DIR" -m "$FINAL_MODE"

echo "========================================="
echo "  处理完成"
echo "  结果保存在: $OUTPUT_DIR"
echo "========================================="