#!/bin/bash

# 引入公共函数
source "$(dirname "$0")/common.sh"

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
DEBUG=false
IPDB=false
SKIP_QUALITY_CHECK=false
PROMPT="<image>\n<|grounding|>Convert the document to markdown."

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
        --debug)
            DEBUG=true
            shift
            ;;
        --ipdb)
            IPDB=true
            shift
            ;;
        --skip-quality-check)
            SKIP_QUALITY_CHECK=true
            shift
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        -*)
            log_error "未知选项: $1"
            echo "使用方法: ./dev/run.sh [--download-models] [--mode MODE] [--debug] [--ipdb] [--skip-quality-check] [--prompt PROMPT] [输入文件] [输出目录]"
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

# 检查项目根目录
check_project_root

# 检查Python环境
check_python

# 检查uv是否已安装
if ! check_uv; then
    install_uv
fi

# 创建虚拟环境（如果不存在）
create_venv

# 激活虚拟环境
activate_venv

# 自动检测GPU类型并安装相应依赖
log_info "检测GPU环境..."
GPU_TYPE=$(detect_gpu)
EXTRA_SUFFIX=$(get_extra_suffix)
RECOMMENDED_MODE=$(python3 dev/detect_gpu.py | grep "推荐的推理模式:" | cut -d' ' -f3)

log_info "安装项目依赖..."
# 使用更精确的依赖安装方式，避免跨平台依赖冲突
install_all_deps

# 检查是否需要下载模型
if [ "$DOWNLOAD_MODELS" = true ]; then
    log_info "下载模型..."
    python3 -m src.cli.download_models -m deepseek-ocr --force
    exit 0
fi

# 检查模型是否存在，如果不存在则下载默认模型
log_info "检查模型..."
if [ ! -d "models/deepseek-ocr" ]; then
    log_info "默认模型不存在，正在下载..."
    python3 -m src.cli.download_models -m deepseek-ocr
else
    log_info "默认模型已存在"
fi

# 如果没有提供输入文件，则退出
if [ -z "$INPUT_FILE" ]; then
    echo "使用方法: ./dev/run.sh [--download-models] [--mode MODE] [--debug] [--ipdb] <输入文件> [输出目录]"
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
    log_error "输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 处理模式参数
FINAL_MODE="$MODE"
if [ -n "$RECOMMENDED_MODE" ] && [ "$RECOMMENDED_MODE" != "auto" ]; then
    # 如果有推荐模式且用户没有明确指定模式，则使用推荐模式
    if [ "$MODE" = "auto" ]; then
        FINAL_MODE="$RECOMMENDED_MODE"
        log_info "使用推荐的推理模式: $FINAL_MODE"
    elif [ "$MODE" != "$RECOMMENDED_MODE" ]; then
        log_warn "当前模式 ($MODE) 与系统推荐模式 ($RECOMMENDED_MODE) 不一致"
    fi
else
    FINAL_MODE="$MODE"
fi

# 运行OCR处理
log_info "处理文件: $INPUT_FILE"
log_info "输出目录: $OUTPUT_DIR"
log_info "推理模式: $FINAL_MODE"

# 构建命令参数
ARGS=("$INPUT_FILE" -o "$OUTPUT_DIR" -m "$FINAL_MODE" --prompt "$PROMPT")
if [ "$DEBUG" = true ]; then
    ARGS+=(--debug)
fi

if [ "$IPDB" = true ]; then
    ARGS+=(--ipdb)
fi

# 检测MPS环境并设置环境变量
if [[ "$FINAL_MODE" == "transformers" ]]; then
    # 检测是否在MPS环境下
    if python3 -c "import torch; print(torch.backends.mps.is_available() and torch.backends.mps.is_built())" | grep -q "True"; then
        log_info "检测到MPS环境，设置CPU回退模式和内存管理"
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    fi
fi

python3 -m src.cli.main "${ARGS[@]}"

echo "========================================="
echo "  处理完成"
echo "  结果保存在: $OUTPUT_DIR"
echo "========================================="