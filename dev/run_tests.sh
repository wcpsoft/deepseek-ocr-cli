#!/bin/bash

# 引入公共函数
source "$(dirname "$0")/common.sh"

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 测试脚本"
echo "========================================="

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

log_info "安装开发依赖..."
# 保持与setup.sh一致的依赖安装逻辑
install_all_deps

# 运行代码质量检查
log_info "运行代码质量检查..."
if ! ./dev/auto_check_code_style.sh --check-only; then
    log_error "代码质量检查失败，终止测试"
    exit 1
fi

# 运行单元测试
log_info "运行单元测试..."
.venv/bin/python -m pytest tests/unit/ -v

# 运行集成测试
log_info "运行集成测试..."
.venv/bin/python -m pytest tests/integration/ -v

# 运行端到端测试
log_info "运行端到端测试..."

# 检查模型是否存在且完整
if [ -d "models" ]; then
    # 检查默认模型目录是否存在
    if [ -d "models/deepseek-ocr" ]; then
        # 检查模型必要文件是否存在
        if [ -f "models/deepseek-ocr/config.json" ]; then
            # 检查是否存在模型权重文件
            if ls models/deepseek-ocr/pytorch_model*.bin 1> /dev/null 2>&1 || ls models/deepseek-ocr/*.safetensors 1> /dev/null 2>&1 || [ -f "models/deepseek-ocr/model.safetensors.index.json" ]; then
                log_info "模型已存在，开始运行端到端测试..."
                if [ -d "samples" ]; then
                    .venv/bin/python -m pytest tests/e2e/ -v
                else
                    log_warn "跳过端到端测试：未找到samples目录"
                    log_success "所有单元测试和集成测试已通过！"
                fi
            else
                log_warn "注意: 端到端测试需要下载模型文件才能正常运行"
                echo "请先运行 'deepseek-ocr --download-models' 下载模型"
            fi
        else
            log_warn "注意: 端到端测试需要下载模型文件才能正常运行"
            echo "请先运行 'deepseek-ocr --download-models' 下载模型"
        fi
    else
        log_warn "注意: 端到端测试需要下载模型文件才能正常运行"
        echo "请先运行 'deepseek-ocr --download-models' 下载模型"
    fi
else
    log_warn "注意: 端到端测试需要下载模型文件才能正常运行"
    echo "请先运行 'deepseek-ocr --download-models' 下载模型"
fi

echo "========================================="
echo "  所有测试完成"
echo "========================================="
echo ""
echo "提示: 如果端到端测试因模型依赖问题失败，请确保已安装正确的Transformers版本"
echo "      并下载了DeepSeek-OCR模型文件到models目录"