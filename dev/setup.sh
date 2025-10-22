#!/bin/bash

# 引入公共函数
source "$(dirname "$0")/common.sh"

# 检查是否在虚拟环境中
if [ "$VIRTUAL_ENV" = "" ]; then
    log_warn "未检测到虚拟环境"
    echo "建议先创建并激活虚拟环境:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    read -p "是否继续安装? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查项目根目录
check_project_root

# 检查Python环境
check_python

# 检查uv是否已安装
if ! check_uv; then
    install_uv
fi

log_info "开始安装DeepSeek-OCR项目依赖..."

# 安装所有依赖
install_all_deps

# 验证安装
if verify_core_deps; then
    log_success "依赖安装完成!"
    echo "安装完成! 可以运行以下命令测试:"
    echo "  ./dev/debug_ocr.py"
else
    log_error "依赖安装验证失败"
    exit 1
fi
