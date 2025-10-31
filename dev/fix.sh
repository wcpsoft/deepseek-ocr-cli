#!/bin/bash

# 引入公共函数
source "$(dirname "$0")/common.sh"

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 代码自动修复脚本"
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

# 安装格式化工具
install_quality_tools

# 运行代码自动修复
log_info "运行代码自动修复..."

# 检查目录是否存在（排除template目录）
SRC_DIRS=""
[ -d "src" ] && SRC_DIRS="$SRC_DIRS src/"
[ -d "tests" ] && SRC_DIRS="$SRC_DIRS tests/"
[ -d "dev" ] && SRC_DIRS="$SRC_DIRS dev/"

if [ -z "$SRC_DIRS" ]; then
    log_warn "未找到源代码目录"
    echo "========================================="
    echo "  代码自动修复完成"
    echo "========================================="
    exit 0
fi

# 1. isort（导入排序）
log_info "运行isort进行导入排序..."
python -m isort $SRC_DIRS --skip template/

# 2. black（代码格式化）
log_info "运行black进行代码格式化..."
python -m black $SRC_DIRS --exclude template/

# 3. ruff（代码检查和修复）
log_info "运行ruff进行代码检查和自动修复..."
python -m ruff check --fix $SRC_DIRS --exclude template/

log_success "代码自动修复完成"

echo "========================================="
echo "  代码自动修复完成"
echo "========================================="