#!/bin/bash

# 引入公共函数
source "$(dirname "$0")/common.sh"

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 代码质量检查与修复脚本"
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

# 解析命令行参数
CHECK_ONLY=false
FIX_ISSUES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --fix)
            FIX_ISSUES=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --check-only  只检查代码质量问题，不进行修复"
            echo "  --fix         自动修复可修复的问题"
            echo "  --help, -h    显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# 运行代码质量检查和修复
log_info "运行代码质量检查和修复..."

# 检查目录是否存在（排除template目录）
SRC_DIRS=""
[ -d "src" ] && SRC_DIRS="$SRC_DIRS src/"
[ -d "tests" ] && SRC_DIRS="$SRC_DIRS tests/"
[ -d "dev" ] && SRC_DIRS="$SRC_DIRS dev/"

if [ -z "$SRC_DIRS" ]; then
    log_warn "未找到源代码目录"
    echo "========================================="
    echo "  代码质量检查与修复完成"
    echo "========================================="
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    # 只检查不修复
    if ! run_quality_check; then
        log_error "代码质量检查失败"
        exit 1
    fi
elif [ "$FIX_ISSUES" = true ]; then
    # 自动修复问题
    log_info "运行isort进行导入排序..."
    python -m isort $SRC_DIRS --skip template/

    log_info "运行black进行代码格式化..."
    python -m black $SRC_DIRS --exclude template/

    log_info "运行ruff进行代码检查和自动修复..."
    python -m ruff check --fix $SRC_DIRS --exclude template/

    log_success "代码自动修复完成"
else
    # 默认行为：先检查，如果有问题则提示用户选择是否修复
    log_info "运行代码质量检查..."
    
    # 1. isort检查（不修改文件）
    log_info "运行isort进行导入排序检查..."
    if ! python -m isort --check-only $SRC_DIRS --skip template/; then
        log_error "isort检查失败"
        echo "运行 './dev/auto_check_code_style.sh --fix' 可自动修复此问题"
        exit 1
    fi
    
    # 2. black检查（不修改文件）
    log_info "运行black进行代码格式检查..."
    if ! python -m black --check $SRC_DIRS --exclude template/; then
        log_error "black检查失败"
        echo "运行 './dev/auto_check_code_style.sh --fix' 可自动修复此问题"
        exit 1
    fi
    
    # 3. ruff检查
    log_info "运行ruff进行代码质量检查..."
    if ! python -m ruff check $SRC_DIRS --exclude template/; then
        log_error "ruff检查失败"
        echo "运行 './dev/auto_check_code_style.sh --fix' 可自动修复此问题"
        exit 1
    fi
    
    # 4. mypy类型检查（只检查存在的目录，排除template目录）
    MY_PY_DIRS=""
    [ -d "src" ] && MY_PY_DIRS="$MY_PY_DIRS src/"
    [ -d "dev" ] && MY_PY_DIRS="$MY_PY_DIRS dev/"
    
    if [ -n "$MY_PY_DIRS" ]; then
        log_info "运行mypy进行类型检查..."
        if ! python -m mypy $MY_PY_DIRS --exclude template/; then
            log_error "mypy类型检查失败"
            exit 1
        fi
    else
        log_warn "未找到Python源代码目录进行类型检查"
    fi
    
    log_success "所有代码质量检查通过"
fi

echo "========================================="
echo "  代码质量检查与修复完成"
echo "========================================="