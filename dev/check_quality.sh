#!/bin/bash

# 引入公共函数
source "$(dirname "$0")/common.sh"

set -e  # 遇到错误时退出

echo "========================================="
echo "  DeepSeek OCR CLI 代码质量检查脚本"
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

# 安装质量检查工具
install_quality_tools

# 运行代码质量检查
if run_quality_check; then
    echo "========================================="
    echo "  所有代码质量检查通过"
    echo "========================================="
else
    echo "========================================="
    echo "  代码质量检查失败"
    echo "========================================="
    exit 1
fi