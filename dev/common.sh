#!/bin/bash

# 公共函数和变量定义
# 用于dev目录下的所有脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 检查是否在项目根目录
check_project_root() {
    if [ ! -f "pyproject.toml" ]; then
        log_error "请在项目根目录运行此脚本"
        exit 1
    fi
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "未找到Python3"
        exit 1
    fi
}

# 检查uv是否已安装
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_warn "未找到uv命令"
        return 1
    fi
    return 0
}

# 安装uv
install_uv() {
    log_info "安装uv..."
    python3 -m pip install uv
}

# 创建虚拟环境（如果不存在）
create_venv() {
    if [ ! -d ".venv" ]; then
        log_info "创建虚拟环境..."
        uv venv
    fi
}

# 激活虚拟环境
activate_venv() {
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        log_error "虚拟环境不存在，请先创建"
        exit 1
    fi
}

# 检测GPU类型
detect_gpu() {
    python3 dev/detect_gpu.py | grep "GPU_TYPE=" | cut -d'=' -f2
}

# 获取额外依赖后缀
get_extra_suffix() {
    python3 dev/detect_gpu.py | grep "EXTRA_SUFFIX=" | cut -d'=' -f2
}

# 安装核心依赖
install_core_deps() {
    log_info "安装核心依赖..."
    uv pip install -e .
}

# 安装开发依赖
install_dev_deps() {
    log_info "安装开发依赖..."
    uv pip install -e .[dev]
}

# 根据GPU类型安装平台特定依赖
install_platform_deps() {
    local gpu_type=$1
    
    case $gpu_type in
        "nvidia")
            log_info "安装NVIDIA GPU特定依赖..."
            uv pip install -r requirements/requirements-nvidia.txt
            ;;
        "amd")
            log_info "安装AMD GPU特定依赖..."
            uv pip install -r requirements/requirements-amd.txt
            ;;
        "mps")
            log_info "安装Apple Silicon MPS特定依赖..."
            uv pip install -r requirements/requirements-mps.txt
            ;;
        "dcu")
            log_info "安装DCU特定依赖..."
            uv pip install -r requirements/requirements-dcu.txt
            ;;
        "cpu"|*)
            log_info "安装CPU特定依赖..."
            uv pip install -r requirements/requirements-cpu.txt
            ;;
    esac
}

# 安装所有依赖（核心+开发+平台特定）
install_all_deps() {
    local gpu_type=$(detect_gpu)
    
    log_info "检测到GPU类型: $gpu_type"
    
    # 安装核心依赖
    install_core_deps
    
    # 安装开发依赖
    install_dev_deps
    
    # 安装平台特定依赖
    install_platform_deps "$gpu_type"
}

# 验证核心依赖安装
verify_core_deps() {
    log_info "验证核心依赖安装..."
    python -c "import fitz; import img2pdf; print('核心依赖验证成功')" 2>/dev/null || {
        log_error "核心依赖验证失败"
        return 1
    }
    return 0
}