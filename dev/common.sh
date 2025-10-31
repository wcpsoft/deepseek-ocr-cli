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
            # 先安装PyTorch相关依赖
            log_info "安装PyTorch..."
            uv pip install torch==2.5.1 torchvision>=0.20,<0.21 torchaudio==2.5.1
            # 再安装flash-attn
            log_info "安装flash-attn..."
            uv pip install flash-attn==2.7.3 --no-build-isolation
            # 安装其他依赖
            uv pip install -r requirements/requirements-nvidia.txt --exclude torch torchvision torchaudio flash-attn
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

# 安装代码质量检查工具
install_quality_tools() {
    log_info "安装代码质量检查工具..."
    uv pip install black isort ruff mypy pre-commit
}

# 运行代码格式化（按正确顺序）
run_format() {
    log_info "运行代码格式化..."
    
    # 检查目录是否存在（排除template目录）
    SRC_DIRS=""
    [ -d "src" ] && SRC_DIRS="$SRC_DIRS src/"
    [ -d "tests" ] && SRC_DIRS="$SRC_DIRS tests/"
    [ -d "dev" ] && SRC_DIRS="$SRC_DIRS dev/"
    
    # 按照最佳实践顺序执行格式化工具
    # 1. isort（导入排序）
    log_info "运行isort进行导入排序..."
    if [ -n "$SRC_DIRS" ]; then
        python -m isort $SRC_DIRS --skip template/
    else
        log_warn "未找到源代码目录"
    fi
    
    # 2. black（代码格式化）
    log_info "运行black进行代码格式化..."
    if [ -n "$SRC_DIRS" ]; then
        python -m black $SRC_DIRS --exclude template/
    else
        log_warn "未找到源代码目录"
    fi
    
    # 3. ruff（代码检查和修复）
    log_info "运行ruff进行代码检查和自动修复..."
    if [ -n "$SRC_DIRS" ]; then
        python -m ruff check --fix $SRC_DIRS --exclude template/
    else
        log_warn "未找到源代码目录"
    fi
    
    log_success "代码格式化完成"
}

# 运行代码质量检查（按正确顺序）
run_quality_check() {
    log_info "运行代码质量检查..."
    
    # 检查目录是否存在（排除template目录）
    SRC_DIRS=""
    [ -d "src" ] && SRC_DIRS="$SRC_DIRS src/"
    [ -d "tests" ] && SRC_DIRS="$SRC_DIRS tests/"
    [ -d "dev" ] && SRC_DIRS="$SRC_DIRS dev/"
    
    if [ -z "$SRC_DIRS" ]; then
        log_warn "未找到源代码目录"
        return 0
    fi
    
    # 按照最佳实践顺序执行检查工具
    # 1. isort检查（不修改文件）
    log_info "运行isort进行导入排序检查..."
    if ! python -m isort --check-only $SRC_DIRS --skip template/; then
        log_error "isort检查失败，请运行格式化脚本修复问题"
        return 1
    fi
    
    # 2. black检查（不修改文件）
    log_info "运行black进行代码格式检查..."
    if ! python -m black --check $SRC_DIRS --exclude template/; then
        log_error "black检查失败，请运行格式化脚本修复问题"
        return 1
    fi
    
    # 3. ruff检查
    log_info "运行ruff进行代码质量检查..."
    if ! python -m ruff check $SRC_DIRS --exclude template/; then
        log_error "ruff检查失败，可通过运行 'dev/format.sh' 脚本自动修复大部分问题"
        return 1
    fi
    
    # 4. mypy类型检查（只检查存在的目录，排除template目录）
    MY_PY_DIRS=""
    [ -d "src" ] && MY_PY_DIRS="$MY_PY_DIRS src/"
    [ -d "dev" ] && MY_PY_DIRS="$MY_PY_DIRS dev/"
    
    if [ -n "$MY_PY_DIRS" ]; then
        log_info "运行mypy进行类型检查..."
        if ! python -m mypy $MY_PY_DIRS --exclude template/; then
            log_error "mypy类型检查失败，请修复类型问题"
            return 1
        fi
    else
        log_warn "未找到Python源代码目录进行类型检查"
    fi
    
    log_success "所有代码质量检查通过"
    return 0
}
