# DeepSeek-OCR-CLI

基于视觉编码器与大语言模型的光学字符识别系统命令行工具

## 项目介绍

DeepSeek-OCR-CLI 是一个基于视觉编码器与大语言模型的光学字符识别系统命令行工具。本项目基于 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 项目改造，提供了增强功能，增加了对多种文档格式的支持，包括 Word、PPT、Excel 等，并提供统一的命令行接口和Web界面进行处理。

### 核心特性

- **多格式支持**：支持PDF、Word、PPT、Excel、图像等多种文档格式
- **双引擎支持**：支持vLLM和Transformers两种推理引擎
- **智能模式切换**：自动检测硬件环境并选择最优推理引擎
- **Web界面**：提供友好的Web操作界面
- **高精度识别**：基于DeepSeek-OCR模型，支持复杂文档布局分析
- **模块化架构**：采用工厂模式设计，vLLM和Transformers引擎解耦，易于扩展和维护

## 功能特性

### 核心OCR功能
- 文档转Markdown
- 图像OCR识别
- 图表内容解析
- 自由文本识别
- 文本定位与高亮

### 增强功能
- 支持 Word (.doc, .docx)、PPT (.ppt, .pptx)、Excel (.xls, .xlsx) 等办公文档格式
- 自动将办公文档转换为PDF，再转换为图像进行OCR处理
- 统一的命令行接口，简化使用流程
- 支持vLLM和Transformers两种推理后端
- 提供Web界面进行可视化操作
- 模块化架构设计，易于扩展和维护

### 支持的文件格式
- Microsoft Office: .doc, .docx, .ppt, .pptx, .xls, .xlsx
- PDF: .pdf
- 图像: .jpg, .jpeg, .png

## 系统要求

### 环境要求
- Python 3.10 或更高版本
- 支持多种硬件加速平台：
  - NVIDIA GPU (CUDA 11.8 + PyTorch 2.4.1)
  - AMD GPU (ROCm + PyTorch)
  - Apple Silicon (MPS + PyTorch 2.4.1)
  - DCU (Direct Compute Unit)
  - CPU (通用处理器)

### Apple Silicon (MPS) 支持说明

在Apple Silicon设备上，项目会自动使用Transformers引擎进行推理，不依赖vLLM和flash_attn库。这是因为在MPS环境下：

- **vLLM不可用**：vLLM目前不支持Apple Silicon设备
- **flash_attn不可用**：flash_attn库在MPS环境下无法编译安装

项目会自动检测MPS环境并选择合适的推理引擎，确保在Mac设备上也能正常运行OCR功能。

### 安装LibreOffice
为了支持办公文档格式转换（Word、PPT、Excel等），需要安装LibreOffice：

- Ubuntu/Debian: `sudo apt-get install libreoffice`
- CentOS/RHEL: `sudo yum install libreoffice`
- macOS: `brew install --cask libreoffice`
- Windows: 从官网下载安装

注意：LibreOffice是一个独立的应用程序，不是Python包，因此不会通过pip或uv安装。

## 安装指南

### 1. 克隆项目

```bash
git clone https://github.com/wcpsoft/deepseek-ocr-cli.git
cd deepseek-ocr-cli
```

### 2. 创建并激活虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 3. 安装依赖

项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理器，提供更快的依赖安装速度。

```bash
# 安装uv (如果尚未安装)
pip install uv

# 自动检测硬件平台并安装相应依赖
./dev/setup.sh
```

或者手动选择平台安装:

```bash
# 安装核心依赖
uv pip install -e .

# 根据硬件平台选择安装相应的依赖
# NVIDIA GPU
uv pip install -r requirements/requirements-nvidia.txt

# AMD GPU
uv pip install -r requirements/requirements-amd.txt

# Apple Silicon
uv pip install -r requirements/requirements-mps.txt

# DCU
uv pip install -r requirements/requirements-dcu.txt

# CPU-only
uv pip install -r requirements/requirements-cpu.txt
```

### 4. 验证安装

```bash
# 验证核心依赖
python -c "import fitz; import img2pdf; print('核心依赖验证成功')"

# 测试基本功能
./dev/debug.py
```

## 快速使用

### 命令行模式

```bash
# 查看帮助信息
deepseek-ocr --help

# 处理Word文档
deepseek-ocr document.docx -o output_dir

# 处理PDF文档
deepseek-ocr document.pdf -o output_dir

# 处理图像文件
deepseek-ocr image.jpg -o output_dir

# 使用Transformers后端
deepseek-ocr document.docx -o output_dir --mode transformers

# 使用vLLM后端（需要安装vLLM，不支持MPS环境）
deepseek-ocr document.docx -o output_dir --mode vllm

# 自动选择后端（默认，MPS环境会自动选择Transformers）
deepseek-ocr document.docx -o output_dir --mode auto

# 自定义提示词
deepseek-ocr image.jpg -o output_dir --prompt "<image>\nOCR this image."

# 使用统一调试脚本
deepseek-ocr-debug

# 或者直接运行脚本
./dev/debug.py
```

### Web API模式

项目支持通过Web API进行OCR处理，提供图形界面和RESTful API接口：

```bash
# 启动Web API服务
python src/app.py

# 或使用uvicorn直接启动
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

# 访问Web界面
# 打开浏览器访问 http://localhost:8000

# 使用API接口
# POST /api/ocr/pdf - 上传PDF文件进行OCR处理
# GET /api/download/{job_id}/{file_type} - 下载处理结果
```

### Python API

```python
from src.core.factory.engine_factory import get_engine

# 创建vLLM引擎
vllm_engine = get_engine("vllm")

# 创建Transformers引擎
transformers_engine = get_engine("transformers")

# 初始化引擎
vllm_engine.initialize()
```

## 代码质量保证

本项目采用严格的代码质量控制措施，使用多种工具确保代码质量和一致性：

### 代码质量工具链

1. **[black](https://github.com/psf/black)** - 代码格式化工具，确保代码风格统一
2. **[isort](https://github.com/PyCQA/isort)** - 导入语句排序和格式化工具
3. **[ruff](https://github.com/charliermarsh/ruff)** - 高性能Python代码检查工具
4. **[mypy](https://github.com/python/mypy)** - 静态类型检查工具

### 统一配置和执行顺序

所有工具均已统一配置，确保无冲突并按最佳实践顺序执行：

1. **isort** - 首先处理导入语句排序
2. **black** - 然后进行代码格式化
3. **ruff** - 接着进行代码质量检查和修复
4. **mypy** - 最后进行类型检查

### 使用方法

#### 自动化脚本

项目提供了统一的自动化脚本用于代码格式化和质量检查：

```bash
# 代码质量检查（检查但不修复问题）
./dev/auto_check_code_style.sh --check-only

# 自动修复代码质量问题
./dev/auto_check_code_style.sh --fix

# 默认行为：检查问题并在发现问题时提示如何修复
./dev/auto_check_code_style.sh
```

#### Pre-commit钩子

项目支持pre-commit钩子，在代码提交前自动执行质量检查：

```bash
# 安装pre-commit钩子
pre-commit install

# 手动运行所有pre-commit检查
pre-commit run --all-files
```

### 配置文件

所有工具的配置均在 [pyproject.toml](file:///Users/zhangruixiang/aiworkspace/DeepSeek-OCR/pyproject.toml) 文件中统一管理，确保各工具间的兼容性：

- 行长度统一设置为120字符
- Python目标版本统一设置为3.10
- isort使用black配置文件确保兼容性
- ruff启用全面的检查规则集
- mypy配置平衡严格性和实用性

通过这套完整的代码质量保证体系，我们确保项目代码的一致性、可读性和可维护性。