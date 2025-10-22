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
from cli.document_processor import DocumentProcessor

# 创建处理器
processor = DocumentProcessor(mode="auto")  # 在MPS环境下会自动选择Transformers模式

# 处理文档
processor.process("input.docx", "output_dir")
```

## 分辨率模式

模型支持以下分辨率模式：
- 固定分辨率：
  - Tiny: 512×512 （64 vision tokens）
  - Small: 640×640 （100 vision tokens）
  - Base: 1024×1024 （256 vision tokens）
  - Large: 1280×1280 （400 vision tokens）
- 动态分辨率：
  - Gundam: n×640×640 + 1×1024×1024

## 提示词示例

```
# 文档转Markdown（默认）
"<image>\n<|grounding|>Convert the document to markdown."

# 图像OCR
"<image>\n<|grounding|>OCR this image."

# 自由OCR（无布局）
"<image>\nFree OCR."

# 图表解析
"<image>\nParse the figure."

# 图像描述
"<image>\nDescribe this image in detail."

# 文本定位
"<image>\nLocate <|ref|>xxxx<|/ref|> in the image."

# 其他常用提示词
"<image>\n先天下之忧而忧"

# 模型文件中的提示词示例
"<image>\n<|grounding|>Given the layout of the image."
"<image>\nExtract the text in the image."
"<image>\nExtract all information from this image and convert them into markdown format."
```

## 项目结构

```
DeepSeek-OCR/
├── cli/                   # 命令行工具
│   ├── __init__.py
│   ├── main.py            # 主入口
│   ├── document_processor.py  # 文档处理核心
│   ├── model_manager.py   # 模型管理
│   ├── pdf_converter.py   # PDF转换器
│   ├── download_models.py # 模型下载
│   ├── example_usage.py   # 使用示例
│   └── test_cli.py        # CLI测试
├── src/                   # 源代码目录
│   ├── __init__.py
│   ├── app.py             # Web API服务入口
│   ├── cli.py             # CLI入口
│   └── core/              # 核心算法实现
│       ├── __init__.py
│       ├── config.py          # 配置文件
│       ├── deepseek_ocr.py    # vLLM模型实现
│       ├── api/               # Web API服务
│       ├── process/           # 处理模块
│       └── deepencoder/       # 编码器模块
├── dev/                   # 开发工具
│   ├── run_tests.sh       # 自动化测试脚本
│   ├── debug.py           # 统一调试脚本
│   ├── detect_gpu.py      # GPU检测工具
│   ├── run.sh             # 运行脚本
│   └── setup.sh           # 安装脚本
├── tests/                 # 测试目录
│   └── test_cli.py        # CLI测试脚本
├── README.md              # 项目说明与使用指南
└── pyproject.toml         # 项目配置
```

## 测试

运行单元测试：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_cli.py

# 运行端到端测试（使用samples目录中的示例文件）
python -m pytest tests/test_e2e.py

# 使用开发脚本运行测试
./dev/run_tests.sh
```

## 许可证

本项目采用 Apache 2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 致谢

我们感谢以下项目提供的宝贵模型和想法：
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)