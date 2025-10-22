# DeepSeek-OCR-cli

## 项目介绍

DeepSeek-OCR-cli 是一个基于视觉编码器与大语言模型的光学字符识别系统命令行工具。本项目基于 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 项目基础上改造，让用户直接可以使用，提供了增强功能，增加了对多种文档格式的支持，包括 Word、PPT、Excel 等，并提供统一的命令行接口进行处理。

**项目作者：Rxzhang**

## 功能特性

### 核心功能
- 文档转Markdown
- 图像OCR
- 图表解析
- 自由描述
- 文本定位

### 增强功能
- 支持 Word (.doc, .docx)、PPT (.ppt, .pptx)、Excel (.xls, .xlsx) 等办公文档格式
- 自动将办公文档转换为PDF，再转换为图像进行OCR处理
- 统一的命令行接口，简化使用流程
- 支持vLLM和Transformers两种推理后端

### 支持的文件格式
- Microsoft Office: .doc, .docx, .ppt, .pptx, .xls, .xlsx
- PDF: .pdf
- 图像: .jpg, .jpeg, .png

## 部署要求

### 环境要求
- Python 3.10 或更高版本
- 支持多种硬件加速平台：
  - NVIDIA GPU (CUDA 11.8 + PyTorch 2.4.1)
  - AMD GPU (ROCm + PyTorch ://download.pytorch.org/whl/rocm6.1
  - Apple Silicon (MPS + PyTorch 2.4.1)
  - DCU (Direct Compute Unit)
- 支持的操作系统：Linux、Windows、macOS

### 安装LibreOffice
为了支持办公文档格式转换（Word、PPT、Excel等），需要安装LibreOffice：

- Ubuntu/Debian: `sudo apt-get install libreoffice`
- CentOS/RHEL: `sudo yum install libreoffice`
- macOS: `brew install --cask libreoffice`
- Windows: 从官网下载安装

注意：LibreOffice是一个独立的应用程序，不是Python包，因此不会通过pip或uv安装。

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
│   └── core/              # 核心算法实现
│       ├── __init__.py
│       ├── config.py          # 配置文件
│       ├── deepseek_ocr.py    # vLLM模型实现
│       ├── process/           # 处理模块
│       └── deepencoder/       # 编码器模块
├── dev/                   # 开发工具
│   ├── run_tests.sh       # 自动化测试脚本
│   ├── debug_ocr.py       # 用户调试脚本
│   ├── detect_gpu.py      # GPU检测工具
│   ├── run.sh             # 运行脚本
│   └── setup.sh           # 安装脚本
├── tests/                 # 测试目录
│   └── test_cli.py        # CLI测试脚本
├── README.md              # 项目说明与使用指南
└── pyproject.toml         # 项目配置
```

## 快速使用

### 安装指南

```bash
# 克隆项目
git clone https://github.com/wcpsoft/deepseek-ocr-cli.git
cd deepseek-ocr-cli

# 使用一键安装脚本（推荐）
./dev/setup.sh

# 或使用开发脚本自动安装
./dev/run.sh samples/1.pdf output
# 或运行测试
./dev/run_tests.sh
```

开发脚本会自动检测系统中的GPU类型并安装相应的依赖包。

### 手动安装步骤

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装基础依赖
uv pip install -e .

# 安装开发依赖
uv pip install -e '.[dev]'

# 根据硬件平台安装相应的依赖：
# NVIDIA GPU:
uv pip install -r requirements/requirements-nvidia.txt
# AMD GPU:
uv pip install -r requirements/requirements-amd.txt
# Apple Silicon (MPS):
uv pip install -r requirements/requirements-mps.txt
# DCU:
uv pip install -r requirements/requirements-dcu.txt
# CPU only:
uv pip install -r requirements/requirements-cpu.txt

# 如果需要从ModelScope下载模型，安装额外依赖：
uv pip install -e '.[modelscope]'
```

### 重要依赖说明

项目依赖 PyMuPDF (fitz) 和 img2pdf 库来处理 PDF 文档和图像转换。这些依赖已添加到所有平台的 requirements 文件中。如果遇到 `ModuleNotFoundError: No module named 'fitz'` 错误，请确保已正确安装依赖：

```bash
# 重新安装所有依赖
uv pip install -e .
uv pip install -r requirements/requirements-<your-platform>.txt
```

### 命令行工具

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
```

### Python API

```python
from cli.document_processor import DocumentProcessor

# 创建处理器
processor = DocumentProcessor(mode="auto")  # 在MPS环境下会自动选择Transformers模式

# 处理文档
processor.process("input.docx", "output_dir")
```

### 开发脚本

项目提供了便捷的开发脚本：

```bash
# 运行测试
./dev/run_tests.sh

# 快速处理文件
./dev/run.sh input.pdf output_dir

# 调试OCR处理
./dev/debug_ocr.py
```

### 下载模型

```bash
# 下载模型到本地
deepseek-ocr-download
```

### 调试工具

项目提供了便捷的调试脚本：

```bash
# 使用调试脚本快速测试OCR功能
python dev/debug_ocr.py

# 或者使用安装后的命令
deepseek-ocr-debug
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