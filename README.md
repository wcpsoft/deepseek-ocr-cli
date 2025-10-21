<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>


<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>



<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>📥 Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 Paper Link</b></a> |
  <a href="DeepSeek_OCR_paper.pdf"><b>📄 Arxiv Paper Link</b></a> |
</p>

<h2>
<p align="center">
  <a href="">DeepSeek-OCR-cli: Contexts Optical Compression CLI Tool</a>
</p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 1000px" align=center>
</p>
<p align="center">
<a href="">Explore the boundaries of visual-text compression.</a>       
</p>

## 项目介绍

DeepSeek-OCR-cli 是一个基于视觉编码器与大语言模型的光学字符识别系统命令行工具。本项目基于 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 项目基础上改造，让用户直接可以使用，提供了增强功能，增加了对多种文档格式的支持，包括 Word、PPT、Excel 等，并提供统一的命令行接口进行处理。

**项目作者：Rxzhang**

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
├── tests/                 # 测试目录
│   └── test_cli.py        # CLI测试脚本
├── dev/                   # 开发工具
│   ├── test.sh            # 测试脚本
│   └── run.sh             # 运行脚本
├── README.md              # 项目说明与使用指南
└── pyproject.toml         # 项目配置
```

## 环境要求

- Python 3.10 或更高版本
- 支持多种硬件加速平台：
  - NVIDIA GPU (CUDA 11.8 + PyTorch 2.4.1)
  - AMD GPU (ROCm + PyTorch ://download.pytorch.org/whl/rocm6.1
  - Apple Silicon (MPS + PyTorch 2.4.1)
  - DCU (Direct Compute Unit)
- 支持的操作系统：Linux、Windows、macOS

## 安装指南

```
# 克隆项目
git clone https://github.com/wcpsoft/deepseek-ocr-cli.git
cd deepseek-ocr-cli

# 使用开发脚本自动安装（推荐）
./dev/run.sh samples/1.pdf output
# 或运行测试
./dev/test.sh
```

开发脚本会自动检测系统中的GPU类型并安装相应的依赖包。

### 手动安装步骤

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 根据硬件平台安装相应的依赖：
# NVIDIA GPU:
uv sync --extra nvidia
# AMD GPU:
uv sync --extra amd
# CPU only:
uv sync --extra cpu
# 默认安装（自动选择合适的PyTorch版本）:
uv sync

# 如果需要从ModelScope下载模型，安装额外依赖：
uv sync --extra modelscope
```

### GPU特定依赖说明

项目通过 `pyproject.toml` 中的可选依赖组管理不同GPU平台的依赖：

- `[nvidia]`: NVIDIA GPU (CUDA 11.8)
- `[amd]`: AMD GPU (ROCm 6.1)
- `[cpu]`: CPU only 版本
- `[modelscope]`: ModelScope支持

> ⚠️ **重要提示**: `[nvidia]`、`[amd]` 和 `[cpu]` 这些依赖组是互斥的，因为它们包含不同版本的 PyTorch。
> 在同一环境中只能安装其中一个依赖组。开发脚本会自动检测硬件类型并安装相应的依赖组。

### 安装LibreOffice

为了支持办公文档格式转换（Word、PPT、Excel等），需要安装LibreOffice：

- Ubuntu/Debian: `sudo apt-get install libreoffice`
- CentOS/RHEL: `sudo yum install libreoffice`
- macOS: `brew install --cask libreoffice`
- Windows: 从官网下载安装

注意：LibreOffice是一个独立的应用程序，不是Python包，因此不会通过pip或uv安装。

### 3. 安装项目依赖

#### 安装项目
```bash
# 安装核心依赖
uv pip install .

# 如果需要vLLM支持
uv pip install .[vllm]

# 如果需要开发依赖
uv pip install .[dev]
```

## 使用方法

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

# 使用vLLM后端（需要安装vLLM）
deepseek-ocr document.docx -o output_dir --mode vllm

# 自动选择后端（默认）
deepseek-ocr document.docx -o output_dir --mode auto

# 自定义提示词
deepseek-ocr image.jpg -o output_dir --prompt "<image>\nOCR this image."
```

### 智能模式选择

DeepSeek OCR CLI支持智能模式选择：

- `auto`（默认）：自动检测系统中可用的推理引擎，优先使用vLLM（如果已安装），否则使用Transformers
- `vllm`：强制使用vLLM引擎（需要先安装vLLM）
- `transformers`：强制使用Transformers引擎

程序会在运行时显示实际使用的推理引擎，方便用户了解当前的处理方式。

### 开发脚本

项目提供了便捷的开发脚本：

```bash
# 运行测试
./dev/test.sh

# 快速处理文件
./dev/run.sh input.pdf output_dir
```

### 下载模型

```bash
# 下载模型到本地
deepseek-ocr-download
```

### Python API

```python
from cli.document_processor import DocumentProcessor

# 创建处理器
processor = DocumentProcessor(mode="vllm")

# 处理文档
processor.process("input.docx", "output_dir")
```

## 支持的文件格式

- Microsoft Office: .doc, .docx, .ppt, .pptx, .xls, .xlsx
- PDF: .pdf
- 图像: .jpg, .jpeg, .png

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
# 文档转Markdown
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
./dev/test.sh
```

## 许可证

本项目采用 Apache 2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 致谢

我们感谢以下项目提供的宝贵模型和想法：
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 

# DeepSeek OCR CLI

基于视觉编码器与大语言模型的光学字符识别系统命令行工具

## 项目简介

DeepSeek OCR CLI 是一个功能强大的命令行工具，用于处理各种文档格式（PDF、Word、PPT、XLSX等）并执行光学字符识别。该工具支持多种硬件加速平台，包括NVIDIA GPU、AMD GPU、Apple Silicon和DCU。

## 功能特性

- 支持多种文档格式处理：PDF、Word、PPT、XLSX等
- 自动文档转换流程：文档 → PDF → 图像 → OCR识别
- 双推理引擎支持：Transformers和vLLM
- 多平台硬件加速：NVIDIA、AMD、Apple Silicon、DCU
- 自动GPU检测和依赖安装
- 模型自动下载和管理（支持Hugging Face和ModelScope）
- 智能模型文件过滤（默认只下载运行必需的文件）
- 命令行界面，易于集成到自动化流程中
- 完整的单元测试和端到端测试覆盖

## 项目结构

```
deepseek-ocr-cli/
├── cli/                 # 命令行接口模块
│   ├── __init__.py
│   ├── main.py          # 主程序入口
│   ├── document_processor.py  # 文档处理核心逻辑
│   ├── pdf_converter.py       # PDF转换器
│   ├── model_manager.py       # 模型管理器
│   ├── download_models.py     # 模型下载工具
│   └── example_usage.py       # 使用示例
├── src/                 # 核心源代码
│   ├── __init__.py
│   └── core/            # 核心OCR实现
├── tests/               # 测试文件
│   ├── __init__.py
│   ├── conftest.py      # pytest配置文件
│   ├── unit/            # 单元测试
│   │   ├── test_cli_imports.py        # CLI模块导入测试
│   │   ├── test_model_manager.py      # 模型管理器测试
│   │   ├── test_model_download_filter.py  # 模型下载过滤测试
│   │   ├── test_project_config.py     # 项目配置测试
│   │   └── test_dependency_installation.py  # 依赖安装测试
│   ├── integration/     # 集成测试
│   │   └── test_gpu_detection_integration.py  # GPU检测集成测试
│   └── e2e/             # 端到端测试
│       ├── test_model_availability.py     # 模型可用性测试
│       └── test_document_processing.py    # 文档处理测试
├── dev/                 # 开发工具
│   ├── detect_gpu.py    # GPU自动检测脚本
│   ├── run.sh           # 运行脚本（自动检测GPU并安装依赖）
│   └── test.sh          # 测试脚本（自动检测GPU并安装依赖）
├── samples/             # 示例文件
├── assets/              # 资源文件
├── models/              # 模型文件（默认下载目录）
├── pyproject.toml       # 项目配置和依赖声明
├── setup.py             # 项目安装配置
├── README.md            # 项目说明文档
└── LICENSE              # 许可证文件
```

## 环境要求

- Python 3.10 或更高版本
- 支持多种硬件加速平台：
  - NVIDIA GPU (CUDA 11.8 + PyTorch 2.4.1)
  - AMD GPU (ROCm + PyTorch 2.4.1)
  - Apple Silicon (MPS + PyTorch 2.4.1)
  - DCU (Direct Compute Unit)
- 支持的操作系统：Linux、Windows、macOS

## 安装指南

```bash
# 克隆项目
git clone https://github.com/wcpsoft/deepseek-ocr-cli.git
cd deepseek-ocr-cli

# 使用开发脚本自动安装（推荐）
./dev/run.sh samples/1.pdf output
# 或运行测试
./dev/test.sh
```

开发脚本会自动检测系统中的GPU类型并安装相应的依赖包。

### 手动安装步骤

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 根据硬件平台安装相应的依赖：
# NVIDIA GPU:
uv sync --extra nvidia
# AMD GPU:
uv sync --extra amd
# CPU only:
uv sync --extra cpu
# 默认安装（自动选择合适的PyTorch版本）:
uv sync

# 如果需要从ModelScope下载模型，安装额外依赖：
uv sync --extra modelscope
```

### GPU特定依赖说明

项目通过 `pyproject.toml` 中的可选依赖组管理不同GPU平台的依赖：

- `[nvidia]`: NVIDIA GPU (CUDA 11.8)
- `[amd]`: AMD GPU (ROCm 6.1)
- `[cpu]`: CPU only 版本
- `[modelscope]`: ModelScope支持

> ⚠️ **重要提示**: `[nvidia]`、`[amd]` 和 `[cpu]` 这些依赖组是互斥的，因为它们包含不同版本的 PyTorch。
> 在同一环境中只能安装其中一个依赖组。开发脚本会自动检测硬件类型并安装相应的依赖组。

## 项目配置测试

项目包含专门的测试来验证 `pyproject.toml` 配置的正确性：

```bash
# 运行项目配置测试
python3 -m pytest tests/unit/test_project_config.py -v

# 运行依赖安装测试
python3 -m pytest tests/unit/test_dependency_installation.py -v

# 运行GPU检测集成测试
python3 -m pytest tests/integration/test_gpu_detection_integration.py -v
```

这些测试确保：
- `pyproject.toml` 文件格式正确
- 所有必需的依赖组都已定义
- GPU检测脚本与依赖组配置一致
- 安装命令生成正确

## 模型管理

### 自动下载模型

```bash
# 下载默认模型到./models目录（默认只下载运行必需的文件）
deepseek-ocr-download

# 下载到自定义目录
deepseek-ocr-download -d /path/to/models

# 下载特定模型
deepseek-ocr-download -m deepseek-ocr

# 强制重新下载
deepseek-ocr-download -f

# 下载完整模型（包括文档、示例等文件）
deepseek-ocr-download --full-download

# 添加自定义模型（支持Hugging Face和ModelScope）
deepseek-ocr-download --add-custom my-model my-hf-username/my-model-repo huggingface
deepseek-ocr-download --add-custom my-ms-model my-ms-model-id modelscope

# 列出自定义模型
deepseek-ocr-download --list-custom

# 列出已下载模型
deepseek-ocr-download --list-downloaded
```

### 模型智能过滤

默认情况下，模型下载器会智能过滤文件，只下载运行OCR任务所必需的文件：
- 忽略文档文件（*.md, README*, LICENSE*等）
- 忽略示例文件和图片（assets/*, examples/*等）
- 忽略脚本和测试文件（scripts/*, tests/*等）
- 只下载模型权重、配置文件和必要的支持文件

这样可以显著减少下载时间和磁盘空间占用。

### 模型目录结构

```
models/
├── deepseek-ocr/           # 默认DeepSeek OCR模型 (deepseek-ai/DeepSeek-OCR)
│   ├── config.json
│   ├── pytorch_model.bin 或 model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── ... (其他运行必需的文件)
├── model_config.json       # 模型配置文件
└── ...                     # 其他自定义模型
```

模型配置文件 `model_config.json` 记录了已下载模型的信息和自定义模型的映射关系。

## 测试

项目包含完整的测试套件，确保功能正确性：

```bash
# 运行所有测试
./dev/test.sh

# 或手动运行特定测试
python3 -m pytest tests/unit/ -v          # 运行单元测试
python3 -m pytest tests/integration/ -v   # 运行集成测试
python3 -m pytest tests/e2e/ -v           # 运行端到端测试
python3 -m pytest tests/ -v               # 运行所有测试

# 运行特定测试文件
python3 -m pytest tests/unit/test_cli_imports.py -v
python3 -m pytest tests/integration/test_gpu_detection_integration.py -v
```

测试套件包括：
- **单元测试**：验证各个模块的基本功能
- **集成测试**：验证模块间的集成和配置正确性
- **端到端测试**：验证完整的文档处理流程
