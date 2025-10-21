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

### 测试脚本修复说明

测试脚本已经过修复，解决了以下问题：

1. 更新了`pyproject.toml`，将弃用的`tool.uv.dev-dependencies`替换为新的`dependency-groups.dev`格式
2. 修复了`test.sh`脚本，确保正确使用虚拟环境中的Python和pytest
3. 改进了错误处理机制，增强了代码的健壮性
4. 添加了对`samples`和`models`目录的检查，避免在缺少必要文件时运行端到端测试

现在运行`./dev/test.sh`将自动执行单元测试和集成测试，只有在检测到必要的示例文件和模型时才会运行端到端测试。

## 许可证

本项目采用 Apache 2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 致谢

我们感谢以下项目提供的宝贵模型和想法：
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 
