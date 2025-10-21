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
├── src/                   # 源代码目录
│   ├── __init__.py
│   └── core/              # 核心算法实现
│       ├── __init__.py
│       ├── config.py          # 配置文件
│       ├── deepseek_ocr.py    # vLLM模型实现
│       ├── vllm_process/      # vLLM处理模块
│       └── vllm_deepencoder/  # vLLM编码器模块
├── cli/                   # 命令行工具
│   ├── __init__.py
│   ├── main.py            # 主入口
│   ├── document_processor.py  # 文档处理核心
│   ├── model_manager.py   # 模型管理
│   ├── pdf_converter.py   # PDF转换器
│   ├── download_models.py # 模型下载
│   ├── example_usage.py   # 使用示例
│   └── test_cli.py        # CLI测试
├── tests/                 # 测试目录
│   └── test_cli.py        # CLI测试脚本
├── README.md              # 项目说明与使用指南
└── pyproject.toml         # 项目配置
```

## 环境要求

- Python 3.10 或更高版本
- CUDA 11.8 + PyTorch 2.6.0
- 支持的系统：Linux、Windows（需要WSL）、macOS

## 安装指南

### 1. 克隆项目
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
```

### 2. 创建Conda环境
```bash
conda create -n deepseek-ocr-cli python=3.12.9 -y
conda activate deepseek-ocr-cli
```

### 3. 安装依赖

#### 基础依赖
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 安装项目
```bash
# 安装核心依赖
pip install .

# 如果需要vLLM支持
pip install .[vllm]

# 如果需要开发依赖
pip install .[dev]
```

#### 安装LibreOffice
为了支持办公文档格式转换，需要安装LibreOffice：
- Ubuntu/Debian: `sudo apt-get install libreoffice`
- CentOS/RHEL: `sudo yum install libreoffice`
- macOS: `brew install --cask libreoffice`
- Windows: 从官网下载安装

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

# 自定义提示词
deepseek-ocr image.jpg -o output_dir --prompt "<image>\nOCR this image."
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

```python
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
```

## 许可证

本项目采用 Apache 2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 致谢

我们感谢以下项目提供的宝贵模型和想法：
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 