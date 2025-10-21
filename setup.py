#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR CLI setup script
"""

from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements
with open("pyproject.toml", "r", encoding="utf-8") as fh:
    pyproject_content = fh.read()

setup(
    name="deepseek-ocr-cli",
    version="1.0.0",
    author="Rxzhang",
    author_email="rxzhang@example.com",
    description="DeepSeek OCR CLI - 基于视觉编码器与大语言模型的光学字符识别系统命令行工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wcpsoft/deepseek-ocr-cli",
    packages=find_packages(where=".", include=["cli*", "src*"]),
    package_data={
        "cli": ["*.py"],
        "src": ["*.py"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=[
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "PyMuPDF",
        "img2pdf",
        "einops",
        "easydict",
        "addict",
        "Pillow",
        "numpy",
        "torch>=2.4.1",
        "huggingface-hub>=0.20.0",
        "libreoffice>=7.0.0",
    ],
    extras_require={
        "vllm": [
            "vllm>=0.8.5",
            "flash-attn>=2.7.3",
        ],
        "cli": [
            "huggingface-hub>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        # GPU特定依赖
        "nvidia": [
            "torch==2.4.1+cu118",
            "torchvision==0.19.1+cu118",
            "torchaudio==2.4.1+cu118",
        ],
        "amd": [
            "torch==2.4.1+rocm6.1",
            "torchvision==0.19.1+rocm6.1",
            "torchaudio==2.4.1+rocm6.1",
        ],
        "mps": [
            "torch==2.4.1",
            "torchvision==0.19.1",
            "torchaudio==2.4.1",
        ],
        "dcu": [
            "torch==2.4.1",
            "torchvision==0.19.1",
            "torchaudio==2.4.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepseek-ocr=cli.main:main",
            "deepseek-ocr-download=cli.download_models:main",
            "deepseek-ocr-detect-gpu=dev.detect_gpu:main",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/wcpsoft/deepseek-ocr-cli",
        "Repository": "https://github.com/wcpsoft/deepseek-ocr-cli.git",
        "Documentation": "https://github.com/wcpsoft/deepseek-ocr-cli/blob/main/README.md",
    },
)