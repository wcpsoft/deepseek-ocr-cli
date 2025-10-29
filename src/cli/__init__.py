"""
DeepSeek OCR CLI 模块
支持多种文档格式的OCR处理命令行工具

本模块是DeepSeek-OCR-cli项目的一部分，提供了命令行接口用于处理各种文档格式的OCR识别。
"""

__version__ = "1.0.0"
__author__ = "Rxzhang"

# 导出主要类
from .document_processor import DocumentProcessor
from .model_manager import ModelManager
from .pdf_converter import PDFConverter

__all__ = [
    "DocumentProcessor",
    "ModelManager",
    "PDFConverter",
]
