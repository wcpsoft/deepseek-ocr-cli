"""
DeepSeek OCR 核心模块
包含基于视觉编码器与大语言模型的光学字符识别系统的核心算法实现

本模块基于DeepSeek-OCR项目，提供了OCR核心算法的实现。
"""

__version__ = "1.0.0"
__author__ = "Rxzhang"

# 导出核心类和函数
from .config import *
from .deepseek_ocr import DeepseekOCRForCausalLM

__all__ = [
    "DeepseekOCRForCausalLM",
]