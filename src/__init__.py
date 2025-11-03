"""
DeepSeek OCR 项目
基于视觉编码器与大语言模型的光学字符识别系统

本项目是DeepSeek-OCR-cli的一部分，包含了OCR系统的核心算法实现。
"""

__version__ = "1.0.0"
__author__ = "Rxzhang"

# 导出核心模块
from src import core

__all__ = [
    "core",
]
