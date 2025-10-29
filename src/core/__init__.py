#!/usr/bin/env python3
"""
DeepSeek OCR核心模块
"""

# 导入配置常量
from .config import BASE_SIZE, CROP_MODE, IMAGE_SIZE, PRINT_NUM_VIS_TOKENS
from .models.model_factory import OCRModelFactory, OCRModelInterface

# 导入主要组件
from .process.image_process import DeepseekOCRProcessor

__all__ = [
    "BASE_SIZE",
    "CROP_MODE",
    "IMAGE_SIZE",
    "PRINT_NUM_VIS_TOKENS",
    "DeepseekOCRProcessor",
    "OCRModelFactory",
    "OCRModelInterface",
]
