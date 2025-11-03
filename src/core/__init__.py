#!/usr/bin/env python3
"""
DeepSeek OCR核心模块
"""

# 导入配置常量
from src.core.config import BASE_SIZE, CROP_MODE, IMAGE_SIZE, PRINT_NUM_VIS_TOKENS
from src.core.models.model_factory import OCRModelFactory, OCRModelInterface
from src.core.process.image_process import DeepseekOCRProcessor

__all__ = [
    "BASE_SIZE",
    "CROP_MODE",
    "IMAGE_SIZE",
    "PRINT_NUM_VIS_TOKENS",
    "DeepseekOCRProcessor",
    "OCRModelFactory",
    "OCRModelInterface",
]
