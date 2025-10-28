#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR核心模块
"""

# 导入主要组件
from .process.image_process import DeepseekOCRProcessor
from .models.model_factory import OCRModelFactory, OCRModelInterface

# 导入配置常量
from .config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS

__all__ = [
    "DeepseekOCRProcessor",
    "OCRModelFactory",
    "OCRModelInterface",
    "IMAGE_SIZE",
    "BASE_SIZE",
    "CROP_MODE",
    "PRINT_NUM_VIS_TOKENS"
]