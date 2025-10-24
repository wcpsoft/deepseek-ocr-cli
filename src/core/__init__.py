#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR核心模块
"""

# 导入主要组件
from .deepseek_ocr import (
    DeepseekOCRVisionEncoder,
    IMAGE_SIZE,
    BASE_SIZE,
    CROP_MODE,
    PRINT_NUM_VIS_TOKENS
)
from .inference.deepseek_ocr_inference import DeepseekOCRInference
from .process.image_process import DeepseekOCRProcessor

__all__ = [
    "DeepseekOCRVisionEncoder",
    "DeepseekOCRInference",
    "DeepseekOCRProcessor",
    "IMAGE_SIZE",
    "BASE_SIZE",
    "CROP_MODE",
    "PRINT_NUM_VIS_TOKENS"
]