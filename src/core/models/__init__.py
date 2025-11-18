#!/usr/bin/env python3
"""
DeepSeek OCR模型模块
"""

from src.core.deepseek_ocr_config import DeepseekV2Config, DeepseekVLV2Config

# 为了向后兼容，也导出DeepseekOCRForCausalLM
from src.core.models.deepseek_ocr_model import _IMAGE_TOKEN, DeepseekOCRForCausalLM
from src.core.models.model_adapter import (
    ModelAdapter,
    TransformersOCRModelAdapter,
    UnifiedOCRModelAdapter,
    VLLMOCRModelAdapter,
)
from src.core.models.model_factory import OCRModelFactory, OCRModelInterface

__all__ = [
    "_IMAGE_TOKEN",
    "DeepseekOCRForCausalLM",
    "DeepseekV2Config",
    "DeepseekVLV2Config",
    "ModelAdapter",
    "OCRModelFactory",
    "OCRModelInterface",
    "TransformersOCRModelAdapter",
    "UnifiedOCRModelAdapter",
    "VLLMOCRModelAdapter",
]
