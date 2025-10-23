#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR核心模块
"""

# 保留原有的导入以确保向后兼容性
from .deepseek_ocr import DeepseekOCRForCausalLM

# 导入新的模块
from .base import BaseOCREngine
from .vllm import VLLMEngine
from .transformers import TransformersEngine
from .factory import EngineFactory, get_engine

__all__ = [
    "DeepseekOCRForCausalLM",
    "BaseOCREngine",
    "VLLMEngine",
    "TransformersEngine",
    "EngineFactory",
    "get_engine"
]