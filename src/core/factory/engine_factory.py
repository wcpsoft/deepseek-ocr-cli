#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR引擎工厂类
负责根据配置创建相应的OCR引擎实例
"""

from typing import Optional
from src.core.base.ocr_engine import BaseOCREngine


class EngineFactory:
    """OCR引擎工厂类"""
    
    @staticmethod
    def create_engine(engine_type: str, model_path: Optional[str] = None, 
                      prompt: Optional[str] = None, base_size: int = 1024, 
                      image_size: int = 640, crop_mode: bool = True) -> BaseOCREngine:
        """
        创建OCR引擎实例
        
        Args:
            engine_type: 引擎类型 ("vllm" 或 "transformers")
            model_path: 模型路径
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否启用裁剪模式
            
        Returns:
            BaseOCREngine: OCR引擎实例
            
        Raises:
            ValueError: 不支持的引擎类型
        """
        if engine_type.lower() == "vllm":
            try:
                from src.core.vllm.vllm_engine import VLLMEngine
                return VLLMEngine(model_path, prompt, base_size, image_size, crop_mode)
            except ImportError as e:
                raise RuntimeError(f"无法导入vLLM引擎: {str(e)}")
        elif engine_type.lower() == "transformers":
            try:
                from src.core.transformers.transformers_engine import TransformersEngine
                return TransformersEngine(model_path, prompt, base_size, image_size, crop_mode)
            except ImportError as e:
                raise RuntimeError(f"无法导入Transformers引擎: {str(e)}")
        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")


def get_engine(engine_type: str, model_path: Optional[str] = None, 
               prompt: Optional[str] = None, base_size: int = 1024, 
               image_size: int = 640, crop_mode: bool = True) -> BaseOCREngine:
    """
    获取OCR引擎实例的便捷函数
    
    Args:
        engine_type: 引擎类型 ("vllm" 或 "transformers")
        model_path: 模型路径
        prompt: 提示词
        base_size: 基础尺寸
        image_size: 图像尺寸
        crop_mode: 是否启用裁剪模式
        
    Returns:
        BaseOCREngine: OCR引擎实例
    """
    return EngineFactory.create_engine(engine_type, model_path, prompt, base_size, image_size, crop_mode)