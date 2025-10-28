#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR引擎工厂单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_ocr_engine_factory_register_engine():
    """测试注册引擎"""
    from src.core.factory.ocr_engine_factory import OCREngineFactory
    
    # 创建一个模拟引擎类
    mock_engine_class = MagicMock()
    
    # 注册引擎
    OCREngineFactory.register_engine("test_engine", mock_engine_class)
    
    # 验证引擎已注册
    assert "test_engine" in OCREngineFactory._engine_classes
    assert OCREngineFactory._engine_classes["test_engine"] == mock_engine_class


def test_ocr_engine_factory_create_engine():
    """测试创建引擎"""
    with patch.dict('sys.modules', {
        'src.core.vllm.vllm_engine': MagicMock(),
        'src.core.vllm.vllm_engine.VLLMEngine': MagicMock()
    }):
        from src.core.factory.ocr_engine_factory import OCREngineFactory
        
        # 创建vLLM引擎
        engine = OCREngineFactory.create_engine("vllm")
        
        # 验证返回了引擎实例
        assert engine is not None


def test_ocr_engine_factory_invalid_engine_type():
    """测试创建不支持的引擎类型"""
    from src.core.factory.ocr_engine_factory import OCREngineFactory
    
    # 尝试创建不支持的引擎类型
    with pytest.raises(ValueError):
        OCREngineFactory.create_engine("invalid_engine")


def test_ocr_engine_factory_get_available_engines():
    """测试获取可用引擎列表"""
    from src.core.factory.ocr_engine_factory import OCREngineFactory
    
    # 获取可用引擎
    available_engines = OCREngineFactory.get_available_engines()
    
    # 验证返回了字典
    assert isinstance(available_engines, dict)


def test_ocr_engine_factory_is_engine_available():
    """测试检查引擎是否可用"""
    from src.core.factory.ocr_engine_factory import OCREngineFactory
    
    # 检查transformers引擎是否可用
    is_available = OCREngineFactory.is_engine_available("transformers")
    
    # 验证返回了布尔值
    assert isinstance(is_available, bool)