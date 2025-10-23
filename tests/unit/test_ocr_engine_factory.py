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


def test_engine_factory_create_vllm_engine():
    """测试创建vLLM引擎"""
    with patch.dict('sys.modules', {
        'src.core.vllm.vllm_engine': MagicMock(),
        'src.core.vllm.vllm_engine.VLLMEngine': MagicMock()
    }):
        from src.core.factory.engine_factory import EngineFactory
        
        # 创建vLLM引擎
        engine = EngineFactory.create_engine("vllm")
        
        # 验证返回了引擎实例
        assert engine is not None


def test_engine_factory_create_transformers_engine():
    """测试创建Transformers引擎"""
    with patch.dict('sys.modules', {
        'src.core.transformers.transformers_engine': MagicMock(),
        'src.core.transformers.transformers_engine.TransformersEngine': MagicMock()
    }):
        from src.core.factory.engine_factory import EngineFactory
        
        # 创建Transformers引擎
        engine = EngineFactory.create_engine("transformers")
        
        # 验证返回了引擎实例
        assert engine is not None


def test_engine_factory_invalid_engine_type():
    """测试创建不支持的引擎类型"""
    from src.core.factory.engine_factory import EngineFactory
    
    # 尝试创建不支持的引擎类型
    with pytest.raises(ValueError):
        EngineFactory.create_engine("invalid_engine")


def test_get_engine_function():
    """测试便捷函数"""
    with patch.dict('sys.modules', {
        'src.core.transformers.transformers_engine': MagicMock(),
        'src.core.transformers.transformers_engine.TransformersEngine': MagicMock()
    }):
        from src.core.factory.engine_factory import get_engine
        
        # 使用便捷函数创建引擎
        engine = get_engine("transformers")
        
        # 验证返回了引擎实例
        assert engine is not None


def test_engine_factory_import_error():
    """测试导入错误处理"""
    with patch.dict('sys.modules', {
        'src.core.vllm.vllm_engine': None  # 模拟导入失败
    }):
        from src.core.factory.engine_factory import EngineFactory
        
        # 尝试创建vLLM引擎应该抛出RuntimeError
        with pytest.raises(RuntimeError):
            EngineFactory.create_engine("vllm")