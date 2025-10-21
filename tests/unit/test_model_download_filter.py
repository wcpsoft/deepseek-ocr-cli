#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载过滤功能测试
"""

import pytest
from pathlib import Path
import tempfile
import inspect

def test_model_download_filter_configuration():
    """测试模型下载过滤配置"""
    from cli.model_manager import ModelManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "models"
        manager = ModelManager(str(model_dir))
        
        # 检查默认模型信息
        assert "deepseek-ocr" in manager.default_models
        default_model = manager.default_models["deepseek-ocr"]
        assert default_model["repo_id"] == "deepseek-ai/DeepSeek-OCR"
        
        # 检查模型下载方法是否存在
        assert hasattr(manager, '_download_from_huggingface')
        assert hasattr(manager, '_download_from_modelscope')

def test_huggingface_download_patterns():
    """测试Hugging Face下载模式配置"""
    from cli.model_manager import ModelManager
    
    # 检查Hugging Face下载方法中的过滤配置
    manager = ModelManager("./test_models")
    download_method = manager._download_from_huggingface
    source_code = inspect.getsource(download_method)
    
    # 检查是否包含过滤模式
    assert "ignore_patterns" in source_code
    assert "allow_patterns" in source_code
    assert "*.md" in source_code
    assert "assets/*" in source_code
    assert "examples/*" in source_code

def test_model_verification_logic():
    """测试模型验证逻辑"""
    from cli.model_manager import ModelManager
    
    manager = ModelManager("./test_models")
    
    # 检查模型验证方法
    assert hasattr(manager, 'verify_model')
    
    # 检查验证逻辑
    verify_method = manager.verify_model
    source_code = inspect.getsource(verify_method)
    
    # 检查是否验证必需文件
    assert "config.json" in source_code
    assert "pytorch_model*.bin" in source_code or "pytorch_model" in source_code
    assert "*.safetensors" in source_code