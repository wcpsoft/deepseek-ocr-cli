#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器单元测试
"""

import pytest
from pathlib import Path
import tempfile

def test_model_manager_initialization():
    """测试模型管理器初始化"""
    from cli.model_manager import ModelManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "models"
        manager = ModelManager(str(model_dir))
        
        # 验证初始化
        assert manager is not None
        assert Path(manager.get_model_dir()).resolve() == model_dir.resolve()

def test_model_manager_default_models():
    """测试默认模型配置"""
    from cli.model_manager import ModelManager
    
    manager = ModelManager("./test_models")
    
    # 验证默认模型信息
    assert "deepseek-ocr" in manager.default_models
    default_model = manager.default_models["deepseek-ocr"]
    assert isinstance(default_model, dict)
    assert default_model["repo_id"] == "deepseek-ai/DeepSeek-OCR"
    assert default_model["source"] == "huggingface"

def test_model_manager_custom_models():
    """测试自定义模型功能"""
    from cli.model_manager import ModelManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "models"
        manager = ModelManager(str(model_dir))
        
        # 添加自定义模型
        manager.add_custom_model("test-model", "test/repo-id", "huggingface")
        custom_models = manager.list_custom_models()
        
        # 验证添加成功
        assert "test-model" in custom_models
        assert custom_models["test-model"]["repo_id"] == "test/repo-id"
        assert custom_models["test-model"]["source"] == "huggingface"
        
        # 移除自定义模型
        manager.remove_custom_model("test-model")
        custom_models = manager.list_custom_models()
        
        # 验证移除成功
        assert "test-model" not in custom_models

def test_model_manager_model_verification():
    """测试模型验证功能"""
    from cli.model_manager import ModelManager
    
    manager = ModelManager("./test_models")
    
    # 验证模型验证方法存在
    assert hasattr(manager, 'verify_model')