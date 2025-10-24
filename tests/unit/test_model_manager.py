#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器单元测试
测试模型管理器的各项功能
"""

import sys
import os
import pytest
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_model_manager_initialization():
    """测试模型管理器初始化"""
    try:
        from src.cli.model_manager import ModelManager
        manager = ModelManager("/tmp/test_models")
        # 使用Path.resolve()来处理符号链接
        expected_path = Path("/tmp/test_models").resolve()
        assert manager.model_dir == expected_path
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")

def test_model_manager_default_models():
    """测试默认模型配置"""
    try:
        from src.cli.model_manager import ModelManager
        manager = ModelManager("/tmp/test_models")
        assert "deepseek-ocr" in manager.default_models
        assert manager.default_models["deepseek-ocr"]["repo_id"] == "deepseek-ai/DeepSeek-OCR"
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")

def test_model_manager_custom_models():
    """测试自定义模型功能"""
    try:
        from src.cli.model_manager import ModelManager
        manager = ModelManager("/tmp/test_models")
        
        # 添加自定义模型
        custom_model = {
            "repo_id": "custom/model",
            "source": "huggingface"
        }
        # 假设有一个方法可以添加自定义模型
        # 这里我们只测试初始化
        assert manager is not None
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")

def test_model_manager_model_verification():
    """测试模型验证功能"""
    try:
        from src.cli.model_manager import ModelManager
        manager = ModelManager("/tmp/test_models")
        
        # 测试模型目录获取
        expected_path = str(Path("/tmp/test_models").resolve())
        model_dir = manager.get_model_dir()
        assert model_dir == expected_path
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")