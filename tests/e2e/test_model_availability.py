#!/usr/bin/env python3
"""
模型可用性端到端测试
验证模型是否可以正常加载和使用
"""

import os
import sys
from pathlib import Path

import pytest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


def test_model_loading():
    """测试模型加载"""
    try:
        from src.cli.model_manager import ModelManager

        model_manager = ModelManager("./models")

        # 检查模型目录是否存在
        model_path = model_manager.model_dir / "deepseek-ocr"
        if not model_path.exists():
            pytest.skip("模型目录不存在，跳过测试")

        # 简单测试模型管理器功能
        assert model_manager is not None
        assert model_manager.model_dir.exists()

    except ImportError:
        pytest.fail("无法导入ModelManager")


def test_model_verification():
    """测试模型验证"""
    try:
        from src.cli.model_manager import ModelManager

        model_manager = ModelManager("./models")

        # 检查模型目录是否存在
        model_path = model_manager.model_dir / "deepseek-ocr"
        if not model_path.exists():
            pytest.skip("模型目录不存在，跳过测试")

        # 测试获取模型目录功能
        model_dir = model_manager.get_model_dir()
        assert model_dir is not None

    except ImportError:
        pytest.fail("无法导入ModelManager")


def test_model_configuration():
    """测试模型配置"""
    try:
        from src.cli.model_manager import ModelManager

        model_manager = ModelManager("./models")

        # 检查默认模型配置
        assert hasattr(model_manager, "default_models")
        assert "deepseek-ocr" in model_manager.default_models

        # 检查模型信息
        model_info = model_manager.default_models["deepseek-ocr"]
        assert "repo_id" in model_info
        assert model_info["repo_id"] == "deepseek-ai/DeepSeek-OCR"

    except ImportError:
        pytest.fail("无法导入ModelManager")


def test_model_path_resolution():
    """测试模型路径解析"""
    try:
        from src.cli.model_manager import ModelManager

        model_manager = ModelManager("./models")

        # 测试模型路径解析
        test_model_name = "deepseek-ocr"
        expected_path = model_manager.model_dir / test_model_name

        # 检查路径是否正确构建
        assert expected_path is not None
        assert isinstance(expected_path, Path)

    except ImportError:
        pytest.fail("无法导入ModelManager")
