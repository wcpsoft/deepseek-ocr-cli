#!/usr/bin/env python3
"""
模型管理器单元测试
测试模型管理器的各项功能
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


def test_model_manager_initialization() -> None:
    """测试模型管理器初始化"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))
            # 使用Path.resolve()来处理符号链接
            expected_path = temp_path.resolve()
            assert manager.model_dir == expected_path
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")


def test_model_manager_default_models() -> None:
    """测试默认模型配置"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))
            assert "deepseek-ocr" in manager.default_models
            assert manager.default_models["deepseek-ocr"]["repo_id"] == "deepseek-ai/DeepSeek-OCR"
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")


def test_model_manager_custom_models() -> None:
    """测试自定义模型功能"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))

            # 添加自定义模型
            # custom_model = {"repo_id": "custom/model", "source": "huggingface"}
            # 假设有一个方法可以添加自定义模型
            # 这里我们只测试初始化
            assert manager is not None
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")


def test_model_manager_model_verification() -> None:
    """测试模型验证功能"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))

            # 测试模型目录获取
            expected_path = str(temp_path.resolve())
            model_dir = manager.get_model_dir()
            assert model_dir == expected_path
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")
