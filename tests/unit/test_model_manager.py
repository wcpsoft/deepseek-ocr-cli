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
            manager.add_custom_model("test-model", "test/repo", "huggingface")

            # 验证自定义模型已添加
            custom_models = manager.list_custom_models()
            assert "test-model" in custom_models
            assert custom_models["test-model"]["repo_id"] == "test/repo"
            assert custom_models["test-model"]["source"] == "huggingface"

            # 测试移除自定义模型
            manager.remove_custom_model("test-model")
            custom_models = manager.list_custom_models()
            assert "test-model" not in custom_models
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


def test_model_manager_model_dir_setter() -> None:
    """测试模型目录设置功能"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))

            # 测试模型目录设置
            with tempfile.TemporaryDirectory() as new_temp_dir:
                new_temp_path = Path(new_temp_dir)
                manager.set_model_dir(str(new_temp_path))
                # 使用Path.resolve()来处理符号链接
                expected_path = str(new_temp_path.resolve())
                actual_path = manager.get_model_dir()
                assert actual_path == expected_path
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")


def test_model_manager_download_models() -> None:
    """测试模型下载功能"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))

            # 验证默认模型列表
            downloaded_models = manager.list_downloaded_models()
            assert isinstance(downloaded_models, list)
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")


def test_model_manager_clean_py_files() -> None:
    """测试清理Python文件功能"""
    try:
        from src.cli.model_manager import ModelManager

        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manager = ModelManager(str(temp_path))

            # 创建测试模型目录
            model_name = "test-model"
            model_path = temp_path / model_name
            model_path.mkdir()

            # 创建一个Python文件
            py_file = model_path / "test.py"
            py_file.write_text("# 测试文件")

            # 验证清理功能
            result = manager.clean_model_py_files(model_name)
            assert result is True
            assert not py_file.exists()
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")
