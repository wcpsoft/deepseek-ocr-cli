#!/usr/bin/env python3
"""
模型可用性端到端测试
验证模型的加载、验证和下载功能
"""


import pytest


def test_model_loading() -> None:
    """测试模型加载"""
    try:
        from src.cli.model_manager import ModelManager

        # 创建模型管理器实例
        model_manager = ModelManager()

        # 验证模型管理器实例
        assert model_manager is not None
        assert model_manager.model_dir is not None

    except ImportError as e:
        pytest.fail(f"模型加载测试失败: {e}")


def test_model_verification() -> None:
    """测试模型验证"""
    try:
        from src.cli.model_manager import ModelManager

        # 创建模型管理器实例
        model_manager = ModelManager()

        # 检查模型目录是否存在
        model_path = model_manager.model_dir / "deepseek-ocr"
        if not model_path.exists():
            pytest.skip("模型目录不存在, 跳过测试")

        # 验证模型文件
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
        ]

        missing_files = []
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            pytest.fail(f"模型文件缺失: {', '.join(missing_files)}")

    except OSError as e:
        pytest.fail(f"模型验证测试失败: {e}")


def test_model_download() -> None:
    """测试模型下载功能"""
    try:
        from src.cli.model_manager import ModelManager

        # 创建模型管理器实例
        model_manager = ModelManager()

        # 检查模型目录是否存在
        model_path = model_manager.model_dir / "deepseek-ocr"
        if not model_path.exists():
            # 如果模型不存在, 测试下载功能
            # 注意: 这里只测试方法是否存在, 不实际下载
            assert hasattr(model_manager, "download_models")
        else:
            # 如果模型存在, 测试验证功能
            assert model_manager.verify_model("deepseek-ocr") is True

    except OSError as e:
        pytest.fail(f"模型下载测试失败: {e}")
