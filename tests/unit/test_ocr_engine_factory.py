#!/usr/bin/env python3
"""
OCR引擎工厂单元测试
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_ocr_engine_factory_register_engine() -> None:
    """测试注册引擎"""
    from src.core.factory.ocr_engine_factory import OCREngineFactory

    # 保存原始注册的引擎类
    original_engine_classes = OCREngineFactory._engine_classes.copy()

    try:
        # 创建模拟引擎类
        mock_engine_class = MagicMock()

        # 注册引擎
        OCREngineFactory.register_engine("test_engine", mock_engine_class)

        # 验证引擎已注册
        assert "test_engine" in OCREngineFactory._engine_classes
        assert OCREngineFactory._engine_classes["test_engine"] == mock_engine_class
    finally:
        # 恢复原始状态
        OCREngineFactory._engine_classes = original_engine_classes


def test_ocr_engine_factory_create_engine() -> None:
    """测试创建引擎"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.vllm.vllm_engine": MagicMock(),
            "src.core.transformers.transformers_engine": MagicMock(),
        },
    ):
        from src.core.factory.ocr_engine_factory import OCREngineFactory

        # 保存原始注册的引擎类
        original_engine_classes = OCREngineFactory._engine_classes.copy()

        try:
            # 测试创建vLLM引擎
            vllm_engine = OCREngineFactory.create_engine("vllm")
            assert vllm_engine is not None

            # 测试创建Transformers引擎
            transformers_engine = OCREngineFactory.create_engine("transformers")
            assert transformers_engine is not None
        finally:
            # 恢复原始状态
            OCREngineFactory._engine_classes = original_engine_classes


def test_ocr_engine_factory_invalid_engine_type() -> None:
    """测试创建不支持的引擎类型"""
    from src.core.factory.ocr_engine_factory import OCREngineFactory

    # 保存原始注册的引擎类
    original_engine_classes = OCREngineFactory._engine_classes.copy()

    try:
        # 测试创建不支持的引擎类型
        try:
            OCREngineFactory.create_engine("invalid_engine")
            raise AssertionError("应该抛出ValueError异常")
        except ValueError:
            pass  # 期望的异常
    finally:
        # 恢复原始状态
        OCREngineFactory._engine_classes = original_engine_classes


def test_ocr_engine_factory_get_available_engines() -> None:
    """测试获取可用引擎列表"""
    # 重新导入模块以确保模拟生效
    if "src.core.factory.ocr_engine_factory" in sys.modules:
        importlib.reload(sys.modules["src.core.factory.ocr_engine_factory"])

    with patch.dict(
        "sys.modules",
        {
            "src.core.vllm.vllm_engine": MagicMock(),
            "src.core.transformers.transformers_engine": MagicMock(),
        },
    ):
        # 如果模块已导入, 重新加载
        if "src.core.factory.ocr_engine_factory" in sys.modules:
            importlib.reload(sys.modules["src.core.factory.ocr_engine_factory"])

        from src.core.factory.ocr_engine_factory import OCREngineFactory

        # 保存原始注册的引擎类
        original_engine_classes = OCREngineFactory._engine_classes.copy()

        try:
            # 获取可用引擎
            available_engines = OCREngineFactory.get_available_engines()

            # 验证至少包含vLLM和Transformers引擎
            assert "vllm" in available_engines
            assert "transformers" in available_engines
        finally:
            # 恢复原始状态
            OCREngineFactory._engine_classes = original_engine_classes


def test_ocr_engine_factory_is_engine_available() -> None:
    """测试检查引擎是否可用"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.vllm.vllm_engine": MagicMock(),
            "src.core.transformers.transformers_engine": MagicMock(),
        },
    ):
        from src.core.factory.ocr_engine_factory import OCREngineFactory

        # 保存原始注册的引擎类
        original_engine_classes = OCREngineFactory._engine_classes.copy()

        try:
            # 测试可用引擎
            assert OCREngineFactory.is_engine_available("vllm") is True
            assert OCREngineFactory.is_engine_available("transformers") is True

            # 测试不可用引擎
            assert OCREngineFactory.is_engine_available("invalid_engine") is False
        finally:
            # 恢复原始状态
            OCREngineFactory._engine_classes = original_engine_classes


def test_ocr_engine_factory_create_engine_with_parameters() -> None:
    """测试创建引擎时传递参数"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.vllm.vllm_engine": MagicMock(),
            "src.core.transformers.transformers_engine": MagicMock(),
        },
    ):
        from src.core.factory.ocr_engine_factory import OCREngineFactory

        # 保存原始注册的引擎类
        original_engine_classes = OCREngineFactory._engine_classes.copy()

        try:
            # 测试创建引擎时传递参数
            engine = OCREngineFactory.create_engine(
                engine_type="vllm",
                model_path="/test/model",
                device="cuda",
                prompt="测试提示词",
                base_size=512,
                image_size=320,
                crop_mode=False,
            )
            assert engine is not None
        finally:
            # 恢复原始状态
            OCREngineFactory._engine_classes = original_engine_classes
