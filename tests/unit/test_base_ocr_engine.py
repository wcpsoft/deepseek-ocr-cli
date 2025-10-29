#!/usr/bin/env python3
"""
基础OCR引擎抽象类单元测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_base_ocr_engine_initialization():
    """测试基础OCR引擎初始化"""
    from src.core.base.ocr_engine import BaseOCREngine

    # 创建基础OCR引擎子类用于测试
    class TestEngine(BaseOCREngine):
        def initialize(self):
            pass

        def process(self, images, output_dir):
            pass

        def cleanup(self):
            pass

    # 创建测试引擎实例
    engine = TestEngine()

    # 验证初始化
    assert engine is not None
    assert engine.model_path is None
    assert engine.prompt is None
    assert engine.base_size == 1024
    assert engine.image_size == 640
    assert engine.crop_mode is True


def test_base_ocr_engine_with_parameters():
    """测试带参数的基础OCR引擎初始化"""
    from src.core.base.ocr_engine import BaseOCREngine

    # 创建基础OCR引擎子类用于测试
    class TestEngine(BaseOCREngine):
        def initialize(self):
            pass

        def process(self, images, output_dir):
            pass

        def cleanup(self):
            pass

    # 创建带参数的测试引擎实例
    engine = TestEngine(
        model_path="/path/to/model",
        prompt="测试提示词",
        base_size=512,
        image_size=512,
        crop_mode=False,
    )

    # 验证参数
    assert engine.model_path == "/path/to/model"
    assert engine.prompt == "测试提示词"
    assert engine.base_size == 512
    assert engine.image_size == 512
    assert engine.crop_mode is False


def test_base_ocr_engine_abstract_methods():
    """测试基础OCR引擎抽象方法"""
    from src.core.base.ocr_engine import BaseOCREngine

    # 验证抽象方法存在
    assert hasattr(BaseOCREngine, "initialize")
    assert hasattr(BaseOCREngine, "process")
    assert hasattr(BaseOCREngine, "cleanup")

    # 创建基础OCR引擎子类用于测试
    class TestEngine(BaseOCREngine):
        def initialize(self):
            pass

        def process(self, images, output_dir):
            pass

        def cleanup(self):
            pass

    # 创建测试引擎实例
    engine = TestEngine()

    # 验证方法可以被调用
    try:
        engine.initialize()
        engine.cleanup()
        # 如果没有抛出异常，则测试通过
        assert True
    except NotImplementedError as e:
        # 如果抛出NotImplementedError，则测试失败
        raise AssertionError("cleanup方法未正确实现") from e
