#!/usr/bin/env python3
"""
基础OCR引擎抽象类单元测试
"""

import unittest

from src.core.base.ocr_engine import BaseOCREngine
from tests.utils import TestUtils


class TestBaseOCREngine(unittest.TestCase):
    """基础OCR引擎测试类"""

    def test_abstract_methods(self) -> None:
        """测试抽象方法"""
        # 尝试实例化抽象类应该抛出TypeError
        with self.assertRaises(TypeError):
            BaseOCREngine()  # type: ignore

    def test_initialize_method(self) -> None:
        """测试initialize方法"""
        # 使用TestUtils创建模拟子类
        engine = TestUtils.create_mock_ocr_engine()

        # 调用initialize方法,应该不会抛出异常
        try:
            engine.initialize()
            # 如果没有抛出异常,则测试通过
            assert True
        except NotImplementedError as e:
            # 如果抛出NotImplementedError,则测试失败
            raise AssertionError("initialize方法未正确实现") from e

    def test_process_method(self) -> None:
        """测试process方法"""
        # 使用TestUtils创建模拟子类
        engine = TestUtils.create_mock_ocr_engine()

        # 调用process方法,应该不会抛出异常
        try:
            engine.process([], "/path/to/output")
            # 如果没有抛出异常,则测试通过
            assert True
        except NotImplementedError as e:
            raise AssertionError("process方法未正确实现") from e

    def test_cleanup_method(self) -> None:
        """测试cleanup方法"""
        # 使用TestUtils创建模拟子类
        engine = TestUtils.create_mock_ocr_engine()

        # 调用cleanup方法,应该不会抛出异常
        try:
            engine.cleanup()
            # 如果没有抛出异常,则测试通过
            assert True
        except NotImplementedError as e:
            # 如果抛出NotImplementedError,则测试失败
            raise AssertionError("cleanup方法未正确实现") from e
