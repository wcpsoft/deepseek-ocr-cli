#!/usr/bin/env python3
"""
CLI工具单元测试
"""

import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from src.cli.utils import (
    check_model_availability,
    get_appropriate_dtype_compat,
    get_compatible_device,
    get_project_root,
    should_use_bfloat16_compat,
)
from tests.conftest import is_model_available


class TestCLIUtils(unittest.TestCase):
    """CLI工具测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """测试类初始化"""
        if not is_model_available():
            pytest.skip("模型文件不可用", allow_module_level=True)

    def setUp(self) -> None:
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_check_model_availability(self) -> None:
        """测试检查模型可用性"""
        # 测试模型可用性检查
        project_root = get_project_root()
        is_available, model_path = check_model_availability(project_root)

        # 验证返回值类型
        self.assertIsInstance(is_available, bool)
        if is_available:
            self.assertIsNotNone(model_path)
            self.assertIsInstance(model_path, str)

    def test_get_project_root(self) -> None:
        """测试获取项目根目录"""
        # 测试获取项目根目录
        project_root = get_project_root()

        # 验证返回值类型和内容
        self.assertIsInstance(project_root, Path)
        self.assertTrue(project_root.exists())
        self.assertTrue(project_root.is_dir())

    def test_get_compatible_device(self) -> None:
        """测试获取兼容设备"""
        # 测试获取兼容设备
        device = get_compatible_device()

        # 验证返回值类型
        self.assertIsInstance(device, torch.device)

    def test_get_appropriate_dtype(self) -> None:
        """测试获取适当数据类型"""
        # 测试获取适当数据类型
        device = get_compatible_device()
        dtype = get_appropriate_dtype_compat(device)

        # 验证返回值类型
        self.assertIsInstance(dtype, torch.dtype)
        self.assertIn(dtype, [torch.float32, torch.float16, torch.bfloat16])

    def test_should_use_bfloat16(self) -> None:
        """测试是否应使用bfloat16"""
        # 测试是否应使用bfloat16
        device = get_compatible_device()
        use_bfloat16 = should_use_bfloat16_compat(device)

        # 验证返回值类型
        self.assertIsInstance(use_bfloat16, bool)


if __name__ == "__main__":
    unittest.main()
