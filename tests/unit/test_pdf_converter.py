#!/usr/bin/env python3
"""
PDF转换器单元测试
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cli.pdf_converter import PDFConverter
from tests.utils import TestUtils


class TestPDFConverter(unittest.TestCase):
    """PDF转换器测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.converter = PDFConverter()

    def tearDown(self) -> None:
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pdf_converter_initialization(self) -> None:
        """测试PDF转换器初始化"""
        self.assertIsNotNone(self.converter)
        self.assertIsInstance(self.converter, PDFConverter)

    @patch("subprocess.run")
    def test_convert_to_images(self, mock_run: MagicMock) -> None:
        """测试PDF转图像"""
        # 使用TestUtils创建模拟subprocess结果
        mock_result = TestUtils.create_mock_subprocess_result(returncode=0)
        mock_run.return_value = mock_result

        # 创建测试PDF文件
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        Path(pdf_path).touch()

        # 测试转换
        output_dir = os.path.join(self.temp_dir, "output")
        result = self.converter.convert_to_images(pdf_path, output_dir)

        # 验证结果
        self.assertTrue(result)
        mock_run.assert_called()

    @patch("subprocess.run")
    def test_convert_to_images_failure(self, mock_run: MagicMock) -> None:
        """测试PDF转图像失败情况"""
        # 使用TestUtils创建模拟subprocess结果
        mock_result = TestUtils.create_mock_subprocess_result(returncode=1, stderr="Error")
        mock_run.return_value = mock_result

        # 创建测试PDF文件
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        Path(pdf_path).touch()

        # 测试转换
        output_dir = os.path.join(self.temp_dir, "output")
        result = self.converter.convert_to_images(pdf_path, output_dir)

        # 验证结果
        self.assertFalse(result)

    def test_check_pdf_valid(self) -> None:
        """测试检查有效PDF"""
        # 创建测试PDF文件
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        Path(pdf_path).touch()

        # 测试检查
        result = self.converter.is_valid_pdf(pdf_path)

        # 验证结果
        self.assertTrue(result)

    def test_check_pdf_invalid(self) -> None:
        """测试检查无效PDF"""
        # 测试不存在的文件
        pdf_path = os.path.join(self.temp_dir, "nonexistent.pdf")
        result = self.converter.is_valid_pdf(pdf_path)

        # 验证结果
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
