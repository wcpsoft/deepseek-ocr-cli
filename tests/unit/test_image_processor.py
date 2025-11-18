#!/usr/bin/env python3
"""
图像处理单元测试
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.process.image_process import DeepseekOCRProcessor
from tests.conftest import is_model_available
from tests.utils import TestUtils


class TestDeepseekOCRProcessor(unittest.TestCase):
    """图像处理器测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """测试类初始化"""
        if not is_model_available():
            pytest.skip("模型文件不可用", allow_module_level=True)

    def setUp(self) -> None:
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DeepseekOCRProcessor()

    def tearDown(self) -> None:
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deepseek_ocr_processor_initialization(self) -> None:
        """测试DeepseekOCRProcessor初始化"""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor, DeepseekOCRProcessor)

    @patch("PIL.Image.open")
    def test_load_image(self, mock_open: MagicMock) -> None:
        """测试加载图像"""
        # 使用TestUtils创建模拟PIL图像
        mock_image = TestUtils.create_mock_pil_image()
        mock_open.return_value = mock_image

        # 测试加载图像
        image_path = os.path.join(self.temp_dir, "test.jpg")
        Path(image_path).touch()  # 创建空文件
        result = self.processor.load_image(image_path)

        # 验证结果
        self.assertEqual(result, mock_image)
        mock_open.assert_called_once_with(image_path)

    @patch("PIL.Image.open")
    def test_deepseek_ocr_processor_with_image(self, mock_image_open: MagicMock) -> None:
        """测试DeepseekOCRProcessor处理图像"""
        # 使用TestUtils创建模拟PIL图像
        mock_image = TestUtils.create_mock_pil_image()
        mock_image.size = (640, 480)
        mock_image_open.return_value = mock_image

        # 测试处理器
        image_path = os.path.join(self.temp_dir, "test.jpg")
        with open(image_path, "w") as f:
            f.write("")

        # 验证处理器可以处理图像
        self.assertIsNotNone(self.processor)

    @patch("PIL.Image.fromarray")
    def test_save_image(self, mock_fromarray: MagicMock) -> None:
        """测试保存图像"""
        # 使用TestUtils创建模拟PIL图像
        mock_image = TestUtils.create_mock_pil_image()
        mock_fromarray.return_value = mock_image

        # 测试保存图像
        import numpy as np

        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        output_path = os.path.join(self.temp_dir, "output.jpg")

        result = self.processor.save_image(image_array, output_path)

        # 验证结果
        self.assertTrue(result)
        mock_image.save.assert_called_once_with(output_path)

    @patch("cv2.resize")
    def test_resize_image(self, mock_resize: MagicMock) -> None:
        """测试调整图像大小"""
        # 使用TestUtils创建模拟cv2图像
        mock_resized = TestUtils.create_mock_cv2_image()
        mock_resize.return_value = mock_resized

        # 测试调整图像大小
        import numpy as np

        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.processor.resize_image(image_array, (50, 50))

        # 验证结果
        self.assertEqual(result, mock_resized)
        mock_resize.assert_called_once()

    @patch("cv2.cvtColor")
    def test_convert_color_space(self, mock_cvtcolor: MagicMock) -> None:
        """测试转换颜色空间"""
        # 使用TestUtils创建模拟cv2图像
        mock_converted = TestUtils.create_mock_cv2_image()
        mock_cvtcolor.return_value = mock_converted

        # 测试转换颜色空间
        import numpy as np

        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.processor.convert_color_space(image_array, "RGB", "GRAY")

        # 验证结果
        self.assertEqual(result, mock_converted)
        mock_cvtcolor.assert_called_once()

    def test_normalize_image(self) -> None:
        """测试图像归一化"""
        # 测试归一化
        import numpy as np

        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.processor.normalize_image(image_array)

        # 验证结果
        self.assertEqual(result.shape, image_array.shape)
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)

    def test_preprocess_for_ocr(self) -> None:
        """测试OCR预处理"""
        # 测试OCR预处理
        import numpy as np

        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.processor.preprocess_for_ocr(image_array)

        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 3)  # 应该仍然是3D数组


if __name__ == "__main__":
    unittest.main()
