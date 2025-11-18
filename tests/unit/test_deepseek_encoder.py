#!/usr/bin/env python3
"""
深度编码器单元测试
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.core.models.deepencoder import ImageEncoderViT
from tests.conftest import is_model_available
from tests.utils import TestUtils


class TestImageEncoderViT(unittest.TestCase):
    """深度编码器测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """测试类初始化"""
        if not is_model_available():
            pytest.skip("模型文件不可用", allow_module_level=True)

    def setUp(self) -> None:
        """测试前准备"""
        self.model_path = "/test/model/path"
        self.device = "cpu"

    @patch("src.core.models.deepencoder.ImageEncoderViT")
    def test_image_encoder_vit_initialization(self, mock_encoder: MagicMock) -> None:
        """测试ImageEncoderViT初始化"""
        # 使用TestUtils创建模拟编码器
        mock_encoder_instance = TestUtils.create_mock_encoder()
        mock_encoder.return_value = mock_encoder_instance

        # 创建编码器实例
        encoder = ImageEncoderViT(self.model_path, self.device)

        # 验证初始化
        self.assertIsNotNone(encoder)
        mock_encoder.assert_called_once_with(self.model_path, self.device)

    @patch("src.core.deepencoder.deepseek_encoder.DeepSeekEncoder")
    def test_encode_image(self, mock_encoder_class: MagicMock) -> None:
        """测试图像编码"""
        # 使用TestUtils创建模拟编码器
        mock_encoder = TestUtils.create_mock_encoder()
        mock_encoder.encode_image.return_value = [[0.1, 0.2, 0.3]]
        mock_encoder_class.return_value = mock_encoder

        # 创建编码器并测试编码
        encoder = mock_encoder_class(self.model_path, self.device)
        result = encoder.encode_image("test_image.jpg")

        # 验证结果
        self.assertEqual(result, [[0.1, 0.2, 0.3]])
        mock_encoder.encode_image.assert_called_once_with("test_image.jpg")

    @patch("src.core.deepencoder.deepseek_encoder.DeepSeekEncoder")
    def test_encode_text(self, mock_encoder_class: MagicMock) -> None:
        """测试文本编码"""
        # 使用TestUtils创建模拟编码器
        mock_encoder = TestUtils.create_mock_encoder()
        mock_encoder.encode_text.return_value = [[0.4, 0.5, 0.6]]
        mock_encoder_class.return_value = mock_encoder

        # 创建编码器并测试编码
        encoder = mock_encoder_class(self.model_path, self.device)
        result = encoder.encode_text("test text")

        # 验证结果
        self.assertEqual(result, [[0.4, 0.5, 0.6]])
        mock_encoder.encode_text.assert_called_once_with("test text")


if __name__ == "__main__":
    unittest.main()
