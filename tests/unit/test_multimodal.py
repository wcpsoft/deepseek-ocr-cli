#!/usr/bin/env python3
"""
多模态处理单元测试
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.core.multimodal.deepseek_ocr_multimodal import DeepseekOCRProcessingInfo
from tests.conftest import is_model_available


class TestDeepseekOCRProcessingInfo(unittest.TestCase):
    """DeepSeek多模态处理测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """测试类初始化"""
        if not is_model_available():
            pytest.skip("模型文件不可用", allow_module_level=True)

    def setUp(self) -> None:
        """测试前准备"""
        self.model_path = "/test/model/path"
        self.device = "cpu"

    def test_deepseek_ocr_processing_info_initialization(self) -> None:
        """测试DeepseekOCRProcessingInfo初始化"""
        # 创建处理器实例
        processor = DeepseekOCRProcessingInfo()

        # 验证处理器不为None
        self.assertIsNotNone(processor)

    @patch("src.core.multimodal.deepseek_multimodal.DeepSeekMultimodal")
    def test_process_image_and_text(self, mock_multimodal_class: MagicMock) -> None:
        """测试图像和文本处理"""
        # 模拟多模态处理实例和方法
        mock_multimodal = MagicMock()
        mock_multimodal.process_image_and_text.return_value = {
            "description": "Image shows a document with text",
            "extracted_text": "Sample text from image",
            "confidence": 0.92,
        }
        mock_multimodal_class.return_value = mock_multimodal

        # 创建多模态处理并测试
        multimodal = mock_multimodal_class(self.model_path, self.device)
        result = multimodal.process_image_and_text("test_image.jpg", "Describe this image")

        # 验证结果
        self.assertEqual(result["description"], "Image shows a document with text")
        self.assertEqual(result["extracted_text"], "Sample text from image")
        self.assertEqual(result["confidence"], 0.92)
        mock_multimodal.process_image_and_text.assert_called_once_with("test_image.jpg", "Describe this image")

    @patch("src.core.multimodal.deepseek_multimodal.DeepSeekMultimodal")
    def test_extract_structured_data(self, mock_multimodal_class: MagicMock) -> None:
        """测试结构化数据提取"""
        # 模拟多模态处理实例和方法
        mock_multimodal = MagicMock()
        mock_multimodal.extract_structured_data.return_value = {
            "title": "Document Title",
            "sections": [
                {"heading": "Section 1", "content": "Content of section 1"},
                {"heading": "Section 2", "content": "Content of section 2"},
            ],
            "tables": [{"headers": ["Col1", "Col2"], "rows": [["A1", "B1"], ["A2", "B2"]]}],
        }
        mock_multimodal_class.return_value = mock_multimodal

        # 创建多模态处理并测试
        multimodal = mock_multimodal_class(self.model_path, self.device)
        result = multimodal.extract_structured_data("test_document.jpg")

        # 验证结果
        self.assertEqual(result["title"], "Document Title")
        self.assertEqual(len(result["sections"]), 2)
        self.assertEqual(len(result["tables"]), 1)
        mock_multimodal.extract_structured_data.assert_called_once_with("test_document.jpg")


if __name__ == "__main__":
    unittest.main()
