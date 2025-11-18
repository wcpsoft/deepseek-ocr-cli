#!/usr/bin/env python3
"""
推理模块单元测试
"""

import unittest
from unittest.mock import MagicMock, patch

from src.core.inference.deepseek_ocr_inference import DeepSeekOCRInference


class TestDeepSeekOCRInference(unittest.TestCase):
    """DeepSeek OCR推理测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.model_path = "/test/model/path"
        self.device = "cpu"

    @patch("src.core.inference.deepseek_ocr_inference.DeepSeekOCRInference")
    def test_inference_initialization(self, mock_inference: MagicMock) -> None:
        """测试推理初始化"""
        # 模拟推理类
        mock_inference_instance = MagicMock()
        mock_inference.return_value = mock_inference_instance

        # 创建推理实例
        inference = DeepSeekOCRInference(self.model_path, self.device)

        # 验证初始化
        self.assertIsNotNone(inference)
        mock_inference.assert_called_once_with(self.model_path, self.device)

    @patch("src.core.inference.deepseek_ocr_inference.DeepSeekOCRInference")
    def test_process_image(self, mock_inference_class: MagicMock) -> None:
        """测试图像处理"""
        # 模拟推理实例和方法
        mock_inference = MagicMock()
        mock_inference.process_image.return_value = {"text": "OCR result", "confidence": 0.95}
        mock_inference_class.return_value = mock_inference

        # 创建推理并测试处理
        inference = mock_inference_class(self.model_path, self.device)
        result = inference.process_image("test_image.jpg")

        # 验证结果
        self.assertEqual(result["text"], "OCR result")
        self.assertEqual(result["confidence"], 0.95)
        mock_inference.process_image.assert_called_once_with("test_image.jpg")

    @patch("src.core.inference.deepseek_ocr_inference.DeepSeekOCRInference")
    def test_process_batch(self, mock_inference_class: MagicMock) -> None:
        """测试批量处理"""
        # 模拟推理实例和方法
        mock_inference = MagicMock()
        mock_inference.process_batch.return_value = [
            {"text": "Result 1", "confidence": 0.9},
            {"text": "Result 2", "confidence": 0.85},
        ]
        mock_inference_class.return_value = mock_inference

        # 创建推理并测试批量处理
        inference = mock_inference_class(self.model_path, self.device)
        images = ["image1.jpg", "image2.jpg"]
        results = inference.process_batch(images)

        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Result 1")
        self.assertEqual(results[1]["text"], "Result 2")
        mock_inference.process_batch.assert_called_once_with(images)


if __name__ == "__main__":
    unittest.main()
