#!/usr/bin/env python3
"""
OCR引擎集成测试
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.factory.ocr_engine_factory import OCREngineFactory
from tests.utils import TestUtils


class TestOCREngineIntegration(unittest.TestCase):
    """OCR引擎集成测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = "/test/model/path"
        self.device = "cpu"

    def tearDown(self) -> None:
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.core.factory.ocr_engine_factory.OCREngineFactory")
    def test_engine_factory_integration(self, mock_factory: MagicMock) -> None:
        """测试引擎工厂集成"""
        # 使用TestUtils创建模拟引擎
        mock_engine = TestUtils.create_mock_ocr_engine()
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_engine.return_value = mock_engine
        mock_factory.return_value = mock_factory_instance

        # 测试创建引擎
        factory = OCREngineFactory()
        engine = factory.create_engine("transformers", model_path=self.model_path, device=self.device)

        # 验证结果
        self.assertEqual(engine, mock_engine)
        mock_factory_instance.create_engine.assert_called_once_with(
            "transformers", model_path=self.model_path, device=self.device
        )

    @patch("src.core.service.ocr_service.OCRService")
    @patch("src.core.process.image_process.ImageProcessor")
    def test_service_processor_integration(
        self, mock_processor_class: MagicMock, mock_service_class: MagicMock
    ) -> None:
        """测试服务与处理器集成"""
        # 使用TestUtils创建模拟图像处理器
        mock_processor = TestUtils.create_mock_image_processor()
        mock_processor_class.return_value = mock_processor

        # 使用TestUtils创建模拟OCR服务
        mock_service = TestUtils.create_mock_ocr_service()
        mock_service_class.return_value = mock_service

        # 创建测试图像
        image_path = os.path.join(self.temp_dir, "test.jpg")
        Path(image_path).touch()

        # 测试集成
        processor = mock_processor_class()
        service = mock_service_class(self.model_path, self.device)

        # 加载并预处理图像
        image = processor.load_image(image_path)
        preprocessed = processor.preprocess_for_ocr(image)

        # 处理图像
        result = service.process_image(preprocessed)

        # 验证结果
        self.assertEqual(result["text"], "OCR result")
        self.assertEqual(result["confidence"], 0.95)

        # 验证方法调用
        mock_processor.load_image.assert_called_once_with(image_path)
        mock_processor.preprocess_for_ocr.assert_called_once_with("mock_image")
        mock_service.process_image.assert_called_once_with("preprocessed_image")

    @patch("src.cli.pdf_converter.PDFConverter")
    @patch("src.core.service.ocr_service.OCRService")
    def test_pdf_converter_service_integration(
        self, mock_service_class: MagicMock, mock_converter_class: MagicMock
    ) -> None:
        """测试PDF转换器与服务集成"""
        # 使用TestUtils创建模拟PDF转换器
        mock_converter = TestUtils.create_mock_pdf_converter()
        mock_converter_class.return_value = mock_converter

        # 使用TestUtils创建模拟OCR服务
        mock_service = TestUtils.create_mock_ocr_service()
        mock_service_class.return_value = mock_service

        # 创建测试PDF
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        Path(pdf_path).touch()

        # 测试集成
        converter = mock_converter_class()
        service = mock_service_class(self.model_path, self.device)

        # 验证PDF有效性
        is_valid = converter.is_valid_pdf(pdf_path)
        self.assertTrue(is_valid)

        # 转换PDF为图像
        output_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(output_dir, exist_ok=True)
        image_paths = converter.convert_to_images(pdf_path, output_dir)

        # 处理每个图像
        results = []
        for image_path in image_paths:
            result = service.process_image(image_path)
            results.append(result)

        # 验证结果
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["text"], "Page text")
            self.assertEqual(result["confidence"], 0.9)

        # 验证方法调用
        mock_converter.is_valid_pdf.assert_called_once_with(pdf_path)
        mock_converter.convert_to_images.assert_called_once_with(pdf_path, output_dir)
        self.assertEqual(mock_service.process_image.call_count, 2)


if __name__ == "__main__":
    unittest.main()
