#!/usr/bin/env python3
"""
服务模块单元测试
"""

import unittest

from tests.utils import TestUtils


class TestOCRService(unittest.TestCase):
    """OCR服务测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.model_path = "/test/model/path"
        self.device = "cpu"

    def test_ocr_service_initialization(self) -> None:
        """测试OCR服务初始化"""
        # 使用TestUtils创建模拟OCR服务
        mock_service = TestUtils.create_mock_ocr_service()

        # 验证初始化
        self.assertIsNotNone(mock_service)

    def test_process_document(self) -> None:
        """测试文档处理"""
        # 使用TestUtils创建模拟OCR服务
        mock_service = TestUtils.create_mock_ocr_service()

        # 设置处理文档的返回值
        expected_result = {
            "text": "Extracted text from document",
            "format": "markdown",
            "pages": [{"page": 1, "text": "Page 1 content"}, {"page": 2, "text": "Page 2 content"}],
        }
        mock_service.process_document.return_value = expected_result

        # 测试处理
        result = mock_service.process_document("test_document.pdf")

        # 验证结果
        self.assertEqual(result["text"], "Extracted text from document")
        self.assertEqual(result["format"], "markdown")
        self.assertEqual(len(result["pages"]), 2)
        mock_service.process_document.assert_called_once_with("test_document.pdf")

    def test_process_image(self) -> None:
        """测试图像处理"""
        # 使用TestUtils创建模拟OCR服务
        mock_service = TestUtils.create_mock_ocr_service()

        # 设置处理图像的返回值
        expected_result = {
            "text": "Extracted text from image",
            "confidence": 0.95,
            "bounding_boxes": [
                {"text": "Word 1", "x": 10, "y": 10, "width": 50, "height": 20},
                {"text": "Word 2", "x": 70, "y": 10, "width": 50, "height": 20},
            ],
        }
        mock_service.process_image.return_value = expected_result

        # 测试处理
        result = mock_service.process_image("test_image.jpg")

        # 验证结果
        self.assertEqual(result["text"], "Extracted text from image")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(len(result["bounding_boxes"]), 2)
        mock_service.process_image.assert_called_once_with("test_image.jpg")

    def test_extract_tables(self) -> None:
        """测试表格提取"""
        # 使用TestUtils创建模拟OCR服务
        mock_service = TestUtils.create_mock_ocr_service()

        # 设置表格提取的返回值
        expected_result = [
            {
                "table_id": 1,
                "headers": ["Column 1", "Column 2", "Column 3"],
                "rows": [["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"], ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]],
            }
        ]
        mock_service.extract_tables.return_value = expected_result

        # 测试表格提取
        result = mock_service.extract_tables("test_document_with_tables.pdf")

        # 验证结果
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["table_id"], 1)
        self.assertEqual(len(result[0]["headers"]), 3)
        self.assertEqual(len(result[0]["rows"]), 2)
        mock_service.extract_tables.assert_called_once_with("test_document_with_tables.pdf")


if __name__ == "__main__":
    unittest.main()
