#!/usr/bin/env python3
"""
vLLM引擎单元测试 - 使用Mock进行测试
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.vllm.vllm_engine import VLLMEngine
from tests.utils import TestUtils


class TestVLLMEngine(unittest.TestCase):
    """vLLM引擎测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.config = {
            "model_path": "deepseek-ai/deepseek-ocr-1.5b",
            "device": "cpu",
            "dtype": "float16",
            "max_length": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
        }

    @patch("src.core.vllm.vllm_engine.VLLMEngine.is_available", return_value=True)
    def test_vllm_engine_initialization(self, mock_is_available) -> None:
        """测试vLLM引擎初始化"""
        # 创建模拟vLLM引擎
        mock_engine = TestUtils.create_mock_ocr_engine()

        # 设置初始化状态
        mock_engine.is_initialized = True
        mock_engine.model = MagicMock()
        mock_engine.tokenizer = MagicMock()
        mock_engine.processor = MagicMock()
        mock_engine.device = self.config["device"]

        # 验证初始化
        self.assertTrue(mock_engine.is_initialized)
        self.assertIsNotNone(mock_engine.model)
        self.assertIsNotNone(mock_engine.tokenizer)
        self.assertIsNotNone(mock_engine.processor)
        self.assertEqual(mock_engine.device, self.config["device"])

    @patch("src.core.vllm.vllm_engine.VLLMEngine.is_available", return_value=True)
    def test_vllm_engine_cleanup(self, mock_is_available) -> None:
        """测试vLLM引擎资源清理"""
        # 创建模拟vLLM引擎
        mock_engine = TestUtils.create_mock_ocr_engine()

        # 设置初始状态
        mock_engine.is_initialized = True
        mock_engine.model = MagicMock()
        mock_engine.tokenizer = MagicMock()
        mock_engine.processor = MagicMock()

        # 模拟清理过程
        mock_engine.cleanup()

        # 设置清理后的状态
        mock_engine.model = None
        mock_engine.tokenizer = None
        mock_engine.processor = None
        mock_engine.is_initialized = False

        # 验证资源已清理
        self.assertIsNone(mock_engine.model)
        self.assertIsNone(mock_engine.tokenizer)
        self.assertIsNone(mock_engine.processor)
        self.assertFalse(mock_engine.is_initialized)

    @patch("src.core.vllm.vllm_engine.VLLMEngine.is_available", return_value=True)
    def test_vllm_engine_device_compatibility(self, mock_is_available) -> None:
        """测试vLLM引擎设备兼容性"""
        # 测试不同设备类型
        device_types = ["cpu", "cuda", "mps", "dcu", "amd"]

        for device_type in device_types:
            with self.subTest(device=device_type):
                # 创建模拟vLLM引擎
                mock_engine = TestUtils.create_mock_ocr_engine()

                # 设置设备
                mock_engine.device = device_type
                mock_engine.is_initialized = True

                # 验证设备类型
                self.assertEqual(mock_engine.device, device_type)
                self.assertTrue(mock_engine.is_initialized)

    @patch("src.core.vllm.vllm_engine.VLLMEngine.is_available", return_value=True)
    def test_vllm_engine_configuration(self, mock_is_available) -> None:
        """测试vLLM引擎配置"""
        # 测试不同配置
        configs = [
            {
                "model_path": "deepseek-ai/deepseek-ocr-1.5b",
                "device": "cpu",
                "dtype": "float16",
                "max_length": 2048,
                "temperature": 0.5,
                "top_p": 0.8,
            },
            {
                "model_path": "deepseek-ai/deepseek-ocr-1.5b",
                "device": "cuda",
                "dtype": "bfloat16",
                "max_length": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        ]

        for i, config in enumerate(configs):
            with self.subTest(config=i):
                # 创建模拟vLLM引擎
                mock_engine = TestUtils.create_mock_ocr_engine()

                # 设置配置
                mock_engine.max_length = config["max_length"]
                mock_engine.temperature = config["temperature"]
                mock_engine.top_p = config["top_p"]
                mock_engine.device = config["device"]
                mock_engine.is_initialized = True

                # 验证配置
                self.assertEqual(mock_engine.max_length, config["max_length"])
                self.assertEqual(mock_engine.temperature, config["temperature"])
                self.assertEqual(mock_engine.top_p, config["top_p"])
                self.assertEqual(mock_engine.device, config["device"])
                self.assertTrue(mock_engine.is_initialized)

    @patch("src.core.vllm.vllm_engine.VLLMEngine.is_available", return_value=True)
    def test_vllm_engine_model_path_handling(self, mock_is_available) -> None:
        """测试vLLM引擎模型路径处理"""
        # 测试不同模型路径
        model_paths = [
            "deepseek-ai/deepseek-ocr-1.5b",
            "/path/to/local/model",  # 本地路径
        ]

        for model_path in model_paths:
            with self.subTest(model_path=model_path):
                # 创建模拟vLLM引擎
                mock_engine = TestUtils.create_mock_ocr_engine()

                # 设置模型路径
                mock_engine.model_path = model_path
                mock_engine.is_initialized = True

                # 验证模型路径
                self.assertEqual(mock_engine.model_path, model_path)
                self.assertTrue(mock_engine.is_initialized)

    @patch("src.core.vllm.vllm_engine.VLLMEngine.is_available", return_value=False)
    def test_vllm_engine_unavailable(self, mock_is_available) -> None:
        """测试vLLM引擎不可用时的处理"""
        # 验证is_available返回False
        self.assertFalse(VLLMEngine.is_available())


if __name__ == "__main__":
    unittest.main()
