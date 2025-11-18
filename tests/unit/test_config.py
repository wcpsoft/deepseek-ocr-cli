#!/usr/bin/env python3
"""
配置模块单元测试
"""

import os
import tempfile
import unittest

from src.core.config.app_config import AppConfig
from src.core.config.settings import Config


class TestAppConfig(unittest.TestCase):
    """应用配置测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

    def tearDown(self) -> None:
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_app_config_initialization(self) -> None:
        """测试应用配置初始化"""
        config = AppConfig()
        self.assertIsNotNone(config)
        self.assertIsInstance(config, AppConfig)

    def test_app_config_get_default_model_path(self) -> None:
        """测试获取默认模型路径"""
        config = AppConfig()
        model_path = config.get_default_model_path()
        self.assertIsInstance(model_path, str)
        self.assertTrue(len(model_path) > 0)

    def test_app_config_get_supported_formats(self) -> None:
        """测试获取支持的格式"""
        config = AppConfig()
        formats = config.get_supported_formats()
        self.assertIsInstance(formats, list)
        self.assertIn("pdf", formats)
        self.assertIn("jpg", formats)


class TestConfig(unittest.TestCase):
    """设置测试类"""

    def test_config_initialization(self) -> None:
        """测试Config初始化"""
        config = Config()
        self.assertIsNotNone(config)
        self.assertIsInstance(config, Config)

    def test_config_get_debug_mode(self) -> None:
        """测试获取调试模式"""
        config = Config()
        debug_mode = config.get_debug_mode()
        self.assertIsInstance(debug_mode, bool)

    def test_config_get_log_level(self) -> None:
        """测试获取日志级别"""
        config = Config()
        log_level = config.get_log_level()
        self.assertIsInstance(log_level, str)
        self.assertIn(log_level, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])


if __name__ == "__main__":
    unittest.main()
