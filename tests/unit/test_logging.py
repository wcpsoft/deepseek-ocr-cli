#!/usr/bin/env python3
"""
日志模块单元测试
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.core.logging import get_logger


class TestLogging(unittest.TestCase):
    """日志模块测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        self.logger = get_logger()

    def test_logger_instance(self) -> None:
        """测试日志记录器实例"""
        self.assertIsNotNone(self.logger)
        self.assertIsInstance(self.logger, object)

    def test_log_levels(self) -> None:
        """测试不同日志级别"""
        # 测试info级别日志
        self.logger.info("测试info日志")

        # 测试warning级别日志
        self.logger.warning("测试warning日志")

        # 测试error级别日志
        self.logger.error("测试error日志")

        # 测试debug级别日志
        # 默认情况下debug日志不会显示,除非设置DEBUG=TRUE
        os.environ["DEBUG"] = "TRUE"
        debug_logger = get_logger()  # 重新获取logger以应用新的环境变量
        debug_logger.debug("测试debug日志")
        os.environ["DEBUG"] = ""

    def test_logger_singleton(self) -> None:
        """测试日志记录器单例模式"""
        logger1 = get_logger()
        logger2 = get_logger()
        self.assertIs(logger1, logger2)  # 应该是同一个实例

    def test_log_file_output(self) -> None:
        """测试日志文件输出"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            # 重新初始化日志系统以使用临时文件
            with patch.dict(os.environ, {"LOG_FILE": log_file}):
                logger = get_logger()
                logger.info("测试日志文件输出")

                # 检查日志文件是否存在
                self.assertTrue(os.path.exists(log_file))
