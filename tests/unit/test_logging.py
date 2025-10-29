#!/usr/bin/env python3
"""
日志模块单元测试
"""

import os
import sys
import unittest
from pathlib import Path

from src.core.logging import get_logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestLogging(unittest.TestCase):
    """日志模块测试类"""

    def setUp(self):
        """测试前准备"""
        self.logger = get_logger()

    def test_logger_instance(self):
        """测试日志记录器实例"""
        self.assertIsNotNone(self.logger)
        self.assertIsInstance(self.logger, object)

    def test_log_levels(self):
        """测试不同日志级别"""
        # 测试info级别日志
        self.logger.info("测试info日志")

        # 测试warning级别日志
        self.logger.warning("测试warning日志")

        # 测试error级别日志
        self.logger.error("测试error日志")

        # 测试debug级别日志
        # 默认情况下debug日志不会显示，除非设置DEBUG=TRUE
        os.environ["DEBUG"] = "TRUE"
        debug_logger = get_logger()  # 重新获取logger以应用新的环境变量
        debug_logger.debug("测试debug日志")

        # 恢复环境变量
        os.environ["DEBUG"] = ""

    def test_logger_singleton(self):
        """测试日志记录器单例模式"""
        logger1 = get_logger()
        logger2 = get_logger()
        self.assertIs(logger1, logger2)


if __name__ == "__main__":
    unittest.main()
