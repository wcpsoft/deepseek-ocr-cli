#!/usr/bin/env python3
"""
日志模块
提供统一的日志记录功能
"""

import logging
import os
from typing import Optional


class OCRLogger:
    """OCR日志记录器"""

    _instance: Optional["OCRLogger"] = None
    _logger: logging.Logger | None = None

    def __new__(cls) -> "OCRLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self) -> None:
        """初始化日志记录器"""
        # 创建logger
        self._logger = logging.getLogger("deepseek_ocr")

        # 设置日志级别
        debug_mode = os.environ.get("DEBUG", "").upper() == "TRUE"
        self._logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        # 避免重复添加处理器
        if not self._logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

            # 创建格式器
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)

            # 添加处理器到logger
            self._logger.addHandler(console_handler)

    def debug(self, message: str) -> None:
        """记录debug级别日志"""
        if self._logger:
            self._logger.debug(message)

    def info(self, message: str) -> None:
        """记录info级别日志"""
        if self._logger:
            self._logger.info(message)

    def warning(self, message: str) -> None:
        """记录warning级别日志"""
        if self._logger:
            self._logger.warning(message)

    def error(self, message: str) -> None:
        """记录error级别日志"""
        if self._logger:
            self._logger.error(message)

    def critical(self, message: str) -> None:
        """记录critical级别日志"""
        if self._logger:
            self._logger.critical(message)


def get_logger() -> OCRLogger:
    """
    获取OCR日志记录器实例

    Returns:
        OCRLogger: 日志记录器实例
    """
    return OCRLogger()


# 兼容旧代码的函数
def setup_logging(level: str = "INFO") -> None:
    """
    设置日志记录（兼容旧代码）

    Args:
        level: 日志级别
    """
    # 根据传入的level设置环境变量
    os.environ["LOG_LEVEL"] = level.upper()

    # 如果level是DEBUG，设置DEBUG环境变量
    if level.upper() == "DEBUG":
        os.environ["DEBUG"] = "TRUE"

    # 初始化日志记录器
    get_logger()


def is_debug_mode() -> bool:
    """
    检查是否启用了调试模式

    Returns:
        bool: 是否启用了调试模式
    """
    return os.environ.get("DEBUG", "").upper() == "TRUE"


def is_ipdb_mode() -> bool:
    """
    检查是否启用了ipdb调试模式

    Returns:
        bool: 是否启用了ipdb调试模式
    """
    return os.environ.get("IPDB", "").upper() == "TRUE"


def debug_trace():
    """调试跟踪函数"""
    # 只有在ipdb模式下才进入调试器
    if not is_ipdb_mode():
        return

    try:
        import ipdb

        ipdb.set_trace()
    except ImportError:
        try:
            import pdb

            pdb.set_trace()
        except ImportError:
            # 如果都没有安装，只打印消息
            print("调试模式已启用，但未安装调试器 (ipdb 或 pdb)")


def debug_wrapper(func):
    """
    调试装饰器
    在函数执行前后设置断点

    Args:
        func: 要调试的函数
    """

    def wrapper(*args, **kwargs):
        if is_debug_mode():
            print(f"进入函数: {func.__name__}")
            debug_trace()

        try:
            result = func(*args, **kwargs)

            if is_debug_mode():
                print(f"函数 {func.__name__} 执行完成")
                debug_trace()

            return result
        except Exception as e:
            if is_debug_mode():
                print(f"函数 {func.__name__} 发生异常: {e}")
                debug_trace()
            raise

    return wrapper
