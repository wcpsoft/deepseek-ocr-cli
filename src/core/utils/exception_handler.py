#!/usr/bin/env python3
"""
异常处理工具
提供统一的异常处理机制，特别是针对内存不足等特殊情况
"""

import traceback
from collections.abc import Callable
from typing import Any

import psutil
import torch

from src.core.logging import get_logger

logger = get_logger()


class OCRError(Exception):
    """
    OCR处理异常基类
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class OCRMemoryError(OCRError):
    """
    内存不足异常
    """

    def __init__(self, message: str = "内存不足", details: dict[str, Any] | None = None):
        super().__init__(message, "MEMORY_ERROR", details)


class ModelLoadError(OCRError):
    """
    模型加载异常
    """

    def __init__(self, message: str = "模型加载失败", details: dict[str, Any] | None = None):
        super().__init__(message, "MODEL_LOAD_ERROR", details)


class ImageProcessError(OCRError):
    """
    图像处理异常
    """

    def __init__(self, message: str = "图像处理失败", details: dict[str, Any] | None = None):
        super().__init__(message, "IMAGE_PROCESS_ERROR", details)


class ExceptionHandler:
    """
    异常处理器
    提供统一的异常处理机制
    """

    @staticmethod
    def handle_exception(func: Callable) -> Callable:
        """
        异常处理装饰器

        Args:
            func: 要装饰的函数

        Returns:
            装饰后的函数
        """

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                # 处理CUDA内存不足
                memory_info = ExceptionHandler._get_memory_info()
                logger.error(f"CUDA内存不足: {e!s}")
                logger.error(f"内存信息: {memory_info}")

                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 抛出内存不足异常
                raise OCRMemoryError(
                    "GPU内存不足，请尝试减小图像尺寸或使用CPU模式",
                    {"memory_info": memory_info, "original_error": str(e)},
                ) from e

            except MemoryError as e:
                # 处理系统内存不足
                memory_info = ExceptionHandler._get_memory_info()
                logger.error(f"系统内存不足: {e!s}")
                logger.error(f"内存信息: {memory_info}")

                # 抛出内存不足异常
                raise OCRMemoryError(
                    "系统内存不足，请尝试减小图像尺寸或关闭其他程序",
                    {"memory_info": memory_info, "original_error": str(e)},
                ) from e

            except Exception as e:
                # 处理其他异常
                error_type = type(e).__name__
                error_msg = str(e)
                error_traceback = traceback.format_exc()

                logger.error(f"处理过程中发生异常: {error_type}: {error_msg}")
                logger.debug(f"异常堆栈: {error_traceback}")

                # 根据异常类型抛出相应的自定义异常
                if "model" in error_msg.lower() or "load" in error_msg.lower():
                    raise ModelLoadError(
                        f"模型加载失败: {error_msg}",
                        details={"error_type": error_type, "original_error": str(e)},
                    ) from e
                elif "image" in error_msg.lower() or "process" in error_msg.lower():
                    raise ImageProcessError(
                        f"图像处理失败: {error_msg}",
                        details={"error_type": error_type, "original_error": str(e)},
                    ) from e
                else:
                    raise OCRError(
                        f"处理失败: {error_msg}",
                        details={"error_type": error_type, "original_error": str(e)},
                    ) from e

        return wrapper

    @staticmethod
    def _get_memory_info() -> dict[str, Any]:
        """
        获取内存信息

        Returns:
            内存信息字典
        """
        memory_info = {}

        # 系统内存
        memory = psutil.virtual_memory()
        memory_info["system_memory"] = {
            "total": f"{memory.total / (1024**3):.2f} GB",
            "available": f"{memory.available / (1024**3):.2f} GB",
            "percent": f"{memory.percent}%",
            "used": f"{memory.used / (1024**3):.2f} GB",
        }

        # GPU内存
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_reserved = torch.cuda.memory_reserved(i)
                gpu_allocated = torch.cuda.memory_allocated(i)

                gpu_memory.append(
                    {
                        "device": i,
                        "total": f"{gpu_mem / (1024**3):.2f} GB",
                        "reserved": f"{gpu_reserved / (1024**3):.2f} GB",
                        "allocated": f"{gpu_allocated / (1024**3):.2f} GB",
                        "free": f"{(gpu_mem - gpu_allocated) / (1024**3):.2f} GB",
                    }
                )

            memory_info["gpu_memory"] = gpu_memory

        return memory_info

    @staticmethod
    def check_memory_availability(required_memory_gb: float = 4.0) -> bool:
        """
        检查是否有足够的内存

        Args:
            required_memory_gb: 所需内存大小（GB）

        Returns:
            是否有足够内存
        """
        # 检查系统内存
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)

        if available_memory_gb < required_memory_gb:
            logger.warning(f"系统可用内存不足: {available_memory_gb:.2f} GB < {required_memory_gb} GB")
            return False

        # 检查GPU内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_allocated = torch.cuda.memory_allocated(i)
                available_gpu_memory_gb = (gpu_mem - gpu_allocated) / (1024**3)

                if available_gpu_memory_gb < required_memory_gb:
                    logger.warning(f"GPU {i} 可用内存不足: {available_gpu_memory_gb:.2f} GB < {required_memory_gb} GB")
                    return False

        return True


class SafeExecution:
    """
    安全执行器
    提供安全执行环境，在异常情况下进行适当处理
    """

    @staticmethod
    def safe_execute(
        func: Callable[..., object], *args: object, stop_on_error: bool = True, **kwargs: object
    ) -> object:
        """
        安全执行函数

        Args:
            func: 要执行的函数
            *args: 位置参数
            stop_on_error: 出错时是否停止
            **kwargs: 关键字参数

        Returns:
            函数执行结果

        Raises:
            OCRError: 执行失败时抛出
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"函数 {func.__name__} 执行失败: {e!s}"
            logger.error(error_msg)
            logger.debug(f"错误详情: {traceback.format_exc()}")

            if stop_on_error:
                raise OCRError(error_msg) from e
            return None
