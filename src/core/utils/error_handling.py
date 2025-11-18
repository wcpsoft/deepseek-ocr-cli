"""
统一的错误处理工具

提供错误处理装饰器、自定义异常类和错误处理工具函数，
用于减少代码中重复的 try-except 块。
"""

import functools
import logging
import traceback
from typing import Any, Callable, Optional, Type, Union, TypeVar, Dict, List
from enum import Enum

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OCRException(Exception):
    """OCR 相关异常的基类"""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 cause: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.cause = cause
        self.context = context or {}
        self.timestamp = None

    def __str__(self) -> str:
        base_msg = f"[{self.severity.value.upper()}] {super().__str__()}"
        if self.context:
            base_msg += f" | Context: {self.context}"
        if self.cause:
            base_msg += f" | Caused by: {self.cause}"
        return base_msg


class ModelLoadError(OCRException):
    """模型加载错误"""

    def __init__(self, message: str, model_path: Optional[str] = None,
                 cause: Optional[Exception] = None):
        context = {"model_path": model_path} if model_path else {}
        super().__init__(message, ErrorSeverity.HIGH, cause, context)


class ImageProcessError(OCRException):
    """图像处理错误"""

    def __init__(self, message: str, image_path: Optional[str] = None,
                 operation: Optional[str] = None, cause: Optional[Exception] = None):
        context = {}
        if image_path:
            context["image_path"] = image_path
        if operation:
            context["operation"] = operation
        super().__init__(message, ErrorSeverity.MEDIUM, cause, context)


class DeviceError(OCRException):
    """设备相关错误"""

    def __init__(self, message: str, device: Optional[str] = None,
                 cause: Optional[Exception] = None):
        context = {"device": device} if device else {}
        super().__init__(message, ErrorSeverity.HIGH, cause, context)


class ConfigurationError(OCRException):
    """配置错误"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 cause: Optional[Exception] = None):
        context = {"config_key": config_key} if config_key else {}
        super().__init__(message, ErrorSeverity.MEDIUM, cause, context)


class ValidationError(OCRException):
    """数据验证错误"""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, cause: Optional[Exception] = None):
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        super().__init__(message, ErrorSeverity.LOW, cause, context)


def handle_ocr_error(
    default_return: Any = None,
    re_raise: bool = False,
    log_level: int = logging.ERROR,
    exception_types: Optional[List[Type[Exception]]] = None,
    context: Optional[Dict[str, Any]] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Callable:
    """
    OCR 错误处理装饰器

    Args:
        default_return: 发生异常时的默认返回值
        re_raise: 是否重新抛出异常
        log_level: 日志级别
        exception_types: 要捕获的异常类型列表，None 表示捕获所有异常
        context: 额外的上下文信息
        severity: 错误严重程度

    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            func_context = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            if context:
                func_context.update(context)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 检查是否为指定的异常类型
                if exception_types and not any(isinstance(e, exc_type) for exc_type in exception_types):
                    raise

                # 构造错误消息
                error_msg = f"函数 {func.__name__} 执行失败: {str(e)}"

                # 记录日志
                logger.log(log_level, error_msg)
                if log_level <= logging.DEBUG:
                    logger.debug(f"异常堆栈:\n{traceback.format_exc()}")

                # 创建 OCR 异常
                if not isinstance(e, OCRException):
                    ocr_exception = OCRException(error_msg, severity, e, func_context)
                    if re_raise:
                        raise ocr_exception
                else:
                    if re_raise:
                        raise

                # 返回默认值
                if default_return is not None:
                    logger.debug(f"返回默认值: {default_return}")
                    return default_return

                # 如果没有默认值且不重新抛出，返回 None
                return None

        return wrapper
    return decorator


def handle_model_error(default_return: Any = None, re_raise: bool = True) -> Callable:
    """模型错误处理装饰器"""
    return handle_ocr_error(
        default_return=default_return,
        re_raise=re_raise,
        exception_types=[ImportError, RuntimeError, ValueError, ModelLoadError],
        context={"component": "model"},
        severity=ErrorSeverity.HIGH
    )


def handle_image_error(default_return: Any = None, re_raise: bool = False) -> Callable:
    """图像处理错误处理装饰器"""
    return handle_ocr_error(
        default_return=default_return,
        re_raise=re_raise,
        exception_types=[FileNotFoundError, ValueError, ImageProcessError],
        context={"component": "image_processing"},
        severity=ErrorSeverity.MEDIUM
    )


def handle_device_error(default_return: Any = None, re_raise: bool = True) -> Callable:
    """设备相关错误处理装饰器"""
    return handle_ocr_error(
        default_return=default_return,
        re_raise=re_raise,
        exception_types=[RuntimeError, DeviceError],
        context={"component": "device"},
        severity=ErrorSeverity.HIGH
    )


def handle_config_error(default_return: Any = None, re_raise: bool = False) -> Callable:
    """配置错误处理装饰器"""
    return handle_ocr_error(
        default_return=default_return,
        re_raise=re_raise,
        exception_types=[KeyError, ValueError, ConfigurationError],
        context={"component": "configuration"},
        severity=ErrorSeverity.MEDIUM
    )


def safe_execute(
    func: Callable[..., T],
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> T:
    """
    安全执行函数，捕获所有异常

    Args:
        func: 要执行的函数
        *args: 位置参数
        default_return: 默认返回值
        log_errors: 是否记录错误
        **kwargs: 关键字参数

    Returns:
        函数执行结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"安全执行 {func.__name__} 失败: {str(e)}")
        return default_return


class ErrorCollector:
    """错误收集器，用于批量处理时的错误收集"""

    def __init__(self):
        self.errors: List[OCRException] = []

    def add_error(self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None):
        """添加错误"""
        if isinstance(error, str):
            error = OCRException(error, context=context)
        elif not isinstance(error, OCRException):
            error = OCRException(str(error), cause=error, context=context)

        self.errors.append(error)

    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[OCRException]:
        """按严重程度获取错误"""
        return [error for error in self.errors if error.severity == severity]

    def get_critical_errors(self) -> List[OCRException]:
        """获取严重错误"""
        return self.get_errors_by_severity(ErrorSeverity.CRITICAL)

    def get_high_errors(self) -> List[OCRException]:
        """获取高级错误"""
        return self.get_errors_by_severity(ErrorSeverity.HIGH)

    def clear(self):
        """清空错误列表"""
        self.errors.clear()

    def __str__(self) -> str:
        if not self.errors:
            return "无错误"

        result = [f"收集到 {len(self.errors)} 个错误:"]
        for i, error in enumerate(self.errors, 1):
            result.append(f"  {i}. {error}")
        return "\n".join(result)


