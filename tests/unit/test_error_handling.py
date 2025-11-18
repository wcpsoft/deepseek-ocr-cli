"""
错误处理工具的单元测试
"""

from unittest.mock import patch

import pytest

from src.core.utils.error_handling import (
    ConfigurationError,
    DeviceError,
    ErrorCollector,
    ErrorSeverity,
    ImageProcessError,
    ModelLoadError,
    OCRException,
    ValidationError,
    handle_device_error,
    handle_image_error,
    handle_model_error,
    handle_ocr_error,
    safe_execute,
)


class TestOCRException:
    """测试 OCR 异常基类"""

    def test_basic_exception(self):
        """测试基本异常创建"""
        exc = OCRException("测试错误")
        assert str(exc) == "[MEDIUM] 测试错误"
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.cause is None
        assert exc.context == {}

    def test_exception_with_severity(self):
        """测试带严重程度的异常"""
        exc = OCRException("严重错误", severity=ErrorSeverity.HIGH)
        assert str(exc) == "[HIGH] 严重错误"
        assert exc.severity == ErrorSeverity.HIGH

    def test_exception_with_cause(self):
        """测试带原因的异常"""
        cause = ValueError("原始错误")
        exc = OCRException("包装错误", cause=cause)
        assert "Caused by: ValueError" in str(exc)
        assert exc.cause == cause

    def test_exception_with_context(self):
        """测试带上下文的异常"""
        context = {"file": "test.jpg", "line": 42}
        exc = OCRException("上下文错误", context=context)
        assert "Context: {'file': 'test.jpg', 'line': 42}" in str(exc)
        assert exc.context == context

    def test_exception_full(self):
        """测试完整异常"""
        cause = RuntimeError("原因")
        context = {"operation": "load_model"}
        exc = OCRException("完整错误", ErrorSeverity.CRITICAL, cause, context)

        assert "[CRITICAL]" in str(exc)
        assert "Context:" in str(exc)
        assert "Caused by:" in str(exc)


class TestSpecificExceptions:
    """测试具体的异常类"""

    def test_model_load_error(self):
        """测试模型加载错误"""
        exc = ModelLoadError("模型加载失败", model_path="/path/to/model")
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.context["model_path"] == "/path/to/model"

    def test_image_process_error(self):
        """测试图像处理错误"""
        exc = ImageProcessError("图像处理失败", image_path="test.jpg", operation="resize")
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.context["image_path"] == "test.jpg"
        assert exc.context["operation"] == "resize"

    def test_device_error(self):
        """测试设备错误"""
        exc = DeviceError("设备不可用", device="cuda:0")
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.context["device"] == "cuda:0"

    def test_configuration_error(self):
        """测试配置错误"""
        exc = ConfigurationError("配置项缺失", config_key="model.path")
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.context["config_key"] == "model.path"

    def test_validation_error(self):
        """测试验证错误"""
        exc = ValidationError("验证失败", field="age", value=-1)
        assert exc.severity == ErrorSeverity.LOW
        assert exc.context["field"] == "age"
        assert exc.context["value"] == "-1"


class TestErrorHandlingDecorator:
    """测试错误处理装饰器"""

    def test_successful_execution(self):
        """测试成功执行的函数"""

        @handle_ocr_error()
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_exception_with_default_return(self):
        """测试异常时返回默认值"""

        @handle_ocr_error(default_return="default")
        def test_func():
            raise ValueError("测试错误")

        result = test_func()
        assert result == "default"

    def test_exception_with_re_raise(self):
        """测试异常时重新抛出"""

        @handle_ocr_error(re_raise=True)
        def test_func():
            raise ValueError("测试错误")

        with pytest.raises(OCRException):
            test_func()

    def test_exception_with_specific_types(self):
        """测试特定异常类型的捕获"""

        @handle_ocr_error(exception_types=[ValueError])
        def test_func():
            raise KeyError("键错误")

        with pytest.raises(KeyError):
            test_func()

    @patch("src.core.utils.error_handling.logger")
    def test_exception_logging(self, mock_logger):
        """测试异常日志记录"""

        @handle_ocr_error()
        def test_func():
            raise ValueError("测试错误")

        test_func()
        mock_logger.error.assert_called_once()

    def test_context_in_decorator(self):
        """测试装饰器中的上下文"""

        @handle_ocr_error(context={"custom": "value"})
        def test_func():
            raise ValueError("测试错误")

        # 这个测试主要检查装饰器不会因为上下文而失败
        result = test_func()
        assert result is None


class TestSpecializedDecorators:
    """测试专用装饰器"""

    @patch("src.core.utils.error_handling.logger")
    def test_model_error_decorator(self, mock_logger):
        """测试模型错误装饰器"""

        @handle_model_error()
        def test_func():
            raise ImportError("模块导入失败")

        result = test_func()
        assert result is None
        mock_logger.error.assert_called_once()

    @patch("src.core.utils.error_handling.logger")
    def test_image_error_decorator(self, mock_logger):
        """测试图像错误装饰器"""

        @handle_image_error(default_return="fallback")
        def test_func():
            raise FileNotFoundError("文件不存在")

        result = test_func()
        assert result == "fallback"
        mock_logger.error.assert_called_once()

    @patch("src.core.utils.error_handling.logger")
    def test_device_error_decorator(self, mock_logger):
        """测试设备错误装饰器"""

        @handle_device_error(re_raise=True)
        def test_func():
            raise RuntimeError("CUDA 错误")

        with pytest.raises(OCRException):
            test_func()
        mock_logger.error.assert_called_once()


class TestSafeExecute:
    """测试安全执行函数"""

    def test_successful_execution(self):
        """测试成功执行"""

        def test_func():
            return "result"

        result = safe_execute(test_func)
        assert result == "result"

    def test_exception_handling(self):
        """测试异常处理"""

        def test_func():
            raise ValueError("错误")

        result = safe_execute(test_func, default_return="default")
        assert result == "default"

    @patch("src.core.utils.error_handling.logger")
    def test_logging_disabled(self, mock_logger):
        """测试禁用日志记录"""

        def test_func():
            raise ValueError("错误")

        safe_execute(test_func, log_errors=False)
        mock_logger.error.assert_not_called()


class TestErrorCollector:
    """测试错误收集器"""

    def test_empty_collector(self):
        """测试空收集器"""
        collector = ErrorCollector()
        assert not collector.has_errors()
        assert len(collector.errors) == 0
        assert str(collector) == "无错误"

    def test_add_string_error(self):
        """测试添加字符串错误"""
        collector = ErrorCollector()
        collector.add_error("测试错误")

        assert collector.has_errors()
        assert len(collector.errors) == 1
        assert isinstance(collector.errors[0], OCRException)

    def test_add_exception_error(self):
        """测试添加异常错误"""
        collector = ErrorCollector()
        original_error = ValueError("原始错误")
        collector.add_error(original_error)

        assert collector.has_errors()
        assert len(collector.errors) == 1
        assert collector.errors[0].cause == original_error

    def test_add_error_with_context(self):
        """测试带上下文的错误"""
        collector = ErrorCollector()
        context = {"file": "test.txt"}
        collector.add_error("测试错误", context)

        assert collector.errors[0].context == context

    def test_get_errors_by_severity(self):
        """测试按严重程度获取错误"""
        collector = ErrorCollector()
        collector.add_error("低级错误", ErrorSeverity.LOW)
        collector.add_error("高级错误", ErrorSeverity.HIGH)

        low_errors = collector.get_errors_by_severity(ErrorSeverity.LOW)
        high_errors = collector.get_errors_by_severity(ErrorSeverity.HIGH)

        assert len(low_errors) == 1
        assert len(high_errors) == 1
        assert low_errors[0].severity == ErrorSeverity.LOW
        assert high_errors[0].severity == ErrorSeverity.HIGH

    def test_critical_errors_helper(self):
        """测试获取严重错误的辅助方法"""
        collector = ErrorCollector()
        collector.add_error("普通错误", ErrorSeverity.MEDIUM)
        collector.add_error("严重错误", ErrorSeverity.CRITICAL)

        critical = collector.get_critical_errors()
        assert len(critical) == 1
        assert critical[0].severity == ErrorSeverity.CRITICAL

    def test_high_errors_helper(self):
        """测试获取高级错误的辅助方法"""
        collector = ErrorCollector()
        collector.add_error("中级错误", ErrorSeverity.MEDIUM)
        collector.add_error("高级错误", ErrorSeverity.HIGH)

        high = collector.get_high_errors()
        assert len(high) == 1
        assert high[0].severity == ErrorSeverity.HIGH

    def test_max_errors_limit(self):
        """测试最大错误数量限制"""
        collector = ErrorCollector(max_errors=3)

        # 添加超过限制的错误
        for i in range(5):
            collector.add_error(f"错误 {i}")

        assert len(collector.errors) == 3
        assert "错误 2" not in [str(error) for error in collector.errors]  # 最早的错误被移除
        assert "错误 4" in [str(error) for error in collector.errors]  # 最新的错误保留

    def test_clear_errors(self):
        """测试清空错误"""
        collector = ErrorCollector()
        collector.add_error("测试错误")
        assert collector.has_errors()

        collector.clear()
        assert not collector.has_errors()
        assert len(collector.errors) == 0

    def test_string_representation(self):
        """测试字符串表示"""
        collector = ErrorCollector()
        collector.add_error("错误 1")
        collector.add_error("错误 2")

        result = str(collector)
        assert "收集到 2 个错误" in result
        assert "错误 1" in result
        assert "错误 2" in result


class TestIntegration:
    """集成测试"""

    @patch("src.core.utils.error_handling.logger")
    def test_decorator_with_error_collector(self, mock_logger):
        """测试装饰器与错误收集器的集成"""
        collector = ErrorCollector()

        @handle_ocr_error(default_return=None)
        def failing_function(index):
            if index == 1:
                raise ValueError(f"错误 {index}")
            return f"成功 {index}"

        # 执行多个操作，收集错误
        for i in range(3):
            try:
                result = failing_function(i)
                if result is None:
                    collector.add_error(f"操作 {i} 失败")
            except Exception as e:
                collector.add_error(f"操作 {i} 异常: {e}")

        assert collector.has_errors()
        assert len(collector.errors) == 1  # 只有 index=1 会失败
