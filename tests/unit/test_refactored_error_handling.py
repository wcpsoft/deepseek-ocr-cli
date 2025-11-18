"""
重构后的错误处理测试

专门测试重构后引入的错误处理框架
"""

import logging
from unittest.mock import patch

import pytest

from src.core.utils.error_handling import (  # 异常类; 装饰器; 工具函数; 别名
    ConfigurationError,
    DeviceError,
    ErrorCollector,
    ErrorSeverity,
    ImageProcessError,
    ModelLoadError,
    OCRException,
    ValidationError,
    config_error_handler,
    device_error_handler,
    handle_config_error,
    handle_device_error,
    handle_image_error,
    handle_model_error,
    handle_ocr_error,
    image_error_handler,
    model_error_handler,
    ocr_error_handler,
    safe_execute,
)


class TestErrorExceptions:
    """测试自定义异常类"""

    def test_ocr_exception_basic(self):
        """测试 OCR 异常基本功能"""
        exc = OCRException("测试消息")

        assert str(exc) == "[MEDIUM] 测试消息"
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.cause is None
        assert exc.context == {}

    def test_ocr_exception_with_severity(self):
        """测试带严重程度的 OCR 异常"""
        exc = OCRException("严重错误", ErrorSeverity.HIGH)

        assert str(exc) == "[HIGH] 严重错误"
        assert exc.severity == ErrorSeverity.HIGH

    def test_ocr_exception_with_cause(self):
        """测试带原因的 OCR 异常"""
        original_error = ValueError("原始错误")
        exc = OCRException("包装错误", cause=original_error)

        assert "Caused by: ValueError" in str(exc)
        assert exc.cause == original_error

    def test_ocr_exception_with_context(self):
        """测试带上下文的 OCR 异常"""
        context = {"file": "test.jpg", "line": 42}
        exc = OCRException("上下文错误", context=context)

        assert "Context:" in str(exc)
        assert exc.context == context

    def test_model_load_error(self):
        """测试模型加载错误"""
        exc = ModelLoadError("模型加载失败", model_path="/path/to/model")

        assert exc.severity == ErrorSeverity.HIGH
        assert exc.context["model_path"] == "/path/to/model"
        assert "ModelLoadError" in str(type(exc))

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


class TestErrorHandlingDecorators:
    """测试错误处理装饰器"""

    def test_handle_ocr_error_default_return(self):
        """测试默认返回值的错误处理"""

        @handle_ocr_error(default_return="fallback")
        def failing_function():
            raise ValueError("测试错误")

        result = failing_function()
        assert result == "fallback"

    def test_handle_ocr_error_re_raise(self):
        """测试重新抛出异常"""

        @handle_ocr_error(re_raise=True)
        def failing_function():
            raise ValueError("测试错误")

        with pytest.raises(OCRException) as exc_info:
            failing_function()

        assert "测试错误" in str(exc_info.value)
        assert exc_info.value.severity == ErrorSeverity.MEDIUM

    def test_handle_ocr_error_specific_exceptions(self):
        """测试特定异常类型处理"""

        @handle_ocr_error(exception_types=[ValueError])
        def function_with_value_error():
            raise ValueError("值错误")

        @handle_ocr_error(exception_types=[ValueError])
        def function_with_key_error():
            raise KeyError("键错误")

        # 应该处理 ValueError
        result1 = function_with_value_error()
        assert result1 is None

        # 应该重新抛出 KeyError
        with pytest.raises(KeyError):
            function_with_key_error()

    def test_handle_ocr_error_with_context(self):
        """测试带上下文的错误处理"""

        @handle_ocr_error(context={"component": "test_component"})
        def failing_function():
            raise ValueError("测试错误")

        with patch("src.core.utils.error_handling.logger") as mock_logger:
            failing_function()

            # 验证日志记录
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "failing_function" in call_args

    @patch("src.core.utils.error_handling.logger")
    def test_handle_ocr_error_logging(self, mock_logger):
        """测试错误处理日志记录"""

        @handle_ocr_error(log_level=logging.WARNING)
        def failing_function():
            raise ValueError("测试错误")

        failing_function()

        mock_logger.warning.assert_called_once()
        assert "failing_function" in mock_logger.warning.call_args[0][0]

    def test_specialized_decorators(self):
        """测试专用装饰器"""

        # 测试模型错误装饰器
        @handle_model_error()
        def model_function():
            raise ImportError("模型导入失败")

        with pytest.raises(OCRException):
            model_function()

        # 测试图像错误装饰器
        @handle_image_error(default_return="image_error")
        def image_function():
            raise FileNotFoundError("图片不存在")

        result = image_function()
        assert result == "image_error"

        # 测试设备错误装饰器
        @handle_device_error(re_raise=True)
        def device_function():
            raise RuntimeError("CUDA错误")

        with pytest.raises(OCRException):
            device_function()

        # 测试配置错误装饰器
        @handle_config_error()
        def config_function():
            raise KeyError("配置键缺失")

        with pytest.raises(OCRException):
            config_function()


class TestErrorCollector:
    """测试错误收集器"""

    def test_error_collector_basic(self):
        """测试错误收集器基本功能"""
        collector = ErrorCollector()

        assert not collector.has_errors()
        assert len(collector.errors) == 0
        assert str(collector) == "无错误"

    def test_error_collector_add_string_errors(self):
        """测试添加字符串错误"""
        collector = ErrorCollector()

        collector.add_error("错误1")
        collector.add_error("错误2")

        assert collector.has_errors()
        assert len(collector.errors) == 2
        assert all(isinstance(error, OCRException) for error in collector.errors)

    def test_error_collector_add_exception_errors(self):
        """测试添加异常错误"""
        collector = ErrorCollector()

        original_error = ValueError("原始错误")
        collector.add_error(original_error)

        assert collector.has_errors()
        assert len(collector.errors) == 1
        assert collector.errors[0].cause == original_error

    def test_error_collector_with_context(self):
        """测试带上下文的错误收集"""
        collector = ErrorCollector()
        context = {"operation": "test"}

        collector.add_error("测试错误", context)

        assert collector.errors[0].context == context

    def test_error_collector_severity_filtering(self):
        """测试按严重程度过滤错误"""
        collector = ErrorCollector()

        collector.add_error("低级错误", severity=ErrorSeverity.LOW)
        collector.add_error("中级错误", severity=ErrorSeverity.MEDIUM)
        collector.add_error("高级错误", severity=ErrorSeverity.HIGH)
        collector.add_error("严重错误", severity=ErrorSeverity.CRITICAL)

        low_errors = collector.get_errors_by_severity(ErrorSeverity.LOW)
        high_errors = collector.get_high_errors()
        critical_errors = collector.get_critical_errors()

        assert len(low_errors) == 1
        assert len(high_errors) == 1
        assert len(critical_errors) == 1
        assert low_errors[0].severity == ErrorSeverity.LOW
        assert high_errors[0].severity == ErrorSeverity.HIGH
        assert critical_errors[0].severity == ErrorSeverity.CRITICAL

    def test_error_collector_max_errors_limit(self):
        """测试最大错误数量限制"""
        collector = ErrorCollector(max_errors=3)

        # 添加超过限制的错误
        for i in range(5):
            collector.add_error(f"错误{i}")

        assert len(collector.errors) == 3
        # 最新的错误应该保留
        error_messages = [str(error).split("]")[1].strip() for error in collector.errors]
        assert "错误2" in error_messages
        assert "错误3" in error_messages
        assert "错误4" in error_messages

    def test_error_collector_clear(self):
        """测试清空错误"""
        collector = ErrorCollector()

        collector.add_error("测试错误")
        assert collector.has_errors()

        collector.clear()
        assert not collector.has_errors()
        assert len(collector.errors) == 0

    def test_error_collector_string_representation(self):
        """测试字符串表示"""
        collector = ErrorCollector()

        collector.add_error("错误1")
        collector.add_error("错误2")

        result = str(collector)
        assert "收集到 2 个错误" in result
        assert "错误1" in result
        assert "错误2" in result


class TestSafeExecute:
    """测试安全执行函数"""

    def test_safe_execute_success(self):
        """测试成功执行"""

        def test_func(x):
            return x * 2

        result = safe_execute(test_func, 5)
        assert result == 10

    def test_safe_execute_with_exception(self):
        """测试异常处理"""

        def failing_func():
            raise ValueError("测试错误")

        result = safe_execute(failing_func, default_return="fallback")
        assert result == "fallback"

    @patch("src.core.utils.error_handling.logger")
    def test_safe_execute_logging_disabled(self, mock_logger):
        """测试禁用日志记录"""

        def failing_func():
            raise ValueError("测试错误")

        safe_execute(failing_func, log_errors=False)
        mock_logger.error.assert_not_called()


class TestDecoratorAliases:
    """测试装饰器别名"""

    def test_decorator_aliases(self):
        """测试装饰器别名是否正常工作"""

        # 这些应该都能正常工作
        @ocr_error_handler(default_return="ocr_fallback")
        def ocr_func():
            raise ValueError("OCR错误")

        @model_error_handler(re_raise=True)
        def model_func():
            raise ImportError("模型错误")

        @image_error_handler(default_return="image_fallback")
        def image_func():
            raise FileNotFoundError("图像错误")

        @device_error_handler(re_raise=True)
        def device_func():
            raise RuntimeError("设备错误")

        @config_error_handler(default_return="config_fallback")
        def config_func():
            raise KeyError("配置错误")

        # 测试结果
        assert ocr_func() == "ocr_fallback"
        assert image_func() == "image_fallback"
        assert config_func() == "config_fallback"

        with pytest.raises(OCRException):
            model_func()

        with pytest.raises(OCRException):
            device_func()


class TestIntegrationScenarios:
    """测试集成场景"""

    def test_nested_error_handling(self):
        """测试嵌套错误处理"""

        @handle_model_error(re_raise=True)
        def outer_function():
            @handle_image_error(default_return="inner_fallback")
            def inner_function():
                raise FileNotFoundError("内部错误")

            return inner_function()

        # 外层装饰器应该捕获内层抛出的异常
        with pytest.raises(OCRException):
            outer_function()

    def test_error_collector_with_decorators(self):
        """测试错误收集器与装饰器的集成"""
        collector = ErrorCollector()

        @handle_ocr_error(default_return=None)
        def process_item(item_id):
            if item_id == 1:
                raise ValueError(f"处理项目 {item_id} 失败")
            return f"成功处理 {item_id}"

        # 处理多个项目，收集错误
        results = []
        for i in range(3):
            result = process_item(i)
            if result is None:
                collector.add_error(f"处理项目 {i} 失败")
            else:
                results.append(result)

        assert len(results) == 2  # 两个成功
        assert collector.has_errors()  # 一个失败

    def test_context_propagation(self):
        """测试上下文传播"""

        @handle_ocr_error(context={"operation": "test_operation"})
        def test_function():
            raise ValueError("测试错误")

        with patch("src.core.utils.error_handling.logger") as mock_logger:
            try:
                test_function()
            except:
                pass  # 忽略异常，只检查日志

            # 验证上下文被正确传递
            mock_logger.error.assert_called_once()
            log_message = mock_logger.error.call_args[0][0]
            assert "test_operation" in log_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
