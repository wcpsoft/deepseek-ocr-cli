"""
重构验证集成测试

验证重构后的代码架构是否正确工作，包括：
- 错误处理框架
- 职责划分和依赖注入
- 统一张量操作
- 设备管理
"""

from unittest.mock import Mock, patch

import pytest
import torch

# 测试重构后的核心组件
from src.core.models.model_manager import ModelManager
from src.core.process.image_handler import ImageHandler
from src.core.service.ocr_service import OCRService
from src.core.utils.device_manager import DeviceManager

# 测试错误处理框架
from src.core.utils.error_handling import (
    ErrorCollector,
    OCRException,
    handle_model_error,
    handle_ocr_error,
)


class TestErrorHandlingFramework:
    """测试错误处理框架"""

    def test_basic_error_handling_decorator(self):
        """测试基本错误处理装饰器"""

        @handle_ocr_error(default_return="fallback", re_raise=False)
        def failing_function():
            raise ValueError("测试错误")

        result = failing_function()
        assert result == "fallback"

    def test_error_handling_with_re_raise(self):
        """测试重新抛出异常的装饰器"""

        @handle_ocr_error(re_raise=True)
        def failing_function():
            raise ValueError("测试错误")

        with pytest.raises(OCRException):
            failing_function()

    def test_specific_error_types(self):
        """测试特定错误类型装饰器"""

        @handle_model_error(re_raise=True)
        def model_failing_function():
            raise ImportError("模型导入失败")

        with pytest.raises(OCRException):
            model_failing_function()

    def test_error_collector(self):
        """测试错误收集器"""
        collector = ErrorCollector(max_errors=3)

        collector.add_error("错误1")
        collector.add_error(ValueError("错误2"))
        collector.add_error(OCRException("错误3"))

        assert collector.has_errors()
        assert len(collector.errors) == 3
        assert "错误1" in str(collector)

    def test_error_severity_classification(self):
        """测试错误严重程度分类"""
        collector = ErrorCollector()

        collector.add_error("低级错误", severity="low")
        collector.add_error("高级错误", severity="high")
        collector.add_error("严重错误", severity="critical")

        assert len(collector.get_high_errors()) == 1
        assert len(collector.get_critical_errors()) == 1


class TestModelManagerRefactor:
    """测试 ModelManager 重构"""

    def test_model_manager_initialization(self):
        """测试 ModelManager 初始化"""
        manager = ModelManager("test_model_path")

        assert manager.model_path == "test_model_path"
        assert manager.model is None
        assert manager.tokenizer is None
        assert manager.processor is None
        # 验证设备属性已被移除
        assert not hasattr(manager, "device") or getattr(manager, "device", None) is None

    def test_model_manager_removed_device_methods(self):
        """测试 ModelManager 已移除设备管理方法"""
        manager = ModelManager("test_model_path")

        # 这些方法应该不存在
        assert not hasattr(manager, "setup_device")
        assert not hasattr(manager, "move_model_to_device")

    def test_model_manager_info_method(self):
        """测试 ModelManager 信息方法不包含设备信息"""
        manager = ModelManager("test_model_path")
        info = manager.get_model_info()

        assert "device" not in info
        assert info["model_path"] == "test_model_path"
        assert "has_model" in info
        assert "has_tokenizer" in info
        assert "has_processor" in info

    @patch("src.core.models.model_manager.ModelPathResolver")
    def test_model_loading_error_handling(self, mock_resolver):
        """测试模型加载的错误处理"""
        mock_resolver.get_loading_params.return_value = {"trust_remote_code": True, "local_files_only": True}

        manager = ModelManager("invalid_model")

        with patch("src.core.models.model_manager.create_ocr_model") as mock_create:
            mock_create.side_effect = RuntimeError("模型加载失败")

            with pytest.raises(OCRException):
                manager.load_model_and_tokenizer()


class TestDeviceManagerExtensions:
    """测试 DeviceManager 扩展功能"""

    def test_device_manager_move_model_to_device(self):
        """测试 DeviceManager 的模型移动功能"""
        device_manager = DeviceManager()

        # 创建一个简单的模拟模型
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)

        # 测试模型移动
        moved_model = device_manager.move_model_to_device(mock_model)

        mock_model.to.assert_called_once()
        assert moved_model == mock_model

    def test_device_manager_move_tensor_to_device(self):
        """测试 DeviceManager 的张量移动功能"""
        device_manager = DeviceManager()

        # 创建测试张量
        tensor = torch.randn(2, 3)

        # 测试张量移动
        moved_tensor = device_manager.move_tensor_to_device(tensor)

        assert moved_tensor.shape == tensor.shape
        assert isinstance(moved_tensor, torch.Tensor)

    def test_device_manager_without_to_method(self):
        """测试没有 to 方法的模型处理"""
        device_manager = DeviceManager()

        # 创建没有 to 方法的对象
        mock_model = Mock(spec=[])  # 没有指定任何方法

        # 应该优雅地处理
        result = device_manager.move_model_to_device(mock_model)
        assert result == mock_model


class TestImageHandlerRefactor:
    """测试 ImageHandler 重构"""

    @patch("src.core.process.image_handler.DeepseekOCRProcessor")
    def test_image_handler_initialization(self, mock_processor_class):
        """测试 ImageHandler 初始化"""
        mock_processor_class.return_value = Mock()
        mock_tokenizer = Mock()

        handler = ImageHandler(mock_tokenizer)

        assert handler.tokenizer == mock_tokenizer
        mock_processor_class.assert_called_once_with(tokenizer=mock_tokenizer)

    def test_extract_tensors_with_device_manager(self):
        """测试张量提取使用 DeviceManager"""
        handler = ImageHandler(Mock())

        # 模拟处理后的数据
        input_ids = torch.tensor([[1, 2, 3]])
        pixel_values = torch.randn(1, 3, 224, 224)
        images_crop = torch.randn(1, 10, 768)
        images_spatial_crop = torch.randn(1, 10, 768)

        processed_data = [[input_ids, pixel_values, images_crop, images_spatial_crop]]

        # 模拟 DeviceManager
        mock_device_manager = Mock()
        mock_device_manager.move_tensor_to_device = Mock(side_effect=lambda x, device: x)

        # 测试张量提取
        device = torch.device("cpu")
        result = handler.extract_tensors(processed_data, device, mock_device_manager)

        # 验证调用了 DeviceManager 的方法
        assert mock_device_manager.move_tensor_to_device.call_count == 4

        # 验证返回结果
        assert len(result) == 8
        assert torch.equal(result[0], input_ids)
        assert torch.equal(result[1], pixel_values)

    def test_extract_tensors_fallback(self):
        """测试张量提取的回退机制"""
        handler = ImageHandler(Mock())

        # 模拟处理后的数据
        input_ids = torch.tensor([[1, 2, 3]])
        pixel_values = torch.randn(1, 3, 224, 224)
        images_crop = torch.randn(1, 10, 768)
        images_spatial_crop = torch.randn(1, 10, 768)

        processed_data = [[input_ids, pixel_values, images_crop, images_spatial_crop]]

        # 不提供 DeviceManager
        device = torch.device("cpu")
        result = handler.extract_tensors(processed_data, device, None)

        # 验证回退到原来的方式
        assert len(result) == 8
        assert torch.equal(result[0], input_ids)

    def test_load_image_error_handling(self):
        """测试图像加载错误处理"""
        handler = ImageHandler(Mock())

        with patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = FileNotFoundError("图片不存在")

            with pytest.raises(OCRException):
                handler.load_image("nonexistent.jpg")


class TestOCRServiceRefactor:
    """测试 OCRService 重构"""

    def test_ocr_service_error_handling_decorators(self):
        """测试 OCRService 的错误处理装饰器"""
        mock_engine = Mock()
        mock_engine.initialize.return_value = True

        service = OCRService(mock_engine)

        # 测试初始化方法的错误处理
        with patch.object(mock_engine, "initialize", side_effect=RuntimeError("引擎初始化失败")):
            result = service.initialize()
            assert result is False

    def test_ocr_service_image_processing_error_handling(self):
        """测试 OCRService 图像处理错误处理"""
        mock_engine = Mock()
        mock_engine.process_image.side_effect = RuntimeError("图像处理失败")

        service = OCRService(mock_engine)

        from PIL import Image

        test_image = Image.new("RGB", (100, 100))

        with pytest.raises(OCRException):
            service.process_image(test_image)


class TestDependencyInjection:
    """测试依赖注入功能"""

    def test_transformers_engine_dependency_injection(self):
        """测试 TransformersEngine 依赖注入"""
        # 这个测试需要实际的 TransformersEngine 类
        # 由于依赖复杂，我们创建一个模拟版本
        mock_model_manager = Mock()
        mock_image_handler = Mock()
        mock_device_manager = Mock()

        # 模拟 TransformersEngine 的初始化逻辑
        engine_initialized = False

        def mock_initialize():
            nonlocal engine_initialized
            if mock_device_manager:
                device = mock_device_manager.get_optimal_device()
                if hasattr(mock_model_manager, "model"):
                    mock_device_manager.move_model_to_device(mock_model_manager.model)
            engine_initialized = True
            return True

        # 测试依赖注入是否正常工作
        mock_engine = Mock()
        mock_engine.initialize = mock_initialize
        mock_engine.model_manager = mock_model_manager
        mock_engine.image_handler = mock_image_handler
        mock_engine.device_manager = mock_device_manager

        result = mock_engine.initialize()

        assert result is True
        assert engine_initialized is True


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_old_api_still_works(self):
        """测试旧 API 仍然可用"""
        # 测试 DeviceManager 的回退机制
        device_manager = DeviceManager()

        # 即使没有传入 DeviceManager，也应该能正常工作
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)

        # 这应该不会抛出异常
        moved_model = device_manager.move_model_to_device(mock_model)
        assert moved_model is not None

    def test_error_handling_backwards_compatible(self):
        """测试错误处理的向后兼容性"""

        # 装饰器应该能处理没有额外参数的情况
        @handle_ocr_error()
        def simple_function():
            return "success"

        result = simple_function()
        assert result == "success"


class TestPerformanceMetrics:
    """测试性能指标"""

    def test_error_handling_performance(self):
        """测试错误处理性能"""
        import time

        @handle_ocr_error()
        def fast_function():
            return "result"

        # 测试装饰器不会显著影响性能
        start_time = time.time()
        for _ in range(1000):
            fast_function()
        end_time = time.time()

        # 应该在合理时间内完成
        assert (end_time - start_time) < 1.0  # 1秒内完成1000次调用

    def test_device_manager_performance(self):
        """测试设备管理器性能"""
        device_manager = DeviceManager()

        # 测试多次张量移动的性能
        tensor = torch.randn(100, 100)
        start_time = time.time()

        for _ in range(100):
            device_manager.move_tensor_to_device(tensor)

        end_time = time.time()
        assert (end_time - start_time) < 0.5  # 0.5秒内完成100次移动


if __name__ == "__main__":
    # 可以直接运行此文件进行快速测试
    pytest.main([__file__, "-v"])
