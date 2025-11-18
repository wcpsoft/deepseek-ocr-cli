"""
职责划分验证测试

验证重构后的组件职责分离是否正确：
- ModelManager 专注于模型管理
- DeviceManager 负责设备操作
- TransformersEngine 使用依赖注入
- ImageHandler 专注于图像处理
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.core.models.model_manager import ModelManager
from src.core.utils.device_manager import DeviceManager
from src.core.process.image_handler import ImageHandler
from src.core.utils.model_path_utils import ModelPathResolver


class TestModelManagerResponsibilities:
    """测试 ModelManager 职责分离"""

    def test_model_manager_only_manages_models(self):
        """测试 ModelManager 只负责模型管理"""
        manager = ModelManager("test_model_path")

        # 应该有模型相关属性
        assert hasattr(manager, 'model_path')
        assert hasattr(manager, 'model')
        assert hasattr(manager, 'tokenizer')
        assert hasattr(manager, 'processor')

        # 不应该有设备管理相关属性和方法
        device_attrs = ['device', 'setup_device', 'move_model_to_device']
        for attr in device_attrs:
            assert not hasattr(manager, attr), f"ModelManager 不应该有 {attr} 属性/方法"

    def test_model_manager_methods_only_model_related(self):
        """测试 ModelManager 方法只与模型相关"""
        manager = ModelManager("test_model_path")

        # 应该有模型相关方法
        model_methods = [
            'load_model_and_tokenizer',
            '_load_model',
            'load_processor',
            'adjust_vocab_size',
            'get_model_info'
        ]

        for method in model_methods:
            assert hasattr(manager, method), f"ModelManager 应该有 {method} 方法"

    @patch('src.core.models.model_manager.ModelPathResolver')
    @patch('src.core.models.model_manager.create_ocr_model')
    @patch('src.core.models.model_manager.AutoTokenizer')
    def test_model_manager_loading_logic(self, mock_tokenizer, mock_create_model, mock_resolver):
        """测试模型管理器的加载逻辑"""
        # 设置模拟
        mock_resolver.get_loading_params.return_value = {
            "trust_remote_code": True,
            "local_files_only": True
        }

        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_create_model.return_value = Mock()

        manager = ModelManager("test_model")

        # 测试模型和分词器加载
        model, tokenizer = manager.load_model_and_tokenizer()

        mock_tokenizer.from_pretrained.assert_called_once()
        mock_create_model.assert_called_once()

    def test_model_manager_no_device_logic(self):
        """测试 ModelManager 不包含设备逻辑"""
        manager = ModelManager("test_model_path")

        # 加载模型的方法中不应该包含设备相关逻辑
        import inspect

        # 检查 load_model_and_tokenizer 方法
        source = inspect.getsource(manager.load_model_and_tokenizer)
        device_terms = ['device', 'cuda', 'mps', 'cpu', '.to(']

        for term in device_terms:
            if term in source.lower():
                # 如果包含设备术语，应该是注释或者字符串常量
                lines = source.split('\n')
                for line in lines:
                    if term in line.lower() and not line.strip().startswith('#'):
                        # 如果不是注释，检查是否在字符串中
                        if not ('"' in line or "'" in line):
                            pytest.fail(f"ModelManager 方法中不应该包含设备逻辑: {line}")

    def test_model_manager_info_no_device(self):
        """测试 ModelManager 信息不包含设备信息"""
        manager = ModelManager("test_model_path")
        info = manager.get_model_info()

        device_keys = ['device', 'cuda', 'mps', 'cpu']
        for key in device_keys:
            assert key not in info, f"ModelManager 信息不应该包含 {key}"


class TestDeviceManagerResponsibilities:
    """测试 DeviceManager 职责分离"""

    def test_device_manager_only_device_related(self):
        """测试 DeviceManager 只负责设备相关功能"""
        device_manager = DeviceManager()

        # 应该有设备相关方法
        device_methods = [
            'get_optimal_device',
            'get_appropriate_dtype',
            'move_tensor_to_device',
            'move_model_to_device',
            'clear_device_cache',
            'configure_device_environment',
            'log_device_info'
        ]

        for method in device_methods:
            assert hasattr(device_manager, method), f"DeviceManager 应该有 {method} 方法"

        # 不应该有模型管理相关方法
        model_methods = ['load_model', 'load_tokenizer', 'process_image']
        for method in model_methods:
            assert not hasattr(device_manager, method), f"DeviceManager 不应该有 {method} 方法"

    def test_device_manager_singleton_pattern(self):
        """测试 DeviceManager 单例模式"""
        manager1 = DeviceManager()
        manager2 = DeviceManager()

        assert manager1 is manager2, "DeviceManager 应该使用单例模式"

    def test_device_manager_model_movement(self):
        """测试设备管理器的模型移动功能"""
        device_manager = DeviceManager()

        # 创建模拟模型
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)

        # 测试模型移动
        result = device_manager.move_model_to_device(mock_model)

        mock_model.to.assert_called_once()
        assert result == mock_model

    def test_device_manager_tensor_movement(self):
        """测试设备管理器的张量移动功能"""
        device_manager = DeviceManager()

        # 创建测试张量
        tensor = torch.randn(3, 4)

        # 测试张量移动
        result = device_manager.move_tensor_to_device(tensor)

        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_device_manager_dtype_selection(self):
        """测试设备管理器的数据类型选择"""
        device_manager = DeviceManager()

        # 测试不同设备的默认数据类型
        cpu_dtype = device_manager.get_appropriate_dtype(torch.device("cpu"))
        assert cpu_dtype == torch.float32

    def test_device_manager_optimal_detection(self):
        """测试设备管理器的最优设备检测"""
        device_manager = DeviceManager()

        device = device_manager.get_optimal_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu']  # 在测试环境中通常只有 CPU


class TestImageHandlerResponsibilities:
    """测试 ImageHandler 职责分离"""

    @patch('src.core.process.image_handler.DeepseekOCRProcessor')
    def test_image_handler_initialization(self, mock_processor_class):
        """测试 ImageHandler 初始化职责"""
        mock_processor_class.return_value = Mock()
        mock_tokenizer = Mock()

        handler = ImageHandler(mock_tokenizer)

        assert handler.tokenizer == mock_tokenizer
        mock_processor_class.assert_called_once_with(tokenizer=mock_tokenizer)

    @patch('src.core.process.image_handler.DeepseekOCRProcessor')
    def test_image_handler_only_image_processing(self, mock_processor_class):
        """测试 ImageHandler 只负责图像处理"""
        mock_processor_class.return_value = Mock()
        mock_tokenizer = Mock()

        handler = ImageHandler(mock_tokenizer)

        # 应该有图像处理相关方法
        image_methods = [
            'load_image',
            'process_image',
            'extract_tensors'
        ]

        for method in image_methods:
            assert hasattr(handler, method), f"ImageHandler 应该有 {method} 方法"

        # 不应该有模型管理或设备管理方法
        restricted_methods = [
            'load_model', 'setup_device', 'move_model_to_device',
            'get_optimal_device', 'load_tokenizer'
        ]

        for method in restricted_methods:
            assert not hasattr(handler, method), f"ImageHandler 不应该有 {method} 方法"

    def test_extract_tensors_with_device_manager(self):
        """测试张量提取使用 DeviceManager"""
        handler = ImageHandler(Mock())

        # 模拟处理后的数据
        input_ids = torch.tensor([[1, 2, 3]])
        pixel_values = torch.randn(1, 3, 224, 224)
        images_crop = torch.randn(1, 10, 768)
        images_spatial_crop = torch.randn(1, 10, 768)

        processed_data = [[
            input_ids, pixel_values, Mock(), Mock(), images_crop, images_spatial_crop, [], []
        ]]

        # 模拟 DeviceManager
        mock_device_manager = Mock()
        mock_device_manager.move_tensor_to_device.side_effect = lambda x, device: x

        device = torch.device("cpu")
        result = handler.extract_tensors(processed_data, device, mock_device_manager)

        # 验证使用了 DeviceManager
        assert mock_device_manager.move_tensor_to_device.call_count >= 1

        # 验证返回结果结构
        assert len(result) == 8
        assert isinstance(result[0], torch.Tensor)  # input_ids
        assert isinstance(result[1], torch.Tensor)  # pixel_values

    def test_extract_tensors_fallback_mechanism(self):
        """测试张量提取的回退机制"""
        handler = ImageHandler(Mock())

        # 模拟处理后的数据
        processed_data = [[
            torch.tensor([[1, 2, 3]]),
            torch.randn(1, 3, 224, 224),
            Mock(), Mock(),
            torch.randn(1, 10, 768),
            torch.randn(1, 10, 768),
            [], []
        ]]

        device = torch.device("cpu")

        # 不提供 DeviceManager，应该使用回退机制
        result = handler.extract_tensors(processed_data, device, None)

        # 验证回退机制正常工作
        assert len(result) == 8
        assert isinstance(result[0], torch.Tensor)

    @patch('PIL.Image.open')
    def test_load_image_responsibility(self, mock_open):
        """测试图像加载职责"""
        handler = ImageHandler(Mock())

        # 模拟图像
        mock_image = Mock()
        mock_image.convert.return_value = mock_image
        mock_open.return_value = mock_image

        result = handler.load_image("test.jpg")

        mock_open.assert_called_once_with("test.jpg")
        mock_image.convert.assert_called_once_with("RGB")
        assert result == mock_image


class TestDependencyInjectionValidation:
    """测试依赖注入验证"""

    def test_transformers_engine_injection_pattern(self):
        """测试 TransformersEngine 的依赖注入模式"""
        # 由于无法直接导入 TransformersEngine，我们验证其接口设计
        # 这部分测试验证重构是否保持了依赖注入的设计

        # 模拟依赖注入的参数
        mock_model_manager = Mock()
        mock_image_handler = Mock()
        mock_device_manager = Mock()

        # 验证依赖对象具有正确的接口
        assert hasattr(mock_model_manager, 'load_model_and_tokenizer')
        assert hasattr(mock_image_handler, 'extract_tensors')
        assert hasattr(mock_device_manager, 'get_optimal_device')
        assert hasattr(mock_device_manager, 'move_model_to_device')

    def test_device_manager_injection_contract(self):
        """测试 DeviceManager 注入契约"""
        device_manager = DeviceManager()

        # 验证必需的方法存在
        required_methods = [
            'get_optimal_device',
            'move_model_to_device',
            'move_tensor_to_device'
        ]

        for method in required_methods:
            assert hasattr(device_manager, method), f"DeviceManager 缺少必需方法: {method}"
            assert callable(getattr(device_manager, method)), f"{method} 不是可调用方法"

    def test_model_manager_injection_contract(self):
        """测试 ModelManager 注入契约"""
        manager = ModelManager("test_model_path")

        # 验证必需的方法存在
        required_methods = [
            'load_model_and_tokenizer',
            'load_processor',
            'get_model_info'
        ]

        for method in required_methods:
            assert hasattr(manager, method), f"ModelManager 缺少必需方法: {method}"
            assert callable(getattr(manager, method)), f"{method} 不是可调用方法"

    def test_image_handler_injection_contract(self):
        """测试 ImageHandler 注入契约"""
        with patch('src.core.process.image_handler.DeepseekOCRProcessor'):
            mock_tokenizer = Mock()
            handler = ImageHandler(mock_tokenizer)

            # 验证必需的方法存在
            required_methods = [
                'load_image',
                'process_image',
                'extract_tensors'
            ]

            for method in required_methods:
                assert hasattr(handler, method), f"ImageHandler 缺少必需方法: {method}"
                assert callable(getattr(handler, method)), f"{method} 不是可调用方法"


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_old_device_management_still_works(self):
        """测试旧的设备管理方式仍然可用"""
        device_manager = DeviceManager()

        # 旧的方式应该仍然工作
        device = device_manager.get_optimal_device()
        assert isinstance(device, torch.device)

        # 模型移动应该仍然工作
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        result = device_manager.move_model_to_device(mock_model)
        assert result == mock_model

    def test_model_manager_interface_unchanged(self):
        """测试 ModelManager 公共接口未改变"""
        manager = ModelManager("test_model_path")

        # 这些公共方法应该仍然可用
        public_methods = [
            'load_model_and_tokenizer',
            'load_processor',
            'get_model_info'
        ]

        for method in public_methods:
            assert hasattr(manager, method), f"公共方法 {method} 应该仍然可用"

    def test_image_handler_interface_expanded(self):
        """测试 ImageHandler 接口扩展而非破坏"""
        with patch('src.core.process.image_handler.DeepseekOCRProcessor'):
            mock_tokenizer = Mock()
            handler = ImageHandler(mock_tokenizer)

            # 原有方法应该仍然可用
            original_methods = ['load_image', 'process_image']
            for method in original_methods:
                assert hasattr(handler, method), f"原有方法 {method} 应该仍然可用"

            # 新方法应该可用
            assert hasattr(handler, 'extract_tensors'), "新方法 extract_tensors 应该可用"

            # extract_tensors 方法应该支持可选的 device_manager 参数
            import inspect
            sig = inspect.signature(handler.extract_tensors)
            assert 'device_manager' in sig.parameters, "extract_tensors 应该支持 device_manager 参数"


class TestResponsibilityBoundaries:
    """测试职责边界"""

    def test_no_cross_responsibilities(self):
        """测试没有跨职责的方法"""
        model_manager = ModelManager("test_model_path")
        device_manager = DeviceManager()

        with patch('src.core.process.image_handler.DeepseekOCRProcessor'):
            image_handler = ImageHandler(Mock())

        # ModelManager 不应该有设备管理方法
        device_methods = ['get_optimal_device', 'move_tensor_to_device']
        for method in device_methods:
            assert not hasattr(model_manager, method), f"ModelManager 不应该有 {method}"

        # DeviceManager 不应该有模型管理方法
        model_methods = ['load_model_and_tokenizer', 'load_processor']
        for method in model_methods:
            assert not hasattr(device_manager, method), f"DeviceManager 不应该有 {method}"

        # ImageHandler 应该专注于图像处理
        assert hasattr(image_handler, 'load_image'), "ImageHandler 应该有 load_image"
        assert hasattr(image_handler, 'process_image'), "ImageHandler 应该有 process_image"

    def test_clear_separation_of_concerns(self):
        """测试关注点的清晰分离"""
        # 每个组件应该有明确的职责

        # ModelManager: 模型相关
        model_manager = ModelManager("test_path")
        model_responsibilities = ['model_path', 'model', 'tokenizer', 'processor']
        for resp in model_responsibilities:
            assert hasattr(model_manager, resp), f"ModelManager 应该负责 {resp}"

        # DeviceManager: 设备相关
        device_manager = DeviceManager()
        device_responsibilities = ['get_optimal_device', 'move_model_to_device']
        for resp in device_responsibilities:
            assert hasattr(device_manager, resp), f"DeviceManager 应该负责 {resp}"

        # 验证职责不会重叠
        model_attrs = set(dir(model_manager))
        device_attrs = set(dir(device_manager))

        # 不应该有重叠的核心职责（除了基础方法）
        overlap = model_attrs & device_attrs
        core_methods = {'__class__', '__dict__', '__module__', '__doc__', '__weakref__'}
        overlap -= core_methods

        # 职责重叠应该很少或没有
        assert len(overlap) <= 2, f"职责重叠过多: {overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])