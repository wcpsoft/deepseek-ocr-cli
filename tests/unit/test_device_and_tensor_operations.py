"""
设备管理和张量操作测试

专门测试重构后的设备管理和统一张量操作功能
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from src.core.utils.device_manager import DeviceManager


class TestDeviceManagerCoreFunctionality:
    """测试 DeviceManager 核心功能"""

    def test_device_manager_singleton(self):
        """测试 DeviceManager 单例模式"""
        manager1 = DeviceManager()
        manager2 = DeviceManager()

        assert manager1 is manager2, "DeviceManager 应该使用单例模式"

    def test_device_manager_initialization(self):
        """测试 DeviceManager 初始化"""
        manager = DeviceManager()

        # 应该有初始化标记
        assert hasattr(manager, 'initialized')
        assert manager.initialized is True

        # 应该有单例实例
        assert DeviceManager._instance is not None

    def test_device_manager_device_detection(self):
        """测试设备检测功能"""
        manager = DeviceManager()

        device = manager.get_optimal_device()
        assert isinstance(device, torch.device)

        # 在测试环境中通常是 CPU
        assert device.type in ['cpu']

    def test_device_manager_dtype_selection(self):
        """测试数据类型选择"""
        manager = DeviceManager()

        # 测试 CPU 数据类型
        cpu_dtype = manager.get_appropriate_dtype(torch.device("cpu"))
        assert cpu_dtype == torch.float32

        # 测试未指定设备的情况
        default_dtype = manager.get_appropriate_dtype()
        assert default_dtype == torch.float32

    def test_device_manager_device_validation(self):
        """测试设备验证功能"""
        manager = DeviceManager()

        # 测试设备验证
        cpu_device = torch.device("cpu")
        assert manager.is_device_available(cpu_device) is True

        # 测试无效设备
        invalid_device = Mock()
        invalid_device.type = "invalid"
        assert manager.is_device_available(invalid_device) is False


class TestTensorMovementOperations:
    """测试张量移动操作"""

    def test_move_tensor_to_device_basic(self):
        """测试基本张量移动"""
        manager = DeviceManager()

        # 创建测试张量
        original_tensor = torch.randn(3, 4)
        original_device = original_tensor.device

        # 移动张量
        moved_tensor = manager.move_tensor_to_device(original_tensor)

        # 验证张量
        assert isinstance(moved_tensor, torch.Tensor)
        assert moved_tensor.shape == original_tensor.shape
        assert torch.equal(moved_tensor, original_tensor)

    def test_move_tensor_to_specific_device(self):
        """测试移动到指定设备"""
        manager = DeviceManager()

        original_tensor = torch.randn(2, 3)
        target_device = torch.device("cpu")

        moved_tensor = manager.move_tensor_to_device(original_tensor, target_device)

        assert moved_tensor.device == target_device
        assert torch.equal(moved_tensor, original_tensor)

    def test_move_tensor_mps_dtype_handling(self):
        """测试 MPS 设备的数据类型处理"""
        manager = DeviceManager()

        # 创建 float16 张量
        float16_tensor = torch.randn(2, 3, dtype=torch.float16)
        mps_device = torch.device("mps")

        # 移动到 MPS 设备应该转换 dtype
        moved_tensor = manager.move_tensor_to_device(float16_tensor, mps_device)

        # MPS 设备上应该使用 float32
        if moved_tensor.device.type == "mps":
            assert moved_tensor.dtype == torch.float32
        else:
            # 在 CPU 上应该保持原样
            assert moved_tensor.dtype == torch.float16

    def test_move_tensor_cuda_dtype_handling(self):
        """测试 CUDA 设备的数据类型处理"""
        manager = DeviceManager()

        original_tensor = torch.randn(2, 3)
        cuda_device = torch.device("cuda")

        # 移动到 CUDA 设备
        moved_tensor = manager.move_tensor_to_device(original_tensor, cuda_device)

        # 验证张量在正确的设备上（如果 CUDA 可用）
        if torch.cuda.is_available():
            assert moved_tensor.device.type == "cuda"
        else:
            # 如果 CUDA 不可用，应该在 CPU 上
            assert moved_tensor.device.type == "cpu"

    def test_move_tensor_preserves_values(self):
        """测试张量移动保持数值不变"""
        manager = DeviceManager()

        # 创建具有特定值的张量
        original_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_device = torch.device("cpu")

        moved_tensor = manager.move_tensor_to_device(original_tensor, target_device)

        assert torch.allclose(moved_tensor, original_tensor)
        assert moved_tensor.shape == original_tensor.shape

    def test_move_tensor_different_dtypes(self):
        """测试不同数据类型的张量移动"""
        manager = DeviceManager()

        dtypes_to_test = [torch.float32, torch.float16, torch.int32, torch.long]
        target_device = torch.device("cpu")

        for dtype in dtypes_to_test:
            original_tensor = torch.randn(2, 3, dtype=dtype)
            moved_tensor = manager.move_tensor_to_device(original_tensor, target_device)

            assert moved_tensor.dtype == dtype
            assert moved_tensor.device == target_device

    def test_move_tensor_edge_cases(self):
        """测试张量移动的边界情况"""
        manager = DeviceManager()

        # 空张量
        empty_tensor = torch.empty(0, 3)
        moved_empty = manager.move_tensor_to_device(empty_tensor)
        assert moved_empty.shape[0] == 0

        # 单元素张量
        single_tensor = torch.tensor([1.0])
        moved_single = manager.move_tensor_to_device(single_tensor)
        assert moved_single.numel() == 1

        # 大张量
        large_tensor = torch.randn(1000, 1000)
        moved_large = manager.move_tensor_to_device(large_tensor)
        assert moved_large.shape == large_tensor.shape


class TestModelMovementOperations:
    """测试模型移动操作"""

    def test_move_model_to_device_basic(self):
        """测试基本模型移动"""
        manager = DeviceManager()

        # 创建简单的模型
        model = nn.Linear(10, 5)
        original_device = next(model.parameters()).device

        # 移动模型
        moved_model = manager.move_model_to_device(model)

        # 验证模型
        assert isinstance(moved_model, nn.Module)
        assert moved_model is model  # 应该是同一个对象

    def test_move_model_to_specific_device(self):
        """测试移动模型到指定设备"""
        manager = DeviceManager()

        model = nn.Linear(10, 5)
        target_device = torch.device("cpu")

        moved_model = manager.move_model_to_device(model, target_device)

        # 验证模型参数在目标设备上
        for param in moved_model.parameters():
            assert param.device == target_device

    def test_move_model_without_to_method(self):
        """测试没有 to 方法的模型处理"""
        manager = DeviceManager()

        # 创建没有 to 方法的对象
        class MockModel:
            def __init__(self):
                self.param = torch.randn(3, 4)

        mock_model = MockModel()

        # 应该优雅地处理
        result = manager.move_model_to_device(mock_model)
        assert result is mock_model

    def test_move_model_complex_network(self):
        """测试复杂网络的移动"""
        manager = DeviceManager()

        # 创建更复杂的模型
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        target_device = torch.device("cpu")
        moved_model = manager.move_model_to_device(model, target_device)

        # 验证所有参数都在目标设备上
        for param in moved_model.parameters():
            assert param.device == target_device

        # 验证缓冲区也在目标设备上
        for buffer in moved_model.buffers():
            assert buffer.device == target_device

    def test_move_model_preserves_functionality(self):
        """测试模型移动后功能保持"""
        manager = DeviceManager()

        # 创建可测试的模型
        model = nn.Linear(5, 1)
        original_bias = model.bias.clone()

        target_device = torch.device("cpu")
        moved_model = manager.move_model_to_device(model, target_device)

        # 测试模型仍然可以工作
        input_tensor = torch.randn(1, 5)
        output = moved_model(input_tensor)

        assert output.shape == (1, 1)
        assert moved_model.bias.device == target_device

    def test_move_model_device_failure_handling(self):
        """测试模型移动时设备失败的处理"""
        manager = DeviceManager()

        model = nn.Linear(5, 1)

        # 模拟设备失败
        with patch.object(model, 'to', side_effect=RuntimeError("设备不可用")):
            # 应该优雅地处理错误
            with pytest.raises(RuntimeError):
                manager.move_model_to_device(model)


class TestDeviceCacheManagement:
    """测试设备缓存管理"""

    def test_clear_device_cache_cpu(self):
        """测试 CPU 设备缓存清理"""
        manager = DeviceManager()

        # CPU 设备的缓存清理应该不会出错
        manager.clear_device_cache()

    def test_clear_device_cache_cuda(self):
        """测试 CUDA 设备缓存清理"""
        manager = DeviceManager()

        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                manager.clear_device_cache()
                mock_empty_cache.assert_called_once()

    def test_clear_device_cache_mps(self):
        """测试 MPS 设备缓存清理"""
        manager = DeviceManager()

        # 模拟 MPS 环境
        with patch('torch.backends.mps.is_available', return_value=True):
            mock_mps = Mock()
            mock_mps.empty_cache = Mock()

            with patch('torch.mps', mock_mps):
                manager.clear_device_cache()
                mock_mps.empty_cache.assert_called_once()


class TestDeviceConfiguration:
    """测试设备配置功能"""

    def test_configure_device_environment(self):
        """测试设备环境配置"""
        manager = DeviceManager()

        # 配置环境应该不会出错
        manager.configure_device_environment()

    def test_log_device_info(self):
        """测试设备信息日志记录"""
        manager = DeviceManager()

        # 日志记录应该不会出错
        manager.log_device_info()

    def test_get_device_config(self):
        """测试设备配置获取"""
        manager = DeviceManager()

        config = manager.get_device_config()
        assert isinstance(config, dict)

        # 应该包含基本配置项
        expected_keys = ['device_type', 'dtype', 'memory_efficient']
        for key in expected_keys:
            assert key in config, f"配置应该包含 {key}"


class TestPerformanceAndOptimization:
    """测试性能和优化"""

    def test_tensor_movement_performance(self):
        """测试张量移动性能"""
        import time

        manager = DeviceManager()
        tensor = torch.randn(1000, 1000)

        # 测试多次移动的性能
        start_time = time.time()
        for _ in range(100):
            moved_tensor = manager.move_tensor_to_device(tensor)
        end_time = time.time()

        # 应该在合理时间内完成
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"张量移动性能不佳: {execution_time:.2f}秒"

    def test_model_movement_performance(self):
        """测试模型移动性能"""
        import time

        manager = DeviceManager()
        model = nn.Linear(1000, 100)

        start_time = time.time()
        for _ in range(10):
            moved_model = manager.move_model_to_device(model)
        end_time = time.time()

        # 应该在合理时间内完成
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"模型移动性能不佳: {execution_time:.2f}秒"

    def test_device_manager_memory_usage(self):
        """测试设备管理器内存使用"""
        import gc
        import sys

        # 创建多个 DeviceManager 实例（应该返回单例）
        managers = [DeviceManager() for _ in range(10)]

        # 验证都是同一个实例
        assert all(manager is managers[0] for manager in managers)

        # 强制垃圾回收
        gc.collect()

        # 内存使用应该很少（因为使用单例）
        manager_size = sys.getsizeof(managers[0])
        assert manager_size < 10000, f"DeviceManager 内存使用过多: {manager_size} 字节"


class TestErrorHandlingAndEdgeCases:
    """测试错误处理和边界情况"""

    def test_invalid_device_handling(self):
        """测试无效设备处理"""
        manager = DeviceManager()

        tensor = torch.randn(2, 3)
        invalid_device = Mock()
        invalid_device.type = "invalid_device"

        # 应该优雅地处理无效设备
        with pytest.raises(Exception):
            manager.move_tensor_to_device(tensor, invalid_device)

    def test_none_tensor_handling(self):
        """测试 None 张量处理"""
        manager = DeviceManager()

        # 移动 None 应该抛出适当的错误
        with pytest.raises(AttributeError):
            manager.move_tensor_to_device(None)

    def test_none_model_handling(self):
        """测试 None 模型处理"""
        manager = DeviceManager()

        # 移动 None 模型应该优雅处理
        result = manager.move_model_to_device(None)
        assert result is None

    def test_device_type_normalization(self):
        """测试设备类型标准化"""
        manager = DeviceManager()

        # 测试不同的设备类型表示
        device_types = ["cpu", "cuda", "mps"]

        for device_type in device_types:
            device = manager.get_optimal_device()
            assert isinstance(device, torch.device)

    def test_concurrent_device_access(self):
        """测试并发设备访问"""
        import threading

        manager = DeviceManager()
        results = []

        def device_access():
            device = manager.get_optimal_device()
            results.append(device)

        # 创建多个线程访问设备管理器
        threads = [threading.Thread(target=device_access) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 所有结果应该相同（单例模式）
        assert all(result == results[0] for result in results)
        assert len(results) == 10


class TestIntegrationScenarios:
    """测试集成场景"""

    def test_image_handler_tensor_operations(self):
        """测试 ImageHandler 中的张量操作集成"""
        from src.core.process.image_handler import ImageHandler

        with patch('src.core.process.image_handler.DeepseekOCRProcessor'):
            mock_tokenizer = Mock()
            handler = ImageHandler(mock_tokenizer)

            # 模拟处理后的数据
            processed_data = [[
                torch.tensor([[1, 2, 3]]),
                torch.randn(1, 3, 224, 224),
                Mock(), Mock(),
                torch.randn(1, 10, 768),
                torch.randn(1, 10, 768),
                [], []
            ]]

            device_manager = DeviceManager()
            device = torch.device("cpu")

            # 测试集成的张量提取
            result = handler.extract_tensors(processed_data, device, device_manager)

            assert len(result) == 8
            assert isinstance(result[0], torch.Tensor)
            assert isinstance(result[1], torch.Tensor)

    def test_model_manager_device_integration(self):
        """测试 ModelManager 与设备管理的集成"""
        from src.core.models.model_manager import ModelManager

        # 这个测试验证 ModelManager 不再管理设备
        manager = ModelManager("test_model_path")

        # 应该没有设备管理方法
        assert not hasattr(manager, 'setup_device')
        assert not hasattr(manager, 'move_model_to_device')

        # 设备管理应该由 DeviceManager 负责
        device_manager = DeviceManager()
        assert hasattr(device_manager, 'get_optimal_device')
        assert hasattr(device_manager, 'move_model_to_device')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])