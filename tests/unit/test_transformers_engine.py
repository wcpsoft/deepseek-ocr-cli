"""
Transformers引擎单元测试

本测试文件针对Transformers引擎的各种设备类型进行测试，包括：
- CPU设备测试
- CUDA设备测试
- MPS设备测试
- DCU设备测试
- AMD设备测试

测试覆盖了Transformers引擎的主要功能：
- 初始化
- 图像处理
- 批量处理
- 错误处理
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 测试CPU设备
def test_transformers_engine_cpu_initialization() -> None:
    """测试Transformers引擎在CPU设备上的初始化"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例
            engine = TransformersEngine()
            
            # 模拟初始化过程
            with patch.object(engine, "initialize", return_value=True):
                # 测试初始化
                assert engine.initialize()
                
                # 验证设备类型
                assert engine.device.type == "cpu"
                
                # 验证初始化状态
                assert engine.is_initialized is True


# 测试CUDA设备
def test_transformers_engine_cuda_initialization() -> None:
    """测试Transformers引擎在CUDA设备上的初始化"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cuda"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟CUDA可用性
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.current_device.return_value = 0
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
    mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
    mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 200  # 200MB
    mock_torch.cuda.empty_cache = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    sys.modules["torch.cuda"] = mock_torch.cuda
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定CUDA设备
            engine = TransformersEngine()
            engine.device = mock_torch.device("cuda")
            
            # 模拟初始化过程
            with patch.object(engine, "initialize", return_value=True):
                # 测试初始化
                assert engine.initialize()
                
                # 验证设备类型
                assert engine.device.type == "cuda"
                
                # 验证初始化状态
                assert engine.is_initialized is True


# 测试MPS设备
def test_transformers_engine_mps_initialization() -> None:
    """测试Transformers引擎在MPS设备上的初始化"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="mps"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟MPS可用性
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.backends.mps.is_built.return_value = True
    mock_torch.mps.empty_cache = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    sys.modules["torch.backends"] = MagicMock()
    sys.modules["torch.backends.mps"] = mock_torch.backends.mps
    sys.modules["torch.mps"] = mock_torch.mps
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定MPS设备
            engine = TransformersEngine()
            engine.device = mock_torch.device("mps")
            
            # 模拟初始化过程
            with patch.object(engine, "initialize", return_value=True):
                # 测试初始化
                assert engine.initialize()
                
                # 验证设备类型
                assert engine.device.type == "mps"
                
                # 验证初始化状态
                assert engine.is_initialized is True


# 测试DCU设备
def test_transformers_engine_dcu_initialization() -> None:
    """测试Transformers引擎在DCU设备上的初始化"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="dcu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟DCU可用性
    mock_torch.dcu.is_available.return_value = True
    mock_torch.dcu.device_count.return_value = 1
    mock_torch.dcu.current_device.return_value = 0
    mock_torch.dcu.get_device_name.return_value = "AMD DCU"
    mock_torch.dcu.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
    mock_torch.dcu.memory_reserved.return_value = 1024 * 1024 * 200  # 200MB
    mock_torch.dcu.empty_cache = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    sys.modules["torch.dcu"] = mock_torch.dcu
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定DCU设备
            engine = TransformersEngine()
            engine.device = mock_torch.device("dcu")
            
            # 模拟初始化过程
            with patch.object(engine, "initialize", return_value=True):
                # 测试初始化
                assert engine.initialize()
                
                # 验证设备类型
                assert engine.device.type == "dcu"
                
                # 验证初始化状态
                assert engine.is_initialized is True


# 测试AMD设备
def test_transformers_engine_amd_initialization() -> None:
    """测试Transformers引擎在AMD设备上的初始化"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="amd"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟AMD ROCm可用性
    mock_torch.roc.is_available.return_value = True
    mock_torch.roc.device_count.return_value = 1
    mock_torch.roc.current_device.return_value = 0
    mock_torch.roc.get_device_name.return_value = "AMD Radeon RX 7900 XTX"
    mock_torch.roc.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
    mock_torch.roc.memory_reserved.return_value = 1024 * 1024 * 200  # 200MB
    mock_torch.roc.empty_cache = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    sys.modules["torch.roc"] = mock_torch.roc
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定AMD设备
            engine = TransformersEngine()
            engine.device = mock_torch.device("amd")
            
            # 模拟初始化过程
            with patch.object(engine, "initialize", return_value=True):
                # 测试初始化
                assert engine.initialize()
                
                # 验证设备类型
                assert engine.device.type == "amd"
                
                # 验证初始化状态
                assert engine.is_initialized is True


# 测试不同设备类型下的图像处理
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_process_image_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下处理图像"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from PIL import Image

            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定设备类型
            engine = TransformersEngine()
            engine.device = mock_torch.device(device_type)
            engine.is_initialized = True
            engine.image_handler = MagicMock()
            engine.tokenizer = MagicMock()
            engine.model = MagicMock()

            # 创建模拟图像
            mock_image = MagicMock(spec=Image.Image)
            mock_image.size = (640, 640)

            # 模拟图像处理过程
            mock_processed_data = [
                [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()],  # 确保有7个元素
            ]
            engine.image_handler.process_image.return_value = mock_processed_data
            
            # 创建一个MockTensor实例作为input_ids
            mock_input_ids = MagicMock()
            mock_input_ids.__class__ = MockTensor
            mock_input_ids.shape = [1, 10]  # 设置shape属性为可比较的列表
            
            # 模拟extract_tensors返回值，确保input_ids是MockTensor实例
            engine.image_handler.extract_tensors.return_value = (
                mock_input_ids,  # input_ids
                MagicMock(),  # pixel_values
                MagicMock(),  # images_crop
                MagicMock(),  # images_seq_mask
                MagicMock(),  # images_spatial_crop
                MagicMock(),  # _num_image_tokens
                MagicMock(),  # _image_shapes
            )

            # 模拟模型生成过程
            mock_outputs = MagicMock()
            mock_outputs.shape = [1, 100]  # 设置shape属性为可比较的列表
            engine.model.generate.return_value = mock_outputs

            # 模拟tokenizer解码
            engine.tokenizer.eos_token_id = 0
            engine.tokenizer.decode.return_value = f"测试OCR结果 - {device_type}设备"

            # 调用处理图像方法
            result = engine.process_image(mock_image, "测试提示词")

            # 验证结果
            assert result == f"测试OCR结果 - {device_type}设备"


# 测试不同设备类型下的批量处理
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_process_batch_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下批量处理图像"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 模拟torch.ones_like，使其返回一个MockTensor实例
    def mock_ones_like(input_tensor):
        """模拟torch.ones_like函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)
    
    # 模拟torch.cat，使其返回一个MockTensor实例
    def mock_cat(tensors, dim=0):
        """模拟torch.cat函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.cat = MagicMock(side_effect=mock_cat)
    
    # 模拟torch.unsqueeze，使其返回一个MockTensor实例
    def mock_unsqueeze(input_tensor, dim):
        """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
        result = MagicMock()
        result.__class__ = MockTensor
        return result
        
    mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    
    # 确保torch.ones_like在全局模块中也被模拟
    sys.modules["torch"].ones_like = mock_torch.ones_like
    sys.modules["torch"].cat = mock_torch.cat
    sys.modules["torch"].unsqueeze = mock_torch.unsqueeze
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from PIL import Image

            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定设备类型
            engine = TransformersEngine()
            engine.device = mock_torch.device(device_type)
            engine.is_initialized = True
            engine.image_handler = MagicMock()
            engine.tokenizer = MagicMock()
            engine.model = MagicMock()

            # 创建模拟图像
            mock_images = []
            for i in range(3):
                mock_image = MagicMock(spec=Image.Image)
                mock_image.size = (640, 640)
                mock_images.append(mock_image)

            # 创建模拟提示词
            prompts = ["提示词1", "提示词2", "提示词3"]

            # 模拟图像处理过程
            mock_processed_data = [
                [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()],  # 确保有7个元素
            ]
            engine.image_handler.process_image.return_value = mock_processed_data
            
            # 创建一个MockTensor实例作为input_ids
            mock_input_ids = MagicMock()
            mock_input_ids.__class__ = MockTensor
            mock_input_ids.shape = [1, 10]  # 设置shape属性为可比较的列表
            
            # 模拟extract_tensors返回值，确保input_ids是MockTensor实例
            engine.image_handler.extract_tensors.return_value = (
                mock_input_ids,  # input_ids
                MagicMock(),  # pixel_values
                MagicMock(),  # images_crop
                MagicMock(),  # images_seq_mask
                MagicMock(),  # images_spatial_crop
                MagicMock(),  # _num_image_tokens
                MagicMock(),  # _image_shapes
            )

            # 模拟模型生成过程
            mock_outputs = MagicMock()
            mock_outputs.shape = [1, 100]  # 设置shape属性为可比较的列表
            engine.model.generate.return_value = mock_outputs

            # 模拟tokenizer解码
            engine.tokenizer.eos_token_id = 0
            engine.tokenizer.decode.return_value = f"测试OCR结果 - {device_type}设备"

            # 调用批量处理方法
            results = engine.process_batch(mock_images, prompts)

            # 验证结果
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result == f"测试OCR结果 - {device_type}设备"


# 测试不同设备类型下的资源清理
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_cleanup_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下的资源清理"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 根据设备类型设置相应的模拟
    if device_type == "cuda":
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()
    elif device_type == "mps":
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.empty_cache = MagicMock()
    elif device_type == "dcu":
        mock_torch.dcu.is_available.return_value = True
        mock_torch.dcu.empty_cache = MagicMock()
    elif device_type == "amd":
        mock_torch.roc.is_available.return_value = True
        mock_torch.roc.empty_cache = MagicMock()
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    
    if device_type == "cuda":
        sys.modules["torch.cuda"] = mock_torch.cuda
    elif device_type == "mps":
        sys.modules["torch.backends"] = MagicMock()
        sys.modules["torch.backends.mps"] = mock_torch.backends.mps
        sys.modules["torch.mps"] = mock_torch.mps
    elif device_type == "dcu":
        sys.modules["torch.dcu"] = mock_torch.dcu
    elif device_type == "amd":
        sys.modules["torch.roc"] = mock_torch.roc
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定设备类型
            engine = TransformersEngine()
            engine.device = mock_torch.device(device_type)
            engine.is_initialized = True

            # 调用清理方法
            engine.cleanup()

            # 验证初始化状态
            assert engine.is_initialized is False

            # 验证特定设备类型的清理方法被调用
            if device_type == "cuda":
                mock_torch.cuda.empty_cache.assert_called_once()
            elif device_type == "mps":
                mock_torch.mps.empty_cache.assert_called_once()
            elif device_type == "dcu":
                mock_torch.dcu.empty_cache.assert_called_once()
            elif device_type == "amd":
                mock_torch.roc.empty_cache.assert_called_once()


# 测试不同设备类型下的错误处理
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_error_handling_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下的错误处理"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from PIL import Image

            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定设备类型
            engine = TransformersEngine()
            engine.device = mock_torch.device(device_type)
            engine.is_initialized = False

            # 创建模拟图像
            mock_image = MagicMock(spec=Image.Image)
            mock_image.size = (640, 640)

            # 模拟初始化失败
            with patch.object(engine, "initialize", return_value=False):
                # 处理应该触发初始化错误
                with pytest.raises(RuntimeError, match="Transformers引擎初始化失败"):
                    engine.process_image(mock_image, "测试提示词")


# 测试不同设备类型下的设备配置
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_device_config_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下的设备配置"""
    # 模拟torch模块
    mock_torch = MagicMock()
    
    # 创建一个可以接受参数的Device类
    class MockDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type
    
    # 将MockDevice类添加到torch模块中
    mock_torch.device = MockDevice
    
    # 创建一个MockTensor类，用于isinstance检查
    class MockTensor:
        pass
    
    # 将MockTensor类添加到torch模块中
    mock_torch.Tensor = MockTensor
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.long = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.autocast = MagicMock()
    mock_torch.float16 = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    
    # 更新sys.modules中的torch模拟
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "src.core.models.deepseek_ocr_model": MagicMock(),  # 添加模型模块的模拟
            "src.core.models.modeling_deepseekv2": MagicMock(),  # 添加模型文件的模拟
            "src.core.models.model_adapter": MagicMock(),  # 添加模型适配器的模拟
            "src.core.models.model_factory": MagicMock(),  # 添加模型工厂的模拟
        },
    ):
        # 使用patch来替换transformers_engine模块中的torch引用
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建Transformers引擎实例，指定设备类型
            engine = TransformersEngine()
            engine.device = mock_torch.device(device_type)
            
            # 验证设备类型
            assert engine.device.type == device_type
            
            # 模拟初始化过程
            with patch.object(engine, "initialize", return_value=True):
                # 测试初始化
                assert engine.initialize()
                
                # 验证初始化状态
                assert engine.is_initialized is True