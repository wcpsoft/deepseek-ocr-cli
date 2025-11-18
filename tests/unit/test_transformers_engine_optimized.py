"""
Transformers引擎单元测试 - 优化版本

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

from tests.utils import TestUtils

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_mock_engine_with_device(device_type: str):
    """创建带有指定设备类型的模拟引擎"""
    # 使用TestUtils创建设备模拟
    mock_torch = TestUtils.create_mock_torch(device_type)

    # 模拟模块
    mock_modules = {
        "src.core.deepseek_ocr": MagicMock(),
        "transformers": MagicMock(),
        "src.cli.utils": MagicMock(),
        "src.core.models.deepseek_ocr_model": MagicMock(),
        "src.core.models.modeling_deepseekv2": MagicMock(),
        "src.core.models.model_adapter": MagicMock(),
        "src.core.models.model_factory": MagicMock(),
        "src.core.utils.device_manager": MagicMock(),
        "src.core.process.image_process": MagicMock(),
    }

    return mock_torch, mock_modules


def setup_engine_for_testing(device_type: str):
    """设置引擎用于测试"""

    from src.core.transformers.transformers_engine import TransformersEngine

    # 创建模拟引擎
    mock_torch, mock_modules = create_mock_engine_with_device(device_type)

    with patch.dict("sys.modules", mock_modules):
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            # 创建引擎实例
            engine = TransformersEngine()
            engine.device = mock_torch.device(device_type)
            engine.is_initialized = True
            engine.image_handler = MagicMock()
            engine.tokenizer = MagicMock()
            engine.model = MagicMock()

            # 设置图像处理模拟
            mock_processed_data = [
                [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()],
            ]
            engine.image_handler.process_image.return_value = mock_processed_data

            # 设置tensor模拟
            mock_input_ids = TestUtils.create_mock_tensor()
            mock_input_ids.shape = [1, 10]
            engine.image_handler.extract_tensors.return_value = (
                mock_input_ids,  # input_ids
                MagicMock(),  # pixel_values
                MagicMock(),  # images_crop
                MagicMock(),  # images_seq_mask
                MagicMock(),  # images_spatial_crop
                MagicMock(),  # _num_image_tokens
                MagicMock(),  # _image_shapes
            )

            # 设置模型生成模拟
            mock_outputs = TestUtils.create_mock_tensor()
            mock_outputs.shape = [1, 100]
            engine.model.generate.return_value = mock_outputs

            # 设置tokenizer解码模拟
            engine.tokenizer.eos_token_id = 0
            engine.tokenizer.decode.return_value = f"测试OCR结果 - {device_type}设备"

            return engine, mock_torch


# 测试不同设备类型下的初始化
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_initialization_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下的初始化"""
    # 创建模拟引擎
    mock_torch, mock_modules = create_mock_engine_with_device(device_type)

    with patch.dict("sys.modules", mock_modules):
        with patch("src.core.utils.device_manager.get_optimal_device", return_value=mock_torch.device(device_type)):
            with patch("src.core.transformers.transformers_engine.torch", mock_torch):
                from src.core.transformers.transformers_engine import TransformersEngine

                # 创建引擎实例
                engine = TransformersEngine()

                # 模拟初始化过程
                with patch.object(engine, "initialize", return_value=True):
                    # 测试初始化
                    assert engine.initialize()

                    # 验证设备类型
                    assert engine.device.type == device_type

                    # 验证初始化状态
                    assert engine.is_initialized is True


# 测试不同设备类型下的图像处理
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_process_image_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下处理图像"""
    # 设置引擎
    engine, mock_torch = setup_engine_for_testing(device_type)

    # 创建模拟图像
    mock_image = MagicMock(spec=MagicMock)
    mock_image.size = (640, 640)

    # 调用处理图像方法
    result = engine.process_image(mock_image, "测试提示词")

    # 验证结果
    assert result == f"测试OCR结果 - {device_type}设备"


# 测试不同设备类型下的批量处理
@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "dcu", "amd"])
def test_transformers_engine_process_batch_with_different_devices(device_type: str) -> None:
    """测试Transformers引擎在不同设备类型下批量处理图像"""
    # 设置引擎
    engine, mock_torch = setup_engine_for_testing(device_type)

    # 创建模拟图像
    mock_images = []
    for i in range(3):
        mock_image = MagicMock(spec=MagicMock)
        mock_image.size = (640, 640)
        mock_images.append(mock_image)

    # 创建模拟提示词
    prompts = ["提示词1", "提示词2", "提示词3"]

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
    # 创建模拟引擎
    mock_torch, mock_modules = create_mock_engine_with_device(device_type)

    with patch.dict("sys.modules", mock_modules):
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建引擎实例
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
    # 创建模拟引擎
    mock_torch, mock_modules = create_mock_engine_with_device(device_type)

    with patch.dict("sys.modules", mock_modules):
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from PIL import Image

            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建引擎实例
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
    # 创建模拟引擎
    mock_torch, mock_modules = create_mock_engine_with_device(device_type)

    with patch.dict("sys.modules", mock_modules):
        with patch("src.core.transformers.transformers_engine.torch", mock_torch):
            from src.core.transformers.transformers_engine import TransformersEngine

            # 创建引擎实例
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
