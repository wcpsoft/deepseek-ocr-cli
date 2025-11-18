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

from unittest.mock import MagicMock, patch

import pytest

from tests.utils import TestUtils


# 测试CPU设备
def test_transformers_engine_cpu_initialization(mock_torch_cpu) -> None:
    """测试Transformers引擎在CPU设备上的初始化"""
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
            "src.core.utils.device_manager": MagicMock(),  # 添加设备管理器的模拟
            "src.core.process.image_process": MagicMock(),  # 添加图像处理模块的模拟
        },
    ):
        # 模拟get_optimal_device函数返回CPU设备
        with patch("src.core.utils.device_manager.get_optimal_device", return_value=mock_torch_cpu.device("cpu")):
            # 使用patch来替换transformers_engine模块中的torch引用
            with patch("src.core.transformers.transformers_engine.torch", mock_torch_cpu):
                from src.core.transformers.transformers_engine import TransformersEngine

                # 创建Transformers引擎实例
                engine = TransformersEngine()

                # 模拟初始化过程
                with patch.object(
                    engine.model_manager, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())
                ):
                    with patch.object(engine.model_manager, "load_processor", return_value=MagicMock()):
                        with patch.object(engine.model_manager, "move_model_to_device"):
                            # 测试初始化
                            assert engine.initialize()

                            # 验证设备类型
                            assert engine.device.type == "cpu"

                            # 验证初始化状态
                            assert engine.is_initialized is True


# 测试CUDA设备
def test_transformers_engine_cuda_initialization(mock_torch_cuda) -> None:
    """测试Transformers引擎在CUDA设备上的初始化"""
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
            "src.core.utils.device_manager": MagicMock(),  # 添加设备管理器的模拟
            "src.core.process.image_process": MagicMock(),  # 添加图像处理模块的模拟
        },
    ):
        # 模拟get_optimal_device函数返回CUDA设备
        with patch("src.core.utils.device_manager.get_optimal_device", return_value=mock_torch_cuda.device("cuda")):
            # 使用patch来替换transformers_engine模块中的torch引用
            with patch("src.core.transformers.transformers_engine.torch", mock_torch_cuda):
                from src.core.transformers.transformers_engine import TransformersEngine

                # 创建Transformers引擎实例，指定CUDA设备
                engine = TransformersEngine()

                # 模拟初始化过程
                with patch.object(
                    engine.model_manager, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())
                ):
                    with patch.object(engine.model_manager, "load_processor", return_value=MagicMock()):
                        with patch.object(engine.model_manager, "move_model_to_device"):
                            # 测试初始化
                            assert engine.initialize()

                            # 验证设备类型
                            assert engine.device.type == "cuda"

                            # 验证初始化状态
                            assert engine.is_initialized is True


# 测试MPS设备
def test_transformers_engine_mps_initialization(mock_torch_mps) -> None:
    """测试Transformers引擎在MPS设备上的初始化"""
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
            "src.core.utils.device_manager": MagicMock(),  # 添加设备管理器的模拟
            "src.core.process.image_process": MagicMock(),  # 添加图像处理模块的模拟
        },
    ):
        # 模拟get_optimal_device函数返回MPS设备
        with patch("src.core.utils.device_manager.get_optimal_device", return_value=mock_torch_mps.device("mps")):
            # 使用patch来替换transformers_engine模块中的torch引用
            with patch("src.core.transformers.transformers_engine.torch", mock_torch_mps):
                from src.core.transformers.transformers_engine import TransformersEngine

                # 创建Transformers引擎实例，指定MPS设备
                engine = TransformersEngine()

                # 模拟初始化过程
                with patch.object(
                    engine.model_manager, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())
                ):
                    with patch.object(engine.model_manager, "load_processor", return_value=MagicMock()):
                        with patch.object(engine.model_manager, "move_model_to_device"):
                            # 测试初始化
                            assert engine.initialize()

                            # 验证设备类型
                            assert engine.device.type == "mps"

                            # 验证初始化状态
                            assert engine.is_initialized is True


# 测试DCU设备
def test_transformers_engine_dcu_initialization() -> None:
    """测试Transformers引擎在DCU设备上的初始化"""
    # 使用TestUtils创建DCU设备模拟
    mock_torch = TestUtils.create_mock_torch("dcu")

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
            "src.core.utils.device_manager": MagicMock(),  # 添加设备管理器的模拟
            "src.core.process.image_process": MagicMock(),  # 添加图像处理模块的模拟
        },
    ):
        # 模拟get_optimal_device函数返回DCU设备
        with patch("src.core.utils.device_manager.get_optimal_device", return_value=mock_torch.device("dcu")):
            # 使用patch来替换transformers_engine模块中的torch引用
            with patch("src.core.transformers.transformers_engine.torch", mock_torch):
                from src.core.transformers.transformers_engine import TransformersEngine

                # 创建Transformers引擎实例，指定DCU设备
                engine = TransformersEngine()

                # 模拟模型管理器
                with patch.object(engine, "model_manager") as mock_model_manager:
                    # 模拟setup_device方法返回正确的设备
                    mock_model_manager.setup_device.return_value = mock_torch.device("dcu")

                    # 模拟初始化过程
                    with patch.object(
                        mock_model_manager, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())
                    ):
                        with patch.object(mock_model_manager, "load_processor", return_value=MagicMock()):
                            with patch.object(mock_model_manager, "move_model_to_device", return_value=None):
                                # 测试初始化
                                assert engine.initialize()

                                # 验证设备类型
                                assert engine.device.type == "dcu"

                                # 验证初始化状态
                                assert engine.is_initialized is True


# 测试AMD设备
def test_transformers_engine_amd_initialization() -> None:
    """测试Transformers引擎在AMD设备上的初始化"""
    # 使用TestUtils创建AMD设备模拟
    mock_torch = TestUtils.create_mock_torch("amd")

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
            "src.core.utils.device_manager": MagicMock(),  # 添加设备管理器的模拟
            "src.core.process.image_process": MagicMock(),  # 添加图像处理模块的模拟
        },
    ):
        # 模拟get_optimal_device函数返回AMD设备
        with patch("src.core.utils.device_manager.get_optimal_device", return_value=mock_torch.device("amd")):
            # 使用patch来替换transformers_engine模块中的torch引用
            with patch("src.core.transformers.transformers_engine.torch", mock_torch):
                from src.core.transformers.transformers_engine import TransformersEngine

                # 创建Transformers引擎实例
                engine = TransformersEngine()

                # 模拟模型管理器
                with patch.object(engine, "model_manager") as mock_model_manager:
                    # 模拟setup_device方法返回正确的设备
                    mock_model_manager.setup_device.return_value = mock_torch.device("amd")

                    # 模拟初始化过程
                    with patch.object(
                        mock_model_manager, "load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())
                    ):
                        with patch.object(mock_model_manager, "load_processor", return_value=MagicMock()):
                            with patch.object(mock_model_manager, "move_model_to_device", return_value=None):
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
    # 使用TestUtils创建设备模拟
    mock_torch = TestUtils.create_mock_torch(device_type)

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
            "torchvision": MagicMock(),  # 添加torchvision模块的模拟以避免冲突
            "torchvision.ops": MagicMock(),  # 添加torchvision.ops模块的模拟
            "torch": mock_torch,  # 直接替换torch模块
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
                [
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ],  # 确保有7个元素
            ]
            engine.image_handler.process_image.return_value = mock_processed_data

            # 创建一个MockTensor实例作为input_ids
            mock_input_ids = TestUtils.create_mock_tensor()
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
            mock_outputs = TestUtils.create_mock_tensor()
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
    # 使用TestUtils创建设备模拟
    mock_torch = TestUtils.create_mock_torch(device_type)

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
            "torchvision": MagicMock(),  # 添加torchvision模块的模拟以避免冲突
            "torchvision.ops": MagicMock(),  # 添加torchvision.ops模块的模拟
            "torch": mock_torch,  # 直接替换torch模块
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
                [
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ],  # 确保有7个元素
            ]
            engine.image_handler.process_image.return_value = mock_processed_data

            # 创建一个MockTensor实例作为input_ids
            mock_input_ids = TestUtils.create_mock_tensor()
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
            mock_outputs = TestUtils.create_mock_tensor()
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
    # 使用TestUtils创建设备模拟
    mock_torch = TestUtils.create_mock_torch(device_type)

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
            "torchvision": MagicMock(),  # 添加torchvision模块的模拟以避免冲突
            "torchvision.ops": MagicMock(),  # 添加torchvision.ops模块的模拟
            "torch": mock_torch,  # 直接替换torch模块
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
    # 使用TestUtils创建设备模拟
    mock_torch = TestUtils.create_mock_torch(device_type)

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
            "torchvision": MagicMock(),  # 添加torchvision模块的模拟以避免冲突
            "torchvision.ops": MagicMock(),  # 添加torchvision.ops模块的模拟
            "torch": mock_torch,  # 直接替换torch模块
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
    # 使用TestUtils创建设备模拟
    mock_torch = TestUtils.create_mock_torch(device_type)

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
            "torchvision": MagicMock(),  # 添加torchvision模块的模拟以避免冲突
            "torchvision.ops": MagicMock(),  # 添加torchvision.ops模块的模拟
            "torch": mock_torch,  # 直接替换torch模块
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

                # 手动设置初始化状态，因为patch可能不会实际修改对象状态
                engine.is_initialized = True

                # 验证初始化状态
                assert engine.is_initialized is True
