#!/usr/bin/env python3
"""
测试工具类
提供通用的测试工具和模拟对象，减少测试代码重复
"""

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch


class TestUtils:
    """测试工具类，提供通用的测试功能"""

    @staticmethod
    def create_mock_torch(device_type: str = "cpu") -> MagicMock:
        """
        创建模拟的torch模块

        Args:
            device_type: 设备类型，如"cpu", "cuda", "mps"等

        Returns:
            模拟的torch模块
        """
        mock_torch = MagicMock()

        # 创建一个可以接受参数的Device类
        class MockDevice:
            def __init__(self, device_type="cpu"):
                self.type = device_type
                self.index = 0

            def __str__(self):
                return self.type

        # 将MockDevice类添加到torch模块中
        mock_torch.device = MockDevice

        # 创建一个MockTensor类，用于isinstance检查
        class MockTensor:
            def __init__(self, data=None):
                self.data = data
                self.shape = (1, 3, 224, 224)  # 默认形状
                self.dtype = torch.float32
                self.device = MockDevice(device_type)

            def to(self, device):
                new_tensor = MockTensor(self.data)
                new_tensor.device = MockDevice(device)
                return new_tensor

            def cpu(self):
                new_tensor = MockTensor(self.data)
                new_tensor.device = MockDevice("cpu")
                return new_tensor

            def numpy(self):
                return np.zeros((224, 224, 3))

            def unsqueeze(self, dim):
                """模拟unsqueeze方法"""
                new_tensor = MockTensor(self.data)
                shape = list(self.shape)
                if dim < 0:
                    dim = len(shape) + 1 + dim
                if dim <= len(shape):
                    shape.insert(dim, 1)
                new_tensor.shape = tuple(shape)
                new_tensor.dtype = self.dtype
                new_tensor.device = self.device
                return new_tensor

            def squeeze(self, dim=None):
                """模拟squeeze方法"""
                new_tensor = MockTensor(self.data)
                shape = list(self.shape)
                if dim is None:
                    # 移除所有大小为1的维度
                    shape = [s for s in shape if s != 1]
                else:
                    # 移除指定维度（如果大小为1）
                    if dim < len(shape) and shape[dim] == 1:
                        shape.pop(dim)
                new_tensor.shape = tuple(shape)
                new_tensor.dtype = self.dtype
                new_tensor.device = self.device
                return new_tensor

        # 将MockTensor类添加到torch模块中
        mock_torch.Tensor = MockTensor
        mock_torch.zeros = MagicMock(return_value=MockTensor())
        mock_torch.ones = MagicMock(return_value=MockTensor())
        mock_torch.tensor = MagicMock(return_value=MockTensor())
        mock_torch.rand = MagicMock(return_value=MockTensor())
        mock_torch.randn = MagicMock(return_value=MockTensor())
        mock_torch.long = MagicMock()
        mock_torch.float32 = MagicMock()
        mock_torch.float16 = MagicMock()
        mock_torch.bfloat16 = MagicMock()
        mock_torch.uint8 = MagicMock()
        mock_torch.autocast = MagicMock()

        # 模拟torch.ones_like，使其返回一个MockTensor实例
        def mock_ones_like(input_tensor):
            """模拟torch.ones_like函数，确保返回MockTensor实例"""
            result = MockTensor()
            if hasattr(input_tensor, "shape"):
                result.shape = input_tensor.shape
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.ones_like = MagicMock(side_effect=mock_ones_like)

        # 模拟torch.zeros_like，使其返回一个MockTensor实例
        def mock_zeros_like(input_tensor):
            """模拟torch.zeros_like函数，确保返回MockTensor实例"""
            result = MockTensor()
            if hasattr(input_tensor, "shape"):
                result.shape = input_tensor.shape
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.zeros_like = MagicMock(side_effect=mock_zeros_like)

        # 模拟torch.cat，使其返回一个MockTensor实例
        def mock_cat(tensors, dim=0):
            """模拟torch.cat函数，确保返回MockTensor实例"""
            result = MockTensor()
            if tensors and hasattr(tensors[0], "shape"):
                # 简单的形状计算
                shape = list(tensors[0].shape)
                if dim < len(shape):
                    shape[dim] = len(tensors) * shape[dim]
                result.shape = tuple(shape)
            if tensors and hasattr(tensors[0], "dtype"):
                result.dtype = tensors[0].dtype
            if tensors and hasattr(tensors[0], "device"):
                result.device = tensors[0].device
            return result

        mock_torch.cat = MagicMock(side_effect=mock_cat)

        # 模拟torch.stack，使其返回一个MockTensor实例
        def mock_stack(tensors, dim=0):
            """模拟torch.stack函数，确保返回MockTensor实例"""
            result = MockTensor()
            if tensors and hasattr(tensors[0], "shape"):
                # 简单的形状计算
                shape = [len(tensors)] + list(tensors[0].shape)
                result.shape = tuple(shape)
            if tensors and hasattr(tensors[0], "dtype"):
                result.dtype = tensors[0].dtype
            if tensors and hasattr(tensors[0], "device"):
                result.device = tensors[0].device
            return result

        mock_torch.stack = MagicMock(side_effect=mock_stack)

        # 模拟torch.unsqueeze，使其返回一个MockTensor实例
        def mock_unsqueeze(input_tensor, dim):
            """模拟torch.unsqueeze函数，确保返回MockTensor实例"""
            result = MockTensor()
            if hasattr(input_tensor, "shape"):
                shape = list(input_tensor.shape)
                if dim < 0:
                    dim = len(shape) + 1 + dim
                if dim <= len(shape):
                    shape.insert(dim, 1)
                result.shape = tuple(shape)
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.unsqueeze = MagicMock(side_effect=mock_unsqueeze)

        # 模拟torch.squeeze，使其返回一个MockTensor实例
        def mock_squeeze(input_tensor, dim=None):
            """模拟torch.squeeze函数，确保返回MockTensor实例"""
            result = MockTensor()
            if hasattr(input_tensor, "shape"):
                shape = list(input_tensor.shape)
                if dim is None:
                    # 移除所有大小为1的维度
                    shape = [s for s in shape if s != 1]
                else:
                    # 移除指定维度（如果大小为1）
                    if dim < len(shape) and shape[dim] == 1:
                        shape.pop(dim)
                result.shape = tuple(shape)
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.squeeze = MagicMock(side_effect=mock_squeeze)

        # 模拟torch.permute，使其返回一个MockTensor实例
        def mock_permute(input_tensor, *dims):
            """模拟torch.permute函数，确保返回MockTensor实例"""
            result = MockTensor()
            if hasattr(input_tensor, "shape"):
                shape = list(input_tensor.shape)
                new_shape = [shape[dim] for dim in dims]
                result.shape = tuple(new_shape)
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.permute = MagicMock(side_effect=mock_permute)

        # 模拟torch.reshape，使其返回一个MockTensor实例
        def mock_reshape(input_tensor, *shape):
            """模拟torch.reshape函数，确保返回MockTensor实例"""
            result = MockTensor()
            result.shape = shape
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.reshape = MagicMock(side_effect=mock_reshape)

        # 模拟torch.view，使其返回一个MockTensor实例
        def mock_view(input_tensor, *shape):
            """模拟torch.view函数，确保返回MockTensor实例"""
            result = MockTensor()
            result.shape = shape
            if hasattr(input_tensor, "dtype"):
                result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                result.device = input_tensor.device
            return result

        mock_torch.view = MagicMock(side_effect=mock_view)

        # 模拟torch.split，使其返回一个MockTensor列表
        def mock_split(input_tensor, split_size_or_sections, dim=0):
            """模拟torch.split函数，确保返回MockTensor列表"""
            results = []
            if hasattr(input_tensor, "shape"):
                shape = list(input_tensor.shape)
                if isinstance(split_size_or_sections, int):
                    # 均匀分割
                    num_splits = shape[dim] // split_size_or_sections
                    for i in range(num_splits):
                        result = MockTensor()
                        result.shape = tuple(shape)
                        results.append(result)
                else:
                    # 按指定大小分割
                    for size in split_size_or_sections:
                        result = MockTensor()
                        result.shape = tuple(shape)
                        results.append(result)

            if hasattr(input_tensor, "dtype"):
                for result in results:
                    result.dtype = input_tensor.dtype
            if hasattr(input_tensor, "device"):
                for result in results:
                    result.device = input_tensor.device

            return results

        mock_torch.split = MagicMock(side_effect=mock_split)

        # 模拟CUDA相关功能
        mock_cuda = MagicMock()
        mock_cuda.is_available = MagicMock(return_value=(device_type == "cuda"))
        mock_cuda.device_count = MagicMock(return_value=1)
        mock_cuda.current_device = MagicMock(return_value=0)
        mock_cuda.get_device_name = MagicMock(return_value="NVIDIA GeForce RTX 4090")
        mock_cuda.memory_allocated = MagicMock(return_value=1024 * 1024 * 100)  # 100MB
        mock_cuda.memory_reserved = MagicMock(return_value=1024 * 1024 * 200)  # 200MB
        mock_cuda.empty_cache = MagicMock()
        mock_cuda.set_device = MagicMock()
        mock_torch.cuda = mock_cuda

        # 模拟MPS相关功能
        mock_mps = MagicMock()
        mock_mps.is_available = MagicMock(return_value=(device_type == "mps"))
        mock_mps.is_built = MagicMock(return_value=(device_type == "mps"))
        mock_mps.empty_cache = MagicMock()
        mock_torch.mps = mock_mps

        # 模拟DCU相关功能
        mock_dcu = MagicMock()
        mock_dcu.is_available = MagicMock(return_value=(device_type == "dcu"))
        mock_dcu.device_count = MagicMock(return_value=1)
        mock_dcu.current_device = MagicMock(return_value=0)
        mock_dcu.get_device_name = MagicMock(return_value="Hygon DCU K100-AI")
        mock_dcu.memory_allocated = MagicMock(return_value=1024 * 1024 * 100)  # 100MB
        mock_dcu.memory_reserved = MagicMock(return_value=1024 * 1024 * 200)  # 200MB
        mock_dcu.empty_cache = MagicMock()
        mock_dcu.set_device = MagicMock()
        mock_torch.dcu = mock_dcu

        # 模拟ROCm相关功能（AMD GPU）
        mock_roc = MagicMock()
        mock_roc.is_available = MagicMock(return_value=(device_type == "amd"))
        mock_roc.device_count = MagicMock(return_value=1)
        mock_roc.current_device = MagicMock(return_value=0)
        mock_roc.get_device_name = MagicMock(return_value="AMD Radeon GPU")
        mock_roc.memory_allocated = MagicMock(return_value=1024 * 1024 * 100)  # 100MB
        mock_roc.memory_reserved = MagicMock(return_value=1024 * 1024 * 200)  # 200MB
        mock_roc.empty_cache = MagicMock()
        mock_roc.set_device = MagicMock()
        mock_torch.roc = mock_roc

        # 模拟分布式功能
        mock_distributed = MagicMock()
        mock_distributed.is_available = MagicMock(return_value=False)
        mock_distributed.is_initialized = MagicMock(return_value=False)
        mock_distributed.get_rank = MagicMock(return_value=0)
        mock_distributed.get_world_size = MagicMock(return_value=1)
        mock_torch.distributed = mock_distributed

        # 模拟nn模块
        mock_nn = MagicMock()
        mock_nn.Module = MagicMock
        mock_nn.functional = MagicMock()
        mock_torch.nn = mock_nn

        # 模拟optim模块
        mock_optim = MagicMock()
        mock_optim.Adam = MagicMock
        mock_optim.SGD = MagicMock
        mock_torch.optim = mock_optim

        return mock_torch

    @staticmethod
    def create_mock_tensor(data=None, shape=None, dtype=None, device_type="cpu") -> "MockTensor":
        """
        创建模拟张量对象

        Args:
            data: 张量数据
            shape: 张量形状
            dtype: 数据类型
            device_type: 设备类型

        Returns:
            模拟张量对象
        """
        return MockTensor(data=data, shape=shape, dtype=dtype, device_type=device_type)

    @staticmethod
    def create_mock_transformers() -> MagicMock:
        """
        创建模拟的transformers模块

        Returns:
            模拟的transformers模块
        """
        mock_transformers = MagicMock()

        # 模拟AutoModelForCausalLM
        mock_auto_model = MagicMock()
        mock_model = MagicMock()
        mock_auto_model.from_pretrained = MagicMock(return_value=mock_model)
        mock_transformers.AutoModelForCausalLM = mock_auto_model

        # 模拟AutoTokenizer
        mock_auto_tokenizer = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        mock_tokenizer.decode = MagicMock(return_value="decoded text")
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.bos_token_id = 2
        mock_tokenizer.unk_token_id = 3
        mock_auto_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
        mock_transformers.AutoTokenizer = mock_auto_tokenizer

        # 模拟AutoProcessor
        mock_auto_processor = MagicMock()
        mock_processor = MagicMock()
        mock_processor.preprocess = MagicMock(return_value={"pixel_values": MockTensor()})
        mock_processor.postprocess = MagicMock(return_value="processed text")
        mock_auto_processor.from_pretrained = MagicMock(return_value=mock_processor)
        mock_transformers.AutoProcessor = mock_auto_processor

        # 模拟AutoConfig
        mock_auto_config = MagicMock()
        mock_config = MagicMock()
        mock_config.model_type = "deepseek"
        mock_config.vocab_size = 100000
        mock_config.hidden_size = 2048
        mock_config.num_attention_heads = 32
        mock_config.num_hidden_layers = 24
        mock_config.max_position_embeddings = 4096
        mock_auto_config.from_pretrained = MagicMock(return_value=mock_config)
        mock_transformers.AutoConfig = mock_auto_config

        # 模拟GenerationConfig
        mock_generation_config = MagicMock()
        mock_generation_config.max_length = 4096
        mock_generation_config.temperature = 0.7
        mock_generation_config.top_p = 0.9
        mock_generation_config.do_sample = True
        mock_transformers.GenerationConfig = MagicMock(return_value=mock_generation_config)

        return mock_transformers

    @staticmethod
    def create_mock_vllm() -> MagicMock:
        """
        创建模拟的vllm模块

        Returns:
            模拟的vllm模块
        """
        mock_vllm = MagicMock()

        # 模拟LLM
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value=["Generated text"])
        mock_vllm.LLM = MagicMock(return_value=mock_llm)

        # 模拟SamplingParams
        mock_sampling_params = MagicMock()
        mock_vllm.SamplingParams = MagicMock(return_value=mock_sampling_params)

        return mock_vllm

    @staticmethod
    def create_mock_pil() -> MagicMock:
        """
        创建模拟的PIL模块

        Returns:
            模拟的PIL模块
        """
        mock_pil = MagicMock()

        # 模拟Image
        mock_image = MagicMock()
        mock_image_instance = MagicMock()
        mock_image_instance.size = (640, 480)
        mock_image_instance.mode = "RGB"
        mock_image_instance.convert = MagicMock(return_value=mock_image_instance)
        mock_image_instance.resize = MagicMock(return_value=mock_image_instance)
        mock_image_instance.crop = MagicMock(return_value=mock_image_instance)
        mock_image_instance.save = MagicMock()
        mock_image.open = MagicMock(return_value=mock_image_instance)
        mock_image.new = MagicMock(return_value=mock_image_instance)
        mock_image.fromarray = MagicMock(return_value=mock_image_instance)
        mock_pil.Image = mock_image

        # 模拟ImageDraw
        mock_image_draw = MagicMock()
        mock_draw_instance = MagicMock()
        mock_image_draw.Draw = MagicMock(return_value=mock_draw_instance)
        mock_pil.ImageDraw = mock_image_draw

        # 模拟ImageFont
        mock_image_font = MagicMock()
        mock_font_instance = MagicMock()
        mock_image_font.truetype = MagicMock(return_value=mock_font_instance)
        mock_image_font.load_default = MagicMock(return_value=mock_font_instance)
        mock_pil.ImageFont = mock_image_font

        return mock_pil

    @staticmethod
    def create_mock_cv2() -> MagicMock:
        """
        创建模拟的cv2模块

        Returns:
            模拟的cv2模块
        """
        mock_cv2 = MagicMock()

        # 模拟图像读取和写入
        mock_cv2.imread = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.imwrite = MagicMock(return_value=True)

        # 模拟图像处理
        mock_cv2.resize = MagicMock(return_value=np.zeros((240, 320, 3), dtype=np.uint8))
        mock_cv2.cvtColor = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.threshold = MagicMock(return_value=(127, np.zeros((480, 640), dtype=np.uint8)))

        # 模拟轮廓检测
        mock_cv2.findContours = MagicMock(return_value=([], None))

        # 模拟绘制
        mock_cv2.rectangle = MagicMock()
        mock_cv2.circle = MagicMock()
        mock_cv2.line = MagicMock()
        mock_cv2.putText = MagicMock()

        # 模拟常量
        mock_cv2.IMREAD_COLOR = 1
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.COLOR_RGB2BGR = 4
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.COLOR_RGB2GRAY = 7
        mock_cv2.THRESH_BINARY = 0
        mock_cv2.THRESH_OTSU = 8

        return mock_cv2

    @staticmethod
    def find_files_by_extension(directory: Path, extensions: list[str]) -> list[Path]:
        """
        查找指定扩展名的文件

        Args:
            directory: 搜索目录
            extensions: 文件扩展名列表

        Returns:
            匹配的文件路径列表
        """
        files = []
        for ext in extensions:
            files.extend(list(directory.glob(f"*{ext}")))
        return files

    @staticmethod
    def create_document_processor(mode: str = "auto"):
        """
        创建文档处理器

        Args:
            mode: 处理器模式

        Returns:
            文档处理器实例
        """
        from src.cli.document_processor import DocumentProcessor
        from src.core.config import DEFAULT_OCR_PROMPT

        return DocumentProcessor(mode=mode, prompt=DEFAULT_OCR_PROMPT)

    @staticmethod
    def create_mock_image_processor():
        """
        创建模拟图像处理器

        Returns:
            模拟的图像处理器实例
        """
        from unittest.mock import MagicMock

        mock_processor = MagicMock()
        mock_processor.load_image.return_value = "mock_image"
        mock_processor.preprocess_for_ocr.return_value = "preprocessed_image"
        return mock_processor

    @staticmethod
    def create_mock_ocr_service():
        """
        创建模拟OCR服务

        Returns:
            模拟的OCR服务实例
        """
        from unittest.mock import MagicMock

        mock_service = MagicMock()
        mock_service.process_image.return_value = {"text": "OCR result", "confidence": 0.95}
        return mock_service

    @staticmethod
    def create_mock_pdf_converter():
        """
        创建模拟PDF转换器

        Returns:
            模拟的PDF转换器实例
        """
        from unittest.mock import MagicMock

        mock_converter = MagicMock()
        mock_converter.convert_to_images.return_value = ["/tmp/page_1.jpg", "/tmp/page_2.jpg"]
        mock_converter.is_valid_pdf.return_value = True
        return mock_converter

    @staticmethod
    def create_mock_encoder():
        """
        创建模拟编码器

        Returns:
            模拟的编码器实例
        """
        from unittest.mock import MagicMock

        mock_encoder = MagicMock()
        mock_encoder.encode_image.return_value = [[0.1, 0.2, 0.3]]
        mock_encoder.encode_text.return_value = [[0.4, 0.5, 0.6]]
        return mock_encoder

    @staticmethod
    def create_mock_ocr_engine():
        """
        创建模拟OCR引擎

        Returns:
            模拟的OCR引擎实例
        """
        from src.core.base.ocr_engine import BaseOCREngine

        class MockEngine(BaseOCREngine):
            def initialize(self) -> None:
                pass

            def process(self, images, output_dir) -> None:
                pass

            def cleanup(self) -> None:
                pass

        return MockEngine()

    @staticmethod
    def create_mock_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = ""):
        """
        创建模拟subprocess.run返回结果

        Args:
            returncode: 返回码
            stdout: 标准输出
            stderr: 标准错误

        Returns:
            模拟的subprocess结果对象
        """
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = returncode
        mock_result.stdout = stdout
        mock_result.stderr = stderr
        return mock_result

    @staticmethod
    def patch_torch(device_type: str = "cpu"):
        """
        上下文管理器，用于模拟torch模块

        Args:
            device_type: 设备类型，如"cpu", "cuda", "mps"等

        Returns:
            上下文管理器
        """
        mock_torch = TestUtils.create_mock_torch(device_type)

        # 更新sys.modules中的torch模拟
        sys.modules["torch"] = mock_torch
        sys.modules["torch.nn"] = mock_torch.nn
        sys.modules["torch.optim"] = mock_torch.optim
        sys.modules["torch.distributed"] = mock_torch.distributed
        sys.modules["torch.cuda"] = mock_torch.cuda
        sys.modules["torch.mps"] = mock_torch.mps

        # 返回一个简单的上下文管理器，不需要patch
        class TorchMockContext:
            def __enter__(self):
                return mock_torch

            def __exit__(self, exc_type, exc_val, exc_tb):
                # 清理sys.modules
                modules_to_remove = ["torch", "torch.nn", "torch.optim", "torch.distributed", "torch.cuda", "torch.mps"]
                for module in modules_to_remove:
                    if module in sys.modules:
                        del sys.modules[module]

        return TorchMockContext()

    @staticmethod
    def patch_transformers():
        """
        上下文管理器，用于模拟transformers模块

        Returns:
            上下文管理器
        """
        mock_transformers = TestUtils.create_mock_transformers()
        return patch("transformers", mock_transformers)

    @staticmethod
    def patch_vllm():
        """
        上下文管理器，用于模拟vllm模块

        Returns:
            上下文管理器
        """
        mock_vllm = TestUtils.create_mock_vllm()
        return patch("vllm", mock_vllm)

    @staticmethod
    def patch_pil():
        """
        上下文管理器，用于模拟PIL模块

        Returns:
            上下文管理器
        """
        mock_pil = TestUtils.create_mock_pil()
        return patch("PIL", mock_pil)

    @staticmethod
    def patch_cv2():
        """
        上下文管理器，用于模拟cv2模块

        Returns:
            上下文管理器
        """
        mock_cv2 = TestUtils.create_mock_cv2()
        return patch("cv2", mock_cv2)

    @staticmethod
    def create_test_image_file(file_path: str | Path, size: tuple = (640, 480), mode: str = "RGB") -> None:
        """
        创建测试图像文件

        Args:
            file_path: 图像文件路径
            size: 图像大小
            mode: 图像模式
        """
        # 创建一个简单的测试图像
        image_array = np.random.randint(0, 256, (*size, 3 if mode == "RGB" else 1), dtype=np.uint8)

        # 使用PIL保存图像
        from PIL import Image

        image = Image.fromarray(image_array, mode)
        image.save(file_path)

    @staticmethod
    def create_test_pdf_file(file_path: str | Path) -> None:
        """
        创建测试PDF文件

        Args:
            file_path: PDF文件路径
        """
        # 创建一个简单的测试PDF文件
        Path(file_path).touch()

    @staticmethod
    def create_test_docx_file(file_path: str | Path) -> None:
        """
        创建测试Word文档文件

        Args:
            file_path: Word文档文件路径
        """
        # 创建一个简单的测试Word文档文件
        Path(file_path).touch()

    @staticmethod
    def create_test_pptx_file(file_path: str | Path) -> None:
        """
        创建测试PowerPoint文件

        Args:
            file_path: PowerPoint文件路径
        """
        # 创建一个简单的测试PowerPoint文件
        Path(file_path).touch()

    @staticmethod
    def create_test_xlsx_file(file_path: str | Path) -> None:
        """
        创建测试Excel文件

        Args:
            file_path: Excel文件路径
        """
        # 创建一个简单的测试Excel文件
        Path(file_path).touch()

    @staticmethod
    def create_temp_dir() -> Generator[Path, None, None]:
        """
        创建临时目录

        Returns:
            临时目录路径的生成器
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @staticmethod
    def create_test_files_in_dir(dir_path: Path, file_types: list[str] = None) -> dict[str, Path]:
        """
        在指定目录中创建测试文件

        Args:
            dir_path: 目录路径
            file_types: 文件类型列表，默认为["jpg", "png", "pdf", "docx", "pptx", "xlsx"]

        Returns:
            文件路径字典
        """
        if file_types is None:
            file_types = ["jpg", "png", "pdf", "docx", "pptx", "xlsx"]

        file_paths = {}

        for file_type in file_types:
            file_name = f"test.{file_type}"
            file_path = dir_path / file_name

            if file_type in ["jpg", "jpeg", "png"]:
                TestUtils.create_test_image_file(file_path)
            elif file_type == "pdf":
                TestUtils.create_test_pdf_file(file_path)
            elif file_type == "docx":
                TestUtils.create_test_docx_file(file_path)
            elif file_type == "pptx":
                TestUtils.create_test_pptx_file(file_path)
            elif file_type == "xlsx":
                TestUtils.create_test_xlsx_file(file_path)

            file_paths[file_type] = file_path

        return file_paths


class MockTensor:
    """模拟Tensor类，用于测试"""

    def __init__(self, data=None, shape=None, dtype=None, device_type="cpu"):
        self.data = data
        self.shape = shape if shape is not None else (1, 3, 224, 224)
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = torch.device(device_type)

    def to(self, device):
        new_tensor = MockTensor(self.data, self.shape, self.dtype)
        new_tensor.device = torch.device(device)
        return new_tensor

    def cpu(self):
        new_tensor = MockTensor(self.data, self.shape, self.dtype)
        new_tensor.device = torch.device("cpu")
        return new_tensor

    def numpy(self):
        return np.zeros(self.shape)

    def __getitem__(self, indices):
        """支持索引操作，返回一个新的MockTensor"""
        # 简单模拟索引操作，返回一个新的MockTensor
        # 这里不进行复杂的形状计算，只返回一个形状较小的MockTensor
        if isinstance(indices, tuple):
            # 如果是元组索引，返回一个形状较小的MockTensor
            new_shape = list(self.shape)
            for i, idx in enumerate(indices):
                if isinstance(idx, int) and i < len(new_shape):
                    new_shape.pop(i)
            return MockTensor(shape=new_shape, dtype=self.dtype, device_type=self.device.type)
        else:
            # 如果是单个索引，返回一个形状较小的MockTensor
            if len(self.shape) > 1:
                new_shape = self.shape[1:]
            else:
                new_shape = (1,)
            return MockTensor(shape=new_shape, dtype=self.dtype, device_type=self.device.type)

    def __str__(self):
        return f"MockTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"
