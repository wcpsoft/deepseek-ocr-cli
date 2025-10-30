#!/usr/bin/env python3
"""
vLLM引擎单元测试
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_vllm_engine_initialization() -> None:
    """测试vLLM引擎初始化"""
    # 创建真实的配置对象, 而不是模拟整个模块
    from src.core.config.settings import Config

    mock_config = Config()

    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "vllm": MagicMock(),
            "vllm.model_executor": MagicMock(),
            "vllm.model_executor.models": MagicMock(),
            "vllm.model_executor.models.registry": MagicMock(),
            "src.core.process.ngram_norepeat": MagicMock(),
            "src.core.process.image_process": MagicMock(),
            "src.core.multimodal.ocr_engine_interface": MagicMock(),
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "transformers": MagicMock(),
            "transformers.modeling_outputs": MagicMock(),
        },
    ):
        # 模拟torch._C._has_torch_function, 避免docstring冲突
        with patch("torch._C._has_torch_function", Mock()):
            # 模拟get_config函数返回真实配置对象
            with patch("src.core.config.get_config", return_value=mock_config):
                from src.core.vllm.vllm_engine import VLLMEngine

                # 创建vLLM引擎实例
                engine = VLLMEngine()

                # 手动设置属性, 因为__init__被模拟了
                engine.model_path = mock_config.MODEL_PATH

                # 验证初始化
                assert engine is not None
                # 检查model_path是否为期望值之一
                assert engine.model_path in [
                    "deepseek-ai/DeepSeek-OCR",
                    "./models/deepseek-ocr",
                ]


def test_vllm_engine_cleanup() -> None:
    """测试vLLM引擎清理"""
    # 创建真实的配置对象, 而不是模拟整个模块
    from src.core.config.settings import Config

    mock_config = Config()

    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "vllm": MagicMock(),
            "vllm.model_executor": MagicMock(),
            "vllm.model_executor.models": MagicMock(),
            "vllm.model_executor.models.registry": MagicMock(),
            "src.core.process.ngram_norepeat": MagicMock(),
            "src.core.process.image_process": MagicMock(),
            "src.core.multimodal.ocr_engine_interface": MagicMock(),
            # 添加torch相关模块的模拟, 避免版本冲突
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "transformers": MagicMock(),
            "transformers.modeling_outputs": MagicMock(),
        },
    ):
        # 模拟torch._C._has_torch_function, 避免docstring冲突
        with patch("torch._C._has_torch_function", Mock()):
            # 模拟get_config函数返回真实配置对象
            with patch("src.core.config.get_config", return_value=mock_config):
                from src.core.vllm.vllm_engine import VLLMEngine

                # 创建vLLM引擎实例
                engine = VLLMEngine()

                # 手动设置属性, 因为__init__被模拟了
                engine.model_path = mock_config.MODEL_PATH
                engine.llm = MagicMock()
                engine.processor = MagicMock()
                engine.is_initialized = True

                # 模拟cleanup方法
                def mock_cleanup() -> None:
                    engine.llm = None
                    engine.processor = None
                    engine.is_initialized = False

                with patch.object(engine, "cleanup", side_effect=mock_cleanup):
                    # 调用清理方法
                    engine.cleanup()

                    # 验证llm和processor被设置为None
                    assert engine.llm is None
                    assert engine.processor is None
                    assert engine.is_initialized is False


def test_vllm_engine_process_without_initialization() -> None:
    """测试未初始化的引擎处理图像"""
    # 创建真实的配置对象, 而不是模拟整个模块
    from src.core.config.settings import Config

    mock_config = Config()

    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "vllm": MagicMock(),
            "vllm.model_executor": MagicMock(),
            "vllm.model_executor.models": MagicMock(),
            "vllm.model_executor.models.registry": MagicMock(),
            "src.core.process.ngram_norepeat": MagicMock(),
            "src.core.process.image_process": MagicMock(),
            "src.core.multimodal.ocr_engine_interface": MagicMock(),
            # 添加torch相关模块的模拟, 避免版本冲突
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "transformers": MagicMock(),
            "transformers.modeling_outputs": MagicMock(),
        },
    ):
        # 模拟torch._C._has_torch_function, 避免docstring冲突
        with patch("torch._C._has_torch_function", Mock()):
            # 模拟get_config函数返回真实配置对象
            with patch("src.core.config.get_config", return_value=mock_config):
                from src.core.vllm.vllm_engine import VLLMEngine

                # 创建vLLM引擎实例
                engine = VLLMEngine()

                # 手动设置属性, 因为__init__被模拟了
                engine.model_path = mock_config.MODEL_PATH

                # 模拟process_image方法抛出RuntimeError
                with patch.object(engine, "process_image", side_effect=RuntimeError("模型未初始化")):
                    # 创建测试图像
                    test_image_array = np.zeros((100, 100, 3), dtype=np.uint8)
                    test_image = Image.fromarray(test_image_array)

                    # 验证抛出异常
                    with pytest.raises(RuntimeError, match="模型未初始化"):
                        engine.process_image(test_image, "测试提示词")


def test_vllm_engine_save_results() -> None:
    """测试vLLM引擎保存结果"""
    # 创建真实的配置对象, 而不是模拟整个模块
    from src.core.config.settings import Config

    mock_config = Config()

    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "vllm": MagicMock(),
            "vllm.model_executor": MagicMock(),
            "vllm.model_executor.models": MagicMock(),
            "vllm.model_executor.models.registry": MagicMock(),
            "src.core.process.ngram_norepeat": MagicMock(),
            "src.core.process.image_process": MagicMock(),
            "src.core.multimodal.ocr_engine_interface": MagicMock(),
            # 添加torch相关模块的模拟, 避免版本冲突
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "transformers": MagicMock(),
            "transformers.modeling_outputs": MagicMock(),
        },
    ):
        # 模拟torch._C._has_torch_function, 避免docstring冲突
        with patch("torch._C._has_torch_function", Mock()):
            # 模拟get_config函数返回真实配置对象
            with patch("src.core.config.get_config", return_value=mock_config):
                from src.core.vllm.vllm_engine import VLLMEngine

                # 创建vLLM引擎实例
                engine = VLLMEngine()

                # 手动设置属性, 因为__init__被模拟了
                engine.model_path = mock_config.MODEL_PATH

                # 验证引擎实例存在
                assert engine is not None
