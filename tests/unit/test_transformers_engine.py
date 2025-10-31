#!/usr/bin/env python3
"""
Transformers引擎单元测试
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_transformers_engine_initialization() -> None:
    """测试Transformers引擎初始化"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
        },
    ):
        from src.core.transformers.transformers_engine import TransformersEngine

        # 创建Transformers引擎实例
        engine = TransformersEngine()

        # 验证初始化
        assert engine is not None
        assert engine.model_path is None
        assert engine.prompt is None
        assert engine.base_size == 1024
        assert engine.image_size == 640
        assert engine.crop_mode is True


@patch("src.core.transformers.transformers_engine.Path")
@patch("builtins.open", create=True)
def test_transformers_engine_process(mock_open: Mock, mock_path: Mock) -> None:
    """测试Transformers引擎处理"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
        },
    ):
        from PIL import Image

        from src.core.transformers.transformers_engine import TransformersEngine

        # 创建Transformers引擎实例
        engine = TransformersEngine()

        # 初始化必要的属性
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()
        engine.image_handler = MagicMock()
        engine.device = MagicMock()

        # 创建模拟图像
        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (640, 640)

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 模拟处理过程
            with patch("src.core.transformers.transformers_engine.EnhancedOCRResultProcessor") as mock_processor:
                mock_processor_instance = MagicMock()
                mock_processor.return_value = mock_processor_instance

                # 模拟Path.resolve()返回Path对象
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = mock_path_instance
                mock_path_instance.__str__ = MagicMock(return_value=str(temp_path))
                mock_path_instance.mkdir = MagicMock()
                mock_path.return_value = mock_path_instance

                # 调用处理方法
                engine.process([mock_image], str(temp_path))

                # 验证结果处理器被创建
                mock_processor.assert_called_once_with(str(temp_path))


def test_transformers_engine_process_without_initialization() -> None:
    """测试未初始化时的处理方法"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
        },
    ):
        from PIL import Image

        from src.core.transformers.transformers_engine import TransformersEngine

        # 创建Transformers引擎实例
        engine = TransformersEngine()

        # 初始化必要的属性
        engine.model = None
        engine.tokenizer = None
        engine.image_handler = None

        # 创建模拟图像
        mock_image = MagicMock(spec=Image.Image)
        # 为图像添加size属性以避免比较错误
        mock_image.size = (640, 640)

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 模拟模型初始化失败的情况
            with patch.object(engine, "initialize", side_effect=RuntimeError("Transformers引擎初始化失败")):
                # 处理应该触发初始化错误
                with pytest.raises(RuntimeError, match="Transformers引擎初始化失败"):
                    engine.process([mock_image], str(temp_path))


def test_transformers_engine_process_image_success() -> None:
    """测试图像处理成功的情况"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
            "torch": MagicMock(),
        },
    ):
        from PIL import Image

        from src.core.transformers.transformers_engine import TransformersEngine

        # 创建Transformers引擎实例
        engine = TransformersEngine()
        engine.is_initialized = True
        engine.image_handler = MagicMock()
        engine.device = MagicMock()
        engine.tokenizer = MagicMock()
        engine.model = MagicMock()

        # 创建模拟图像
        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (640, 640)

        # 模拟图像处理过程
        mock_processed_data = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        engine.image_handler.process_image.return_value = mock_processed_data
        engine.image_handler.extract_tensors.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # 模拟模型生成过程
        mock_outputs = MagicMock()
        mock_outputs.shape = [1, 100]
        engine.model.generate.return_value = mock_outputs

        # 模拟tokenizer解码
        engine.tokenizer.eos_token_id = 0
        engine.tokenizer.decode.return_value = "测试OCR结果"

        # 调用处理图像方法
        result = engine.process_image(mock_image, "测试提示词")

        # 验证结果
        assert result == "测试OCR结果"


def test_transformers_engine_process_image_not_initialized() -> None:
    """测试未初始化时处理图像"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.deepseek_ocr": MagicMock(),
            "transformers": MagicMock(),
            "src.cli.utils": MagicMock(),
        },
    ):
        from PIL import Image

        from src.core.transformers.transformers_engine import TransformersEngine

        # 创建Transformers引擎实例
        engine = TransformersEngine()
        engine.is_initialized = False

        # 创建模拟图像
        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (640, 640)

        # 模拟初始化失败
        with patch.object(engine, "initialize", return_value=False):
            # 处理应该触发初始化错误
            with pytest.raises(RuntimeError, match="Transformers引擎初始化失败"):
                engine.process_image(mock_image, "测试提示词")
