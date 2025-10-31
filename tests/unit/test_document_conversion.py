#!/usr/bin/env python3
"""
文档转换和图像处理单元测试
测试文档转换为图像、PDF处理、图像OCR等核心处理流程
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def test_pdf_to_images() -> None:
    """测试PDF转换为图像功能"""
    with patch.dict(
        "sys.modules",
        {
            "fitz": MagicMock(),  # PyMuPDF
        },
    ):
        from src.cli.document_processor import DocumentProcessor

        # 创建文档处理器实例
        processor = DocumentProcessor()

        # 创建模拟PDF文档
        mock_pdf_document = MagicMock()
        mock_pdf_document.page_count = 2
        
        # 创建模拟页面和像素图
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.width = 800
        mock_pixmap.height = 600
        mock_pixmap.samples = b"fake_image_data"
        
        mock_pdf_document.__getitem__.return_value = mock_page
        mock_page.get_pixmap.return_value = mock_pixmap

        # 模拟fitz.open返回我们的模拟文档
        with patch("fitz.open", return_value=mock_pdf_document):
            # 创建临时PDF文件路径
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
                pdf_path = Path(temp_pdf.name)
                
                # 调用PDF转图像方法
                images = processor._pdf_to_images(pdf_path)
                
                # 验证结果
                assert len(images) == 2
                assert all(isinstance(img, Image.Image) for img in images)
                assert images[0].size == (800, 600)


def test_convert_to_pdf() -> None:
    """测试文档转换为PDF功能"""
    from src.cli.document_processor import DocumentProcessor

    # 创建文档处理器实例
    processor = DocumentProcessor()

    # 模拟subprocess.run返回成功结果
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    # 创建临时目录和文件
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "test.docx"
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        
        # 创建一个空的输入文件
        input_path.touch()
        
        # 创建模拟的PDF输出文件
        temp_pdf_dir = tempfile.mkdtemp()
        mock_pdf_file = Path(temp_pdf_dir) / "test.pdf"
        mock_pdf_file.touch()
        
        # 模拟Path.glob返回我们的模拟PDF文件
        with patch("pathlib.Path.glob", return_value=[mock_pdf_file]):
            with patch("subprocess.run", return_value=mock_result):
                with patch("shutil.which", return_value="/usr/bin/libreoffice"):
                    # 调用转换方法
                    pdf_path = processor._convert_to_pdf(input_path, output_dir)
                    
                    # 验证结果
                    assert pdf_path.exists()
                    assert pdf_path.suffix == ".pdf"


def test_perform_ocr() -> None:
    """测试执行OCR功能"""
    with patch.dict(
        "sys.modules",
        {
            "src.core.factory.ocr_engine_factory": MagicMock(),
            "src.core.api.utils.helpers": MagicMock(),
        },
    ):
        from src.cli.document_processor import DocumentProcessor

        # 创建文档处理器实例
        processor = DocumentProcessor()

        # 创建模拟图像
        image = Image.new("RGB", (100, 100), color="red")

        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # 模拟OCR引擎工厂和引擎
            mock_engine_factory = MagicMock()
            mock_engine = MagicMock()
            mock_engine_factory.create_engine.return_value = mock_engine
            
            # 模拟create_job_directories函数
            with patch("src.core.api.utils.helpers.create_job_directories", return_value=(Path(temp_dir), Path(temp_dir) / "task")):
                # 模拟OCR引擎工厂
                with patch("src.core.factory.ocr_engine_factory.OCREngineFactory", mock_engine_factory):
                    # 调用执行OCR方法
                    processor._perform_ocr([image], output_dir)
                    
                    # 验证引擎方法被调用
                    mock_engine.initialize.assert_called_once()
                    mock_engine.process.assert_called_once()
                    mock_engine.cleanup.assert_called_once()


def test_determine_mode() -> None:
    """测试模式确定功能"""
    from src.cli.document_processor import DocumentProcessor

    # 测试auto模式在不同环境下的行为
    processor_auto = DocumentProcessor(mode="auto")
    
    # 模拟非MPS环境且vLLM可用
    with patch.object(processor_auto, "_is_mps_environment", return_value=False):
        with patch.object(processor_auto, "_is_vllm_available", return_value=True):
            mode = processor_auto._determine_mode()
            assert mode == "vllm"
    
    # 模拟MPS环境
    with patch.object(processor_auto, "_is_mps_environment", return_value=True):
        with patch.object(processor_auto, "_is_vllm_available", return_value=True):
            mode = processor_auto._determine_mode()
            assert mode == "transformers"
    
    # 测试指定模式
    processor_transformers = DocumentProcessor(mode="transformers")
    mode = processor_transformers._determine_mode()
    assert mode == "transformers"
    
    processor_vllm = DocumentProcessor(mode="vllm")
    mode = processor_vllm._determine_mode()
    assert mode == "vllm"


def test_is_vllm_available() -> None:
    """测试vLLM可用性检查"""
    from src.cli.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    
    # 模拟vLLM导入成功
    with patch("importlib.import_module", return_value=MagicMock()):
        assert processor._is_vllm_available() is True
    
    # 模拟vLLM导入失败
    with patch("importlib.import_module", side_effect=ImportError):
        assert processor._is_vllm_available() is False


def test_is_mps_environment() -> None:
    """测试MPS环境检查"""
    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
        },
    ):
        from src.cli.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        
        # 模拟MPS可用
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.backends.mps.is_built", return_value=True):
                assert processor._is_mps_environment() is True
        
        # 模拟MPS不可用
        with patch("torch.backends.mps.is_available", return_value=False):
            assert processor._is_mps_environment() is False
        
        # 模拟torch导入失败
        with patch.dict("sys.modules", {"torch": None}):
            assert processor._is_mps_environment() is False


def test_convert_to_images() -> None:
    """测试转换为图像列表功能"""
    with patch.dict(
        "sys.modules",
        {
            "fitz": MagicMock(),  # PyMuPDF
        },
    ):
        from src.cli.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        
        # 创建临时PDF文件
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            pdf_path = Path(temp_pdf.name)
            
            # 模拟PDF转图像方法
            mock_images = [Image.new("RGB", (100, 100), color="red")]
            with patch.object(processor, "_pdf_to_images", return_value=mock_images):
                # 调用转换方法
                images = processor.convert_to_images(str(pdf_path))
                
                # 验证结果
                assert len(images) == 1
                assert isinstance(images[0], Image.Image)


if __name__ == "__main__":
    pytest.main([__file__])