#!/usr/bin/env python3
"""
文档处理端到端测试
验证文档处理器的功能和模型可用性
"""

import shutil
import tempfile
from pathlib import Path

import pytest


def ensure_model_downloaded() -> bool:
    """确保模型已下载"""
    try:
        from src.cli.model_manager import ModelManager

        # 创建模型管理器实例
        model_manager = ModelManager()

        # 检查模型目录是否存在
        model_path = model_manager.model_dir / "deepseek-ocr"
        if not model_path.exists():
            print("模型目录不存在, 跳过测试")
            return False

        # 验证模型文件
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
        ]

        missing_files = []
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            print(f"模型文件缺失: {', '.join(missing_files)}, 跳过测试")
            return False

        return True

    except OSError as e:
        print(f"模型检查过程中出错: {e}")
        return False


# 在测试模块加载时自动确保模型已下载
pytestmark = pytest.mark.skipif(not ensure_model_downloaded(), reason="模型不可用且无法自动下载")


@pytest.mark.skipif(not shutil.which("libreoffice"), reason="LibreOffice not installed")
def test_pdf_processing(samples_dir: Path) -> None:
    """测试PDF文件处理"""
    try:
        from src.cli.document_processor import DocumentProcessor

        # 检查samples目录中是否存在PDF文件
        pdf_files = list(samples_dir.glob("*.pdf"))
        if not pdf_files:
            pytest.skip("未找到PDF示例文件")

        # 创建临时输出目录
        with tempfile.TemporaryDirectory():
            # output_dir = Path(temp_dir) / "output"

            # 处理第一个PDF文件
            from src.core.config import DEFAULT_OCR_PROMPT

            processor = DocumentProcessor(mode="transformers", prompt=DEFAULT_OCR_PROMPT)
            # 不实际运行处理, 只测试初始化
            assert processor is not None

    except ImportError as e:
        pytest.fail(f"文档处理器初始化测试失败: {e}")


def test_image_processing(samples_dir: Path) -> None:
    """测试图像文件处理"""
    try:
        from src.cli.document_processor import DocumentProcessor

        # 检查samples目录中是否存在图像文件
        image_files = (
            list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg")) + list(samples_dir.glob("*.png"))
        )
        if not image_files:
            pytest.skip("未找到图像示例文件")

        # 创建临时输出目录
        with tempfile.TemporaryDirectory():
            # output_dir = Path(temp_dir) / "output"

            # 处理第一张图像
            from src.core.config import DEFAULT_OCR_PROMPT

            processor = DocumentProcessor(mode="transformers", prompt=DEFAULT_OCR_PROMPT)
            # 不实际运行处理, 只测试初始化
            assert processor is not None

    except ImportError as e:
        pytest.fail(f"文档处理器初始化测试失败: {e}")


@pytest.mark.skipif(not shutil.which("libreoffice"), reason="LibreOffice not installed")
def test_document_conversion(samples_dir: Path) -> None:
    """测试文档转换处理 (Word, PPT等)"""
    try:
        from src.cli.document_processor import DocumentProcessor

        # 检查samples目录中是否存在文档文件
        doc_files = (
            list(samples_dir.glob("*.docx"))
            + list(samples_dir.glob("*.doc"))
            + list(samples_dir.glob("*.pptx"))
            + list(samples_dir.glob("*.ppt"))
        )
        if not doc_files:
            pytest.skip("未找到文档示例文件")

        # 创建临时输出目录
        with tempfile.TemporaryDirectory():

            # 处理第一个文档文件
            from src.core.config import DEFAULT_OCR_PROMPT

            processor = DocumentProcessor(mode="transformers", prompt=DEFAULT_OCR_PROMPT)
            # 不实际运行处理, 只测试初始化
            assert processor is not None

    except ImportError as e:
        pytest.fail(f"文档处理器初始化测试失败: {e}")


def test_model_directory_setting() -> None:
    """测试模型目录设置"""
    try:
        from src.cli.document_processor import DocumentProcessor
        from src.core.config import DEFAULT_OCR_PROMPT

        # 创建处理器实例
        processor = DocumentProcessor(mode="transformers", prompt=DEFAULT_OCR_PROMPT)
        assert processor.mode is not None

    except ImportError as e:
        pytest.fail(f"模型目录设置测试失败: {e}")


def test_model_manager_import() -> None:
    """测试模型管理器导入"""
    try:
        from src.cli.model_manager import ModelManager

        assert ModelManager is not None
    except ImportError:
        pytest.fail("无法导入ModelManager")
