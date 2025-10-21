#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理端到端测试
"""

import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.mark.skipif(not shutil.which("libreoffice"), reason="LibreOffice not installed")
def test_pdf_processing(samples_dir):
    """测试PDF文件处理"""
    from cli.document_processor import DocumentProcessor
    
    # 检查samples目录中是否存在PDF文件
    pdf_files = list(samples_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("未找到PDF示例文件")
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        
        # 处理第一个PDF文件
        processor = DocumentProcessor(mode="transformers")  # 使用Transformers模式避免vLLM依赖
        processor.process(str(pdf_files[0]), str(output_dir))
        
        # 检查输出文件是否存在
        result_file = output_dir / "result.mmd"
        assert result_file.exists(), "未生成结果文件"

def test_image_processing(samples_dir):
    """测试图像文件处理"""
    from cli.document_processor import DocumentProcessor
    
    # 检查samples目录中是否存在图像文件
    image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg")) + list(samples_dir.glob("*.png"))
    if not image_files:
        pytest.skip("未找到图像示例文件")
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        
        # 处理第一张图像
        processor = DocumentProcessor(mode="transformers")  # 使用Transformers模式避免vLLM依赖
        processor.process(str(image_files[0]), str(output_dir))
        
        # 检查输出文件是否存在
        result_file = output_dir / "result.mmd"
        assert result_file.exists(), "未生成结果文件"

@pytest.mark.skipif(not shutil.which("libreoffice"), reason="LibreOffice not installed")
def test_document_conversion(samples_dir):
    """测试文档转换处理（Word、PPT等）"""
    from cli.document_processor import DocumentProcessor
    
    # 检查samples目录中是否存在文档文件
    doc_files = list(samples_dir.glob("*.docx")) + list(samples_dir.glob("*.doc")) + list(samples_dir.glob("*.pptx")) + list(samples_dir.glob("*.ppt"))
    if not doc_files:
        pytest.skip("未找到文档示例文件")
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        
        # 处理第一个文档文件
        processor = DocumentProcessor(mode="transformers")  # 使用Transformers模式避免vLLM依赖
        processor.process(str(doc_files[0]), str(output_dir))
        
        # 检查输出文件是否存在
        result_file = output_dir / "result.mmd"
        assert result_file.exists(), "未生成结果文件"