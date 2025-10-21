#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理端到端测试
"""

import pytest
from pathlib import Path
import tempfile
import shutil

def ensure_model_downloaded():
    """确保模型已下载"""
    try:
        from cli.model_manager import ModelManager
        model_manager = ModelManager("./models")
        
        # 检查模型是否存在且完整
        model_path = model_manager.get_model_path("deepseek-ocr")
        if model_path and model_manager.verify_model("deepseek-ocr"):
            return True
            
        # 如果模型不存在或不完整，下载模型
        print("正在自动下载OCR模型...")
        model_manager.download_models(["deepseek-ocr"])
        
        # 验证下载是否成功
        model_path = model_manager.get_model_path("deepseek-ocr")
        if model_path and model_manager.verify_model("deepseek-ocr"):
            print("模型下载完成")
            return True
        else:
            print("模型下载失败")
            return False
    except Exception as e:
        print(f"模型下载过程中出错: {e}")
        return False

# 在测试模块加载时自动确保模型已下载
pytestmark = pytest.mark.skipif(
    not ensure_model_downloaded(),
    reason="模型不可用且无法自动下载"
)

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