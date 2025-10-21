#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端测试
使用samples目录中的示例文件进行集成测试
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_model_availability():
    """测试模型可用性"""
    try:
        from cli.model_manager import ModelManager
        
        # 检查默认模型是否存在
        manager = ModelManager("./models")
        model_path = manager.get_model_path("deepseek-ocr")
        
        if model_path and Path(model_path).exists():
            print("✓ 默认模型可用")
            return True
        else:
            print("⚠ 默认模型不存在，测试将继续但可能无法完成OCR处理")
            return True  # 不强制要求模型存在，因为这是端到端测试而非模型下载测试
            
    except Exception as e:
        print(f"✗ 模型可用性检查失败: {str(e)}")
        return True  # 不强制要求模型存在

def test_pdf_processing():
    """测试PDF文件处理"""
    try:
        from cli.document_processor import DocumentProcessor
        
        # 检查samples目录中是否存在PDF文件
        samples_dir = Path(__file__).parent.parent / "samples"
        pdf_files = list(samples_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("跳过PDF处理测试：未找到PDF示例文件")
            return True
            
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            
            # 处理第一个PDF文件
            processor = DocumentProcessor(mode="transformers")  # 使用Transformers模式避免vLLM依赖
            processor.process(str(pdf_files[0]), str(output_dir))
            
            # 检查输出文件是否存在
            result_file = output_dir / "result.mmd"
            if result_file.exists():
                print(f"✓ PDF处理测试通过: {pdf_files[0].name}")
                return True
            else:
                print(f"✗ PDF处理测试失败: 未生成结果文件")
                return False
                
    except Exception as e:
        print(f"✗ PDF处理测试失败: {str(e)}")
        return False

def test_image_processing():
    """测试图像文件处理"""
    try:
        from cli.document_processor import DocumentProcessor
        
        # 检查samples目录中是否存在图像文件
        samples_dir = Path(__file__).parent.parent / "samples"
        image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg")) + list(samples_dir.glob("*.png"))
        
        if not image_files:
            print("跳过图像处理测试：未找到图像示例文件")
            return True
            
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            
            # 处理第一张图像
            processor = DocumentProcessor(mode="transformers")  # 使用Transformers模式避免vLLM依赖
            processor.process(str(image_files[0]), str(output_dir))
            
            # 检查输出文件是否存在
            result_file = output_dir / "result.mmd"
            if result_file.exists():
                print(f"✓ 图像处理测试通过: {image_files[0].name}")
                return True
            else:
                print(f"✗ 图像处理测试失败: 未生成结果文件")
                return False
                
    except Exception as e:
        print(f"✗ 图像处理测试失败: {str(e)}")
        return False

def test_document_conversion():
    """测试文档转换处理（Word、PPT等）"""
    try:
        from cli.document_processor import DocumentProcessor
        
        # 检查samples目录中是否存在文档文件
        samples_dir = Path(__file__).parent.parent / "samples"
        doc_files = list(samples_dir.glob("*.docx")) + list(samples_dir.glob("*.doc")) + list(samples_dir.glob("*.pptx")) + list(samples_dir.glob("*.ppt"))
        
        if not doc_files:
            print("跳过文档转换测试：未找到文档示例文件")
            return True
            
        # 检查LibreOffice是否可用
        if not shutil.which("libreoffice"):
            print("跳过文档转换测试：未安装LibreOffice")
            return True
            
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            
            # 处理第一个文档文件
            processor = DocumentProcessor(mode="transformers")  # 使用Transformers模式避免vLLM依赖
            processor.process(str(doc_files[0]), str(output_dir))
            
            # 检查输出文件是否存在
            result_file = output_dir / "result.mmd"
            if result_file.exists():
                print(f"✓ 文档转换测试通过: {doc_files[0].name}")
                return True
            else:
                print(f"✗ 文档转换测试失败: 未生成结果文件")
                return False
                
    except Exception as e:
        print(f"✗ 文档转换测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("DeepSeek OCR CLI 端到端测试")
    print("=" * 30)
    
    # 首先检查模型可用性
    test_model_availability()
    
    tests = [
        test_pdf_processing,
        test_image_processing,
        test_document_conversion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n端到端测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有端到端测试通过")
        return 0
    else:
        print("✗ 部分端到端测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())