#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI工具测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试导入是否正常"""
    try:
        from cli.main import main
        from cli.document_processor import DocumentProcessor
        from cli.model_manager import ModelManager
        from cli.pdf_converter import PDFConverter
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_document_processor():
    """测试文档处理器"""
    try:
        from cli.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        print("✓ DocumentProcessor类创建成功")
        return True
    except Exception as e:
        print(f"✗ DocumentProcessor类创建失败: {e}")
        return False

def test_model_manager():
    """测试模型管理器"""
    try:
        from cli.model_manager import ModelManager
        manager = ModelManager("./test_models")
        print("✓ ModelManager类创建成功")
        return True
    except Exception as e:
        print(f"✗ ModelManager类创建失败: {e}")
        return False

def test_pdf_converter():
    """测试PDF转换器"""
    try:
        from cli.pdf_converter import PDFConverter
        converter = PDFConverter()
        print("✓ PDFConverter类创建成功")
        return True
    except Exception as e:
        print(f"✗ PDFConverter类创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("DeepSeek OCR CLI工具测试")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_document_processor,
        test_model_manager,
        test_pdf_converter
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过，CLI工具可以正常使用")
        return 0
    else:
        print("✗ 部分测试失败，请检查代码")
        return 1

if __name__ == "__main__":
    sys.exit(main())