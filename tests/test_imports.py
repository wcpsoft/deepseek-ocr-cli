#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入测试
验证项目结构和模块导入是否正常工作
"""

import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_core_imports():
    """测试核心模块导入"""
    try:
        from src.core import deepseek_ocr
        from src.core import config
        print("✓ 核心模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 核心模块导入失败: {e}")
        return False

def test_cli_imports():
    """测试CLI模块导入"""
    try:
        from cli import document_processor
        from cli import model_manager
        from cli import pdf_converter
        print("✓ CLI模块导入成功")
        return True
    except Exception as e:
        print(f"✗ CLI模块导入失败: {e}")
        return False

def test_core_classes():
    """测试核心类"""
    try:
        from src.core.deepseek_ocr import DeepseekOCRForCausalLM
        print("✓ 核心类导入成功")
        return True
    except Exception as e:
        print(f"✗ 核心类导入失败: {e}")
        return False

def test_cli_classes():
    """测试CLI类"""
    try:
        from cli.document_processor import DocumentProcessor
        from cli.model_manager import ModelManager
        from cli.pdf_converter import PDFConverter
        print("✓ CLI类导入成功")
        return True
    except Exception as e:
        print(f"✗ CLI类导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("DeepSeek OCR CLI 导入测试")
    print("=" * 30)
    
    tests = [
        test_core_imports,
        test_cli_imports,
        test_core_classes,
        test_cli_classes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有导入测试通过")
        return 0
    else:
        print("✗ 部分导入测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())