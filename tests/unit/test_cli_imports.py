#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI模块导入测试
"""

import pytest

def test_cli_module_imports():
    """测试CLI模块导入"""
    # 测试核心CLI模块导入
    from cli.document_processor import DocumentProcessor
    from cli.model_manager import ModelManager
    from cli.pdf_converter import PDFConverter
    
    # 验证导入成功
    assert DocumentProcessor is not None
    assert ModelManager is not None
    assert PDFConverter is not None

def test_cli_class_instantiation():
    """测试CLI类实例化"""
    from cli.document_processor import DocumentProcessor
    from cli.model_manager import ModelManager
    from cli.pdf_converter import PDFConverter
    
    # 测试类实例化
    processor = DocumentProcessor()
    manager = ModelManager()
    converter = PDFConverter()
    
    # 验证实例化成功
    assert processor is not None
    assert manager is not None
    assert converter is not None