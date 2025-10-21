#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI模块导入测试
验证CLI模块可以正确导入
"""

import pytest
from pathlib import Path

def test_document_processor_import():
    """测试DocumentProcessor导入"""
    from cli.document_processor import DocumentProcessor
    
    # 验证类存在
    assert DocumentProcessor is not None
    
    # 验证可以创建实例
    processor = DocumentProcessor()
    assert processor is not None
    
    # 验证默认模式
    assert hasattr(processor, 'mode')
    
    # 验证支持的模式
    processor_auto = DocumentProcessor(mode="auto")
    processor_vllm = DocumentProcessor(mode="vllm")
    processor_transformers = DocumentProcessor(mode="transformers")
    
    assert processor_auto.mode == "auto"
    assert processor_vllm.mode == "vllm"
    assert processor_transformers.mode == "transformers"

def test_model_manager_import():
    """测试ModelManager导入"""
    from cli.model_manager import ModelManager
    
    # 验证类存在
    assert ModelManager is not None
    
    # 验证可以创建实例
    manager = ModelManager()
    assert manager is not None

def test_main_import():
    """测试主模块导入"""
    # 这个测试主要是确保主模块可以导入而不会出错
    import cli.main
    assert cli.main is not None

def test_mode_determination():
    """测试模式确定功能"""
    from cli.document_processor import DocumentProcessor
    
    # 测试auto模式
    processor_auto = DocumentProcessor(mode="auto")
    actual_mode = processor_auto._determine_mode()
    
    # 在当前测试环境中，vLLM不可用，所以应该返回transformers
    assert actual_mode in ["vllm", "transformers"]
    
    # 测试指定模式
    processor_vllm = DocumentProcessor(mode="vllm")
    assert processor_vllm._determine_mode() == "vllm"
    
    processor_transformers = DocumentProcessor(mode="transformers")
    assert processor_transformers._determine_mode() == "transformers"