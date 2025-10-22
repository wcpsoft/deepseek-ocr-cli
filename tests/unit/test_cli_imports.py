#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI导入测试
确保所有CLI模块可以正确导入
"""

import pytest
from unittest.mock import patch

def test_document_processor_import():
    """测试DocumentProcessor导入"""
    from cli.document_processor import DocumentProcessor
    assert DocumentProcessor is not None

def test_model_manager_import():
    """测试ModelManager导入"""
    from cli.model_manager import ModelManager
    assert ModelManager is not None

def test_main_import():
    """测试main模块导入"""
    import cli.main
    assert cli.main is not None

def test_mode_determination():
    """测试模式确定功能"""
    from cli.document_processor import DocumentProcessor
    
    # 测试auto模式（在没有vLLM的环境中应该返回transformers）
    processor_auto = DocumentProcessor(mode="auto")
    actual_mode = processor_auto._determine_mode()
    # 在当前测试环境中，vLLM不可用，所以应该返回transformers
    assert actual_mode == "transformers"
    
    # 测试在非MPS环境下且vLLM可用的情况
    processor_vllm = DocumentProcessor(mode="vllm")
    # 直接模拟方法返回值
    processor_vllm._is_mps_environment = lambda: False
    processor_vllm._is_vllm_available = lambda: True
    assert processor_vllm._determine_mode() == "vllm"
    
    processor_transformers = DocumentProcessor(mode="transformers")
    processor_transformers._is_mps_environment = lambda: False
    processor_transformers._is_vllm_available = lambda: True
    assert processor_transformers._determine_mode() == "transformers"
    
    # 测试auto模式在vLLM可用且非MPS环境时应该选择vLLM
    processor_auto_vllm = DocumentProcessor(mode="auto")
    processor_auto_vllm._is_mps_environment = lambda: False
    processor_auto_vllm._is_vllm_available = lambda: True
    assert processor_auto_vllm._determine_mode() == "vllm"
    
    # 测试在MPS环境下的特殊处理
    processor_vllm_on_mps = DocumentProcessor(mode="vllm")
    processor_vllm_on_mps._is_mps_environment = lambda: True
    processor_vllm_on_mps._is_vllm_available = lambda: False
    # 即使强制使用vLLM，在MPS环境下也应该回退到transformers
    assert processor_vllm_on_mps._determine_mode() == "transformers"
    
    # 在MPS环境下使用transformers模式
    processor_transformers_on_mps = DocumentProcessor(mode="transformers")
    processor_transformers_on_mps._is_mps_environment = lambda: True
    processor_transformers_on_mps._is_vllm_available = lambda: False
    assert processor_transformers_on_mps._determine_mode() == "transformers"
    
    # 测试auto模式在MPS环境下应该选择transformers
    processor_auto_mps = DocumentProcessor(mode="auto")
    processor_auto_mps._is_mps_environment = lambda: True
    processor_auto_mps._is_vllm_available = lambda: False
    assert processor_auto_mps._determine_mode() == "transformers"