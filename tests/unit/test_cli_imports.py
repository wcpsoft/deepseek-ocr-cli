#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI模块导入测试
验证CLI模块的正确导入和基本功能
"""

import sys
import os
import pytest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_document_processor_import():
    """测试文档处理器模块导入"""
    try:
        from src.cli.document_processor import DocumentProcessor
        assert DocumentProcessor is not None
    except ImportError as e:
        pytest.fail(f"无法导入DocumentProcessor: {e}")

def test_model_manager_import():
    """测试模型管理器模块导入"""
    try:
        from src.cli.model_manager import ModelManager
        assert ModelManager is not None
    except ImportError as e:
        pytest.fail(f"无法导入ModelManager: {e}")

def test_main_import():
    """测试主模块导入"""
    try:
        from src.cli.main import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"无法导入main: {e}")