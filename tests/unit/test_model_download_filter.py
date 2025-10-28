#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载过滤配置测试
验证模型下载时的文件过滤配置是否正确
"""

import sys
import os
import pytest
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_model_download_filter_configuration():
    """测试模型下载过滤配置"""
    try:
        from src.cli.model_manager import ModelManager, MODEL_DOWNLOAD_PATTERNS
        # 验证忽略模式配置
        assert "!*.md" in MODEL_DOWNLOAD_PATTERNS
        assert "!README*" in MODEL_DOWNLOAD_PATTERNS
        assert "!tests/*" in MODEL_DOWNLOAD_PATTERNS
    except ImportError:
        pytest.fail("无法导入模型管理器")

def test_huggingface_download_patterns():
    """测试Hugging Face下载模式配置"""
    try:
        from src.cli.model_manager import ModelManager, MODEL_DOWNLOAD_PATTERNS
        # 验证关键的忽略模式
        assert "!*.md" in MODEL_DOWNLOAD_PATTERNS  # 文档文件
        assert "!tests/*" in MODEL_DOWNLOAD_PATTERNS  # 测试文件
        assert "!examples/*" in MODEL_DOWNLOAD_PATTERNS  # 示例文件
    except ImportError:
        pytest.fail("无法导入模型管理器")

def test_model_verification_logic():
    """测试模型验证逻辑"""
    try:
        from src.cli.model_manager import ModelManager
        manager = ModelManager("/tmp/test_models")
        # 测试模型目录设置
        manager.set_model_dir("/tmp/new_models")
        # 使用Path.resolve()来处理符号链接
        expected_path = str(Path("/tmp/new_models").resolve())
        actual_path = manager.get_model_dir()
        assert actual_path == expected_path
    except ImportError:
        pytest.fail("无法导入模型管理器")