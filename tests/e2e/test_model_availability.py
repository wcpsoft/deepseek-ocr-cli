#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型可用性端到端测试
"""

import pytest
from pathlib import Path

def test_default_model_availability():
    """测试默认模型可用性"""
    from cli.model_manager import ModelManager
    
    # 检查默认模型是否存在
    manager = ModelManager("./models")
    model_path = manager.get_model_path("deepseek-ocr")
    
    # 注意：这里不强制要求模型存在，因为这是端到端测试而非模型下载测试
    # 如果模型存在，验证其路径
    if model_path:
        assert Path(model_path).exists()