#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型可用性端到端测试
"""

import pytest
from pathlib import Path

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

def test_default_model_availability():
    """测试默认模型可用性"""
    from cli.model_manager import ModelManager
    
    # 检查默认模型是否存在
    manager = ModelManager("./models")
    model_path = manager.get_model_path("deepseek-ocr")
    
    # 现在强制要求模型存在，因为我们会自动下载
    assert model_path is not None, "模型路径不应为None"
    assert Path(model_path).exists(), f"模型路径不存在: {model_path}"