#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformers引擎单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_transformers_engine_initialization():
    """测试Transformers引擎初始化"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'transformers': MagicMock(),
        'src.cli.utils': MagicMock(),
    }):
        from src.core.transformers.transformers_engine import TransformersEngine
        
        # 创建Transformers引擎实例
        engine = TransformersEngine()
        
        # 验证初始化
        assert engine is not None
        assert engine.model_path is None
        assert engine.prompt is None
        assert engine.base_size == 1024
        assert engine.image_size == 640
        assert engine.crop_mode is True


def test_transformers_engine_cleanup():
    """测试Transformers引擎清理"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'transformers': MagicMock(),
        'src.cli.utils': MagicMock(),
    }):
        from src.core.transformers.transformers_engine import TransformersEngine
        
        # 创建Transformers引擎实例
        engine = TransformersEngine()
        
        # 调用清理方法（应该不执行任何操作）
        engine.cleanup()
        
        # 验证没有异常
        assert True


@patch('src.core.transformers.transformers_engine.Path')
@patch('builtins.open', create=True)
def test_transformers_engine_save_results(mock_open, mock_path):
    """测试Transformers引擎保存结果"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'transformers': MagicMock(),
        'src.cli.utils': MagicMock(),
    }):
        from src.core.transformers.transformers_engine import TransformersEngine
        
        # 创建Transformers引擎实例
        engine = TransformersEngine()
        
        # 创建模拟结果
        results = ["测试结果1", "测试结果2"]
        
        # 调用保存结果方法
        engine._save_results(results, Path("/tmp"))
        
        # 验证文件被正确打开
        mock_open.assert_called_once_with(Path("/tmp/result.mmd"), 'w', encoding='utf-8')


def test_transformers_engine_process_without_initialization():
    """测试未初始化时的处理方法"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'transformers': MagicMock(),
        'src.cli.utils': MagicMock(),
    }):
        from src.core.transformers.transformers_engine import TransformersEngine
        from PIL import Image
        
        # 创建Transformers引擎实例
        engine = TransformersEngine()
        
        # 创建模拟图像
        mock_image = MagicMock(spec=Image.Image)
        # 为图像添加size属性以避免比较错误
        mock_image.size = (640, 640)
        
        # 模拟模型初始化失败的情况
        with patch.object(engine, 'initialize', side_effect=RuntimeError("Transformers引擎初始化失败")):
            # 处理应该触发初始化错误
            with pytest.raises(RuntimeError, match="Transformers引擎初始化失败"):
                engine.process([mock_image], "/tmp")