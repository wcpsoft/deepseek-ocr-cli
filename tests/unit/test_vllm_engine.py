#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM引擎单元测试
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


def test_vllm_engine_initialization():
    """测试vLLM引擎初始化"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'vllm': MagicMock(),
        'vllm.model_executor': MagicMock(),
        'vllm.model_executor.models': MagicMock(),
        'vllm.model_executor.models.registry': MagicMock(),
        'src.core.process.ngram_norepeat': MagicMock(),
        'src.core.process.image_process': MagicMock(),
    }):
        from src.core.vllm.vllm_engine import VLLMEngine
        
        # 创建vLLM引擎实例
        engine = VLLMEngine()
        
        # 验证初始化
        assert engine is not None
        assert engine.model_path is None
        assert engine.prompt is None
        assert engine.base_size == 1024
        assert engine.image_size == 640
        assert engine.crop_mode is True


def test_vllm_engine_cleanup():
    """测试vLLM引擎清理"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'vllm': MagicMock(),
        'vllm.model_executor': MagicMock(),
        'vllm.model_executor.models': MagicMock(),
        'vllm.model_executor.models.registry': MagicMock(),
        'src.core.process.ngram_norepeat': MagicMock(),
        'src.core.process.image_process': MagicMock(),
    }):
        from src.core.vllm.vllm_engine import VLLMEngine
        
        # 创建vLLM引擎实例
        engine = VLLMEngine()
        
        # 调用清理方法（应该不执行任何操作）
        engine.cleanup()
        
        # 验证没有异常
        assert True


@patch('src.core.vllm.vllm_engine.Path')
@patch('builtins.open', create=True)
def test_vllm_engine_save_results(mock_open, mock_path):
    """测试vLLM引擎保存结果"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'vllm': MagicMock(),
        'vllm.model_executor': MagicMock(),
        'vllm.model_executor.models': MagicMock(),
        'vllm.model_executor.models.registry': MagicMock(),
        'src.core.process.ngram_norepeat': MagicMock(),
        'src.core.process.image_process': MagicMock(),
    }):
        from src.core.vllm.vllm_engine import VLLMEngine
        
        # 创建vLLM引擎实例
        engine = VLLMEngine()
        
        # 创建模拟输出
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "测试结果"
        outputs_list = [mock_output]
        
        # 调用保存结果方法
        engine._save_results(outputs_list, Path("/tmp"))
        
        # 验证文件被正确打开
        mock_open.assert_called_once_with(Path("/tmp/result.mmd"), 'w', encoding='utf-8')


def test_vllm_engine_process_without_initialization():
    """测试未初始化时的处理方法"""
    with patch.dict('sys.modules', {
        'src.core.deepseek_ocr': MagicMock(),
        'vllm': MagicMock(),
        'vllm.model_executor': MagicMock(),
        'vllm.model_executor.models': MagicMock(),
        'vllm.model_executor.models.registry': MagicMock(),
        'src.core.process.ngram_norepeat': MagicMock(),
        'src.core.process.image_process': MagicMock(),
    }):
        from src.core.vllm.vllm_engine import VLLMEngine
        from PIL import Image
        
        # 创建vLLM引擎实例
        engine = VLLMEngine()
        
        # 创建模拟图像
        mock_image = MagicMock(spec=Image.Image)
        # 为图像添加size属性以避免比较错误
        mock_image.size = (640, 640)
        
        # 处理应该触发初始化
        with pytest.raises(RuntimeError):
            engine.process([mock_image], "/tmp")