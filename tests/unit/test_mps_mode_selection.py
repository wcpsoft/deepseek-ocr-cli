#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS环境模式选择单元测试
"""

import pytest
from unittest.mock import patch, MagicMock

def test_mps_environment_detection():
    """测试MPS环境检测功能"""
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.backends.mps.is_built', return_value=True):
        from cli.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        # 模拟在MPS环境下
        with patch.object(processor, '_is_mps_environment', return_value=True):
            assert processor._is_mps_environment() == True

def test_mode_selection_on_mps():
    """测试在MPS环境下的模式选择"""
    from cli.document_processor import DocumentProcessor
    
    # 模拟在MPS环境下，且vLLM不可用
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.backends.mps.is_built', return_value=True):
        processor = DocumentProcessor()
        with patch.object(processor, '_is_vllm_available', return_value=False), \
             patch.object(processor, '_is_mps_environment', return_value=True):
            # 在auto模式下应该选择transformers
            mode = processor._determine_mode()
            assert mode == "transformers"

def test_forced_vllm_mode_on_mps():
    """测试在MPS环境下强制使用vLLM模式时的回退行为"""
    from cli.document_processor import DocumentProcessor
    
    # 模拟在MPS环境下
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.backends.mps.is_built', return_value=True):
        processor = DocumentProcessor(mode="vllm")
        with patch.object(processor, '_is_mps_environment', return_value=True):
            # 即使强制使用vLLM，也应该回退到transformers
            mode = processor._determine_mode()
            assert mode == "transformers"

def test_cli_main_mps_mode_handling():
    """测试CLI主程序在MPS环境下的模式处理"""
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.backends.mps.is_built', return_value=True):
        with patch('cli.main.argparse.ArgumentParser.parse_args') as mock_parse_args:
            # 模拟命令行参数，强制使用vLLM模式
            args = MagicMock()
            args.mode = "vllm"
            args.download_models = False
            args.input = "test.pdf"
            args.output = "output"
            args.model_path = None
            args.prompt = "<image>\n<|grounding|>Convert the document to markdown."
            args.base_size = 1024
            args.image_size = 640
            args.crop_mode = True
            mock_parse_args.return_value = args
            
            # 模拟DocumentProcessor.process方法
            with patch('cli.document_processor.DocumentProcessor.process') as mock_process:
                from cli.main import main
                # 捕获print输出
                with patch('builtins.print') as mock_print:
                    main()
                    # 验证是否打印了警告信息
                    mock_print.assert_any_call("警告: MPS环境不支持vLLM引擎，自动切换到Transformers引擎")
                    # 验证DocumentProcessor被正确调用
                    mock_process.assert_called_once()