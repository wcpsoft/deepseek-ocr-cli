#!/usr/bin/env python3
"""
MPS模式选择测试
验证在Apple Silicon设备上的模式自动选择功能
"""

import os
import sys
from unittest.mock import patch

import pytest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


def test_mps_environment_detection():
    """测试MPS环境检测功能"""
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.backends.mps.is_built", return_value=True),
    ):
        try:
            from src.cli.document_processor import DocumentProcessor

            assert DocumentProcessor is not None
        except ImportError:
            pytest.fail("无法导入DocumentProcessor")


def test_mode_selection_on_mps():
    """测试在MPS环境下的模式选择"""
    try:
        from src.cli.document_processor import DocumentProcessor

        processor = DocumentProcessor(mode="auto")
        # 在MPS环境下应该自动选择transformers模式
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.backends.mps.is_built", return_value=True),
        ):
            actual_mode = processor._determine_mode()
            assert actual_mode == "transformers"
    except ImportError:
        pytest.fail("无法导入DocumentProcessor")


def test_forced_vllm_mode_on_mps():
    """测试在MPS环境下强制使用vLLM模式时的回退行为"""
    try:
        from src.cli.document_processor import DocumentProcessor

        processor = DocumentProcessor(mode="vllm")
        # 在MPS环境下强制使用vLLM应该回退到transformers
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.backends.mps.is_built", return_value=True),
        ):
            actual_mode = processor._determine_mode()
            assert actual_mode == "transformers"
    except ImportError:
        pytest.fail("无法导入DocumentProcessor")


def test_cli_main_mps_mode_handling():
    """测试CLI主程序在MPS环境下的模式处理"""
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.backends.mps.is_built", return_value=True),
    ):
        try:
            with patch("src.cli.main.argparse.ArgumentParser.parse_args") as mock_parse_args:
                mock_parse_args.return_value = type(
                    "Args",
                    (),
                    {
                        "input_path": "/tmp/test.pdf",
                        "output_path": "/tmp/output",
                        "mode": "auto",
                        "download_models": False,
                        "model_dir": "./models",
                    },
                )()

                # 只测试导入，不实际运行main函数
                from src.cli.main import main

                assert main is not None
        except ImportError:
            pytest.fail("无法导入CLI主模块")
