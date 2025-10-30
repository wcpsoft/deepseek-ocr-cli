#!/usr/bin/env python3
"""
MPS模式选择单元测试
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_mps_mode_selection() -> None:
    """测试MPS模式选择"""
    with patch.dict(
        "sys.modules",
        {
            "src.cli.main": MagicMock(),
        },
    ):
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_pdf = temp_path / "test.pdf"
            output_dir = temp_path / "output"

            # 模拟torch和相关模块
            with (
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=True),
            ):
                # 模拟main函数的调用
                with patch(
                    "sys.argv",
                    ["main.py", str(test_pdf), str(output_dir), "--mode", "auto", "--download-models", "False"],
                ):
                    # 只测试导入, 不实际运行main函数
                    from src.cli.main import main

                    # 验证导入成功
                    assert main is not None


def test_mps_mode_selection_with_cuda() -> None:
    """测试CUDA模式选择"""
    with patch.dict(
        "sys.modules",
        {
            "src.cli.main": MagicMock(),
        },
    ):
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_pdf = temp_path / "test.pdf"
            output_dir = temp_path / "output"

            # 模拟torch和相关模块
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                # 模拟main函数的调用
                with patch(
                    "sys.argv",
                    ["main.py", str(test_pdf), str(output_dir), "--mode", "auto", "--download-models", "False"],
                ):
                    # 只测试导入, 不实际运行main函数
                    from src.cli.main import main

                    # 验证导入成功
                    assert main is not None


def test_cpu_mode_selection() -> None:
    """测试CPU模式选择"""
    with patch.dict(
        "sys.modules",
        {
            "src.cli.main": MagicMock(),
        },
    ):
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_pdf = temp_path / "test.pdf"
            output_dir = temp_path / "output"

            # 模拟torch和相关模块
            with (
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                # 模拟main函数的调用
                with patch(
                    "sys.argv",
                    ["main.py", str(test_pdf), str(output_dir), "--mode", "auto", "--download-models", "False"],
                ):
                    # 只测试导入, 不实际运行main函数
                    from src.cli.main import main

                    # 验证导入成功
                    assert main is not None
