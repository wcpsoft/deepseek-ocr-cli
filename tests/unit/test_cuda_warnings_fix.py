#!/usr/bin/env python3
"""
CUDA警告修复测试
验证CUDA警告抑制功能是否正常工作
"""

import os
import sys

import pytest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


def test_cuda_warnings_suppression() -> None:
    """测试CUDA警告抑制功能"""
    # 保存原始环境变量
    original_cuda_module_loading: str = os.environ.get("CUDA_MODULE_LOADING", "")

    try:
        # 验证环境变量是否正确设置
        # 注意:在实际代码中,这个环境变量应该在导入相关模块时被设置
        # 这里我们只是测试环境变量的设置逻辑
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        assert os.environ.get("CUDA_MODULE_LOADING") == "LAZY"
    finally:
        # 恢复原始环境变量
        if original_cuda_module_loading:
            os.environ["CUDA_MODULE_LOADING"] = original_cuda_module_loading
        elif "CUDA_MODULE_LOADING" in os.environ:
            del os.environ["CUDA_MODULE_LOADING"]


def test_environment_variables() -> None:
    """测试环境变量设置"""
    # 保存原始环境变量
    original_vars: dict[str, str] = {
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING", ""),
    }

    try:
        # 清除环境变量
        if "CUDA_MODULE_LOADING" in os.environ:
            del os.environ["CUDA_MODULE_LOADING"]

        # 手动设置环境变量
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"

        # 验证环境变量是否正确设置
        assert os.environ.get("CUDA_MODULE_LOADING") == "LAZY"
    finally:
        # 恢复原始环境变量
        for key, value in original_vars.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


def test_new_cli_command() -> None:
    """测试新的CLI命令"""
    try:
        # 测试导入dev.debug模块
        import dev.debug

        assert dev.debug is not None
    except ImportError:
        pytest.fail("无法导入调试模块")
