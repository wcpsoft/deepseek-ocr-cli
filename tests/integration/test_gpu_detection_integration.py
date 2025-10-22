#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU检测集成测试
验证GPU检测脚本与pyproject.toml依赖组的集成
"""

import pytest
import subprocess
import sys
from pathlib import Path

def test_gpu_detection_script_exists():
    """测试GPU检测脚本存在"""
    project_root = Path(__file__).parent.parent.parent
    detect_script = project_root / "dev" / "detect_gpu.py"
    assert detect_script.exists(), "GPU检测脚本不存在"

def test_gpu_detection_imports():
    """测试GPU检测脚本可以正确导入"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from dev.detect_gpu import get_gpu_type, get_extra_require_suffix
        assert get_gpu_type is not None
        assert get_extra_require_suffix is not None
    finally:
        sys.path.pop(0)

def test_gpu_detection_functions():
    """测试GPU检测函数"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from dev.detect_gpu import get_extra_require_suffix
        
        # 测试不同GPU类型的依赖后缀
        test_cases = [
            ("nvidia", "[nvidia]"),
            ("amd", "[amd]"),
            ("mps", "[mps]"),
            ("dcu", "[dcu]"),
            ("cpu", ""),
            ("unknown", "")
        ]
        
        for gpu_type, expected_suffix in test_cases:
            suffix = get_extra_require_suffix(gpu_type)
            assert suffix == expected_suffix, f"GPU类型 {gpu_type} 的依赖后缀不正确: 期望 {expected_suffix}, 实际 {suffix}"
            
    finally:
        sys.path.pop(0)

def test_pyproject_toml_dependency_groups():
    """测试pyproject.toml中定义的依赖组与GPU检测脚本的一致性"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 读取pyproject.toml内容
    content = pyproject_path.read_text()
    
    # 验证GPU检测脚本中使用的依赖组在pyproject.toml中都有定义
    required_groups = ["nvidia", "amd", "mps", "dcu"]
    
    for group in required_groups:
        # 检查组定义存在
        assert f"{group} = [" in content, f"依赖组 {group} 未在pyproject.toml中定义"
        
        # 验证组名在GPU检测脚本中被使用
        detect_script = project_root / "dev" / "detect_gpu.py"
        script_content = detect_script.read_text()
        assert f"\"{group}\"" in script_content, f"GPU检测脚本中未使用依赖组 {group}"

def test_installation_command_generation():
    """测试安装命令生成"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from dev.detect_gpu import get_pytorch_install_cmd, get_extra_require_suffix
        
        # 测试不同GPU类型的安装命令
        test_cases = [
            ("nvidia", "uv pip install -e .[nvidia]"),
            ("amd", "uv pip install -e .[amd]"),
            ("mps", "uv pip install -e .[mps]"),
            ("dcu", "uv pip install -e .[dcu]"),
            ("cpu", "uv pip install -e .")
        ]
        
        for gpu_type, expected_cmd in test_cases:
            cmd = get_pytorch_install_cmd(gpu_type)
            assert cmd == expected_cmd, f"GPU类型 {gpu_type} 的安装命令不正确: 期望 {expected_cmd}, 实际 {cmd}"
            
            # 验证依赖后缀
            if gpu_type == "cpu":
                expected_suffix = ""
            else:
                expected_suffix = f"[{gpu_type}]"
            suffix = get_extra_require_suffix(gpu_type)
            assert suffix == expected_suffix, f"GPU类型 {gpu_type} 的依赖后缀不正确: 期望 {expected_suffix}, 实际 {suffix}"
            
    finally:
        sys.path.pop(0)

def test_run_sh_integration():
    """测试run.sh脚本中的GPU检测集成"""
    project_root = Path(__file__).parent.parent.parent
    run_sh_path = project_root / "dev" / "run.sh"
    
    # 验证run.sh中正确使用了GPU检测
    content = run_sh_path.read_text()
    
    # 检查是否调用GPU检测脚本
    assert "detect_gpu.py" in content, "run.sh中未调用GPU检测脚本"
    
    # 检查是否使用了EXTRA_SUFFIX
    assert "EXTRA_SUFFIX" in content, "run.sh中未使用EXTRA_SUFFIX变量"
    
    # 检查是否正确使用依赖安装命令
    assert "-e ." in content, "run.sh中未正确使用依赖安装命令"

def test_test_sh_integration():
    """测试run_tests.sh脚本中的GPU检测集成"""
    project_root = Path(__file__).parent.parent.parent
    test_sh_path = project_root / "dev" / "run_tests.sh"
    
    # 验证run_tests.sh中正确使用了GPU检测
    content = test_sh_path.read_text()
    
    # 检查是否调用GPU检测脚本
    assert "detect_gpu.py" in content, "run_tests.sh中未调用GPU检测脚本"
    
    # 检查是否使用了EXTRA_SUFFIX
    assert "EXTRA_SUFFIX" in content, "run_tests.sh中未使用EXTRA_SUFFIX变量"
    
    # 检查是否正确使用依赖安装命令
    assert "-e .[dev" in content, "run_tests.sh中未正确使用开发依赖安装命令"