#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖安装测试
验证通过uv安装不同依赖组的功能
"""

import pytest
import subprocess
import sys
import tempfile
from pathlib import Path

def run_uv_command(cmd, cwd=None):
    """运行uv命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令执行超时"
    except Exception as e:
        return False, "", str(e)

@pytest.mark.skip(reason="需要实际安装依赖，耗时较长")
def test_base_dependencies_installation():
    """测试基础依赖安装"""
    # 创建临时目录进行测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 复制pyproject.toml到临时目录
        project_root = Path(__file__).parent.parent.parent
        pyproject_src = project_root / "pyproject.toml"
        pyproject_dst = temp_path / "pyproject.toml"
        pyproject_dst.write_text(pyproject_src.read_text())
        
        # 创建虚拟环境
        success, stdout, stderr = run_uv_command("uv venv", cwd=temp_dir)
        assert success, f"创建虚拟环境失败: {stderr}"
        
        # 激活虚拟环境并安装基础依赖
        activate_script = temp_path / ".venv" / "bin" / "activate"
        install_cmd = f"source {activate_script} && uv pip install -e ."
        success, stdout, stderr = run_uv_command(install_cmd, cwd=temp_dir)
        assert success, f"基础依赖安装失败: {stderr}"

@pytest.mark.skip(reason="需要实际安装依赖，耗时较长")
def test_gpu_dependencies_installation():
    """测试GPU依赖组安装"""
    # 创建临时目录进行测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 复制pyproject.toml到临时目录
        project_root = Path(__file__).parent.parent.parent
        pyproject_src = project_root / "pyproject.toml"
        pyproject_dst = temp_path / "pyproject.toml"
        pyproject_dst.write_text(pyproject_src.read_text())
        
        # 创建虚拟环境
        success, stdout, stderr = run_uv_command("uv venv", cwd=temp_dir)
        assert success, f"创建虚拟环境失败: {stderr}"
        
        # 测试NVIDIA依赖组安装
        activate_script = temp_path / ".venv" / "bin" / "activate"
        install_cmd = f"source {activate_script} && uv pip install -e .[nvidia]"
        success, stdout, stderr = run_uv_command(install_cmd, cwd=temp_dir)
        assert success, f"NVIDIA依赖组安装失败: {stderr}"
        
        # 测试AMD依赖组安装
        install_cmd = f"source {activate_script} && uv pip install -e .[amd]"
        success, stdout, stderr = run_uv_command(install_cmd, cwd=temp_dir)
        assert success, f"AMD依赖组安装失败: {stderr}"

@pytest.mark.skip(reason="需要实际安装依赖，耗时较长")
def test_dev_dependencies_installation():
    """测试开发依赖组安装"""
    # 创建临时目录进行测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 复制pyproject.toml到临时目录
        project_root = Path(__file__).parent.parent.parent
        pyproject_src = project_root / "pyproject.toml"
        pyproject_dst = temp_path / "pyproject.toml"
        pyproject_dst.write_text(pyproject_src.read_text())
        
        # 创建虚拟环境
        success, stdout, stderr = run_uv_command("uv venv", cwd=temp_dir)
        assert success, f"创建虚拟环境失败: {stderr}"
        
        # 激活虚拟环境并安装开发依赖
        activate_script = temp_path / ".venv" / "bin" / "activate"
        install_cmd = f"source {activate_script} && uv pip install -e .[dev]"
        success, stdout, stderr = run_uv_command(install_cmd, cwd=temp_dir)
        assert success, f"开发依赖组安装失败: {stderr}"

def test_dependency_groups_exist():
    """测试依赖组在pyproject.toml中存在"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 读取pyproject.toml内容
    content = pyproject_path.read_text()
    
    # 验证必需的依赖组存在（注意格式是"group = ["而不是"[group]"）
    required_groups = [
        "nvidia = [",
        "amd = [", 
        "mps = [",
        "dcu = [",
        "dev = [",
        "vllm = [",
        "modelscope = ["
    ]
    
    for group in required_groups:
        assert group in content, f"依赖组 {group} 未在pyproject.toml中定义"

def test_torch_versions_in_groups():
    """测试不同组中的PyTorch版本配置"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    content = pyproject_path.read_text()
    
    # 验证NVIDIA组中的PyTorch版本
    assert "torch==2.4.1+cu118" in content, "NVIDIA组缺少正确的PyTorch版本"
    assert "torchvision==0.19.1+cu118" in content, "NVIDIA组缺少正确的torchvision版本"
    assert "torchaudio==2.4.1+cu118" in content, "NVIDIA组缺少正确的torchaudio版本"
    
    # 验证AMD组中的PyTorch版本
    assert "torch==2.4.1+rocm6.1" in content, "AMD组缺少正确的PyTorch版本"
    assert "torchvision==0.19.1+rocm6.1" in content, "AMD组缺少正确的torchvision版本"
    assert "torchaudio==2.4.1+rocm6.1" in content, "AMD组缺少正确的torchaudio版本"
    
    # 验证MPS和DCU组中的PyTorch版本
    assert "torch==2.4.1" in content, "MPS/DCU组缺少正确的PyTorch版本"
    assert "torchvision==0.19.1" in content, "MPS/DCU组缺少正确的torchvision版本"
    assert "torchaudio==2.4.1" in content, "MPS/DCU组缺少正确的torchaudio版本"