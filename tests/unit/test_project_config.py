#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置测试
验证pyproject.toml配置和不同依赖组的正确性
"""

import pytest
import toml
from pathlib import Path

def test_pyproject_toml_exists():
    """测试pyproject.toml文件存在"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml文件不存在"

def test_pyproject_toml_valid():
    """测试pyproject.toml文件格式有效"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 解析TOML文件
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # 验证基本结构
    assert "project" in config, "缺少[project]节"
    assert "name" in config["project"], "缺少项目名称"
    assert config["project"]["name"] == "deepseek-ocr-cli", "项目名称不正确"
    
    assert "dependencies" in config["project"], "缺少dependencies配置"
    assert "optional-dependencies" in config["project"], "缺少optional-dependencies配置"

def test_core_dependencies():
    """测试核心依赖配置"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 解析TOML文件
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    dependencies = config["project"]["dependencies"]
    
    # 验证必需的核心依赖
    required_deps = [
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "PyMuPDF",
        "img2pdf",
        "torch>=2.4.1",
        "huggingface-hub>=0.20.0",
        "libreoffice>=7.0.0"
    ]
    
    for dep in required_deps:
        assert dep in dependencies, f"缺少核心依赖: {dep}"

def test_optional_dependencies():
    """测试可选依赖组配置"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 解析TOML文件
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    optional_deps = config["project"]["optional-dependencies"]
    
    # 验证必需的可选依赖组
    required_groups = ["vllm", "cli", "dev", "modelscope", "nvidia", "amd", "mps", "dcu"]
    
    for group in required_groups:
        assert group in optional_deps, f"缺少可选依赖组: {group}"
        
        # 验证每个组都有依赖项
        assert len(optional_deps[group]) > 0, f"可选依赖组 {group} 为空"

def test_gpu_specific_dependencies():
    """测试GPU特定依赖配置"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 解析TOML文件
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    optional_deps = config["project"]["optional-dependencies"]
    
    # 验证NVIDIA依赖
    nvidia_deps = optional_deps["nvidia"]
    nvidia_required = [
        "torch==2.4.1+cu118",
        "torchvision==0.19.1+cu118",
        "torchaudio==2.4.1+cu118"
    ]
    for dep in nvidia_required:
        assert dep in nvidia_deps, f"NVIDIA依赖缺失: {dep}"
    
    # 验证AMD依赖
    amd_deps = optional_deps["amd"]
    amd_required = [
        "torch==2.4.1+rocm6.1",
        "torchvision==0.19.1+rocm6.1",
        "torchaudio==2.4.1+rocm6.1"
    ]
    for dep in amd_required:
        assert dep in amd_deps, f"AMD依赖缺失: {dep}"
    
    # 验证MPS和DCU依赖
    mps_deps = optional_deps["mps"]
    dcu_deps = optional_deps["dcu"]
    
    # MPS和DCU使用相同的CPU版本依赖
    cpu_deps = [
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1"
    ]
    
    for dep in cpu_deps:
        assert dep in mps_deps, f"MPS依赖缺失: {dep}"
        assert dep in dcu_deps, f"DCU依赖缺失: {dep}"

def test_dev_dependencies():
    """测试开发依赖配置"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 解析TOML文件
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    dev_deps = config["project"]["optional-dependencies"]["dev"]
    
    # 验证必需的开发依赖
    required_dev_deps = [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0"
    ]
    
    for dep in required_dev_deps:
        assert dep in dev_deps, f"开发依赖缺失: {dep}"

def test_entry_points():
    """测试入口点配置"""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    # 解析TOML文件
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # 验证入口点配置
    assert "scripts" in config["project"], "缺少[project.scripts]节"
    
    scripts = config["project"]["scripts"]
    required_scripts = [
        "deepseek-ocr",
        "deepseek-ocr-download",
        "deepseek-ocr-detect-gpu"
    ]
    
    for script in required_scripts:
        assert script in scripts, f"缺少入口点: {script}"