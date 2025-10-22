#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI工具函数模块
包含公共的工具函数和模型检测逻辑
"""

from pathlib import Path
from typing import Optional
import sys


def check_model_availability(project_root: Path) -> tuple[bool, Optional[str]]:
    """
    检查模型是否可用
    
    Args:
        project_root: 项目根目录路径
        
    Returns:
        tuple[bool, Optional[str]]: (模型是否可用, 模型路径)
    """
    try:
        from cli.model_manager import ModelManager
        
        # 创建模型管理器实例
        model_manager = ModelManager(str(project_root / "models"))
        
        # 检查默认模型是否存在且完整
        default_model_name = "deepseek-ocr"
        model_path = model_manager.get_model_path(default_model_name)
        
        if model_path and model_manager.verify_model(default_model_name):
            return True, model_path
        else:
            return False, None
    except Exception as e:
        print(f"检查模型时发生错误: {e}")
        return False, None


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        Path: 项目根目录路径
    """
    # 从当前文件位置向上查找项目根目录
    current_path = Path(__file__).parent
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    
    # 如果找不到pyproject.toml，使用当前工作目录
    return Path.cwd()


def ensure_model_available(project_root: Path) -> bool:
    """
    确保模型可用，如果不可用则提示用户下载
    
    Args:
        project_root: 项目根目录路径
        
    Returns:
        bool: 模型是否可用
    """
    model_available, _ = check_model_availability(project_root)
    
    if not model_available:
        print("错误: 模型目录不存在或不完整，请先下载模型")
        print("运行命令: deepseek-ocr --download-models")
        return False
    
    return True