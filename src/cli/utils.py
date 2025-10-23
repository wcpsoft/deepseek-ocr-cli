#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI工具函数模块
包含公共的工具函数和模型检测逻辑
"""

from pathlib import Path
from typing import Optional, Tuple
import sys
import torch

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()


def check_model_availability(project_root: Path) -> tuple[bool, Optional[str]]:
    """
    检查模型是否可用
    
    Args:
        project_root: 项目根目录路径
        
    Returns:
        tuple[bool, Optional[str]]: (模型是否可用, 模型路径)
    """
    try:
        from src.cli.model_manager import ModelManager
        
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
        logger.error(f"检查模型时发生错误: {e}")
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
        logger.error("模型目录不存在或不完整，请先下载模型")
        logger.info("运行命令: deepseek-ocr --download-models")
        return False
    
    return True


def get_compatible_device() -> torch.device:
    """
    获取兼容的计算设备
    
    Returns:
        torch.device: 兼容的计算设备
    """
    # 首先检查CUDA
    if torch.cuda.is_available():
        try:
            # 尝试创建CUDA设备以验证是否真正可用
            device = torch.device("cuda")
            # 尝试在设备上创建一个张量来验证
            test_tensor = torch.zeros(1).to(device)
            # 清理测试张量
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            logger.warning(f"CUDA设备不可用 ({str(e)})，尝试其他设备...")
            # 如果CUDA不可用，继续检查其他设备
            pass
    
    # 检查MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # 尝试创建MPS设备以验证是否真正可用
            device = torch.device("mps")
            # 尝试在设备上创建一个张量来验证
            test_tensor = torch.zeros(1).to(device)
            # 清理测试张量
            del test_tensor
            return device
        except Exception as e:
            logger.warning(f"MPS设备不可用 ({str(e)})，使用CPU...")
            # 如果MPS不可用，继续使用CPU
            pass
    
    # 默认使用CPU
    logger.info("使用CPU设备进行推理")
    return torch.device("cpu")


def get_appropriate_dtype(device: torch.device) -> torch.dtype:
    """
    根据设备类型获取适当的数据类型
    
    Args:
        device: 计算设备
        
    Returns:
        torch.dtype: 适当的数据类型
    """
    if device.type == "mps":
        # MPS上避免使用bfloat16和float16，使用float32以确保兼容性
        logger.info("在MPS设备上运行，使用float32数据类型以确保兼容性")
        return torch.float32
    elif device.type == "cuda":
        # 在CUDA设备上可以使用bfloat16（如果支持）
        if torch.cuda.is_bf16_supported():
            logger.info("在CUDA设备上运行，使用bfloat16数据类型")
            return torch.bfloat16
        else:
            logger.info("在CUDA设备上运行，使用float32数据类型")
            return torch.float32
    else:
        # 在CPU上使用float32
        logger.info("在CPU设备上运行，使用float32数据类型")
        return torch.float32


def should_use_bfloat16(device: torch.device) -> bool:
    """
    判断是否应该使用bfloat16数据类型
    
    Args:
        device: 计算设备
        
    Returns:
        bool: 是否应该使用bfloat16
    """
    if device.type == "mps":
        # MPS不支持bfloat16
        return False
    elif device.type == "cuda":
        # 在CUDA设备上可以使用bfloat16（如果支持）
        return torch.cuda.is_bf16_supported()
    else:
        # 在CPU上不使用bfloat16
        return False