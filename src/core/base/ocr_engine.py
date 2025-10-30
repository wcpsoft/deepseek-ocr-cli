#!/usr/bin/env python3
"""
OCR引擎抽象基类
定义所有OCR引擎的通用接口
"""

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class BaseOCREngine(ABC):
    """OCR引擎抽象基类"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        prompt: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        device: Optional[str] = None,
        *,
        crop_mode: bool = True,
    ):
        """
        初始化OCR引擎

        Args:
            model_path: 模型路径
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            device: 设备类型
            crop_mode: 是否启用裁剪模式
        """
        self.model_path = model_path
        self.prompt = prompt
        self.base_size = base_size
        self.image_size = image_size
        self.device = device
        self.crop_mode = crop_mode
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化引擎

        Returns:
            是否初始化成功
        """

    @abstractmethod
    def process(self, images: list[Image.Image], output_dir: str) -> None:
        """
        处理图像列表并保存结果到输出目录

        Args:
            images: 图像列表
            output_dir: 输出目录路径
        """

    @abstractmethod
    def cleanup(self) -> None:
        """
        清理资源
        """

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()