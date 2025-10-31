#!/usr/bin/env python3
"""
统一的OCR引擎接口
定义所有OCR引擎应该实现的标准接口
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

from src.core.logging import get_logger

logger = get_logger()


class OCREngineInterface(ABC):
    """
    OCR引擎接口
    定义所有OCR引擎应该实现的标准接口
    """

    def __init__(self, model_path: str, device: str | None = None):
        """
        初始化OCR引擎

        Args:
            model_path: 模型路径
            device: 设备类型
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化模型和处理器

        Returns:
            是否初始化成功
        """

    @abstractmethod
    def process_image(self, image: Image.Image | torch.Tensor, prompt: str) -> str:
        """
        处理单个图像

        Args:
            image: 图像对象
            prompt: 提示词

        Returns:
            OCR结果
        """

    @abstractmethod
    def process_batch(self, images: list[Image.Image | torch.Tensor], prompts: list[str]) -> list[str]:
        """
        批量处理图像

        Args:
            images: 图像列表
            prompts: 提示词列表

        Returns:
            OCR结果列表
        """

    @abstractmethod
    def cleanup(self) -> None:
        """
        清理资源
        """

    def is_available(self) -> bool:
        """
        检查引擎是否可用

        Returns:
            引擎是否可用
        """
        return self.is_initialized and self.model is not None

    def get_model_info(self) -> dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "engine_type": self.__class__.__name__,
        }


class BaseOCREngine(OCREngineInterface):
    """
    基础OCR引擎实现
    提供通用的功能和默认实现
    """

    def __init__(self, model_path: str, device: str | None = None):
        """
        初始化基础OCR引擎

        Args:
            model_path: 模型路径
            device: 设备类型
        """
        super().__init__(model_path, device)
        self._setup_device()

    def _setup_device(self) -> None:
        """
        设置设备
        """
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"使用设备: {self.device}")

    def process_image(self, image: Image.Image | torch.Tensor, prompt: str) -> str:
        """
        处理单个图像

        Args:
            image: 图像对象
            prompt: 提示词

        Returns:
            OCR结果
        """
        if not self.is_available():
            raise RuntimeError("OCR引擎未初始化或不可用")

        try:
            return self._process_single_image(image, prompt)
        except Exception as e:
            logger.error(f"处理图像时发生错误: {e!s}")
            raise

    def process_batch(self, images: list[Image.Image | torch.Tensor], prompts: list[str]) -> list[str]:
        """
        批量处理图像

        Args:
            images: 图像列表
            prompts: 提示词列表

        Returns:
            OCR结果列表
        """
        if not self.is_available():
            raise RuntimeError("OCR引擎未初始化或不可用")

        if len(images) != len(prompts):
            raise ValueError("图像数量和提示词数量不匹配")

        try:
            return self._process_batch_images(images, prompts)
        except Exception as e:
            logger.error(f"批量处理图像时发生错误: {e!s}")
            raise

    @abstractmethod
    def _process_single_image(self, image: Image.Image | torch.Tensor, prompt: str) -> str:
        """
        处理单个图像的具体实现

        Args:
            image: 图像对象
            prompt: 提示词

        Returns:
            OCR结果
        """

    @abstractmethod
    def _process_batch_images(self, images: list[Image.Image | torch.Tensor], prompts: list[str]) -> list[str]:
        """
        批量处理图像的具体实现

        Args:
            images: 图像列表
            prompts: 提示词列表

        Returns:
            OCR结果列表
        """

    def cleanup(self) -> None:
        """
        清理资源
        """
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
            self.processor = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_initialized = False

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 清理MPS内存
        elif hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("OCR引擎资源已清理")
