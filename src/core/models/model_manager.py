#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
专门负责模型的加载、初始化和设备管理
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from src.core.logging import get_logger
from src.core.utils.device_manager import get_optimal_device

logger = get_logger()


class ModelManager:
    """专门负责模型管理的类"""

    def __init__(self, model_path: str):
        """
        初始化模型管理器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None

    def load_model_and_tokenizer(self, trust_remote_code: bool = True) -> Tuple[Any, Any]:
        """
        加载模型和分词器

        Args:
            trust_remote_code: 是否信任远程代码

        Returns:
            (模型, 分词器) 元组
        """
        try:
            # 检查是否是本地路径
            is_remote_repo = self._is_remote_repo(self.model_path)
            local_files_only = not is_remote_repo
            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用trust_remote_code=True
            trust_remote_code_for_tokenizer = is_remote_repo and trust_remote_code

            # 加载tokenizer
            logger.info(f"加载tokenizer: {self.model_path}")
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=trust_remote_code_for_tokenizer, local_files_only=local_files_only
            )

            # 加载模型
            logger.info(f"加载模型: {self.model_path}")
            self.model = self._load_model(trust_remote_code, local_files_only, is_remote_repo)

            logger.info("模型和分词器加载完成")
            return self.model, self.tokenizer
        except Exception as e:
            logger.error(f"加载模型和分词器失败: {str(e)}")
            raise

    def _is_remote_repo(self, model_path: str) -> bool:
        """检查是否是远程仓库路径"""
        return (
            model_path.startswith(("http://", "https://"))
            or model_path.startswith("deepseek-ai/")
            or model_path.startswith("huggingface.co/")
            or "/" not in model_path  # 单个名称可能是远程仓库名
            or (not os.path.exists(model_path) and not os.path.exists(os.path.expanduser(model_path)))
        )

    def _load_model(self, trust_remote_code: bool, local_files_only: bool, is_remote_repo: bool) -> Any:
        """加载模型的具体实现"""
        try:
            # 使用模型工厂创建模型实例
            from src.core.models.model_factory import create_ocr_model

            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用trust_remote_code=True
            trust_remote_code_for_model = is_remote_repo and trust_remote_code

            model = create_ocr_model(
                model_type="transformers",
                model_path=self.model_path,
                trust_remote_code=trust_remote_code_for_model,
                torch_dtype=torch.float16 if self.device and self.device.type == "cuda" else torch.float32,
                local_files_only=local_files_only,
            )
            logger.info("成功加载OCR模型")
            return model
        except Exception as e:
            logger.error(f"使用模型工厂加载失败: {str(e)}")
            # 如果模型工厂不可用，不尝试使用AutoModelForCausalLM，直接抛出异常
            raise RuntimeError(f"无法加载模型: {str(e)}")

    def load_processor(self) -> Any:
        """加载图像处理器"""
        try:
            logger.info(f"加载图像处理器: {self.model_path}")
            from src.core.process.image_process import DeepseekOCRProcessor

            self.processor = DeepseekOCRProcessor.from_pretrained(self.model_path)
            logger.info("成功加载DeepseekOCRProcessor")
            return self.processor
        except Exception as e:
            logger.error(f"无法加载图像处理器: {str(e)}")
            self.processor = None
            return None

    def setup_device(self) -> torch.device:
        """设置并返回设备"""
        self.device = get_optimal_device()
        logger.info(f"使用设备: {self.device.type.upper()}")
        return self.device

    def move_model_to_device(self) -> None:
        """将模型移到设备上"""
        if self.model and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
            logger.info(f"模型已移动到设备: {self.device}")

        if self.model and hasattr(self.model, "eval"):
            self.model = self.model.eval()
            logger.info("模型已设置为评估模式")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "device": str(self.device) if self.device else None,
            "has_model": self.model is not None,
            "has_tokenizer": self.tokenizer is not None,
            "has_processor": self.processor is not None,
        }
