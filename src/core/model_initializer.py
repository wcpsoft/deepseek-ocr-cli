#!/usr/bin/env python3
"""
模型初始化器
统一管理模型和分词器的初始化
"""

from typing import Any

import torch
import torch.nn as nn

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()


class ModelInitializer:
    """统一的模型初始化器 - 已重构为适配器模式"""

    @staticmethod
    def initialize_transformers_model_and_tokenizer(
        model_path: str, *, trust_remote_code: bool = True
    ) -> tuple[object, object]:
        """
        初始化Transformers模型和分词器 - 委托给ModelManager处理

        Args:
            model_path: 模型路径
            trust_remote_code: 是否信任远程代码

        Returns:
            模型和分词器元组
        """
        try:
            # 使用ModelManager处理模型加载，避免重复实现
            from src.core.models.model_manager import ModelManager

            logger.info("使用ModelManager初始化Transformers模型和分词器")
            model_manager = ModelManager(model_path)
            model, tokenizer = model_manager.load_model_and_tokenizer(trust_remote_code=trust_remote_code)

            # 移动模型到适当的设备
            model_manager.setup_device()
            model_manager.move_model_to_device()

            logger.info("Transformers模型和分词器初始化完成")
            return model, tokenizer
        except Exception as e:
            logger.error(f"初始化Transformers模型和分词器失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化Transformers模型和分词器失败: {e!s}") from e

    @staticmethod
    def initialize_vllm_model(model_path: str, prompt: str | None = None, *, trust_remote_code: bool = True) -> object:
        """
        初始化vLLM模型 - 委托给ModelManager处理

        Args:
            model_path: 模型路径
            prompt: 提示词
            trust_remote_code: 是否信任远程代码

        Returns:
            vLLM模型实例
        """
        try:
            # 使用ModelManager处理模型加载，避免重复实现
            from src.core.models.model_manager import ModelManager

            logger.info("使用ModelManager初始化vLLM模型")
            model_manager = ModelManager(model_path)
            # 注意：这里需要特殊处理vLLM模型
            # 由于vLLM的特殊性，我们直接调用模型工厂
            from src.core.models.model_factory import create_ocr_model

            # 检查是否是本地路径
            is_remote_repo = model_manager._is_remote_repo(model_path)
            local_files_only = not is_remote_repo

            model = create_ocr_model(
                model_type="vllm",
                model_path=model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )

            logger.info("vLLM模型初始化完成")
            return model
        except Exception as e:
            logger.error(f"初始化vLLM模型失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化vLLM模型失败: {e!s}") from e

    @staticmethod
    def move_model_to_device(model: nn.Module | Any, device: torch.device | None = None) -> nn.Module | Any:
        """
        将模型移到指定设备 - 委托给专门的工具处理

        Args:
            model: 模型实例
            device: 目标设备，如果为None则自动选择最优设备

        Returns:
            移动后的模型实例
        """
        try:
            # 使用专门的设备管理工具
            from src.core.utils.device_manager import get_optimal_device

            # 如果没有指定设备，自动选择最优设备
            if device is None:
                device = get_optimal_device()

            logger.info(f"将模型移到 {device.type.upper()} 设备")
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model = model.eval()

            logger.debug(f"模型已移到设备: {device}")
            return model
        except Exception as e:
            logger.error(f"将模型移到设备失败: {e!s}")
            raise RuntimeError(f"将模型移到设备失败: {e!s}") from e
