#!/usr/bin/env python3
"""
生成配置工具类
提供统一的生成配置获取方法，避免代码重复
"""

import logging
from typing import Any

from ..deepseek_ocr_config import DEFAULT_GENERATION_CONFIG

logger = logging.getLogger(__name__)


class GenerationConfigManager:
    """生成配置管理器，用于统一管理模型的生成配置"""

    @staticmethod
    def get_generation_config(model, default_max_new_tokens: int = None) -> dict[str, Any]:
        """
        获取模型的生成配置参数

        Args:
            model: 模型实例
            default_max_new_tokens: 默认最大新token数量，如果为None则使用配置文件中的值

        Returns:
            生成配置参数字典
        """
        # 使用配置文件中的默认值
        if default_max_new_tokens is None:
            default_max_new_tokens = DEFAULT_GENERATION_CONFIG.get("max_new_tokens", 8192)
        
        # 默认生成配置
        default_config = {
            "max_new_tokens": default_max_new_tokens,
            "do_sample": DEFAULT_GENERATION_CONFIG.get("do_sample", False),
            "temperature": DEFAULT_GENERATION_CONFIG.get("temperature", 0.0),
            "top_p": DEFAULT_GENERATION_CONFIG.get("top_p", 0.7),
            "top_k": DEFAULT_GENERATION_CONFIG.get("top_k", 50),
            "frequency_penalty": DEFAULT_GENERATION_CONFIG.get("frequency_penalty", 0.0),
        }

        # 如果模型有生成配置，使用模型的配置
        if model is not None and hasattr(model, "generation_config") and model.generation_config is not None:

            model_config = model.generation_config

            # 更新配置参数
            if hasattr(model_config, "max_new_tokens"):
                default_config["max_new_tokens"] = model_config.max_new_tokens
                logger.debug(f"使用模型的max_new_tokens配置: {model_config.max_new_tokens}")

            if hasattr(model_config, "do_sample"):
                default_config["do_sample"] = model_config.do_sample
                logger.debug(f"使用模型的do_sample配置: {model_config.do_sample}")

            if hasattr(model_config, "temperature"):
                default_config["temperature"] = model_config.temperature
                logger.debug(f"使用模型的temperature配置: {model_config.temperature}")

            if hasattr(model_config, "top_p"):
                default_config["top_p"] = model_config.top_p
                logger.debug(f"使用模型的top_p配置: {model_config.top_p}")

        return default_config

    @staticmethod
    def get_generation_config_from_config(config, default_max_new_tokens: int = None) -> dict[str, Any]:
        """
        从配置对象获取生成配置参数

        Args:
            config: 配置对象
            default_max_new_tokens: 默认最大新token数量，如果为None则使用配置文件中的值

        Returns:
            生成配置参数字典
        """
        # 使用配置文件中的默认值
        if default_max_new_tokens is None:
            default_max_new_tokens = DEFAULT_GENERATION_CONFIG.get("max_new_tokens", 8192)
            
        # 默认生成配置
        default_config = {
            "max_new_tokens": default_max_new_tokens,
            "do_sample": DEFAULT_GENERATION_CONFIG.get("do_sample", False),
            "temperature": DEFAULT_GENERATION_CONFIG.get("temperature", 0.0),
            "top_p": DEFAULT_GENERATION_CONFIG.get("top_p", 0.7),
            "top_k": DEFAULT_GENERATION_CONFIG.get("top_k", 50),
            "frequency_penalty": DEFAULT_GENERATION_CONFIG.get("frequency_penalty", 0.0),
        }

        # 如果配置不为空，尝试从配置中获取参数
        if config is not None:
            # 更新配置参数
            if hasattr(config, "max_new_tokens"):
                default_config["max_new_tokens"] = getattr(config, "max_new_tokens", default_max_new_tokens)
                logger.debug(f"使用配置的max_new_tokens: {default_config['max_new_tokens']}")

            if hasattr(config, "do_sample"):
                default_config["do_sample"] = getattr(config, "do_sample", False)
                logger.debug(f"使用配置的do_sample: {default_config['do_sample']}")

            if hasattr(config, "temperature"):
                default_config["temperature"] = getattr(config, "temperature", 1.0)
                logger.debug(f"使用配置的temperature: {default_config['temperature']}")

            if hasattr(config, "top_p"):
                default_config["top_p"] = getattr(config, "top_p", 1.0)
                logger.debug(f"使用配置的top_p: {default_config['top_p']}")

        return default_config
