#!/usr/bin/env python3
"""
OCR引擎工厂类
根据配置创建不同类型的OCR引擎
"""

from typing import Any

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger()


class OCREngineFactory:
    """
    OCR引擎工厂类
    根据配置创建不同类型的OCR引擎
    """

    _engines = {}
    _engine_classes = {}

    @classmethod
    def register_engine(cls, name: str, engine_class) -> None:
        """
        注册OCR引擎类

        Args:
            name: 引擎名称
            engine_class: 引擎类
        """
        cls._engine_classes[name] = engine_class
        logger.info(f"已注册OCR引擎: {name}")

    @classmethod
    def create_engine(cls, engine_type: str | None = None, **kwargs) -> Any:
        """
        创建OCR引擎

        Args:
            engine_type: 引擎类型
            **kwargs: 引擎初始化参数

        Returns:
            OCR引擎实例
        """
        config = get_config()
        engine_type = engine_type or config.ENGINE_TYPE

        if engine_type not in cls._engine_classes:
            raise ValueError(f"不支持的OCR引擎类型: {engine_type}")

        try:
            # 创建引擎实例
            engine_class = cls._engine_classes[engine_type]
            engine = engine_class(**kwargs)

            logger.info(f"已创建OCR引擎: {engine_type}")
            return engine

        except Exception as e:
            logger.error(f"创建OCR引擎失败: {engine_type}, 错误: {e!s}")
            raise

    @classmethod
    def get_available_engines(cls) -> dict[str, str]:
        """
        获取可用的OCR引擎列表

        Returns:
            可用引擎字典
        """
        return {name: engine_class.__name__ for name, engine_class in cls._engine_classes.items()}

    @classmethod
    def is_engine_available(cls, engine_type: str) -> bool:
        """
        检查指定引擎是否可用

        Args:
            engine_type: 引擎类型

        Returns:
            是否可用
        """
        return engine_type in cls._engine_classes


# 自动注册所有可用的OCR引擎
def _register_engines():
    """
    自动注册所有可用的OCR引擎
    """
    try:
        # 注册Transformers引擎
        from src.core.inference.deepseek_ocr_inference import DeepSeekOCRInference

        OCREngineFactory.register_engine("transformers", DeepSeekOCRInference)
    except ImportError as e:
        logger.warning(f"无法注册Transformers引擎: {e!s}")

    try:
        # 注册vLLM引擎
        from src.core.vllm.vllm_engine import VLLMEngine

        OCREngineFactory.register_engine("vllm", VLLMEngine)
    except ImportError as e:
        logger.warning(f"无法注册vLLM引擎: {e!s}")


# 执行自动注册
_register_engines()
