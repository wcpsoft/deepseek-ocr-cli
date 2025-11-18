#!/usr/bin/env python3

"""
OCR引擎工厂模块
提供OCR引擎的创建和管理功能
"""

import logging
from typing import ClassVar, Optional

from src.core.config.app_config import get_app_config

# 获取日志记录器
logger = logging.getLogger(__name__)


class OCREngineFactory:
    """
    OCR引擎工厂类
    根据配置创建不同类型的OCR引擎
    """

    _engines: ClassVar[dict] = {}
    _engine_classes: ClassVar[dict] = {}

    @classmethod
    def register_engine(cls, name: str, engine_class: type) -> None:
        """
        注册OCR引擎类

        Args:
            name: 引擎名称
            engine_class: 引擎类
        """
        cls._engine_classes[name] = engine_class
        logger.info(f"已注册OCR引擎: {name}")

    @classmethod
    def create_engine(
        cls,
        engine_type: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        prompt: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        *,
        crop_mode: bool = True,
        model_manager: Optional[Any] = None,
        image_handler: Optional[Any] = None,
        device_manager: Optional[Any] = None,
    ):
        """
        创建OCR引擎

        Args:
            engine_type: 引擎类型
            model_path: 模型路径
            device: 设备类型
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否启用裁剪模式
            model_manager: 注入的模型管理器
            image_handler: 注入的图像处理器
            device_manager: 注入的设备管理器

        Returns:
            OCR引擎实例
        """
        config = get_app_config()
        engine_type = engine_type or config.model_path  # 使用model_path作为默认引擎类型

        if engine_type not in cls._engine_classes:
            raise ValueError(f"不支持的OCR引擎类型: {engine_type}")

        try:
            # 创建引擎实例
            engine_class = cls._engine_classes[engine_type]

            # 准备创建引擎的参数
            engine_kwargs = {
                "model_path": model_path,
                "device": device,
                "prompt": prompt,
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode,
            }

            # 只有当引擎支持依赖注入时才添加这些参数
            if hasattr(engine_class, "__init__"):
                import inspect

                sig = inspect.signature(engine_class.__init__)
                for param_name, dep in [
                    ("model_manager", model_manager),
                    ("image_handler", image_handler),
                    ("device_manager", device_manager),
                ]:
                    if dep and param_name in sig.parameters:
                        engine_kwargs[param_name] = dep

            engine = engine_class(**engine_kwargs)

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
        from src.core.transformers.transformers_engine import TransformersEngine

        OCREngineFactory.register_engine("transformers", TransformersEngine)
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
