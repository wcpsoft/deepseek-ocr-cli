#!/usr/bin/env python3
"""
OCR模型工厂
用于创建和管理不同类型的OCR模型
"""

from typing import Any, Optional

from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()


class OCRModelInterface:
    """OCR模型接口类"""

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        """
        初始化OCR模型接口

        Args:
            config: 模型配置
            **kwargs: 其他参数
        """
        self.config = config

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """生成方法"""
        raise NotImplementedError

    def to(self, *args: Any, **kwargs: Any) -> None:
        """设备转换方法"""
        raise NotImplementedError

    def eval(self) -> None:
        """评估模式设置方法"""
        raise NotImplementedError


class OCRModelFactory:
    """OCR模型工厂类"""

    _models: dict[str, type[OCRModelInterface]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[OCRModelInterface] | type) -> None:
        """
        注册模型类

        Args:
            name: 模型名称
            model_class: 模型类
        """
        # 放宽类型检查以避免循环导入问题
        cls._models[name] = model_class  # type: ignore
        logger.debug(f"注册模型类: {name}")

    @classmethod
    def create(cls, name: str, config: Optional[Any] = None, **kwargs: Any) -> OCRModelInterface:
        """
        创建模型实例

        Args:
            name: 模型名称
            config: 模型配置
            **kwargs: 其他参数

        Returns:
            模型实例
        """
        if name not in cls._models:
            raise ValueError(f"Unknown model type: {name}")
        return cls._models[name](config, **kwargs)  # type: ignore

    @classmethod
    def get_available_models(cls) -> list[str]:
        """
        获取可用的模型列表

        Returns:
            可用的模型列表
        """
        return list(cls._models.keys())


# 注册模型适配器
def _register_model_adapters():
    """注册所有模型适配器"""
    # 将导入移到函数内部以避免循环导入
    from src.core.models.model_adapter import TransformersOCRModelAdapter, VLLMOCRModelAdapter

    # 只注册两种核心模型类型
    OCRModelFactory.register("vllm", VLLMOCRModelAdapter)
    OCRModelFactory.register("transformers", TransformersOCRModelAdapter)


# 自动注册模型适配器
_register_model_adapters()


def create_ocr_model(model_type: str, model_path: str, **kwargs: Any) -> OCRModelInterface:
    """
    创建OCR模型实例

    Args:
        model_type: 模型类型
        model_path: 模型路径
        **kwargs: 其他参数

    Returns:
        OCR模型实例
    """
    # 使用工厂创建模型实例
    return OCRModelFactory.create(model_type, model_path=model_path, **kwargs)
