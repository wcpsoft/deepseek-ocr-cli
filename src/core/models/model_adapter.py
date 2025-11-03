#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型适配器类
用于适配不同的模型实现
"""

import os
import sys
from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.nn as nn

# 导入应用程序配置
from src.core.config.app_config import get_default_model, get_model_auto_map
from src.core.logging import get_logger

# 延迟导入以避免循环导入
# from src.core.models.model_factory import OCRModelInterface


# 获取日志记录器
logger = get_logger()


# 定义本地接口类以避免循环导入
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


class VLLMOCRModelAdapter(OCRModelInterface):
    """vLLM OCR模型适配器"""

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        """
        初始化vLLM OCR模型适配器

        Args:
            config: 模型配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._model: Optional[Any] = None
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs: Any) -> None:
        """初始化模型"""
        try:
            # vLLM模型可能需要特殊处理，这里简化处理
            logger.warning("vLLM模型适配器初始化被跳过（MPS环境下不可用）")
            self._model = None
        except Exception as e:
            logger.error(f"初始化vLLM OCR模型失败: {e}")
            raise

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """生成方法"""
        if self._model is None:
            raise RuntimeError("模型未初始化")
        return self._model.generate(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> None:
        """设备转换方法"""
        if self._model is not None and hasattr(self._model, "to"):
            self._model = self._model.to(*args, **kwargs)

    def eval(self) -> None:
        """评估模式设置方法"""
        if self._model is not None and hasattr(self._model, "eval"):
            self._model = self._model.eval()

    def __getattr__(self, name: str) -> Any:
        """代理其他方法到底层模型"""
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TransformersOCRModelAdapter(OCRModelInterface):
    """Transformers OCR模型适配器"""

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        """
        初始化Transformers OCR模型适配器

        Args:
            config: 模型配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._model: Optional[Any] = None
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs: Any) -> None:
        """初始化模型"""
        try:
            model_path = kwargs.get("model_path", None)

            # 如果没有提供模型路径，使用默认路径
            if not model_path:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                model_path = os.path.join(project_root, "models", "deepseek-ocr")

            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = False
            if model_path:
                is_remote_repo = (
                    model_path.startswith(("http://", "https://"))
                    or model_path.startswith("deepseek-ai/")
                    or model_path.startswith("huggingface.co/")
                    or "/" not in model_path  # 单个名称可能是远程仓库名
                    or (not os.path.exists(model_path) and not os.path.exists(os.path.expanduser(model_path)))
                )

            # 对于本地模型，也需要trust_remote_code=True，因为模型配置文件包含自定义代码
            # 但确保local_files_only=True，这样就不会尝试从远程下载
            trust_remote_code = kwargs.get("trust_remote_code", True)

            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo

            # 如果是本地路径，直接从src目录加载模型
            if not is_remote_repo:
                # 确保src目录在系统路径中
                src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)

                # 获取app.yaml中的auto_map配置
                default_model = get_default_model()
                app_auto_map = get_model_auto_map(default_model)

                # 使用默认的模型类和配置类
                from src.core.models.modeling_deepseekocr import DeepseekOCRConfig, DeepseekOCRForCausalLM

                # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
                config = DeepseekOCRConfig.from_pretrained(model_path)
                self._model = DeepseekOCRForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=kwargs.get("torch_dtype", torch.float32),
                    local_files_only=local_files_only,
                )
            else:
                # 远程仓库直接加载模型
                from transformers import AutoConfig, AutoModelForCausalLM

                config = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only
                )

                # 使用修改后的配置加载模型
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=kwargs.get("torch_dtype", torch.float32),
                    local_files_only=local_files_only,
                )

            logger.info("成功初始化Transformers OCR模型")
        except Exception as e:
            logger.error(f"初始化Transformers OCR模型失败: {e}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """生成方法"""
        if self._model is None:
            raise RuntimeError("模型未初始化")
        return self._model.generate(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> None:
        """设备转换方法"""
        if self._model is not None and hasattr(self._model, "to"):
            self._model = self._model.to(*args, **kwargs)

    def eval(self) -> None:
        """评估模式设置方法"""
        if self._model is not None and hasattr(self._model, "eval"):
            self._model = self._model.eval()

    def __getattr__(self, name: str) -> Any:
        """代理其他方法到底层模型"""
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class UnifiedOCRModelAdapter(OCRModelInterface):
    """统一OCR模型适配器，根据环境自动选择最佳实现"""

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        """
        初始化统一OCR模型适配器

        Args:
            config: 模型配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._model: Optional[Any] = None
        self._model_type: Optional[str] = None
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs: Any) -> None:
        """初始化模型"""
        # 根据环境自动选择模型类型
        import torch

        if torch.backends.mps.is_available():
            # MPS环境下使用Transformers模型
            self._model_type = "transformers"
            # 直接创建适配器实例而不是导入类
            self._model = TransformersOCRModelAdapter(config=self.config, **kwargs)
        elif torch.cuda.is_available():
            # CUDA环境下使用vLLM模型
            self._model_type = "vllm"
            # 直接创建适配器实例而不是导入类
            self._model = VLLMOCRModelAdapter(config=self.config, **kwargs)
        else:
            # CPU环境下使用Transformers模型
            self._model_type = "transformers"
            # 直接创建适配器实例而不是导入类
            self._model = TransformersOCRModelAdapter(config=self.config, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """生成方法"""
        if self._model is None:
            raise RuntimeError("模型未初始化")
        return self._model.generate(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> None:
        """设备转换方法"""
        if self._model is not None:
            self._model.to(*args, **kwargs)

    def eval(self) -> None:
        """评估模式设置方法"""
        if self._model is not None:
            self._model.eval()

    def __getattr__(self, name: str) -> Any:
        """代理其他方法到底层模型"""
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# 保持向后兼容的ModelAdapter类
class ModelAdapter:
    """
    模型适配器类
    用于统一不同模型实现的接口
    """

    def __init__(self, model):
        """
        初始化模型适配器

        Args:
            model: 原始模型实例
        """
        self._model = model
        # 确保这些属性存在
        if not hasattr(self._model, "sam_model"):
            self._model.sam_model = None
        if not hasattr(self._model, "vision_model"):
            self._model.vision_model = None
        if not hasattr(self._model, "projector"):
            self._model.projector = None
        if not hasattr(self._model, "image_newline"):
            self._model.image_newline = None
        if not hasattr(self._model, "view_seperator"):
            self._model.view_seperator = None
        if not hasattr(self._model, "image_token_id"):
            self._model.image_token_id = None

    def __getattr__(self, name):
        """
        代理访问原始模型的属性和方法

        Args:
            name: 属性或方法名

        Returns:
            原始模型的属性或方法
        """
        # 如果适配器本身有这个属性，返回适配器的属性
        if name in self.__dict__:
            return self.__dict__[name]
        # 否则返回原始模型的属性
        return getattr(self._model, name)

    @property
    def sam_model(self) -> Optional[nn.Module]:
        """获取SAM模型"""
        sam_model = getattr(self._model, "sam_model", None)
        # 如果sam_model是None但模型有sam_model属性，返回模型的sam_model属性
        if sam_model is None and hasattr(self._model, "sam_model"):
            return self._model.sam_model
        return sam_model

    @property
    def vision_model(self) -> Optional[nn.Module]:
        """获取视觉模型"""
        vision_model = getattr(self._model, "vision_model", None)
        # 如果vision_model是None但模型有vision_model属性，返回模型的vision_model属性
        if vision_model is None and hasattr(self._model, "vision_model"):
            return self._model.vision_model
        return vision_model

    @property
    def projector(self) -> Optional[nn.Module]:
        """获取投影器"""
        projector = getattr(self._model, "projector", None)
        # 如果projector是None但模型有projector属性，返回模型的projector属性
        if projector is None and hasattr(self._model, "projector"):
            return self._model.projector
        return projector

    @property
    def image_newline(self) -> Optional[torch.Tensor]:
        """获取图像换行符"""
        image_newline = getattr(self._model, "image_newline", None)
        # 如果image_newline是None但模型有image_newline属性，返回模型的image_newline属性
        if image_newline is None and hasattr(self._model, "image_newline"):
            return self._model.image_newline
        return image_newline

    @property
    def view_seperator(self) -> Optional[torch.Tensor]:
        """获取视图分隔符"""
        view_seperator = getattr(self._model, "view_seperator", None)
        # 如果view_seperator是None但模型有view_seperator属性，返回模型的view_seperator属性
        if view_seperator is None and hasattr(self._model, "view_seperator"):
            return self._model.view_seperator
        return view_seperator

    @property
    def image_token_id(self) -> Optional[int]:
        """获取图像token ID"""
        image_token_id = getattr(self._model, "image_token_id", None)
        # 如果image_token_id是None但模型有image_token_id属性，返回模型的image_token_id属性
        if image_token_id is None and hasattr(self._model, "image_token_id"):
            return self._model.image_token_id
        return image_token_id

    def generate(self, *args, **kwargs):
        """
        生成方法

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            生成结果
        """
        if hasattr(self._model, "generate"):
            return self._model.generate(*args, **kwargs)
        else:
            raise AttributeError("模型没有generate方法")

    def to(self, *args, **kwargs):
        """
        将模型移到指定设备

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            self
        """
        if hasattr(self._model, "to"):
            self._model = self._model.to(*args, **kwargs)
        return self

    def eval(self):
        """
        设置模型为评估模式

        Returns:
            self
        """
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()
        return self

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        获取模型参数

        Args:
            recurse: 是否递归获取子模块参数

        Returns:
            模型参数迭代器
        """
        if hasattr(self._model, "parameters"):
            return self._model.parameters(recurse)
        return iter([])

    def get_input_embeddings(self):
        """
        获取输入嵌入层

        Returns:
            输入嵌入层
        """
        if hasattr(self._model, "get_input_embeddings"):
            return self._model.get_input_embeddings()
        elif hasattr(self._model, "language_model") and self._model.language_model is not None:
            return self._model.language_model.get_input_embeddings()
        else:
            raise AttributeError("模型没有get_input_embeddings方法")

    def merge_multimodal_embeddings(self, input_ids, inputs_embeds, multimodal_embeddings, image_token_id):
        """
        合并多模态嵌入向量

        Args:
            input_ids: 输入ID
            inputs_embeds: 输入嵌入向量
            multimodal_embeddings: 多模态嵌入向量
            image_token_id: 图像token ID

        Returns:
            合并后的嵌入向量
        """
        if hasattr(self._model, "merge_multimodal_embeddings"):
            return self._model.merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, image_token_id
            )
        else:
            # 简单实现：直接返回输入嵌入
            return inputs_embeds

    def _get_generation_config(self):
        """
        获取模型的生成配置参数

        Returns:
            生成配置参数字典
        """
        from src.core.utils.generation_config import GenerationConfigManager

        # 如果模型有_get_generation_config方法，调用它
        if hasattr(self._model, "_get_generation_config"):
            return self._model._get_generation_config()
        # 否则使用统一的生成配置管理器
        elif hasattr(self._model, "model") and self._model.model is not None:
            return GenerationConfigManager.get_generation_config(self._model.model)
        else:
            # 如果模型未初始化，返回默认配置
            return GenerationConfigManager.get_generation_config(None)
