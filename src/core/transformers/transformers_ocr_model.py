#!/usr/bin/env python3
"""
Transformers OCR模型实现

该模块实现了基于Transformers的OCR模型，支持Apple Silicon(MPS)设备。
"""

import logging
from typing import Any

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
)

from src.core.deepseek_ocr_config import (
    DeepseekV2Config as ConfigDeepseekV2Config,
)
from src.core.deepseek_ocr_config import (
    DeepseekVLV2Config as ConfigDeepseekVLV2Config,
)
from src.core.logging import get_logger
from src.core.models.base_ocr_model import BaseDeepseekOCRForCausalLM

# 获取日志记录器
logger = get_logger()

# 确保在模块加载时就注册配置类和模型类
try:
    # 动态注册配置类（如果尚未注册）
    if "deepseek_vl_v2" not in CONFIG_MAPPING:
        CONFIG_MAPPING._extra_content["deepseek_vl_v2"] = ConfigDeepseekVLV2Config
    if "deepseek_v2" not in CONFIG_MAPPING:
        CONFIG_MAPPING._extra_content["deepseek_v2"] = ConfigDeepseekV2Config

    # 尝试注册到AutoConfig
    try:
        # 使用更安全的方式访问_model_mapping
        if not hasattr(AutoConfig, "_model_mapping"):
            # 使用CONFIG_MAPPING代替_model_mapping
            pass

        # 使用getattr来获取属性
        # 直接使用CONFIG_MAPPING
        CONFIG_MAPPING["deepseek_vl_v2"] = ConfigDeepseekVLV2Config
        CONFIG_MAPPING["deepseek_v2"] = ConfigDeepseekV2Config

        try:
            # 注册DeepSeekVLV2模型配置
            try:
                # 使用本地定义的DeepseekVLV2Config而不是尝试从外部模块导入
                from src.core.deepseek_ocr_config import DeepseekVLV2Config
                CONFIG_MAPPING["deepseek_vl_v2"] = DeepseekVLV2Config
                logger.info("成功注册本地DeepseekVLV2Config")
            except Exception as e:
                logger.warning(f"注册DeepSeekVLV2模型配置失败: {e}")
        except Exception as e:
            logger.warning(f"注册模型配置时发生未知错误: {e}")
    except Exception as e:
        logger.warning(f"初始化模型映射时发生错误: {e}")
except Exception as e:
    logger.warning(f"配置模型时发生未知错误: {e}")

# 导入modeling_deepseekocr模块中的模型类
try:
    from src.core.models.modeling_deepseekocr import (
        DeepseekOCRForCausalLM as DirectDeepseekOCRForCausalLM,
    )

    logger.info("成功从src.core.models.modeling_deepseekocr导入DeepseekOCRForCausalLM")
except ImportError as e:
    logger.error(f"无法从src.core.models.modeling_deepseekocr导入DeepseekOCRForCausalLM: {e}")
    DirectDeepseekOCRForCausalLM = None


# 为保持向后兼容性，创建别名
class DeepseekOCRForCausalLM(BaseDeepseekOCRForCausalLM):
    """
    DeepSeek OCR因果语言模型Transformers实现
    """

    def __init__(self, config: object | None = None) -> None:
        """
        初始化Transformers实现

        Args:
            config: 模型配置
        """
        # 调用父类初始化
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "DeepseekOCRForCausalLM":
        """
        从预训练模型加载模型

        Args:
            pretrained_model_name_or_path: 预训练模型名称或路径
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            DeepseekOCRForCausalLM实例
        """

        logger = logging.getLogger(__name__)

        # 如果有直接可用的模型类，优先使用
        if DirectDeepseekOCRForCausalLM is not None:
            logger.info("使用直接导入的DeepseekOCRForCausalLM类")
            try:
                # 修复类型不匹配问题
                result = DirectDeepseekOCRForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                return result  # type: ignore
            except Exception as e:
                logger.error(f"直接导入的模型类加载失败，回退到基类实现: {e}")
        else:
            logger.warning("直接导入的DeepseekOCRForCausalLM类不可用，使用基类实现")

        # 先创建实例，但不初始化视觉组件
        model = cls(config=None)

        # 加载配置
        from transformers import LlamaConfig

        try:
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            # 尝试使用AutoConfig
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        model.config = config

        # 加载语言模型
        try:
            logger.info("正在加载语言模型")

            # 移除auto_map参数，避免循环导入问题
            kwargs.pop("auto_map", None)

            # 尝试直接加载语言模型
            language_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
            logger.info("配置的model_type: %s", config.model_type)
            model.language_model = language_model
            model.image_token_id = 200005
        except Exception as e:
            logger.error(f"加载语言模型失败: {e}")
            # 尝试使用语言配置加载
            try:
                if hasattr(config, "language_config"):
                    language_config = config.language_config

                    # 检查language_config是字典还是对象
                    if isinstance(language_config, dict):
                        # 从字典中获取配置
                        architectures = language_config.get("architectures", ["DeepseekV2ForCausalLM"])
                        bos_token_id = language_config.get("bos_token_id", 0)
                        eos_token_id = language_config.get("eos_token_id", 1)
                        hidden_size = language_config.get("hidden_size", 1280)
                        intermediate_size = language_config.get("intermediate_size", 6848)
                        max_position_embeddings = language_config.get("max_position_embeddings", 8192)
                        num_attention_heads = language_config.get("num_attention_heads", 10)
                        num_hidden_layers = language_config.get("num_hidden_layers", 12)
                        num_key_value_heads = language_config.get("num_key_value_heads", 10)
                        vocab_size = language_config.get("vocab_size", 129280)
                        torch_dtype = language_config.get("torch_dtype", "bfloat16")
                    else:
                        # 从对象中获取配置
                        architectures = language_config.architectures
                        bos_token_id = language_config.bos_token_id
                        eos_token_id = language_config.eos_token_id
                        hidden_size = language_config.hidden_size
                        intermediate_size = language_config.intermediate_size
                        max_position_embeddings = language_config.max_position_embeddings
                        num_attention_heads = language_config.num_attention_heads
                        num_hidden_layers = language_config.num_hidden_layers
                        num_key_value_heads = language_config.num_key_value_heads
                        vocab_size = language_config.vocab_size
                        torch_dtype = language_config.torch_dtype

                    # 创建一个新的配置对象，只包含语言模型相关的配置
                    lang_config = LlamaConfig(
                        architectures=architectures,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        max_position_embeddings=max_position_embeddings,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=num_hidden_layers,
                        num_key_value_heads=num_key_value_heads,
                        vocab_size=vocab_size,
                        torch_dtype=torch_dtype,
                    )

                    # 使用语言配置加载模型
                    language_model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path,
                        config=lang_config,
                        **kwargs,
                    )
                    model.language_model = language_model
                    model.image_token_id = 200005
                    logger.info("使用语言配置成功加载语言模型")
                else:
                    raise ValueError("配置中没有language_config")
            except Exception as e2:
                logger.error(f"使用语言配置加载模型也失败: {e2}")
                raise

        # 初始化视觉组件（在语言模型加载后）
        model._initialize_vision_components()

        return model

    def generate(self, *args, **kwargs):
        """
        生成方法，调用实际模型的生成方法

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            生成结果
        """
        if hasattr(self, "language_model") and self.language_model is not None:
            return self.language_model.generate(*args, **kwargs)
        else:
            logger.error("语言模型未初始化")
            raise ValueError("语言模型未初始化")

    def to(self, *args, **kwargs):
        """
        将模型移到指定设备

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            self
        """
        if len(args) > 0:
            self.device = args[0]
        elif "device" in kwargs:
            self.device = kwargs["device"]

        # 调用父类的to方法处理视觉模型等组件
        super().to(*args, **kwargs)

        # 处理语言模型
        if hasattr(self, "language_model") and self.language_model is not None:
            self.language_model = self.language_model.to(*args, **kwargs)
        return self

    def eval(self):
        """
        设置模型为评估模式

        Returns:
            self
        """
        # 调用父类的eval方法
        super().eval()

        # 设置语言模型为评估模式
        if hasattr(self, "language_model") and self.language_model is not None:
            self.language_model = self.language_model.eval()
        return self

    def parameters(self, recurse: bool = True):  # type: ignore
        """
        获取模型参数

        Args:
            recurse: 是否递归获取参数

        Returns:
            模型参数迭代器
        """
        # 获取父类参数
        params = super().parameters(recurse)

        # 获取语言模型参数
        if hasattr(self, "language_model") and self.language_model is not None:
            # 将参数合并为一个迭代器
            import itertools

            return itertools.chain(params, self.language_model.parameters(recurse))

        return params


# 延迟注册模型类，避免循环导入问题
def _register_model_classes():
    """延迟注册模型类，避免循环导入问题"""
    try:
        # 确保使用正确的配置类
        from src.core.deepseek_ocr_config import DeepseekVLV2Config
        
        class DeepseekVLV2ForCausalLM(DeepseekOCRForCausalLM):
            """
            DeepSeek VLV2因果语言模型
            这是DeepseekOCRForCausalLM的别名，用于兼容模型配置
            """

        # 注册模型类到AutoModelForCausalLM
        try:
            AutoModelForCausalLM.register(DeepseekVLV2Config, DeepseekVLV2ForCausalLM)
            logger.info("已注册DeepseekVLV2ForCausalLM到AutoModelForCausalLM")
        except Exception as e:
            logger.warning(f"注册模型类到AutoModelForCausalLM失败: {e}")

    except Exception as e:
        logger.warning(f"注册模型类失败: {e}")


# 在模块导入后执行注册
_register_model_classes()
