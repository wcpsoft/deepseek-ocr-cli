#!/usr/bin/env python3
"""
DeepSeek OCR模型基类
提供与HuggingFace兼容的模型接口基类实现
"""

# 标准库导入
from typing import Any, Optional, TypedDict

import torch
import torch.nn as nn
from addict import Dict

# 第三方库导入
from transformers import GenerationMixin

# 配置导入
# 项目内部导入
from src.core.deepencoder.build_linear import MlpProjector
from src.core.deepseek_ocr_config import DeepseekV2Config, DeepseekVLV2Config
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()


class AddSpecialTokensOutput(TypedDict):
    """添加特殊token的输出类型"""

    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    images_seq_mask: torch.BoolTensor
    images_spatial_crop: torch.LongTensor


class BaseDeepseekOCRForCausalLM(nn.Module, GenerationMixin):
    """
    DeepSeek OCR因果语言模型基类
    为Transformers实现提供基础类
    """

    def __init__(self, config: object | None = None) -> None:
        """
        初始化基类

        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.language_model = None
        self.image_token_id = None

    def _initialize_vision_components(self) -> None:
        """
        初始化视觉组件
        子类应该重写此方法以初始化特定的视觉组件
        """


class DeepseekOCRCausalBaseModel(nn.Module):
    """
    DeepSeek OCR因果语言模型基类
    提供与HuggingFace兼容的模型接口基类实现
    """

    def __init__(self, config: DeepseekV2Config | DeepseekVLV2Config, **kwargs):
        """
        初始化DeepSeek OCR因果语言模型基类

        Args:
            config: 模型配置
            **kwargs: 其他参数
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self._init_weights = False

        # 初始化投影器
        n_embed = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
        self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))

        logger.info("DeepseekOCRCausalBaseModel初始化完成")

    def _common_init_weights(self, module: nn.Module) -> None:
        """
        通用权重初始化方法

        Args:
            module: 模块
        """
        if not self._init_weights:
            return

        from transformers import AutoModelForCausalLM

        # 使用AutoModelForCausalLM的权重初始化逻辑
        AutoModelForCausalLM._init_weights(self, module)  # type: ignore

    def get_input_embeddings(self) -> nn.Module:
        """
        获取输入嵌入层

        Returns:
            输入嵌入层
        """
        raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module) -> None:
        """
        设置输入嵌入层

        Args:
            value: 输入嵌入层
        """
        raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        """
        获取输出嵌入层

        Returns:
            输出嵌入层
        """
        raise NotImplementedError

    def set_output_embeddings(self, value: nn.Module) -> None:
        """
        设置输出嵌入层

        Args:
            value: 输出嵌入层
        """
        raise NotImplementedError

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        调整token嵌入层大小

        Args:
            new_num_tokens: 新的token数量
            pad_to_multiple_of: 填充到倍数

        Returns:
            调整后的嵌入层
        """
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list[list[torch.FloatTensor]] | tuple] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.BoolTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> dict:
        """
        为生成准备输入

        Args:
            input_ids: 输入ID
            past_key_values: 过去的键值
            attention_mask: 注意力掩码
            inputs_embeds: 输入嵌入
            images: 图像
            images_seq_mask: 图像序列掩码
            images_spatial_crop: 图像空间裁剪
            **kwargs: 其他参数

        Returns:
            准备好的输入字典
        """
        # 如果有past_key_values，只取最后一个token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # type: ignore

        # 如果有输入嵌入，使用它
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 添加其他参数
        additional_inputs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
        model_inputs.update(additional_inputs)  # type: ignore
        model_inputs.update(kwargs)  # type: ignore

        return model_inputs

    def add_special_tokens(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        images_seq_mask: torch.BoolTensor,
        images_spatial_crop: torch.LongTensor,
    ) -> AddSpecialTokensOutput:
        """
        添加特殊token

        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            images_seq_mask: 图像序列掩码
            images_spatial_crop: 图像空间裁剪

        Returns:
            添加特殊token后的输出
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

    def _get_visual_token_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """
        获取视觉token掩码

        Args:
            input_ids: 输入ID

        Returns:
            视觉token掩码
        """
        # 这里应该根据实际的视觉token ID来实现
        # 暂时返回全False掩码
        return torch.zeros_like(input_ids, dtype=torch.bool)  # type: ignore
