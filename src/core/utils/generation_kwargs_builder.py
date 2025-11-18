#!/usr/bin/env python3
"""
生成参数构建器

提供灵活的模型生成参数构建功能
"""

from typing import Any

import torch


class GenerationKwargsBuilder:
    """
    生成参数构建器

    提供流畅的接口来构建模型生成参数
    """

    def __init__(self):
        """初始化构建器"""
        self._params = {}
        self._defaults = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,
            "eos_token_id": None,
            "repetition_penalty": 1.0,
        }

    def reset(self) -> "GenerationKwargsBuilder":
        """
        重置构建器

        Returns:
            GenerationKwargsBuilder: 重置后的构建器
        """
        self._params = {}
        return self

    def with_max_length(self, max_length: int) -> "GenerationKwargsBuilder":
        """
        设置最大生成长度

        Args:
            max_length: 最大长度

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["max_new_tokens"] = max_length
        return self

    def with_temperature(self, temperature: float) -> "GenerationKwargsBuilder":
        """
        设置温度参数

        Args:
            temperature: 温度值 (0.0-1.0)

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["temperature"] = max(0.0, min(temperature, 2.0))
        return self

    def with_top_p(self, top_p: float) -> "GenerationKwargsBuilder":
        """
        设置top-p采样参数

        Args:
            top_p: top-p值 (0.0-1.0)

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["top_p"] = max(0.0, min(top_p, 1.0))
        return self

    def with_top_k(self, top_k: int) -> "GenerationKwargsBuilder":
        """
        设置top-k采样参数

        Args:
            top_k: top-k值

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["top_k"] = max(0, top_k)
        return self

    def with_no_sampling(self) -> "GenerationKwargsBuilder":
        """
        设置为贪婪解码

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["do_sample"] = False
        return self

    def with_beam_search(self, num_beams: int = 5) -> "GenerationKwargsBuilder":
        """
        设置束搜索

        Args:
            num_beams: 束数量

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["num_beams"] = max(1, num_beams)
        self._params["do_sample"] = False
        return self

    def with_repetition_penalty(self, penalty: float) -> "GenerationKwargsBuilder":
        """
        设置重复惩罚

        Args:
            penalty: 惩罚值 (>1.0 抑制重复, <1.0 鼓励重复)

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["repetition_penalty"] = float(penalty)
        return self

    def with_length_penalty(self, penalty: float) -> "GenerationKwargsBuilder":
        """
        设置长度惩罚

        Args:
            penalty: 长度惩罚值 (>1.0 奖励长序列, <1.0 抑制长序列)

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["length_penalty"] = float(penalty)
        return self

    def with_early_stopping(self, enabled: bool = True) -> "GenerationKwargsBuilder":
        """
        设置早期停止

        Args:
            enabled: 是否启用早期停止

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["early_stopping"] = enabled
        return self

    def with_pad_token(self, pad_token_id: int) -> "GenerationKwargsBuilder":
        """
        设置填充token ID

        Args:
            pad_token_id: 填充token ID

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["pad_token_id"] = pad_token_id
        return self

    def with_eos_token(self, eos_token_id: int) -> "GenerationKwargsBuilder":
        """
        设置结束token ID

        Args:
            eos_token_id: 结束token ID

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["eos_token_id"] = eos_token_id
        return self

    def with_attention_mask(self, attention_mask: torch.Tensor) -> "GenerationKwargsBuilder":
        """
        设置注意力掩码

        Args:
            attention_mask: 注意力掩码张量

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params["attention_mask"] = attention_mask
        return self

    def with_custom_param(self, key: str, value: Any) -> "GenerationKwargsBuilder":
        """
        添加自定义参数

        Args:
            key: 参数键
            value: 参数值

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params[key] = value
        return self

    def from_dict(self, params: dict[str, Any]) -> "GenerationKwargsBuilder":
        """
        从字典导入参数

        Args:
            params: 参数字典

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        self._params.update(params)
        return self

    def with_default_params(self) -> "GenerationKwargsBuilder":
        """
        应用默认参数

        Returns:
            GenerationKwargsBuilder: 构建器自身
        """
        for key, value in self._defaults.items():
            if key not in self._params:
                self._params[key] = value
        return self

    def build(self) -> dict[str, Any]:
        """
        构建最终的参数字典

        Returns:
            Dict[str, Any]: 生成参数字典
        """
        return self._params.copy()

    def build_with_defaults(self) -> dict[str, Any]:
        """
        构建包含默认值的参数字典

        Returns:
            Dict[str, Any]: 包含默认值的生成参数字典
        """
        result = self._defaults.copy()
        result.update(self._params)
        return result

    def validate(self) -> list[str]:
        """
        验证参数的有效性

        Returns:
            List[str]: 验证错误列表，空列表表示无错误
        """
        errors = []

        # 温度验证
        if "temperature" in self._params:
            temp = self._params["temperature"]
            if not isinstance(temp, (int, float)):
                errors.append("temperature必须是数字")
            elif temp < 0 or temp > 2:
                errors.append("temperature应该在0.0-2.0范围内")

        # top_p验证
        if "top_p" in self._params:
            top_p = self._params["top_p"]
            if not isinstance(top_p, (int, float)):
                errors.append("top_p必须是数字")
            elif top_p < 0 or top_p > 1:
                errors.append("top_p应该在0.0-1.0范围内")

        # top_k验证
        if "top_k" in self._params:
            top_k = self._params["top_k"]
            if not isinstance(top_k, int):
                errors.append("top_k必须是整数")
            elif top_k < 0:
                errors.append("top_k必须大于等于0")

        # 束搜索验证
        if "num_beams" in self._params:
            beams = self._params["num_beams"]
            if not isinstance(beams, int):
                errors.append("num_beams必须是整数")
            elif beams < 1:
                errors.append("num_beams必须大于等于1")

        return errors

    def is_valid(self) -> bool:
        """
        检查参数是否有效

        Returns:
            bool: 参数是否有效
        """
        return len(self.validate()) == 0

    def copy(self) -> "GenerationKwargsBuilder":
        """
        创建构建器的副本

        Returns:
            GenerationKwargsBuilder: 构建器副本
        """
        new_builder = GenerationKwargsBuilder()
        new_builder._params = self._params.copy()
        return new_builder

    def get_prompt_template(self) -> str:
        """
        获取参数描述模板

        Returns:
            str: 参数描述
        """
        descriptions = []

        if "max_new_tokens" in self._params:
            descriptions.append(f"最大长度: {self._params['max_new_tokens']}")
        if "temperature" in self._params:
            descriptions.append(f"温度: {self._params['temperature']}")
        if "top_p" in self._params:
            descriptions.append(f"Top-p: {self._params['top_p']}")
        if "top_k" in self._params:
            descriptions.append(f"Top-k: {self._params['top_k']}")
        if "do_sample" in self._params:
            mode = "采样" if self._params["do_sample"] else "贪婪"
            descriptions.append(f"模式: {mode}")
        if "num_beams" in self._params:
            descriptions.append(f"束搜索: {self._params['num_beams']}束")

        return ", ".join(descriptions) if descriptions else "默认参数"

    @classmethod
    def create_text_generation(cls) -> "GenerationKwargsBuilder":
        """
        创建文本生成配置

        Returns:
            GenerationKwargsBuilder: 配置了文本生成参数的构建器
        """
        return (
            cls()
            .with_max_length(2048)
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_top_k(50)
            .with_repetition_penalty(1.1)
        )

    @classmethod
    def create_translation(cls) -> "GenerationKwargsBuilder":
        """
        创建翻译任务配置

        Returns:
            GenerationKwargsBuilder: 配置了翻译参数的构建器
        """
        return cls().with_max_length(512).with_temperature(0.3).with_no_sampling().with_early_stopping()

    @classmethod
    def create_summary(cls) -> "GenerationKwargsBuilder":
        """
        创建摘要生成配置

        Returns:
            GenerationKwargsBuilder: 配置了摘要生成参数的构建器
        """
        return cls().with_max_length(256).with_temperature(0.6).with_top_p(0.9).with_length_penalty(1.2)

    @classmethod
    def create_creative_writing(cls) -> "GenerationKwargsBuilder":
        """
        创建创意写作配置

        Returns:
            GenerationKwargsBuilder: 配置了创意写作参数的构建器
        """
        return (
            cls()
            .with_max_length(1024)
            .with_temperature(1.0)
            .with_top_p(0.92)
            .with_top_k(100)
            .with_repetition_penalty(1.2)
        )


# 便捷函数
def create_generation_builder() -> GenerationKwargsBuilder:
    """
    创建新的生成参数构建器

    Returns:
        GenerationKwargsBuilder: 新的构建器实例
    """
    return GenerationKwargsBuilder()


def build_text_generation_params(
    max_length: int = 2048, temperature: float = 0.8, top_p: float = 0.95, **kwargs
) -> dict[str, Any]:
    """
    快速构建文本生成参数

    Args:
        max_length: 最大长度
        temperature: 温度
        top_p: top-p值
        **kwargs: 其他参数

    Returns:
        Dict[str, Any]: 参数字典
    """
    return (
        GenerationKwargsBuilder()
        .with_max_length(max_length)
        .with_temperature(temperature)
        .with_top_p(top_p)
        .with_custom_params(kwargs)
        .build()
    )
