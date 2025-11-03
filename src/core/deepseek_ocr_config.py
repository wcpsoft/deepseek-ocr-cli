#!/usr/bin/env python3
"""
DeepSeek OCR配置类
定义模型配置相关的类
"""

from transformers import PretrainedConfig


# 默认生成配置
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 8192,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.0,
    "no_repeat_ngram_size": 35,
}


class DeepseekVLV2Config(PretrainedConfig):
    """
    DeepSeek VLV2模型配置类
    """

    model_type = "deepseek_vl_v2"

    def __init__(
        self,
        vision_config=None,
        projector_config=None,
        text_config=None,
        tile_tag="2D",
        global_view_pos="head",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.projector_config = projector_config
        self.text_config = text_config
        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos


class DeepseekV2Config(PretrainedConfig):
    """
    DeepSeek V2模型配置类
    """

    model_type = "deepseek_v2"

    def __init__(
        self,
        hidden_size=1280,
        intermediate_size=6848,
        num_hidden_layers=12,
        num_attention_heads=10,
        num_key_value_heads=10,
        max_position_embeddings=8192,
        vocab_size=129280,
        first_k_dense_replace=1,
        kv_lora_rank=None,
        q_lora_rank=None,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
        *,
        use_mla=False,
        topk_method="greedy",
        topk_group=1,
        n_group=1,
        n_routed_experts=64,
        n_shared_experts=2,
        num_experts_per_tok=6,
        moe_intermediate_size=896,
        rm_head=False,
        lm_head=True,
        bos_token_id=0,
        eos_token_id=1,
        torch_dtype="bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.first_k_dense_replace = first_k_dense_replace
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.use_mla = use_mla
        self.topk_method = topk_method
        self.topk_group = topk_group
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.rm_head = rm_head
        self.lm_head = lm_head
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.torch_dtype = torch_dtype


# 立即注册模型配置类
def _register_configs():
    """注册配置类到transformers库"""
    try:
        from transformers import AutoConfig

        # 注册到AutoConfig中
        AutoConfig.register("deepseek_vl_v2", DeepseekVLV2Config)
        AutoConfig.register("deepseek_v2", DeepseekV2Config)

    except Exception as e:
        # 忽略注册过程中的错误，但记录日志
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"配置类注册过程中出现警告: {e}")


# 立即执行注册
_register_configs()


# 确保这些类在模块级别可用
DeepseekV2Model = None
DeepseekV2ForCausalLM = None
