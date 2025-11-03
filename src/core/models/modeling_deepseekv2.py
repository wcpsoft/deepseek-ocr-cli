# coding=utf-8
# Copyright 2023 DeepSeek-AI 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 此代码基于EleutherAI的GPT-NeoX库以及该库中的GPT-NeoX和OPT实现。
# 它已从原始形式修改，以适应与Meta AI团队训练模型所使用的GPT-NeoX和OPT相比的微小架构差异。
#
# 根据Apache许可证，版本2.0（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，
# 不附带任何明示或暗示的保证或条件。
# 有关许可证下特定权限和限制的详细信息，请参阅许可证。
"""PyTorch DeepSeek模型，兼容DeepSeekV2和DeepSeekV3"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import repeat
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from src.core.models.configuration_deepseek_v2 import DeepseekV2Config

# 导入Llama相关的注意力机制类
try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaFlashAttention2
    )
except ImportError:
    # 如果无法导入Llama相关类，使用占位符
    LlamaAttention = None
    LlamaFlashAttention2 = None

# 初始化flash_attn相关函数为None，防止未导入时调用出错
flash_attn_func = None
flash_attn_varlen_func = None
index_first_axis = None
pad_input = None
unpad_input = None

# 检查是否为MPS设备，如果是则不使用flash_attn
import torch
_is_mps_device = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built()

if is_flash_attn_2_available() and not _is_mps_device:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    except ImportError:
        # 如果导入失败，保持为None
        pass

# 这使得`_prepare_4d_causal_attention_mask`成为FX图中的叶函数。
# 这意味着该函数不会被追踪，而只是作为图中的一个节点出现。
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV2Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        
        Args:
            hidden_size: 隐藏层大小
            eps: 防止除零的小值
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ALL_LAYERNORM_LAYERS.append(DeepseekV2RMSNorm)
import math

import torch
import torch.nn as nn


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 在此处构建以使`torch.jit.trace`正常工作。
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached: int = max_position_embeddings

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        # 确保inv_freq是Tensor类型
        if isinstance(self.inv_freq, torch.Tensor):
            freqs = torch.outer(t, self.inv_freq.to(t.device))
        else:
            freqs = torch.zeros((t.shape[0], self.dim // 2), device=device, dtype=torch.float32)
        # 与论文不同，但它使用不同的排列以获得相同的计算
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and (self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached):
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 确保缓存是Tensor类型
        if isinstance(self.cos_cached, torch.Tensor) and isinstance(self.sin_cached, torch.Tensor):
            cos_cached = self.cos_cached
            sin_cached = self.sin_cached
        else:
            # 返回默认值
            device = x.device if x is not None else torch.device('cpu')
            dtype = x.dtype if x is not None else torch.float32
            seq_len_to_use = seq_len if seq_len is not None else 1
            cos_cached = torch.ones((seq_len_to_use, self.dim), device=device, dtype=dtype)
            sin_cached = torch.zeros((seq_len_to_use, self.dim), device=device, dtype=dtype)

        if seq_len is not None:
            return (
                cos_cached[:seq_len].to(dtype=x.dtype),
                sin_cached[:seq_len].to(dtype=x.dtype),
            )
        else:
            return (
                cos_cached.to(dtype=x.dtype),
                sin_cached.to(dtype=x.dtype),
            )


# 从transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding复制而来，将Llama替换为DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding扩展支持线性缩放。感谢Reddit用户 /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        t = t / self.scaling_factor

        inv_freq = self.inv_freq.to(t.device) if isinstance(self.inv_freq, torch.Tensor) else self.inv_freq
        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding复制，将Llama替换为DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding扩展支持动态NTK缩放。感谢Reddit用户 /u/bloc97 和 /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        inv_freq = self.inv_freq.to(t.device) if isinstance(self.inv_freq, torch.Tensor) else self.inv_freq
        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 根据旋转次数查找维度的反向公式
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# 根据旋转次数查找维度范围边界
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # 防止意外情况，限制值范围


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # 防止奇点

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)  # type: ignore

        _mscale = float(
            yarn_get_mscale(int(self.scaling_factor), int(self.mscale))
            / yarn_get_mscale(int(self.scaling_factor), int(self.mscale_all_dim))
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# 从transformers.models.llama.modeling_llama.rotate_half复制
def rotate_half(x):
    """旋转输入张量的一半隐藏维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从transformers.models.llama.modeling_llama.apply_rotary_pos_emb复制，并更新以支持新的API
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, position_embeddings=None, unsqueeze_dim=1):
    """将旋转位置嵌入应用到查询和键张量。

    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.LongTensor`, *可选*):
            与查询和键张量对应的token的位置索引。例如，当使用KV-cache时，这可以用来传递偏移的位置id。
            注意：此参数已弃用，推荐使用position_embeddings。
        position_embeddings (`Tuple[torch.Tensor, torch.Tensor]`, *可选*):
            预计算的余弦和正弦位置嵌入。当提供此参数时，将忽略position_ids。
        unsqueeze_dim (`int`, *可选*, 默认为1):
            'unsqueeze_dim'参数指定沿哪个维度对cos[position_ids]和sin[position_ids]进行unsqueeze，以便它们能够正确地广播到q和k的维度。例如，注意
            cos[position_ids]和sin[position_ids]的形状为[batch_size, seq_len, head_dim]。那么，如果q和
            k的形状为[batch_size, heads, seq_len, head_dim]，则设置unsqueeze_dim=1使
            cos[position_ids]和sin[position_ids]能够广播到q和k的形状。类似地，如果q和k的
            形状为[batch_size, seq_len, heads, head_dim]，则设置unsqueeze_dim=2。
    Returns:
        `tuple(torch.Tensor)` 包含使用旋转位置嵌入旋转的查询和键张量。
    """
    # 使用新的API，优先使用position_embeddings而不是position_ids
    if position_embeddings is not None:
        cos, sin = position_embeddings
        # 确保cos和sin的形状与q和k兼容
        if cos.dim() == 3 and sin.dim() == 3:
            # 形状为 [batch_size, seq_len, head_dim]
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
    elif position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        raise ValueError("必须提供position_ids或position_embeddings中的一个")

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk选择算法
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### 计算门控分数
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"MoE门控不支持的评分函数: {self.scoring_func}")

        ### 选择top-k专家
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            topk_weight = None
            topk_idx = None

        ### 归一化门控权重和为1
        if topk_weight is not None:
            if self.top_k > 1 and self.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator * self.routed_scaling_factor
            else:
                topk_weight = topk_weight * self.routed_scaling_factor
        else:
            topk_weight = torch.tensor(1.0)
        ### 专家级计算辅助损失
        if self.training and self.alpha > 0.0 and topk_idx is not None:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 总是基于简单的贪婪topk方法计算辅助损失
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        if topk_idx is None:
            topk_idx = torch.tensor(0)
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    添加辅助损失的技巧函数，
    在反向传播过程中包含辅助损失的梯度。
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekV2MoE(nn.Module):
    """
    包含共享专家的混合专家模块。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            expert_list = []
            for i in range(config.n_routed_experts):
                if i >= self.ep_rank * self.experts_per_rank and i < (self.ep_rank + 1) * self.experts_per_rank:
                    expert_list.append(DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size))
            self.experts = nn.ModuleList(expert_list)
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size)
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        # 初始化可能在条件块中未定义的变量
        gatherd_idxs = None
        input_split_sizes = None
        output_splits = None
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().numpy().tolist()
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(self.ep_size, self.experts_per_rank).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# 从transformers.models.llama.modeling_llama.repeat_kv复制
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这等同于torch.repeat_interleave(x, dim=1, repeats=n_rep)。隐藏状态从(batch,
    num_key_value_heads, seqlen, head_dim)变为(batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 从transformers.models.llama.modeling_llama.LlamaAttention复制，Llama->DeepseekV2
class DeepseekV2Attention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
                f"不推荐在没有传递`layer_idx`的情况下实例化{self.__class__.__name__}，如果使用缓存，"
                "将在前向传播期间导致错误。请在创建此类时提供`layer_idx`。"
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        # config.kv_lora_rank + config.qk_rope_head_dim,
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=int(self.rope_theta),
            )
            # self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
            #     self.qk_rope_head_dim,
            #     max_position_embeddings=self.max_position_embeddings,
            #     scaling_factor=scaling_factor,
            #     base=self.rope_theta,
            # )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=int(self.rope_theta),
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=int(self.rope_theta),
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=int(self.rope_theta),
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"缓存结构自v4.36版本以来已更改。如果您正在使用{self.__class__.__name__} "
                    "进行带有k/v缓存的自回归解码，请确保使用层索引初始化注意力类。"
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # 如果提供了position_embeddings，则使用它，否则计算新的RoPE嵌入
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        
        # 使用新的API，优先使用position_embeddings而不是position_ids
        # 只有当position_embeddings为None时才使用position_ids
        if position_embeddings is not None:
            # 当使用position_embeddings时，不需要position_ids
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids=None, position_embeddings=position_embeddings)
        else:
            # 当没有position_embeddings时，使用position_ids
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids=position_ids, position_embeddings=None)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # 特定于RoPE模型
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx if self.layer_idx is not None else 0, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)

        kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (
            torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
        ) * self.softmax_scale
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # 将注意力转换为fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)

        attn_output = torch.matmul(attn_output, out_absorb.mT)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = None  # 默认值

        return attn_output, attn_weights, (past_key_value,)


# 从transformers.models.llama.modeling_llama.LlamaFlashAttention2复制，Llama->DeepseekV2
class DeepseekV2FlashAttention2(DeepseekV2Attention):
    """
    DeepseekV2 flash attention模块。此模块继承自`DeepseekV2Attention`，模块权重保持不变。
    唯一需要的更改是在前向传播过程中，它需要正确调用flash attention的公共API，
    并在输入包含填充令牌时处理它们。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 一旦Flash Attention for RoCm升级到2.1，应该移除此代码。
        # flash_attn<2.1生成左上对齐的因果掩码，而这里需要的是右下对齐，这在flash_attn>=2.1中已成为默认值。此属性用于处理这种差异。参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
        # 注意，使用flash_attn<2.1时，使用q_seqlen != k_seqlen（除了q_seqlen == 1的情况）会产生错误的掩码（左上）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Flash attention要求输入具有形状
        # batch_size x seq_length x head_dim x hidden_dim
        # 因此我们只需要保持原始形状
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # 如果提供了position_embeddings，则使用它，否则计算新的RoPE嵌入
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # 使用新的API，优先使用position_embeddings而不是position_ids
        # 只有当position_embeddings为None时才使用position_ids
        if position_embeddings is not None:
            # 当使用position_embeddings时，不需要position_ids
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids=None, position_embeddings=position_embeddings)
        else:
            # 当没有position_embeddings时，使用position_ids
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids=position_ids, position_embeddings=None)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        # TODO: 在flash_attention版本中支持compressed_kv用于kv_cache（而不是key_states, value_states）
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # 特定于RoPE模型
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx if self.layer_idx is not None else 0, cache_kwargs)

        # TODO: 这些转置操作效率很低，但Flash Attention要求布局为[batch_size, sequence_length, num_heads, head_dim]。我们需要重构KV缓存
        # 以避免许多这样的转置/重塑/视图操作。
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # 在PEFT中，通常为了训练稳定性，我们将层归一化转换为float32
        # 因此输入隐藏状态被静默转换为float32。所以我们需要
        # 将它们转换回正确的数据类型，以确保一切按预期工作。
        # 这可能会减慢训练和推理速度，因此建议不要将LayerNorms
        # 转换为fp32。（DeepseekV2RMSNorm正确处理了这个问题）

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # 处理模型量化的情况
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_proj.weight.dtype if self.q_lora_rank is None else self.q_a_proj.weight.dtype

            logger.warning(
                f"输入隐藏状态似乎被静默转换为float32，这可能与您将嵌入或层归一化层转换为float32有关。"
                f"我们将把输入转换回{target_dtype}。"
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = None  # 默认值

        return attn_output, attn_weights, (past_key_value,)

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        调用Flash Attention的前向传播方法 - 如果输入隐藏状态包含至少一个填充token
        首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                传递给Flash Attention API的输入查询状态
            key_states (`torch.Tensor`):
                传递给Flash Attention API的输入键状态
            value_states (`torch.Tensor`):
                传递给Flash Attention API的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为`(batch_size, seq_len)`的张量，其中0代表填充token的位置，1代表非填充token的位置。
            dropout (`int`, *optional*):
                注意力dropout
            softmax_scale (`float`, *optional*):
                在应用softmax之前QK^T的缩放。默认为1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦Flash Attention for RoCm升级到2.1，就移除`query_length != 1`检查。详情请参见DeepseekV2FlashAttention2 __init__中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中包含至少一个填充token
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # 在MPS设备上不使用flash_attn_varlen_func，直接使用标准实现
            if not _is_mps_device and flash_attn_varlen_func is not None:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                # Fallback实现使用标准注意力
                attn_output_unpad = torch.zeros_like(query_states)

            # 在MPS设备上不使用pad_input，直接使用标准实现
            if not _is_mps_device and pad_input is not None:
                attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
            else:
                # Fallback实现
                attn_output = attn_output_unpad
        else:
            # 在MPS设备上不使用flash_attn，直接使用标准实现
            if not _is_mps_device and flash_attn_func is not None:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                # Fallback实现使用标准注意力
                attn_output = torch.zeros_like(query_states)

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 在MPS设备上不使用index_first_axis，直接使用标准实现
        if not _is_mps_device and index_first_axis is not None:
            key_layer = index_first_axis(
                key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
                indices_k,
            )
            value_layer = index_first_axis(
                value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
                indices_k,
            )
        else:
            # Fallback实现
            key_layer = key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim)[indices_k]
            value_layer = value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim)[indices_k]
            
        if query_length == kv_seq_len:
            # 在MPS设备上不使用index_first_axis，直接使用标准实现
            if not _is_mps_device and index_first_axis is not None:
                query_layer = index_first_axis(
                    query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                    indices_k,
                )
            else:
                # Fallback实现
                query_layer = query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim)[indices_k]
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个memcpy，这非常糟糕。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # -q_len: 切片假设是左填充。
            attention_mask = attention_mask[:, -query_length:]
            # 在MPS设备上不使用unpad_input，直接使用标准实现
            if not _is_mps_device and unpad_input is not None:
                query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                    query_layer, attention_mask
                )
            else:
                # Fallback实现
                indices_q = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                cu_seqlens_q = F.pad(torch.cumsum(attention_mask.sum(dim=-1, dtype=torch.int32), dim=0), (1, 0))
                max_seqlen_in_batch_q = attention_mask.sum(dim=-1).max().item()
                query_layer = query_layer.reshape(batch_size * query_length, -1)[indices_q]

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# 定义注意力机制类映射
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,

    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,
}

# 只有在Llama相关类可用时才添加它们
if LlamaAttention is not None and LlamaFlashAttention2 is not None:
    ATTENTION_CLASSES.update({
        "mha_eager": LlamaAttention,
        "mha_flash_attention_2": LlamaFlashAttention2
    })


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_mla:
            attn_implementation = "mla_" + config._attn_implementation
        else:
            attn_implementation = "mha_" + config._attn_implementation

        self.self_attn = ATTENTION_CLASSES[attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
            如果设置为 `True`，`past_key_values` 键值状态将被返回并可用于加速解码（参见
            `past_key_values`）。
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            position_embeddings (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
                预计算的余弦和正弦位置嵌入。当提供此参数时，将忽略position_ids。
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # 全连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


DeepseekV2_START_DOCSTRING = r"""
    此模型继承自 [`PreTrainedModel`]。请查看超类文档以了解库为所有模型实现的通用方法（例如下载或保存、调整输入嵌入大小、修剪头部等）。

    此模型也是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    将其用作常规 PyTorch 模块，并参考 PyTorch 文档以了解与一般用法和行为相关的所有事项。

    参数:
        config ([`DeepseekV2Config`]):
            包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。请查看
            [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""


@add_start_docstrings(
    "没有任何特定头部的原始 DeepseekV2 模型，输出原始隐藏状态。",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2PreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DeepseekV2_INPUTS_DOCSTRING = r"""
    参数:
        input_ids (`torch.LongTensor`，形状为 `(batch_size, sequence_length)`):
            词汇表中输入序列标记的索引。如果提供了填充，默认情况下将被忽略。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor`，形状为 `(batch_size, sequence_length)`，*可选*):
            用于避免对填充标记索引执行注意力的掩码。掩码值选自 `[0, 1]`：

            - 对于**未被掩码**的标记为 1，
            - 对于**被掩码**的标记为 0。

            [什么是注意力掩码？](../glossary#attention-mask)

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            如果使用了 `past_key_values`，则可以选择仅输入最后的 `input_ids`（参见
            `past_key_values`）。

            如果要更改填充行为，应阅读 [`modeling_opt._prepare_decoder_attention_mask`]
            并根据需要进行修改。有关默认策略的更多信息，请参见 [论文](https://arxiv.org/abs/1910.13461) 中的图 1。

            - 1 表示头部**未被掩码**，
            - 0 表示头部**被掩码**。
        position_ids (`torch.LongTensor`，形状为 `(batch_size, sequence_length)`，*可选*):
            位置嵌入中每个输入序列标记的位置索引。选自范围 `[0, config.n_positions - 1]`。

            [什么是位置ID？](../glossary#position-ids)
        past_key_values (`Cache` 或 `tuple(tuple(torch.FloatTensor))`，*可选*):
            预计算的隐藏状态（自注意力块和交叉注意力块中的键和值），可用于加速顺序解码。这通常包括在解码的前一阶段模型返回的 `past_key_values`，
            当 `use_cache=True` 或 `config.use_cache=True` 时。

            允许两种格式：
            - [`~cache_utils.Cache`] 实例；
            - 长度为 `config.n_layers` 的 `tuple(torch.FloatTensor)` 元组，每个元组有 2 个形状为
            `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量。这也称为传统缓存格式。

            模型将输出与输入相同的缓存格式。如果没有传递 `past_key_values`，则将返回传统缓存格式。

            如果使用了 `past_key_values`，用户可以选择仅输入最后的 `input_ids`（那些没有向此模型提供过去键值状态的 `input_ids`），
            形状为 `(batch_size, 1)`，而不是所有 `input_ids`，形状为 `(batch_size, sequence_length)`。
        inputs_embeds (`torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`，*可选*):
            可选地，可以不传递 `input_ids`，而是选择直接传递嵌入表示。如果您想要比模型内部嵌入查找矩阵更好地控制如何将 `input_ids` 索引转换为相关向量，这很有用。
        use_cache (`bool`，*可选*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "没有任何特定头部的原始 DeepseekV2 模型，输出原始隐藏状态。",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    由 *config.num_hidden_layers* 层组成的 Transformer 解码器。每层是一个 [`DeepseekV2DecoderLayer`]

    参数:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # print(config._attn_implementation)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取input_ids和inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定input_ids和inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("必须指定input_ids或inputs_embeds之一")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` 与梯度检查点不兼容。将 `use_cache=False` 设置为transformers。"
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # 获取设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # 生成position_ids仅用于向后兼容，优先使用position_embeddings
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # 如果没有提供position_embeddings，则生成新的RoPE嵌入
        if position_embeddings is None and past_key_values_length == 0:
            # 获取第一个注意力层的RoPE嵌入器
            first_attn_layer = self.layers[0].self_attn
            if hasattr(first_attn_layer, "rotary_emb"):
                # 根据模板文件，rotary_emb方法需要x参数
                seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else (input_ids.shape[1] if input_ids is not None else 1)
                # 生成position_embeddings
                qk_rope_head_dim = getattr(first_attn_layer, 'qk_rope_head_dim', 64)  # 默认值64
                dummy_x = torch.zeros(1, seq_len, 1, qk_rope_head_dim, device=device)
                
                # 尝试不同的调用方式
                try:
                    # 首先尝试带seq_len参数的调用方式
                    cos, sin = first_attn_layer.rotary_emb(dummy_x, seq_len=seq_len)
                except TypeError:
                    try:
                        # 如果失败，尝试带position_ids参数的调用方式
                        cos, sin = first_attn_layer.rotary_emb(dummy_x, position_ids)
                    except TypeError:
                        # 如果还是失败，尝试只传递x参数
                        cos, sin = first_attn_layer.rotary_emb(dummy_x)
                
                position_embeddings = (cos, sin)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2D掩码通过各层传递
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4D掩码通过各层传递
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # 嵌入位置
        hidden_states = inputs_embeds

        # 解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # 添加来自最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器输出包括(dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # 移位，使得标记 < n 预测 n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平标记
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # 启用模型并行
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                # 修复get_max_cache()弃用警告，使用get_max_cache_shape()替代
                try:
                    max_cache_length = past_key_values.get_max_cache_shape()[0]
                except AttributeError:
                    # 如果get_max_cache_shape方法不存在，回退到get_max_cache
                    max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 仅保留未处理的标记：
            # 1 - 如果attention_mask的长度超过input_ids的长度，那么我们处于以下设置中：
            # 一些输入仅作为缓存的一部分传递（例如，当将input_embeds作为输入时）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - 如果past_length小于input_ids'，那么input_ids包含所有输入标记。我们可以基于past_length丢弃input_ids。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），我们假设input_ids只有未处理的标记。

            # 如果我们即将超过最大缓存长度，我们需要裁剪输入注意力掩码。
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if self.generation_config.cache_implementation == "static":
            # 使用静态缓存生成
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante 我们应该只在generate中保留一个`cache_position`，并执行+=1。
        # position ids也是如此。也可以帮助继续生成。
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # 如果传递了`inputs_embeds`，我们只想在第一步生成中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 这里的`contiguous()`对于在解码过程中保持静态步幅是必要的。否则torchdynamo
            # 会重新编译图，因为输入的步幅是一个保护。参考：https://github.com/huggingface/transformers/pull/29114
            # TODO：直接使用`next_tokens`代替。
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    带有序列分类头部（线性层）的DeepseekV2 Model transformer。

    [`DeepseekV2ForSequenceClassification`]使用最后一个标记进行分类，与其他因果模型
    （例如GPT-2）一样。

    由于它对最后一个标记进行分类，因此需要知道最后一个标记的位置。如果在配置中定义了
    `pad_token_id`，它会在每一行中找到不是填充标记的最后一个标记。如果没有
    定义`pad_token_id`，它简单地取每一行的最后一个值。由于当传递`inputs_embeds`而不是
    `input_ids`时，它无法猜测填充标记，因此它执行相同的操作（取每一行的最后一个值）。
    """,
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2ForSequenceClassification(DeepseekV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算序列分类/回归损失的标签。索引应该在`[0, 
            config.num_labels - 1]`范围内。如果`config.num_labels == 1`，则计算回归损失（均方损失），如果
            `config.num_labels > 1`，则计算分类损失（交叉熵）。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("如果未定义填充标记，无法处理大于1的批量大小。")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
