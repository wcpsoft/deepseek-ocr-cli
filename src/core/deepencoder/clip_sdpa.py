"""
CLIP视觉编码器实现（使用SDPA优化）
基于Vision Transformer的CLIP视觉编码器实现
"""

import logging
import math

import torch

# from megatron.model import LayerNorm
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

# 延迟导入flash_attn，避免在不支持的平台上报错
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_qkvpacked_func = None
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


class LayerNormfp32(torch.nn.LayerNorm):
    """继承torch的LayerNorm以处理fp16。"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            input: 输入张量

        Returns:
            输出张量
        """
        return F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(input.dtype)


def get_abs_pos(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
    """
    获取绝对位置编码

    Args:
        abs_pos: 绝对位置编码
        tgt_size: 目标尺寸

    Returns:
        调整后的绝对位置编码
    """
    # abs_pos: 位置编码，L:序列长度, C:通道数
    # tgt_size: 目标尺寸，M:目标序列长度
    # return: 调整后的位置编码，M:目标序列长度, C:通道数

    # 打印目标尺寸
    # 打印位置编码的形状
    # 退出程序
    dim = abs_pos.size(-1)
    # 打印维度
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).contiguous()
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


@torch.jit.script
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    快速GELU激活函数

    Args:
        x: 输入张量

    Returns:
        激活后的张量
    """
    return x * torch.sigmoid(1.702 * x)


class CLIPVisionEmbeddings(nn.Module):
    """
    CLIP视觉嵌入层
    处理图像输入并生成视觉嵌入向量
    """

    def __init__(
        self, hidden_size: int = 1024, image_size: int = 224, patch_size: int = 14, num_channels: int = 3
    ) -> None:
        """
        初始化CLIP视觉嵌入层

        Args:
            hidden_size: 隐藏层大小
            image_size: 图像尺寸
            patch_size: 图像块大小
            num_channels: 图像通道数
        """
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = torch.nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = torch.nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.Tensor, patch_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播函数

        Args:
            pixel_values: 像素值张量
            patch_embeds: 图像块嵌入

        Returns:
            视觉嵌入向量
        """
        batch_size = pixel_values.shape[0]

        if patch_embeds is not None:
            patch_embeds = patch_embeds
        else:
            patch_embeds = self.patch_embedding(pixel_values)
            
        # 确保patch_embeds不为None再进行操作
        if patch_embeds is not None:
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        else:
            # 如果patch_embeds仍然为None，创建一个默认的张量
            # 这种情况不应该发生，但为了代码健壮性添加检查
            patch_embeds = torch.zeros(batch_size, self.num_patches, self.embed_dim, device=pixel_values.device)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + get_abs_pos(self.position_embedding(self.position_ids), embeddings.size(1))
        return embeddings


class NoTPFeedForward(nn.Module):
    """
    无张量并行前馈网络
    简化的前馈网络实现
    """

    def __init__(
        self,
        cfg,
        dim: int,
        hidden_dim: int,
    ):
        """
        初始化前馈网络

        Args:
            cfg: 配置对象
            dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        output = self.fc2(quick_gelu(self.fc1(x)))
        return output


class NoTPAttention(torch.nn.Module):
    """
    无张量并行注意力机制
    实现标准的自注意力机制
    """

    def __init__(self, cfg):
        """
        初始化注意力机制

        Args:
            cfg: 配置对象
        """
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.n_local_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn

        self.qkv_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

        # self.core_attention = CoreAttention(cfg, AttnType.self_attn)

        self.attn_drop = cfg.attention_dropout

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        前向传播函数

        Args:
            x: 输入张量

        Returns:
            注意力处理后的张量
        """
        bsz, seqlen, _ = x.shape
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)

        if self.use_flash_attention and FLASH_ATTN_AVAILABLE and flash_attn_qkvpacked_func is not None:
            output = flash_attn_qkvpacked_func(xqkv)
            output = output.view(bsz, seqlen, -1)
        else:
            # 使用标准的注意力机制作为替代
            xq, xk, xv = torch.split(xqkv, 1, dim=2)
            xq = xq.squeeze(2)
            xk = xk.squeeze(2)
            xv = xv.squeeze(2)
            # （B:批次大小, num_head:注意力头数, S:序列长度, head_size:头维度)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xv = xv.permute(0, 2, 1, 3)
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
            output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        output = self.out_proj(output)
        return output


class NoTPTransformerBlock(nn.Module):
    """
    无张量并行Transformer块
    实现标准的Transformer块结构
    """

    def __init__(self, cfg, layer_id: int, multiple_of=256):
        """
        初始化Transformer块

        Args:
            cfg: 配置对象
            layer_id: 层ID
            multiple_of: 倍数参数
        """
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.self_attn = NoTPAttention(cfg)
        self.mlp = NoTPFeedForward(cfg, dim=cfg.hidden_size, hidden_dim=cfg.ffn_hidden_size)
        self.layer_id = layer_id
        self.layer_norm1 = torch.nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.layer_norm2 = torch.nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        residual = self.self_attn.forward(self.layer_norm1(x))
        h = x + residual
        out = h + self.mlp.forward(self.layer_norm2(h))
        return out


class NoTPTransformer(nn.Module):
    """
    无张量并行Transformer
    实现完整的Transformer结构
    """

    def __init__(self, cfg):
        """
        初始化Transformer

        Args:
            cfg: 配置对象
        """
        super().__init__()

        self.cfg = cfg
        # self.recompute_list = self.cfg.get("recompute_list", [])
        self.num_layers = cfg.num_layers  # _get_num_layers(cfg)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.layers.append(
                NoTPTransformerBlock(
                    cfg,
                    layer_id + 1,
                )
            )

    def forward(
        self,
        hidden_states,
    ):
        """
        前向传播函数

        Args:
            hidden_states: 隐藏状态张量

        Returns:
            处理后的隐藏状态张量
        """

        for _lid, layer in enumerate(self.layers):
            # if lid in self.recompute_list:
            #     def custom(layer_id):
            #         def custom_forward(*args, **kwargs):
            #             x_ = self.layers[layer_id](*args, **kwargs)
            #             return x_

            #         return custom_forward

            #     assert hidden_states.requires_grad == True, logger.warning(
            #         "When using recalculation, the input must have grad fn"
            #     )
            #     hidden_states = tensor_parallel.checkpoint(
            #         custom(lid),
            #         False,
            #         hidden_states.contiguous()
            #     )
            # else:
            hidden_states = layer(hidden_states)

        return hidden_states


# from megatron.core.tensor_parallel.layers import non_tensor_paralleled, local_dp_reduce, local_dp_scatter


class VitModel(nn.Module):
    """
    Vision Transformer模型
    基于Vision Transformer的视觉模型实现
    """

    def __init__(self, cfg, *, freeze_embed=False, freeze_pre_norm=False) -> None:
        """
        初始化Vision Transformer模型

        Args:
            cfg: 配置对象
            freeze_embed: 是否冻结嵌入层
            freeze_pre_norm: 是否冻结预归一化层
        """
        super().__init__()

        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=cfg.hidden_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
        )

        if freeze_embed:
            for _name, param in self.embeddings.named_parameters():
                param.requires_grad = False

        self.transformer = NoTPTransformer(cfg=cfg)

        if cfg.get("fp32norm", False):
            logging.info("Load fp32 layernorm for ViT.")
            self.pre_layrnorm = LayerNormfp32(
                cfg.hidden_size,
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )
        else:
            self.pre_layrnorm = torch.nn.LayerNorm(
                cfg.hidden_size,
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )

        # self.pre_layrnorm = RMSNorm(
        #     cfg.hidden_size,
        #     eps=cfg.get("pre_layernorm_epsilon", 1e-5),
        #     sequence_parallel=False,
        #     use_fp32=True,
        #     use_optimus=True,
        # )

        if freeze_pre_norm:
            for _name, param in self.pre_layrnorm.named_parameters():
                param.requires_grad = False

        for p in self.parameters():
            # 为参数添加自定义属性
            setattr(p, 'micro_dp', True)

    def set_input_tensor(self, input_tensor):
        """
        设置输入张量

        Args:
            input_tensor: 输入张量
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        # 注意：这里假设transformer有set_input_tensor方法
        # 如果没有，可以忽略或根据实际情况调整

    def __str__(self) -> str:
        """
        字符串表示

        Returns:
            模型名称字符串
        """
        return "open_clip"

    def forward(self, x, patch_embeds):
        """
        前向传播函数

        Args:
            x: 输入张量
            patch_embeds: 图像块嵌入

        Returns:
            处理后的张量
        """
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)

        # hidden_states, dis = local_dp_scatter(hidden_states)
        output = self.transformer(hidden_states)

        # output = local_dp_reduce(output, dis)

        return output


vit_model_cfg = EasyDict(
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    num_attention_heads=16,
    ffn_hidden_size=4096,
    seq_length=256,
    max_position_embeddings=256,
    use_flash_attn=False,
    understand_projector_stride=2,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    no_persist_layer_norm=False,
    layernorm_epsilon=1e-5,
    pre_layernorm_epsilon=1e-5,
    image_size=224,
    patch_size=14,
    recompute_list=[],
)


def build_clip_l():
    """
    构建CLIP-L模型

    Returns:
        CLIP-L模型实例
    """
    return VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )


if __name__ == "__main__":

    vit_model_cfg = EasyDict(
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        seq_length=256,
        max_position_embeddings=256,
        use_flash_attn=False,
        understand_projector_stride=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        no_persist_layer_norm=False,
        layernorm_epsilon=1e-5,
        pre_layernorm_epsilon=1e-5,
        image_size=224,
        patch_size=14,
        recompute_list=[],
    )

    vision_model = VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )

    x = torch.zeros(2, 3, 1024, 1024)

    with torch.no_grad():
        pass
