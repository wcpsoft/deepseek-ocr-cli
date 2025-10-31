#!/usr/bin/env python3
"""
MLP投影器模块
用于将视觉特征投影到语言模型的嵌入空间
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpProjector(nn.Module):
    """
    MLP投影器
    用于将视觉特征投影到语言模型的嵌入空间
    """

    def __init__(self, cfg):
        """
        初始化MLP投影器

        Args:
            cfg: 配置对象，包含投影器的相关配置参数
        """
        super().__init__()

        self.cfg = cfg

        # 初始化modules变量
        modules: Any = None

        modules = self._initialize_modules(cfg)

        # 只有在需要时才设置self.layers
        if cfg.projector_type != "low_high_split_mlp_gelu":
            if cfg.get("token_pooling", False):
                self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

            if cfg.get("conv_fusion_high_low_features", False):
                self.fusion_layer = nn.Linear(cfg.input_dim, cfg.input_dim)
            self.layers = modules

    def _initialize_modules(self, cfg):
        """初始化模块"""
        if cfg.projector_type == "identity":
            return nn.Identity()

        if cfg.projector_type == "linear":
            return nn.Linear(cfg.input_dim, cfg.n_embed)

        if cfg.projector_type == "mlp_gelu":
            return self._create_mlp_gelu(cfg)

        if cfg.projector_type == "normlayer_downsample_mlp_gelu":
            return self._create_normlayer_downsample_mlp_gelu(cfg)

        if cfg.projector_type == "downsample_mlp_gelu":
            return self._create_downsample_mlp_gelu(cfg)

        if cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            return self._create_low_high_hybrid_split_mlp_gelu(cfg)

        if cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            return self._create_hybrid_split_feature_mlp_gelu(cfg)

        if cfg.projector_type == "low_high_split_mlp_gelu":
            return self._create_low_high_split_mlp_gelu(cfg)

        raise ValueError(f"未知的投影器类型: {cfg.projector_type}")

    def _create_mlp_gelu(self, cfg):
        """创建mlp_gelu类型的模块"""
        mlp_depth = cfg.get("depth", 1)
        modules_list_1: list[nn.Linear | nn.GELU] = [nn.Linear(cfg.input_dim, cfg.n_embed)]
        for _ in range(1, mlp_depth):
            modules_list_1.append(nn.GELU())
            modules_list_1.append(nn.Linear(cfg.n_embed, cfg.n_embed))
        return nn.Sequential(*modules_list_1)

    def _create_normlayer_downsample_mlp_gelu(self, cfg):
        """创建normlayer_downsample_mlp_gelu类型的模块"""
        mlp_depth = cfg.get("depth", 1)
        mlp_ratio = cfg.get("mlp_ratio", 1)
        modules_list_2: list[nn.Module] = [
            nn.LayerNorm(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio),
            nn.Linear(
                cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                cfg.n_embed * mlp_ratio,
            ),
        ]
        for _ in range(1, mlp_depth - 1):
            modules_list_2.append(nn.GELU())
            modules_list_2.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
        modules_list_2.append(nn.GELU())
        modules_list_2.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
        return nn.Sequential(*modules_list_2)

    def _create_downsample_mlp_gelu(self, cfg):
        """创建downsample_mlp_gelu类型的模块"""
        mlp_depth = cfg.get("depth", 1)
        mlp_ratio = cfg.get("mlp_ratio", 1)
        modules_list_3: list[nn.Module] = [
            nn.Linear(
                cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                cfg.n_embed * mlp_ratio,
            )
        ]
        for _ in range(1, mlp_depth - 1):
            modules_list_3.append(nn.GELU())
            modules_list_3.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
        modules_list_3.append(nn.GELU())
        modules_list_3.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
        return nn.Sequential(*modules_list_3)

    def _create_low_high_hybrid_split_mlp_gelu(self, cfg):
        """创建low_high_hybrid_split_mlp_gelu类型的模块"""
        mlp_depth = cfg.get("depth", 1)
        self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
        self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

        modules_list_4: list[nn.Module] = []
        for _ in range(1, mlp_depth):
            modules_list_4.append(nn.GELU())
            modules_list_4.append(nn.Linear(cfg.n_embed, cfg.n_embed))
        return nn.Sequential(*modules_list_4)

    def _create_hybrid_split_feature_mlp_gelu(self, cfg):
        """创建hybrid_split_feature_mlp_gelu类型的模块"""
        mlp_depth = cfg.get("depth", 1)
        channel_div = cfg.get("channel_div", 0.5)
        self.high_up_proj = nn.Linear(cfg.input_dim[0], int(cfg.n_embed * channel_div))
        self.low_up_proj = nn.Linear(cfg.input_dim[1], cfg.n_embed - int(cfg.n_embed * channel_div))

        modules_list_5: list[nn.Module] = []
        for _ in range(1, mlp_depth):
            modules_list_5.append(nn.GELU())
            modules_list_5.append(nn.Linear(cfg.n_embed, cfg.n_embed))
        return nn.Sequential(*modules_list_5)

    def _create_low_high_split_mlp_gelu(self, cfg):
        """创建low_high_split_mlp_gelu类型的模块"""
        mlp_depth = cfg.get("depth", 1)
        modules_list_high: list[nn.Module] = []
        for _ in range(1, mlp_depth):
            modules_list_high.append(nn.GELU())
            modules_list_high.append(nn.Linear(cfg.n_embed // 2, cfg.n_embed // 2))

        modules_list_low: list[nn.Module] = []
        for _ in range(1, mlp_depth):
            modules_list_low.append(nn.GELU())
            modules_list_low.append(nn.Linear(cfg.n_embed // 2, cfg.n_embed // 2))

        self.high_layers = nn.Sequential(*modules_list_high)
        self.low_layers = nn.Sequential(*modules_list_low)
        # 对于这种类型，不需要设置self.layers
        return None

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        x = self._handle_token_pooling(x)
        x = self._handle_conv_fusion(x)
        x = self._handle_projector_types(x)
        x = self._handle_downsample_types(x)
        return x

    def _handle_token_pooling(self, x):
        """处理token pooling"""
        if self.cfg.get("token_pooling", False):
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)
        return x

    def _handle_conv_fusion(self, x):
        """处理卷积融合"""
        if self.cfg.get("conv_fusion_high_low_features", False):
            x = self.fusion_layer(x[:, 0]) + x[:, 1]
        return x

    def _handle_projector_types(self, x):
        """处理投影器类型"""
        if self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            high_x = x[..., : self.cfg.input_dim[0]]
            low_x = x[..., self.cfg.input_dim[0] :]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.cfg.projector_type == "low_high_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
            return x

        return x

    def _handle_downsample_types(self, x):
        """处理下采样类型"""
        if (
            self.cfg.projector_type == "downsample_mlp_gelu"
            or self.cfg.projector_type == "normlayer_downsample_mlp_gelu"
        ):
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """计算填充"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4合1拼接"""
            x = x.permute(0, 3, 1, 2)  # B:批次大小, C:通道数, H:高度, W:宽度
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            )  # B:批次大小, C*4:通道数乘以4, HW // 4:高度宽度除以4
            x = x.permute(0, 2, 1)

        # 只有在需要时才返回self.layers(x)
        if self.cfg.projector_type != "low_high_split_mlp_gelu":
            return self.layers(x)
        return x

    @staticmethod
    def get_flops_per_sample(cfg):
        """
        计算每个样本的浮点运算次数

        Args:
            cfg: 配置对象

        Returns:
            每个样本的浮点运算次数
        """
        if cfg.projector_type == "linear":
            fwd = 2 * cfg.input_dim * cfg.n_embed

        elif "mlp_gelu" in cfg.projector_type:
            mlp_depth = cfg.get("depth", 1)
            downsample_ratio = cfg.get("downsample_ratio", 1)
            input_dim = sum(cfg.input_dim) if isinstance(cfg.input_dim, list) else cfg.input_dim
            input_dim = input_dim * downsample_ratio * downsample_ratio
            fwd = 2 * input_dim * cfg.n_embed + (mlp_depth - 1) * 2 * cfg.n_embed * cfg.n_embed
        else:
            fwd = 0

        return fwd * 3
