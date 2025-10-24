#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-OCR核心模块
提供视觉编码器和图像处理功能
"""

# 标准库导入
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

# 项目内部导入
from src.core.deepencoder.sam_vary_sdpa import build_sam_vit_b
from src.core.deepencoder.clip_sdpa import build_clip_l
from src.core.deepencoder.build_linear import MlpProjector
from addict import Dict

# 配置导入
from src.core.config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS

# 常量定义
_IMAGE_TOKEN = "<image>"


class DeepseekOCRVisionEncoder(nn.Module):
    """
    DeepSeek OCR视觉编码器
    负责处理图像并生成视觉嵌入向量
    """
    
    def __init__(self):
        """
        初始化视觉编码器
        """
        super().__init__()
        
        # 初始化视觉编码器和投影器
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        
        n_embed = 1280
        self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        self.tile_tag = "2D"
        self.global_view_pos = True
    
        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )
    
    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> torch.Tensor:
        """
        将像素值转换为嵌入向量
        
        Args:
            pixel_values: 像素值张量
            images_crop: 图像裁剪张量
            images_spatial_crop: 图像空间裁剪张量
            
        Returns:
            嵌入向量张量
        """

        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch = []

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # 根据设备类型决定是否使用bfloat16
                patches = images_crop[jdx][0]
                image_ori = pixel_values[jdx]
                
                # 检查设备类型，MPS上避免使用bfloat16以确保兼容性
                if patches.device.type != "mps":
                    patches = patches.to(torch.bfloat16)
                if image_ori.device.type != "mps":
                    image_ori = image_ori.to(torch.bfloat16)

                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:  # if all values = 0, no crop
                    local_features_1 = self.sam_model(patches)
                    local_features_2 = self.vision_model(patches, local_features_1)  

                    local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('PATCHES: ', local_features.shape)
                        print('=====================')

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)

                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(int(h), int(w), int(n_dim))

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(int(h), 1, int(n_dim))], dim=1
                    )

                    global_features = global_features.view(-1, int(n_dim))

                    local_features = local_features.view(int(height_crop_num), int(width_crop_num), int(h2), int(w2), int(n_dim2)).permute(0, 2, 1, 3, 4).reshape(int(height_crop_num)*int(h2), int(width_crop_num)*int(w2), int(n_dim2))
                    local_features = torch.cat(
                        [local_features, self.image_newline[None, None, :].expand(int(height_crop_num) * int(h2), 1, int(n_dim2))], dim=1
                    )
                    local_features = local_features.view(-1, int(n_dim2))

                    global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                
                else:
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('NO PATCHES')
                        print('=====================')

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    global_features = global_features.view(int(h), int(w), int(n_dim))

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(int(h), 1, int(n_dim))], dim=1
                    )

                    global_features = global_features.view(-1, int(n_dim))

                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                images_in_this_batch.append(global_local_features)

        return torch.stack(images_in_this_batch, dim=0)

    def process_image_input(
            self, image_input: List[torch.Tensor]) -> torch.Tensor:
        """
        处理图像输入
        
        Args:
            image_input: 图像输入数据，包含[pixel_values, images_crop, images_spatial_crop]
            
        Returns:
            处理后的图像张量
        """

        # image_input: [pixel_values, images_crop, images_spatial_crop]
        pixel_values = image_input[0]
        images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # 检查设备类型，MPS上避免使用bfloat16以确保兼容性
        if pixel_values.device.type != "mps":
            pixel_values = pixel_values.to(torch.bfloat16)
        if images_crop.device.type != "mps":
            images_crop = images_crop.to(torch.bfloat16)

        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values, images_crop=images_crop, images_spatial_crop=images_spatial_crop)

        return vision_features