#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型初始化器
统一管理模型和分词器的初始化
"""

import torch
from typing import Optional, Tuple
from pathlib import Path

# 导入日志模块
from src.core.logging import get_logger

# 导入MPS优化工具
from src.core.utils.mps_utils import get_optimal_device, is_mps_device

# 获取日志记录器
logger = get_logger()

class ModelInitializer:
    """统一的模型初始化器"""
    
    @staticmethod
    def initialize_transformers_model_and_tokenizer(model_path: str, trust_remote_code: bool = True) -> Tuple[object, object]:
        """
        初始化Transformers模型和分词器
        
        Args:
            model_path: 模型路径
            trust_remote_code: 是否信任远程代码
            
        Returns:
            (模型, 分词器) 元组
        """
        try:
            # 加载tokenizer
            from transformers import AutoTokenizer
            logger.info("开始加载tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)  # 不使用远程代码
            logger.info("tokenizer加载完成")
            
            # 加载模型 - 实现一个简单的自定义模型
            logger.info("开始加载模型")
            
            # 导入必要的模块
            from transformers import LlamaModel, LlamaConfig
            import torch.nn as nn
            
            # 创建一个简单的OCR模型类，基于原始DeepSeek-OCR项目的逻辑
            class SimpleOCRModel(nn.Module):
                def __init__(self, model_path):
                    super().__init__()
                    # 直接加载基础的LlamaModel
                    self.model = LlamaModel.from_pretrained(model_path, trust_remote_code=False)
                    # 添加语言模型头
                    self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
                    self.config = self.model.config
                    
                    # 初始化视觉编码器和投影器（基于原始DeepSeek-OCR项目）
                    from src.core.deepencoder.sam_vary_sdpa import build_sam_vit_b
                    from src.core.deepencoder.clip_sdpa import build_clip_l
                    from src.core.deepencoder.build_linear import MlpProjector
                    from addict import Dict
                    
                    self.sam_model = build_sam_vit_b()
                    self.vision_model = build_clip_l()
                    
                    n_embed = 1280
                    self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
                    self.tile_tag = "2D"  # 使用默认值
                    self.global_view_pos = True  # 使用默认值
                    
                    # special token for image token sequence format
                    embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
                    self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
                    self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
                    
                    logger.info("创建基于原始DeepSeek-OCR逻辑的OCR模型完成")
                
                def get_input_embeddings(self, input_ids, pixel_values=None, images_crop=None, images_spatial_crop=None):
                    """
                    获取输入嵌入向量，包括文本和图像的嵌入
                    
                    Args:
                        input_ids: 输入ID张量
                        pixel_values: 像素值张量
                        images_crop: 图像裁剪张量
                        images_spatial_crop: 图像空间裁剪张量
                        
                    Returns:
                        输入嵌入向量张量
                    """
                    # 先获取文本嵌入 - LlamaModel.get_input_embeddings()不接受参数
                    embed_tokens = self.model.get_input_embeddings()
                    text_embeds = embed_tokens(input_ids)
                    
                    # 如果有图像输入，处理图像并融合嵌入
                    if pixel_values is not None and images_crop is not None:
                        try:
                            # 使用视觉编码器处理图像
                            with torch.no_grad():
                                # 检查设备类型
                                device = next(self.parameters()).device
                                is_mps = device.type == "mps"
                                
                                # MPS设备上的预处理优化
                                if is_mps:
                                    # 确保输入张量的批次大小一致
                                    batch_size = min(pixel_values.shape[0], images_crop.shape[0])
                                    if pixel_values.shape[0] != batch_size:
                                        pixel_values = pixel_values[:batch_size]
                                        logger.debug(f"MPS设备调整pixel_values批次大小为: {batch_size}")
                                    if images_crop.shape[0] != batch_size:
                                        images_crop = images_crop[:batch_size]
                                        logger.debug(f"MPS设备调整images_crop批次大小为: {batch_size}")
                                    
                                    # 确保数据类型为float32以提高MPS兼容性
                                    pixel_values = pixel_values.to(torch.float32)
                                    images_crop = images_crop.to(torch.float32)
                                
                                # SAM模型处理图像
                                sam_features = self.sam_model(pixel_values)
                                
                                # 在MPS设备上，确保sam_features和images_crop的尺寸兼容
                                if is_mps:
                                    # 检查并调整sam_features的尺寸以匹配images_crop
                                    if sam_features.shape[0] != images_crop.shape[0]:
                                        logger.warning(f"MPS设备上sam_features批次大小({sam_features.shape[0]})与images_crop批次大小({images_crop.shape[0]})不匹配，进行调整")
                                        min_batch_size = min(sam_features.shape[0], images_crop.shape[0])
                                        sam_features = sam_features[:min_batch_size]
                                        images_crop = images_crop[:min_batch_size]
                                
                                # CLIP模型处理图像，需要传入patch_embeds
                                clip_features = self.vision_model(images_crop, sam_features)
                                
                                # 使用投影器将图像特征投影到文本嵌入空间
                                projected_features = self.projector(clip_features)
                                
                                # MPS设备上的特征优化
                                if is_mps:
                                    # 限制图像特征序列长度以减少内存使用
                                    max_tokens = 100
                                    if projected_features.shape[1] > max_tokens:
                                        # 均匀采样减少token数量
                                        step = projected_features.shape[1] // max_tokens + 1
                                        projected_features = projected_features[:, ::step, :]
                                        logger.debug(f"MPS设备采样减少图像token，新形状: {projected_features.shape}")
                                    
                                    # 如果嵌入维度过大，进行降维
                                    if projected_features.shape[2] > 1024:
                                        if not hasattr(self, 'mps_dim_reducer'):
                                            self.mps_dim_reducer = torch.nn.Linear(projected_features.shape[2], 1024).to(device)
                                        projected_features = self.mps_dim_reducer(projected_features)
                                        logger.debug(f"MPS设备降维图像特征，新形状: {projected_features.shape}")
                                
                                # 获取图像token的位置
                                # 从tokenizer获取图像token ID
                                image_token_id = 32000  # 默认值，如果无法从tokenizer获取
                                try:
                                    # 尝试从tokenizer获取图像token ID
                                    if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'vocab') and '<image>' in self.tokenizer.vocab:
                                        image_token_id = self.tokenizer.vocab['<image>']
                                except:
                                    # 如果获取失败，使用默认值
                                    pass
                                
                                # 找到所有图像token的位置
                                image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
                                
                                # 创建新的嵌入向量
                                inputs_embeds = text_embeds.clone()
                                
                                # 改进的图像token处理逻辑
                                if len(image_positions[0]) > 0 and projected_features.shape[0] > 0:
                                    # 获取图像特征的形状
                                    batch_size, num_image_tokens, embed_dim = projected_features.shape
                                    
                                    # 获取当前批次的图像token位置
                                    batch_image_positions = {}
                                    for batch_idx, seq_idx in zip(*image_positions):
                                        if batch_idx not in batch_image_positions:
                                            batch_image_positions[batch_idx] = []
                                        batch_image_positions[batch_idx].append(seq_idx)
                                    
                                    # 为每个批次处理图像token替换
                                    for batch_idx in range(batch_size):
                                        if batch_idx in batch_image_positions:
                                            positions = batch_image_positions[batch_idx]
                                            # 替换图像token位置的嵌入
                                            for i, seq_idx in enumerate(positions):
                                                if i < num_image_tokens:
                                                    inputs_embeds[batch_idx, seq_idx] = projected_features[batch_idx, i]
                                        else:
                                            # 如果该批次没有图像token，在序列开头添加图像特征
                                            if projected_features.shape[1] > 0:
                                                # 在序列开头插入图像特征
                                                image_embeds = projected_features[batch_idx:min(batch_idx+1, projected_features.shape[0])]
                                                if len(image_embeds) > 0:
                                                    inputs_embeds[batch_idx] = torch.cat([image_embeds[0], inputs_embeds[batch_idx]], dim=0)
                                else:
                                    # 如果没有找到图像token位置，但仍有图像特征，在序列开头添加图像特征
                                    if projected_features.shape[0] > 0:
                                        batch_size = min(projected_features.shape[0], inputs_embeds.shape[0])
                                        for batch_idx in range(batch_size):
                                            # 添加图像特征到序列开头
                                            if projected_features.shape[1] > 0:
                                                image_embeds = projected_features[batch_idx]
                                                inputs_embeds[batch_idx] = torch.cat([image_embeds, inputs_embeds[batch_idx]], dim=0)
                            
                            return inputs_embeds
                        except Exception as e:
                            logger.error(f"处理图像特征时发生错误: {str(e)}")
                            import traceback
                            logger.error(f"错误堆栈: {traceback.format_exc()}")
                            # 如果图像处理失败，只返回文本嵌入
                            return text_embeds
                    else:
                        return text_embeds
                
                def forward(self, input_ids=None, attention_mask=None, pixel_values=None, images_crop=None, 
                           images_spatial_crop=None, **kwargs):
                    """
                    前向传播，支持多模态输入
                    
                    Args:
                        input_ids: 输入ID张量
                        attention_mask: 注意力掩码
                        pixel_values: 像素值张量
                        images_crop: 图像裁剪张量
                        images_spatial_crop: 图像空间裁剪张量
                        **kwargs: 其他参数
                        
                    Returns:
                        模型输出
                    """
                    # 获取多模态嵌入
                    inputs_embeds = self.get_input_embeddings(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        images_crop=images_crop,
                        images_spatial_crop=images_spatial_crop
                    )
                    
                    # 如果有图像嵌入，需要调整注意力掩码
                    if pixel_values is not None and images_crop is not None and attention_mask is not None:
                        # 计算图像特征的数量
                        with torch.no_grad():
                            # 使用SAM模型获取图像特征
                            sam_features = self.sam_model(pixel_values)
                            # 使用CLIP模型处理图像以获取特征数量
                            clip_features = self.vision_model(images_crop, sam_features)
                            # 使用投影器将图像特征投影到文本嵌入空间
                            projected_features = self.projector(clip_features)
                            num_image_tokens = projected_features.shape[1]
                        
                        # 检查是否有图像token在input_ids中
                        image_token_id = 32000  # 默认值，如果无法从tokenizer获取
                        try:
                            # 尝试从tokenizer获取图像token ID
                            if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'vocab') and '<image>' in self.tokenizer.vocab:
                                image_token_id = self.tokenizer.vocab['<image>']
                        except:
                            # 如果获取失败，使用默认值
                            pass
                        
                        # 检查input_ids中是否有图像token
                        has_image_tokens = (input_ids == image_token_id).any().item() if input_ids is not None else False
                        
                        # 如果没有图像token，但有图像特征，需要在注意力掩码开头添加图像注意力
                        if not has_image_tokens:
                            # 为图像特征添加注意力掩码，添加到开头
                            num_image_tokens = image_embeds.shape[1] if image_embeds is not None else 0
                            image_attention = torch.ones(
                                (attention_mask.shape[0], num_image_tokens), 
                                dtype=attention_mask.dtype, 
                                device=attention_mask.device
                            )
                            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
                        else:
                            # 如果有图像token，在注意力掩码末尾添加图像注意力
                            num_image_tokens = image_embeds.shape[1] if image_embeds is not None else 0
                            image_attention = torch.ones(
                                (attention_mask.shape[0], num_image_tokens), 
                                dtype=attention_mask.dtype, 
                                device=attention_mask.device
                            )
                            attention_mask = torch.cat([attention_mask, image_attention], dim=1)
                    
                    # 使用语言模型处理嵌入
                    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
                    logits = self.lm_head(outputs.last_hidden_state)
                    return type('obj', (object,), {'logits': logits, 'last_hidden_state': outputs.last_hidden_state})
                    
                def generate(self, input_ids, attention_mask=None, pixel_values=None, images_crop=None, 
                            images_spatial_crop=None, **kwargs):
                    """
                    使用加载的模型进行真实的文本生成，支持多模态输入
                    
                    Args:
                        input_ids: 输入的token ids
                        attention_mask: 注意力掩码
                        pixel_values: 像素值张量
                        images_crop: 图像裁剪张量
                        images_spatial_crop: 图像空间裁剪张量
                        **kwargs: 其他参数
                    
                    Returns:
                        生成的token ids序列
                    """
                    logger.info("使用真实模型进行文本生成")
                    
                    # 确定设备
                    if torch.backends.mps.is_available():
                        device = torch.device('mps')
                    elif torch.cuda.is_available():
                        device = torch.device('cuda')
                    else:
                        device = torch.device('cpu')
                    
                    # 确保输入在正确的设备上
                    if input_ids is not None:
                        input_ids = input_ids.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(device)
                    if images_crop is not None:
                        images_crop = images_crop.to(device)
                    if images_spatial_crop is not None:
                        images_spatial_crop = images_spatial_crop.to(device)
                    
                    # 设置生成参数
                    max_new_tokens = kwargs.get('max_new_tokens', 8192)  # 恢复到原始设置，在CUDA设备上可以获得完整结果
                    max_length = kwargs.get('max_length', max_new_tokens + input_ids.shape[1] if input_ids is not None else 100)
                    temperature = kwargs.get('temperature', 0.7)
                    top_k = kwargs.get('top_k', 50)
                    top_p = kwargs.get('top_p', 0.95)
                    
                    # 如果没有输入，使用模型的bos_token作为起始
                    if input_ids is None:
                        if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'bos_token_id'):
                            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
                        else:
                            # 如果没有bos_token_id，使用空输入
                            logger.warning("分词器没有bos_token_id，使用空输入")
                            return torch.tensor([[]], device=device)
                    
                    # 生成文本
                    output = input_ids.clone()
                    self.eval()  # 设置为评估模式
                    
                    # 预处理图像特征（只在第一次处理）
                    image_embeds = None
                    if pixel_values is not None and images_crop is not None:
                        try:
                            with torch.no_grad():
                                logger.info("预处理图像特征...")
                                
                                # 使用SAM模型获取图像特征
                                sam_features = self.sam_model(pixel_values)
                                # 使用CLIP模型处理图像以获取特征数量
                                clip_features = self.vision_model(images_crop, sam_features)
                                # 拼接CLIP特征和SAM特征
                                concat_features = torch.cat((clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)), dim=-1)
                                # 使用投影器将拼接特征投影到文本嵌入空间
                                image_embeds = self.projector(concat_features)
                                logger.info(f"图像特征预处理完成，形状: {image_embeds.shape}")
                        except Exception as e:
                            logger.error(f"图像特征预处理失败: {str(e)}")
                            import traceback
                            logger.error(f"错误堆栈: {traceback.format_exc()}")
                            image_embeds = None
                    
                    # 使用贪婪解码进行简单的生成
                    try:
                        with torch.no_grad():
                            # 循环生成token，直到达到最大新token数或遇到结束token
                            for i in range(max_new_tokens):
                                logger.debug(f"生成第 {i+1} 个token")
                                
                                # 获取文本嵌入
                                inputs_embeds = self.get_input_embeddings(input_ids)
                                
                                # 如果有图像嵌入，需要调整注意力掩码
                                if image_embeds is not None and attention_mask is not None:
                                    # 检查是否有图像token在input_ids中
                                    image_token_id = 32000  # 默认值，如果无法从tokenizer获取
                                    try:
                                        # 尝试从tokenizer获取图像token ID
                                        if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'vocab') and '<image>' in self.tokenizer.vocab:
                                            image_token_id = self.tokenizer.vocab['<image>']
                                    except:
                                        # 如果获取失败，使用默认值
                                        pass
                                    
                                    # 检查input_ids中是否有图像token
                                    has_image_tokens = (input_ids == image_token_id).any().item() if input_ids is not None else False
                                    
                                    # 如果有图像token，将图像嵌入与文本嵌入结合
                                    if has_image_tokens:
                                        # 找到图像token的位置
                                        image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[1]
                                        
                                        # 为每个图像token位置插入图像嵌入
                                        if len(image_token_positions) > 0:
                                            # 只在第一次生成时插入图像嵌入
                                            if i == 0:
                                                # 创建新的嵌入序列
                                                new_embeds = []
                                                new_attention_mask = []
                                                
                                                # 遍历input_ids，将图像token替换为图像嵌入
                                                prev_pos = 0
                                                for pos in image_token_positions:
                                                    # 添加图像token之前的文本嵌入
                                                    new_embeds.append(inputs_embeds[:, prev_pos:pos, :])
                                                    new_attention_mask.append(attention_mask[:, prev_pos:pos])
                                                    
                                                    # 添加图像嵌入
                                                    batch_size = inputs_embeds.shape[0]
                                                    new_embeds.append(image_embeds)
                                                    new_attention_mask.append(torch.ones((batch_size, image_embeds.shape[1]), device=device))
                                                    
                                                    prev_pos = pos + 1  # 跳过图像token
                                                
                                                # 添加剩余的文本嵌入
                                                new_embeds.append(inputs_embeds[:, prev_pos:, :])
                                                new_attention_mask.append(attention_mask[:, prev_pos:])
                                                
                                                # 合并所有嵌入和注意力掩码
                                                inputs_embeds = torch.cat(new_embeds, dim=1)
                                                attention_mask = torch.cat(new_attention_mask, dim=1)
                                    else:
                                        # 如果没有图像token，在开头添加图像嵌入
                                        if i == 0:
                                            batch_size = inputs_embeds.shape[0]
                                            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
                                            image_attention = torch.ones((batch_size, image_embeds.shape[1]), device=device)
                                            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
                                
                                # 使用语言模型处理嵌入
                                outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                                logits = self.lm_head(outputs.last_hidden_state)
                                
                                # 获取最后一个token的logits
                                last_token_logits = logits[:, -1, :]
                                
                                # 使用贪婪解码选择下一个token
                                next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
                                
                                # 检查是否是结束token
                                if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                                    logger.info("遇到结束token，停止生成")
                                    break
                                
                                # 将新token添加到输出中
                                output = torch.cat([output, next_token], dim=1)
                                
                                # 更新attention mask
                                if attention_mask is not None:
                                    new_mask = torch.ones((attention_mask.size(0), 1), device=device)
                                    attention_mask = torch.cat([attention_mask, new_mask], dim=1)
                                
                                # 图像特征只在第一次前向传播时使用，后续不再使用
                                # 但是我们需要保持input_ids的长度与attention_mask一致
                                image_embeds = None  # 确保后续循环不再使用图像嵌入
                    except Exception as e:
                        logger.error(f"生成过程中出现错误: {str(e)}")
                        import traceback
                        logger.error(f"错误堆栈: {traceback.format_exc()}")
                        # 如果生成失败，返回原始输入或空输出
                        return input_ids if input_ids is not None else torch.tensor([[]], device=device)
                    
                    logger.info(f"生成完成，返回模型生成的token序列，长度: {output.shape[1]}")
                    return output
                    
                def infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False):
                    """
                    OCR推理方法 - 基于原始DeepSeek-OCR项目的推理逻辑
                    
                    Args:
                        tokenizer: 分词器
                        prompt: 提示词
                        image_file: 图像文件路径
                        output_path: 输出路径
                        base_size: 基础尺寸
                        image_size: 图像尺寸
                        crop_mode: 是否启用裁剪模式
                        test_compress: 是否测试压缩
                        save_results: 是否保存结果
                        
                    Returns:
                        OCR结果文本
                    """
                    import torch
                    from PIL import Image
                    # 导入MPS优化工具
                    from src.core.utils.mps_utils import is_mps_device, get_optimal_device, optimize_tensor_for_mps
                    # 使用项目中的图像处理模块
                    from src.core.process.image_process import DeepseekOCRProcessor
                    
                    # 确保tokenizer对象已设置到模型中
                    if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                        self.tokenizer = tokenizer
                    
                    # 确保模型有tokenizer对象，以便在get_input_embeddings方法中可以访问
                    if not hasattr(self.model, 'tokenizer') or self.model.tokenizer is None:
                        self.model.tokenizer = tokenizer
                    
                    try:
                        logger.debug(f"开始infer方法，image_file: {image_file}")
                        # 加载图像
                        image = Image.open(image_file).convert('RGB')
                        logger.debug(f"图像加载完成，尺寸: {image.size}")
                        
                        # 处理图像
                        # 使用项目中的图像处理模块
                        from src.core.process.image_process import DeepseekOCRProcessor
                        processor = DeepseekOCRProcessor(tokenizer=tokenizer)
                        logger.debug("DeepseekOCRProcessor初始化完成")
                        logger.debug("即将调用tokenize_with_images方法")
                        processed_data = processor.tokenize_with_images(
                            images=[image],
                            bos=True,
                            eos=True,
                            cropping=crop_mode
                        )
                        logger.debug(f"tokenize_with_images方法调用完成，processed_data类型: {type(processed_data)}")
                        
                        # 提取处理后的数据
                        if processed_data is None or len(processed_data) == 0 or processed_data[0] is None or len(processed_data[0]) < 7:
                            raise ValueError("图像处理失败，未生成有效的输入数据")
                        
                        # 根据原始仓库的数据结构正确提取元素（索引0-6）
                        input_ids = processed_data[0][0]
                        pixel_values = processed_data[0][1]
                        images_crop = processed_data[0][2]
                        images_seq_mask = processed_data[0][3]
                        images_spatial_crop = processed_data[0][4]
                        num_image_tokens = processed_data[0][5]
                        image_shapes = processed_data[0][6]
                        
                        # 确保数据在正确的设备上
                        device = next(self.parameters()).device
                        
                        # 移动张量到设备
                        input_ids = input_ids.to(device)
                        pixel_values = pixel_values.to(device)
                        images_crop = images_crop.to(device)
                        images_spatial_crop = images_spatial_crop.to(device)
                        
                        # 生成结果
                        with torch.no_grad():
                            # 构造注意力掩码
                            attention_mask = torch.ones_like(input_ids)
                            
                            # 获取生成配置参数
                            generation_config = {
                                "max_new_tokens": 8192,  # 恢复到原始设置，在CUDA设备上可以获得完整结果
                                "do_sample": False,
                                "temperature": 1.0,
                                "top_p": 1.0
                            }
                            
                            generate_kwargs = {
                                "input_ids": input_ids,
                                "max_new_tokens": generation_config.get("max_new_tokens", 8192),
                                "do_sample": generation_config.get("do_sample", False),
                                "pad_token_id": tokenizer.eos_token_id if tokenizer is not None else 0,
                                "attention_mask": attention_mask,
                                # 传递图像特征给模型
                                "pixel_values": pixel_values,
                                "images_crop": images_crop,
                                "images_spatial_crop": images_spatial_crop
                            }
                            
                            # 只有在do_sample为True时才添加temperature和top_p参数
                            if generate_kwargs["do_sample"]:
                                generate_kwargs["temperature"] = generation_config.get("temperature", 1.0)
                                generate_kwargs["top_p"] = generation_config.get("top_p", 1.0)
                            
                            # 对于Transformers模式，需要先通过视觉编码器处理图像
                            # 这里我们简化处理，直接使用语言模型生成文本
                            # 在实际应用中，应该将图像特征通过视觉编码器和投影器处理后
                            # 再与文本嵌入结合输入到语言模型中
                            
                            # 尝试生成结果
                            logger.debug("开始模型生成")
                            outputs = self.generate(**generate_kwargs)
                            logger.debug("模型生成完成")
                            
                            # 解码输出
                            logger.debug("开始解码输出")
                            if tokenizer is not None:
                                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                logger.debug("解码完成")
                                return result
                            else:
                                raise RuntimeError("解码失败：缺少tokenizer")
                                
                    except Exception as e:
                        logger.error(f"图像处理过程中发生错误: {str(e)}")
                        import traceback
                        logger.error(f"错误堆栈: {traceback.format_exc()}")
                        raise RuntimeError(f"图像处理失败: {str(e)}") from e
            
            # 创建模型实例
            model = SimpleOCRModel(model_path)
            
            # 将tokenizer对象设置到模型中，确保在get_input_embeddings方法中可以访问
            model.tokenizer = tokenizer
            
            # 移动模型到适当的设备
            device = get_optimal_device()
            logger.info(f"使用设备: {device.type}")
            
            model.to(device)
            logger.info("模型加载完成")
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"初始化Transformers模型和分词器失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化Transformers模型和分词器失败: {str(e)}")
    
    @staticmethod
    def initialize_vllm_model(model_path: str, prompt: Optional[str] = None) -> object:
        """
        初始化vLLM模型
        
        Args:
            model_path: 模型路径
            prompt: 提示词
            
        Returns:
            vLLM模型实例
        """
        try:
            from src.core.models.deepseek_ocr_model import DeepseekOCRForCausalLM
            # vLLM相关导入（延迟导入，避免在不支持的平台上报错）
            try:
                from vllm.model_executor.models.registry import ModelRegistry
                from vllm import LLM, SamplingParams
            except ImportError:
                # 在不支持vLLM的平台上设置占位符
                ModelRegistry = None
                LLM = object
                SamplingParams = object
                raise RuntimeError("vLLM未安装或不支持当前平台")
            
            # 注册模型 - 使用正确的模型类型
            ModelRegistry.register_model("DeepseekVLV2ForCausalLM", DeepseekOCRForCausalLM)
            
            # 初始化vLLM模型
            llm = LLM(
                model=model_path,
                hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
                block_size=256,
                enforce_eager=False,
                trust_remote_code=True, 
                max_model_len=8192,
                swap_space=0,
                max_num_seqs=100,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                disable_mm_preprocessor_cache=True
            )
            
            return llm
        except Exception as e:
            logger.error(f"初始化vLLM模型失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化vLLM模型失败: {str(e)}")
    
    @staticmethod
    def move_model_to_device(model: object, device: Optional[torch.device] = None) -> object:
        """
        将模型移到指定设备
        
        Args:
            model: 模型实例
            device: 目标设备，如果为None则自动选择最优设备
            
        Returns:
            移动后的模型实例
        """
        try:
            # 如果没有指定设备，自动选择最优设备
            if device is None:
                device = get_optimal_device()
            
            logger.info(f"将模型移到 {device.type.upper()} 设备")
            if hasattr(model, 'to'):
                model = model.to(device)
            if hasattr(model, 'eval'):
                model = model.eval()
            
            # 根据设备类型选择合适的数据类型
            use_bfloat16 = device.type == "cuda"  # 仅在CUDA设备上使用bfloat16
            if use_bfloat16:
                # 检查模型是否有to方法
                if hasattr(model, 'to') and callable(getattr(model, 'to', None)):
                    model = model.to(torch.bfloat16)
            
            logger.debug(f"模型已移到设备: {device}")
            return model
        except Exception as e:
            logger.error(f"将模型移到设备失败: {str(e)}")
            raise RuntimeError(f"将模型移到设备失败: {str(e)}")