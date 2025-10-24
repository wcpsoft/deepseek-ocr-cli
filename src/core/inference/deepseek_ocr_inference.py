#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR模型推理实现
"""

import math
import time
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any, List

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()

# 从模型文件导入必要的类和函数
# 修改导入以支持新的模型实现
try:
    from src.core.models.deepseek_ocr_model import DeepseekOCRForCausalLM, _IMAGE_TOKEN
except ImportError:
    # 如果原始模型不可用，使用新的实现
    from src.core.models.deepseek_ocr_impl import DeepseekOCRForCausalLM
    _IMAGE_TOKEN = "<image>"

# 导入模型适配器
from src.core.models.model_adapter import ModelAdapter

# 配置导入
from src.core.config.config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS, PROMPT


class DeepseekOCRInference:
    """
    DeepSeek OCR推理类
    处理模型推理相关的功能
    """
    
    def __init__(self, model: DeepseekOCRForCausalLM):
        """
        初始化推理类
        
        Args:
            model: DeepseekOCRForCausalLM模型实例
        """
        # 如果模型不是适配器，创建一个适配器
        if not hasattr(model, 'sam_model'):
            from src.core.models.model_adapter import ModelAdapter
            self.model = ModelAdapter(model)
        else:
            self.model = model
    
    def _parse_and_validate_image_input(self, **kwargs: object):
        """
        解析和验证图像输入
        
        Args:
            **kwargs: 输入参数
            
        Returns:
            解析后的图像输入或None
        """
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)


        if pixel_values is None or (isinstance(pixel_values, torch.Tensor) and torch.sum(pixel_values).item() == 0):
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                f"Got type: {type(pixel_values)}")

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                f"Got type: {type(images_spatial_crop)}")
            
            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image crop. "
                                f"Got type: {type(images_crop)}")

            return [pixel_values, images_crop, images_spatial_crop]


        raise AssertionError("This line should be unreachable.")
    
    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        将像素值转换为嵌入向量
        
        Args:
            pixel_values: 像素值张量
            images_crop: 图像裁剪张量
            images_spatial_crop: 图像空间裁剪张量
            
        Returns:
            嵌入向量张量列表
        """
        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch: List[torch.Tensor] = []

        # print(type(images_crop))
        # print(pixel_values.shape)

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # with torch.set_grad_enabled(False):
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
                    # P, C, H, W = patches.shape
                    # crop_flag = 1
                    if self.model.sam_model is not None:
                        # 确保sam_model是一个可调用的对象
                        if callable(self.model.sam_model):
                            local_features_1 = self.model.sam_model(patches)
                        elif hasattr(self.model.sam_model, 'forward'):
                            # 如果不是可调用的，尝试使用forward方法
                            local_features_1 = self.model.sam_model.forward(patches)
                        else:
                            # 如果既不是可调用的也没有forward方法，直接调用
                            local_features_1 = self.model.sam_model(patches)
                    else:
                        raise RuntimeError("SAM模型未初始化")
                    #TODO del patches 
                    # torch.compiler.cudagraph_mark_step_begin()
                    if self.model.vision_model is not None:
                        # 确保vision_model是一个可调用的对象
                        if callable(self.model.vision_model):
                            local_features_2 = self.model.vision_model(patches, local_features_1)
                        elif hasattr(self.model.vision_model, 'forward'):
                            # 如果不是可调用的，尝试使用forward方法
                            local_features_2 = self.model.vision_model.forward(patches, local_features_1)
                        else:
                            # 如果既不是可调用的也没有forward方法，直接调用
                            local_features_2 = self.model.vision_model(patches, local_features_1)
                    else:
                        raise RuntimeError("视觉模型未初始化")


                    if self.model.projector is not None:
                        local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        # 确保projector是一个可调用的对象
                        if callable(self.model.projector):
                            local_features = self.model.projector(local_features)
                        elif hasattr(self.model.projector, 'forward'):
                            # 如果不是可调用的，尝试使用forward方法
                            local_features = self.model.projector.forward(local_features)
                        else:
                            # 如果既不是可调用的也没有forward方法，直接调用
                            local_features = self.model.projector(local_features)
                    else:
                        raise RuntimeError("投影器未初始化")


                    if self.model.sam_model is not None:
                        # 确保sam_model是一个可调用的对象
                        if callable(self.model.sam_model):
                            global_features_1 = self.model.sam_model(image_ori)
                        elif hasattr(self.model.sam_model, 'forward'):
                            # 如果不是可调用的，尝试使用forward方法
                            global_features_1 = self.model.sam_model.forward(image_ori)
                        else:
                            # 如果既不是可调用的也没有forward方法，直接调用
                            global_features_1 = self.model.sam_model(image_ori)
                    else:
                        raise RuntimeError("SAM模型未初始化")
                    if self.model.vision_model is not None:
                        # 确保vision_model是一个可调用的对象
                        if callable(self.model.vision_model):
                            global_features_2 = self.model.vision_model(image_ori, global_features_1)
                        elif hasattr(self.model.vision_model, 'forward'):
                            # 如果不是可调用的，尝试使用forward方法
                            global_features_2 = self.model.vision_model.forward(image_ori, global_features_1)
                        else:
                            # 如果既不是可调用的也没有forward方法，直接调用
                            global_features_2 = self.model.vision_model(image_ori, global_features_1)
                    else:
                        raise RuntimeError("视觉模型未初始化") 
                    if self.model.projector is not None:
                        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        # 确保projector是一个可调用的对象
                        if callable(self.model.projector):
                            global_features = self.model.projector(global_features)
                        elif hasattr(self.model.projector, 'forward'):
                            # 如果不是可调用的，尝试使用forward方法
                            global_features = self.model.projector.forward(global_features)
                        else:
                            # 如果既不是可调用的也没有forward方法，直接调用
                            global_features = self.model.projector(global_features)
                    else:
                        raise RuntimeError("投影器未初始化")

                    if PRINT_NUM_VIS_TOKENS:
                        logger.debug('=====================')
                        logger.debug('BASE: %s' % str(global_features.shape))
                        logger.debug('PATCHES: %s' % str(local_features.shape))
                        logger.debug('=====================')

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)

                    width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])

                    global_features = global_features.view(h, w, n_dim)

                    # 确保image_newline不为None
                    if self.model.image_newline is not None:
                        global_features = torch.cat(
                            (global_features, self.model.image_newline[None, None, :].expand(int(h), 1, int(n_dim))), dim=1
                        )
                    else:
                        # 如果image_newline为None，创建一个零张量
                        zero_tensor = torch.zeros(int(h), 1, int(n_dim), device=global_features.device)
                        global_features = torch.cat((global_features, zero_tensor), dim=1)

                    global_features = global_features.view(-1, n_dim)

                    # 确保view_seperator不为None
                    if self.model.view_seperator is not None:
                        global_local_features = torch.cat([global_features, self.model.view_seperator[None, :]], dim=0)
                    else:
                        # 如果view_seperator为None，只返回global_features
                        global_local_features = global_features
                else:
                    if self.model.sam_model is not None:
                        if callable(self.model.sam_model):
                            global_features_1 = self.model.sam_model(image_ori)
                        elif hasattr(self.model.sam_model, 'forward'):
                            global_features_1 = self.model.sam_model.forward(image_ori)
                        else:
                            global_features_1 = self.model.sam_model(image_ori)
                    else:
                        raise RuntimeError("SAM模型未初始化")
                    if self.model.vision_model is not None:
                        if callable(self.model.vision_model):
                            global_features_2 = self.model.vision_model(image_ori, global_features_1)
                        elif hasattr(self.model.vision_model, 'forward'):
                            global_features_2 = self.model.vision_model.forward(image_ori, global_features_1)
                        else:
                            global_features_2 = self.model.vision_model(image_ori, global_features_1)
                    else:
                        raise RuntimeError("视觉模型未初始化") 
                    if self.model.projector is not None:
                        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        if callable(self.model.projector):
                            global_features = self.model.projector(global_features)
                        elif hasattr(self.model.projector, 'forward'):
                            global_features = self.model.projector.forward(global_features)
                        else:
                            global_features = self.model.projector(global_features)
                    else:
                        raise RuntimeError("投影器未初始化")

                    if PRINT_NUM_VIS_TOKENS:
                        logger.debug('=====================')
                        logger.debug('BASE: %s' % str(global_features.shape))
                        logger.debug('NO PATCHES')
                        logger.debug('=====================')

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    global_features = global_features.view(h, w, n_dim)

                    # 确保image_newline不为None
                    if self.model.image_newline is not None:
                        global_features = torch.cat(
                            [global_features, self.model.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )
                    else:
                        # 如果image_newline为None，创建一个零张量
                        zero_tensor = torch.zeros(h, 1, n_dim, device=global_features.device)
                        global_features = torch.cat((global_features, zero_tensor), dim=1)

                    global_features = global_features.view(-1, n_dim)

                    # 确保view_seperator不为None
                    if self.model.view_seperator is not None:
                        global_local_features = torch.cat([global_features, self.model.view_seperator[None, :]], dim=0)
                    else:
                        # 如果view_seperator为None，只返回global_features
                        global_local_features = global_features

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(self, image_input) -> List[torch.Tensor]:
        """
        处理图像输入
        
        Args:
            image_input: 图像输入数据
            
        Returns:
            处理后的图像张量列表
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

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values, images_crop = images_crop,  images_spatial_crop=images_spatial_crop)

        # local_total_time = time.time() - local_start

        # print('encoder_time: ', local_total_time)
        # exit()
        return vision_features

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[List[torch.Tensor]]:
        """
        获取多模态嵌入向量
        
        Args:
            **kwargs: 输入参数
            
        Returns:
            多模态嵌入向量或None
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        获取输入嵌入向量
        
        Args:
            input_ids: 输入ID张量
            multimodal_embeddings: 多模态嵌入向量
            
        Returns:
            输入嵌入向量张量
        """
        # 获取语言模型的输入嵌入
        if hasattr(self.model, 'language_model') and self.model.language_model is not None:
            # 获取语言模型的嵌入层
            embed_tokens = self.model.language_model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids)
        elif hasattr(self.model, 'get_input_embeddings'):
            embed_tokens = self.model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids)
        else:
            raise RuntimeError("无法获取输入嵌入")

        if multimodal_embeddings is not None:
            # 合并多模态嵌入
            if hasattr(self.model, 'merge_multimodal_embeddings'):
                inputs_embeds = self.model.merge_multimodal_embeddings(
                    input_ids, inputs_embeds, multimodal_embeddings,
                    self.model.image_token_id)
            # print(len(multimodal_embeddings))
            # print(input_ids.shape)
            # print(type(inputs_embeds))
            # print(inputs_embeds.shape)
            
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[object] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):
        """
        前向传播函数
        
        Args:
            input_ids: 输入ID张量
            positions: 位置张量
            intermediate_tensors: 中间张量
            inputs_embeds: 输入嵌入向量
            **kwargs: 其他参数
            
        Returns:
            前向传播结果
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                    vision_embeddings)
            input_ids = None

        # 通过语言模型进行前向传播
        if hasattr(self.model, 'language_model') and self.model.language_model is not None:
            hidden_states = self.model.language_model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds=inputs_embeds
            )
        else:
            # 如果没有language_model属性，直接调用模型
            hidden_states = self.model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds=inputs_embeds
            )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: object,
    ) -> Optional[torch.Tensor]:
        """
        计算logits
        
        Args:
            hidden_states: 隐藏状态张量
            sampling_metadata: 采样元数据
            
        Returns:
            logits张量或None
        """
        if hasattr(self.model, 'language_model') and self.model.language_model is not None:
            return self.model.language_model.compute_logits(hidden_states,
                                                        sampling_metadata)
        else:
            # 如果没有language_model属性，直接返回None
            return None

    def infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False):
        """
        推理方法，用于处理图像并生成OCR结果
        
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
            OCR结果
        """
        import torch
        from PIL import Image
        # 使用项目中的图像处理模块
        from src.core.process.image_process import DeepseekOCRProcessor
        
        try:
            logger.debug(f"开始infer方法，image_file: {image_file}")
            # 加载图像
            image = Image.open(image_file).convert('RGB')
            logger.debug(f"图像加载完成，尺寸: {image.size}")
            
            # 处理图像
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
            
            # 添加调试信息
            logger.debug(f"processed_data类型: {type(processed_data)}")
            logger.debug(f"processed_data长度: {len(processed_data) if processed_data else 'N/A'}")
            
            # 提取处理后的数据
            logger.debug("开始检查processed_data的有效性")
            if processed_data is None:
                raise ValueError("processed_data为None")
            if len(processed_data) == 0:
                raise ValueError("processed_data为空列表")
            if processed_data[0] is None:
                raise ValueError("processed_data[0]为None")
            
            logger.debug(f"processed_data[0]类型: {type(processed_data[0])}")
            logger.debug(f"processed_data[0]长度: {len(processed_data[0]) if processed_data[0] else 'N/A'}")
            
            if len(processed_data) > 0 and processed_data[0] is not None and len(processed_data[0]) >= 7:
                # 注意：这里的索引需要根据实际的数据结构进行调整
                # 确保processed_data[0]有足够的元素
                if len(processed_data[0]) < 7:
                    raise ValueError(f"处理后的数据结构不完整，期望至少7个元素，实际只有{len(processed_data[0])}个元素")
                
                logger.debug("开始提取数据元素")
                # 根据原始仓库的数据结构正确提取元素（索引0-6）
                logger.debug("提取input_ids")
                input_ids = processed_data[0][0]
                logger.debug(f"input_ids提取完成，类型: {type(input_ids)}")
                logger.debug("提取pixel_values")
                pixel_values = processed_data[0][1]
                logger.debug(f"pixel_values提取完成，类型: {type(pixel_values)}")
                logger.debug("提取images_crop")
                images_crop = processed_data[0][2]
                logger.debug(f"images_crop提取完成，类型: {type(images_crop)}")
                logger.debug("提取images_seq_mask")
                images_seq_mask = processed_data[0][3]
                logger.debug(f"images_seq_mask提取完成，类型: {type(images_seq_mask)}")
                logger.debug("提取images_spatial_crop")
                images_spatial_crop = processed_data[0][4]
                logger.debug(f"images_spatial_crop提取完成，类型: {type(images_spatial_crop)}")
                logger.debug("提取num_image_tokens")
                num_image_tokens = processed_data[0][5]
                logger.debug(f"num_image_tokens提取完成，类型: {type(num_image_tokens)}")
                logger.debug("提取image_shapes")
                image_shapes = processed_data[0][6]
                logger.debug(f"image_shapes提取完成，类型: {type(image_shapes)}")
            else:
                raise ValueError("图像处理失败，未生成有效的输入数据")
                
            # 验证提取的元素不为None
            logger.debug("开始验证提取的元素")
            if input_ids is None:
                raise ValueError("input_ids为None")
            if pixel_values is None:
                raise ValueError("pixel_values为None")
            if images_crop is None:
                raise ValueError("images_crop为None")
            if images_seq_mask is None:
                raise ValueError("images_seq_mask为None")
            if images_spatial_crop is None:
                raise ValueError("images_spatial_crop为None")
            if num_image_tokens is None:
                raise ValueError("num_image_tokens为None")
            if image_shapes is None:
                raise ValueError("image_shapes为None")
                
            # 确保数据在正确的设备上
            try:
                logger.debug("开始获取设备信息")
                device = next(self.model.parameters()).device if len(list(self.model.parameters())) > 0 else self.model.device
            except Exception:
                # 如果无法获取参数设备，使用默认设备
                device = torch.device("cpu")
                
            logger.debug(f"设备信息获取完成: {device}")
            # 确保张量在正确的设备上
            try:
                logger.debug("开始将张量移动到设备")
                input_ids = input_ids.to(device)
                pixel_values = pixel_values.to(device)
                images_crop = images_crop.to(device)
                images_spatial_crop = images_spatial_crop.to(device)
            except Exception as e:
                raise ValueError(f"无法将张量移动到设备{device}: {e}")
                
            # 在MPS设备上避免使用bfloat16，使用float32以确保兼容性
            if device.type == "mps":
                # MPS设备上使用float32以确保兼容性
                try:
                    logger.debug("在MPS设备上转换数据类型")
                    pixel_values = pixel_values.to(torch.float32)
                    images_crop = images_crop.to(torch.float32)
                except Exception as e:
                    logger.warning(f"在MPS设备上转换数据类型时出错: {e}")
            else:
                # 其他设备上可以使用bfloat16（如果支持）
                try:
                    if torch.cuda.is_bf16_supported():
                        logger.debug("在CUDA设备上转换数据类型")
                        pixel_values = pixel_values.to(torch.bfloat16)
                        images_crop = images_crop.to(torch.bfloat16)
                except Exception as e:
                    logger.warning(f"转换数据类型时出错: {e}")
                
            # 生成结果
            with torch.no_grad():
                # 构造注意力掩码
                try:
                    logger.debug("开始构造注意力掩码")
                    attention_mask = torch.ones_like(input_ids)
                except Exception as e:
                    raise ValueError(f"创建注意力掩码时出错: {e}")
                    
                # 获取模型的生成配置参数
                try:
                    logger.debug("开始获取生成配置")
                    generation_config = self.model._get_generation_config()
                except Exception as e:
                    logger.warning(f"获取生成配置时出错，使用默认配置: {e}")
                    generation_config = {
                        "max_new_tokens": 8192,
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_p": 1.0
                    }

                logger.debug(f"生成配置获取完成: {generation_config}")
                generate_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": generation_config.get("max_new_tokens", 8192),
                    "do_sample": generation_config.get("do_sample", False),
                    "pad_token_id": tokenizer.eos_token_id if tokenizer is not None else 0,
                    "attention_mask": attention_mask
                }
                    
                # 只有在do_sample为True时才添加temperature和top_p参数
                if generate_kwargs["do_sample"]:
                    generate_kwargs["temperature"] = generation_config.get("temperature", 1.0)
                    generate_kwargs["top_p"] = generation_config.get("top_p", 1.0)
                    
                logger.debug(f"生成参数构造完成: {generate_kwargs}")
                # 检查模型是否正确初始化
                if not hasattr(self.model, 'model') or self.model.model is None:
                    raise RuntimeError("模型未正确初始化")
                
                # 判断模型类型并相应地处理图像参数
                # 检查是否是vLLM模型（具有处理图像的特殊方法）
                is_vllm_model = hasattr(self.model, 'get_multimodal_embeddings') and callable(self.model.get_multimodal_embeddings)
                
                if is_vllm_model:
                    # vLLM模式下添加图像相关参数
                    generate_kwargs["images"] = [[pixel_values, images_crop, images_spatial_crop]]
                    generate_kwargs["images_seq_mask"] = images_seq_mask
                    generate_kwargs["images_spatial_crop"] = images_spatial_crop
                else:
                    # Transformers模式下，需要先通过视觉编码器处理图像
                    # 对于Transformers模型，我们暂时跳过图像处理，只处理文本
                    logger.debug("使用Transformers模式，跳过图像参数传递")
                    # 注意：这会导致模型无法正确处理图像内容
                
                # 尝试生成结果
                try:
                    logger.debug("开始模型生成")
                    logger.debug(f"生成参数: {generate_kwargs.keys()}")
                    outputs = self.model.model.generate(**generate_kwargs)
                    logger.debug("模型生成完成")
                except Exception as generate_error:
                    logger.error(f"模型生成过程中出错: {generate_error}")
                    import traceback
                    logger.error(f"错误堆栈: {traceback.format_exc()}")
                    raise RuntimeError(f"模型生成过程中出错: {generate_error}") from generate_error
            
            # 解码输出 - 根据官方实现进行修正
            try:
                logger.debug("开始解码输出")
                # 标准解码方式 - 根据Hugging Face官方示例
                if hasattr(outputs, 'sequences') and tokenizer is not None:
                    # 对于vLLM输出格式
                    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    logger.debug("解码完成(vLLM格式)")
                    return result
                elif hasattr(outputs, 'output_ids') and tokenizer is not None:
                    # 对于Transformers输出格式
                    if isinstance(outputs.output_ids, list):
                        result = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
                    else:
                        result = tokenizer.decode(outputs.output_ids, skip_special_tokens=True)
                    logger.debug("解码完成(Transformers格式)")
                    return result
                elif tokenizer is not None:
                    # 尝试直接解码outputs
                    if isinstance(outputs, torch.Tensor):
                        # 如果outputs是tensor，尝试解码
                        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        logger.debug("解码完成(Tensor格式)")
                        return result
                    elif hasattr(outputs, '__iter__'):
                        # 如果outputs是可迭代的，尝试解码第一个元素
                        result = tokenizer.decode(list(outputs)[0], skip_special_tokens=True)
                        logger.debug("解码完成(可迭代格式)")
                        return result
                    else:
                        # 尝试直接解码outputs对象
                        result = tokenizer.decode(outputs, skip_special_tokens=True)
                        logger.debug("解码完成(其他格式)")
                        return result
                else:
                    raise RuntimeError("解码失败：缺少tokenizer")
            except Exception as decode_error:
                # 返回详细的解码错误信息
                error_msg = f"解码算法错误: {str(decode_error)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from decode_error
        except Exception as e:
            logger.error(f"图像处理过程中发生错误: {str(e)}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"图像处理失败: {str(e)}") from e