#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的DeepSeek OCR推理类
符合单一职责原则，专注于OCR推理逻辑
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import logging

from transformers import AutoTokenizer, AutoProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin

from src.core.logging import get_logger
from src.core.multimodal.ocr_engine_interface import BaseOCREngine
from src.core.config import get_config

logger = get_logger()


class DeepSeekOCRInference(BaseOCREngine):
    """
    DeepSeek OCR推理引擎
    专注于OCR推理逻辑，符合单一职责原则
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化DeepSeek OCR推理引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型
        """
        config = get_config()
        model_path = model_path or config.MODEL_PATH
        
        super().__init__(model_path, device)
        self.processor = None
        self.tokenizer = None
        self.prompt = config.prompt
    
    def initialize(self) -> bool:
        """
        初始化模型和处理器
        
        Returns:
            是否初始化成功
        """
        try:
            logger.info("初始化DeepSeek OCR推理引擎...")
            
            # 加载tokenizer
            logger.info(f"加载tokenizer: {self.model_path}")
            
            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = (
                self.model_path.startswith(("http://", "https://")) or
                self.model_path.startswith("deepseek-ai/") or
                self.model_path.startswith("huggingface.co/") or
                "/" not in self.model_path or  # 单个名称可能是远程仓库名
                (not os.path.exists(self.model_path) and not os.path.exists(os.path.expanduser(self.model_path)))
            )
            
            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo
            
            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用trust_remote_code=True
            trust_remote_code_for_tokenizer = is_remote_repo
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code_for_tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=local_files_only
            )
            
            # 加载模型
            logger.info(f"加载模型: {self.model_path}")
            try:
                # 使用模型工厂创建模型实例
                from src.core.models.model_factory import create_ocr_model
                
                # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
                # 对于远程模型，使用trust_remote_code=True
                trust_remote_code_for_model = is_remote_repo
                
                self.model = create_ocr_model(
                    model_type="transformers",  # 使用transformers模型类型
                    model_path=self.model_path,
                    trust_remote_code=trust_remote_code_for_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    local_files_only=local_files_only
                )
                logger.info("成功加载OCR模型")
            except Exception as e:
                logger.error(f"无法加载OCR模型: {str(e)}")
                # 如果模型工厂不可用，尝试使用AutoModelForCausalLM
                try:
                    from transformers import AutoModelForCausalLM
                    
                    # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
                    # 对于远程模型，使用trust_remote_code=True
                    trust_remote_code_for_model = is_remote_repo
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=trust_remote_code_for_model,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        local_files_only=local_files_only
                    )
                    logger.info("成功加载AutoModelForCausalLM模型")
                except Exception as e2:
                    logger.error(f"无法加载AutoModelForCausalLM: {str(e2)}")
                    raise RuntimeError(f"无法加载模型: {str(e2)}")
            
            # 加载图像处理器
            logger.info(f"加载图像处理器: {self.model_path}")
            try:
                from src.core.process.image_process import DeepseekOCRProcessor
                self.processor = DeepseekOCRProcessor.from_pretrained(self.model_path)
                logger.info("成功加载DeepseekOCRProcessor")
            except Exception as e:
                logger.error(f"无法加载图像处理器: {str(e)}")
                # 如果无法加载专用处理器，设置为None
                self.processor = None
            
            # 设置设备
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
                logger.info(f"模型已移动到设备: {self.device}")
            
            # 设置评估模式
            if hasattr(self.model, 'eval'):
                self.model.eval()
                logger.info("模型已设置为评估模式")
            
            # 标记初始化成功
            self.is_initialized = True
            
            logger.info("DeepSeek OCR推理引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化DeepSeek OCR推理引擎失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False
    
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
        try:
            # 加载图像
            image = Image.open(image_file).convert('RGB')
            logger.debug(f"图像加载完成，尺寸: {image.size}")
            
            # 处理图像
            try:
                from src.core.process.image_process import DeepseekOCRProcessor
                processor = DeepseekOCRProcessor(tokenizer=tokenizer)
            except ImportError:
                logger.error("无法导入DeepseekOCRProcessor")
                raise ImportError("无法导入DeepseekOCRProcessor")
            
            # 构建对话文本，包含图像标记
            conversation = f"{processor.image_token}"
            processed_data = processor.tokenize_with_images(
                conversation=conversation,
                images=[image],
                bos=True,
                eos=True,
                cropping=crop_mode
            )
            
            # 提取处理后的数据
            if processed_data is None:
                raise ValueError("processed_data为None")
            if len(processed_data) == 0:
                raise ValueError("processed_data为空列表")
            if processed_data[0] is None:
                raise ValueError("processed_data[0]为None")
            
            if len(processed_data) > 0 and processed_data[0] is not None and len(processed_data[0]) >= 7:
                # 确保processed_data[0]有足够的元素
                if len(processed_data[0]) < 7:
                    raise ValueError(f"处理后的数据结构不完整，期望至少7个元素，实际只有{len(processed_data[0])}个元素")
                
                # 根据原始仓库的数据结构正确提取元素（索引0-6）
                input_ids = processed_data[0][0]
                pixel_values = processed_data[0][1]
                images_crop = processed_data[0][2]
                images_seq_mask = processed_data[0][3]
                images_spatial_crop = processed_data[0][4]
                num_image_tokens = processed_data[0][5]
                image_shapes = processed_data[0][6]
            else:
                raise ValueError("图像处理失败，未生成有效的输入数据")
                
            # 验证提取的元素不为None
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
                # 使用统一的设备选择工具
                from src.core.utils.mps_utils import get_optimal_device
                device = get_optimal_device()
            except Exception:
                # 如果无法获取参数设备，使用默认设备
                device = torch.device("cpu")
                
            # 确保张量在正确的设备上
            try:
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
                    pixel_values = pixel_values.to(torch.float32)
                    images_crop = images_crop.to(torch.float32)
                except Exception as e:
                    logger.warning(f"在MPS设备上转换数据类型时出错: {e}")
            else:
                # 其他设备上可以使用bfloat16（如果支持）
                try:
                    if torch.cuda.is_bf16_supported():
                        pixel_values = pixel_values.to(torch.bfloat16)
                        images_crop = images_crop.to(torch.bfloat16)
                except Exception as e:
                    logger.warning(f"转换数据类型时出错: {e}")
                
            # 生成结果
            with torch.no_grad():
                # 构造注意力掩码
                try:
                    # 创建与input_ids相同形状的注意力掩码
                    attention_mask = torch.ones_like(input_ids)
                    
                    # 如果有tokenizer，检查pad_token_id和eos_token_id是否相同
                    if tokenizer is not None:
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        eos_token_id = tokenizer.eos_token_id
                        
                        # 如果pad_token_id和eos_token_id相同，需要特殊处理
                        if pad_token_id == eos_token_id:
                            # 创建一个更精确的注意力掩码，只对实际输入内容设置注意力
                            attention_mask = torch.ones_like(input_ids)
                            # 确保所有非填充位置的注意力为1
                            attention_mask[input_ids != pad_token_id] = 1
                except Exception as e:
                    logger.warning(f"创建注意力掩码时出错，使用默认掩码: {e}")
                    attention_mask = torch.ones_like(input_ids)
                except Exception as e:
                    raise ValueError(f"创建注意力掩码时出错: {e}")
                    
                # 获取模型的生成配置参数
                try:
                    generation_config = self.model._get_generation_config()
                except Exception as e:
                    logger.warning(f"获取生成配置时出错，使用默认配置: {e}")
                    from src.core.utils.generation_config import GenerationConfigManager
                    generation_config = GenerationConfigManager.get_generation_config(None)

                generate_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": getattr(generation_config, "max_new_tokens", 8192),
                    "do_sample": getattr(generation_config, "do_sample", False),
                    "pad_token_id": tokenizer.eos_token_id if tokenizer is not None else 0,
                    "attention_mask": attention_mask
                }
                    
                # 只有在do_sample为True时才添加temperature和top_p参数
                if generate_kwargs["do_sample"]:
                    generate_kwargs["temperature"] = getattr(generation_config, "temperature", 1.0)
                    generate_kwargs["top_p"] = getattr(generation_config, "top_p", 1.0)
                    
                # 检查模型是否正确初始化
                if not hasattr(self.model, 'language_model') or self.model.language_model is None:
                    # 如果没有language_model属性，尝试使用模型本身
                    if not hasattr(self.model, 'generate') or not callable(self.model.generate):
                        raise RuntimeError("模型未正确初始化，缺少language_model属性且模型本身不支持generate方法")
                    else:
                        # 直接使用模型本身进行生成
                        model_to_use = self.model
                else:
                    # 使用模型的language_model属性
                    model_to_use = self.model.language_model
                
                # 判断模型类型并相应地处理图像参数
                # 检查是否是vLLM模型（具有处理图像的特殊方法）
                is_vllm_model = hasattr(model_to_use, 'get_multimodal_embeddings') and callable(model_to_use.get_multimodal_embeddings)
                
                if is_vllm_model:
                    # vLLM模式下添加图像相关参数
                    generate_kwargs["images"] = [[pixel_values, images_crop, images_spatial_crop]]
                    generate_kwargs["images_seq_mask"] = images_seq_mask
                    generate_kwargs["images_spatial_crop"] = images_spatial_crop
                
                # 尝试生成结果
                try:
                    # 修复：直接调用模型的generate方法，不需要弹出input_ids
                    outputs = model_to_use.generate(**generate_kwargs)
                except Exception as generate_error:
                    logger.error(f"模型生成过程中出错: {generate_error}")
                    import traceback
                    logger.error(f"错误堆栈: {traceback.format_exc()}")
                    raise RuntimeError(f"模型生成过程中出错: {generate_error}") from generate_error
            
            # 解码输出 - 根据官方实现进行修正
            try:
                # 标准解码方式 - 根据Hugging Face官方示例
                if hasattr(outputs, 'sequences') and tokenizer is not None:
                    # 对于vLLM输出格式
                    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    return result
                elif hasattr(outputs, 'output_ids') and tokenizer is not None:
                    # 对于Transformers输出格式
                    if isinstance(outputs.output_ids, list):
                        result = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
                    else:
                        result = tokenizer.decode(outputs.output_ids, skip_special_tokens=True)
                    return result
                elif tokenizer is not None:
                    # 尝试直接解码outputs
                    if isinstance(outputs, torch.Tensor):
                        # 如果outputs是tensor，尝试解码
                        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        return result
                    elif hasattr(outputs, '__iter__'):
                        # 如果outputs是可迭代的，尝试解码第一个元素
                        result = tokenizer.decode(list(outputs)[0], skip_special_tokens=True)
                        return result
                    else:
                        # 尝试直接解码outputs对象
                        result = tokenizer.decode(outputs, skip_special_tokens=True)
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
    
    def _process_single_image(self, image: Union[Image.Image, torch.Tensor], prompt: str) -> str:
        """
        处理单个图像的具体实现
        
        Args:
            image: 图像对象
            prompt: 提示词
            
        Returns:
            OCR结果
        """
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 构建输入
            inputs = self._build_inputs(processed_image, prompt)
            
            # 使用统一的设备选择工具
            from src.core.utils.mps_utils import get_optimal_device
            device = get_optimal_device()
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码结果
            result = self._decode_outputs(outputs, inputs)
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像时发生错误: {str(e)}")
            raise
    
    def _process_batch_images(self, images: List[Any], prompts: List[str]) -> List[str]:
        """
        批量处理图像
        
        Args:
            images: 图像列表
            prompts: 提示词列表
            
        Returns:
            识别结果列表
        """
        try:
            # 确保图像和提示词数量一致
            if len(images) != len(prompts):
                logger.warning(f"图像数量({len(images)})与提示词数量({len(prompts)})不一致")
                # 如果提示词较少，用最后一个提示词填充
                if len(prompts) < len(images):
                    prompts = prompts + [prompts[-1]] * (len(images) - len(prompts))
                # 如果图像较少，截断提示词
                else:
                    prompts = prompts[:len(images)]
                    images = images[:len(prompts)]
            
            # 预处理图像
            processed_images = []
            for img in images:
                try:
                    processed_img = self._preprocess_image(img)
                    processed_images.append(processed_img)
                except Exception as e:
                    logger.error(f"预处理图像时出错: {str(e)}")
                    processed_images.append(None)
            
            # 构建批量输入
            batch_inputs = self._build_batch_inputs(processed_images, prompts)
            
            # 确保输入在正确的设备上
            device = self._get_device()
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].to(device)
            
            # 过滤模型输入，只保留模型接受的参数
            filtered_inputs = self._filter_model_inputs(batch_inputs)
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **filtered_inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码输出
            input_length = batch_inputs.get('input_ids', torch.tensor([])).shape[1] if 'input_ids' in batch_inputs else 0
            results = self._decode_outputs(outputs, batch_inputs, input_length)
            
            # 确保结果数量与图像数量一致
            while len(results) < len(images):
                results.append("")
            
            return results
            
        except Exception as e:
            logger.error(f"批量处理图像时发生错误: {str(e)}")
            # 返回与图像数量相同的空字符串列表
            return [""] * len(images)
    
    def _filter_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤模型输入，只保留模型接受的参数
        
        Args:
            inputs: 原始输入
            
        Returns:
            过滤后的输入
        """
        # 获取模型生成方法的参数签名
        import inspect
        try:
            generate_signature = inspect.signature(self.model.generate)
            valid_params = set(generate_signature.parameters.keys())
            logger.info(f"模型generate方法接受的参数: {valid_params}")
        except Exception as e:
            logger.warning(f"无法获取模型generate方法的参数签名: {str(e)}")
            # 使用常见的参数作为后备
            valid_params = {'input_ids', 'attention_mask', 'max_new_tokens', 'do_sample', 
                           'temperature', 'top_p', 'pad_token_id', 'eos_token_id'}
        
        # 过滤输入，只保留模型接受的参数
        filtered_inputs = {}
        for key, value in inputs.items():
            if key in valid_params:
                filtered_inputs[key] = value
            else:
                logger.debug(f"过滤掉模型不接受的参数: {key}")
        
        return filtered_inputs
    
    def _preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> Image.Image:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        if isinstance(image, torch.Tensor):
            # 将tensor转换为PIL图像
            image = self._tensor_to_pil(image)
        
        # 确保图像是RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将tensor转换为PIL图像
        
        Args:
            tensor: 输入tensor
            
        Returns:
            PIL图像
        """
        # 如果tensor在GPU上，先移到CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 如果tensor有梯度，去除梯度
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        # 转换为numpy数组
        if tensor.dim() == 3:
            # CHW格式
            numpy_image = tensor.permute(1, 2, 0).numpy()
        elif tensor.dim() == 4:
            # BCHW格式，取第一个batch
            numpy_image = tensor[0].permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"不支持的tensor维度: {tensor.dim()}")
        
        # 确保值在0-255范围内
        if numpy_image.max() <= 1.0:
            numpy_image = (numpy_image * 255).astype(np.uint8)
        else:
            numpy_image = numpy_image.astype(np.uint8)
        
        # 转换为PIL图像
        return Image.fromarray(numpy_image)
    
    def _build_inputs(self, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        """
        构建模型输入
        
        Args:
            image: 图像
            prompt: 提示词
            
        Returns:
            模型输入字典
        """
        # 使用统一的设备选择工具
        from src.core.utils.mps_utils import get_optimal_device
        device = get_optimal_device()
        
        if self.processor is not None:
            # 使用处理器处理输入
            inputs = self.processor(prompt, [image], return_tensors="pt")
        else:
            # 如果没有处理器，使用tokenizer处理文本，手动处理图像
            text_inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 简单的图像处理 - 调整大小并转换为tensor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
            
            inputs = {
                "input_ids": text_inputs["input_ids"].to(device),
                "attention_mask": text_inputs["attention_mask"].to(device),
                "pixel_values": image_tensor.to(device)
            }
        
        # 确保所有输入都在正确的设备上
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
        
        return inputs
    
    def _build_batch_inputs(self, images: List[Image.Image], prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        构建批量模型输入
        
        Args:
            images: 图像列表
            prompts: 提示词列表
            
        Returns:
            模型输入字典
        """
        # 使用统一的设备选择工具
        from src.core.utils.mps_utils import get_optimal_device
        device = get_optimal_device()
        
        try:
            if self.processor is not None and hasattr(self.processor, 'tokenize_with_images'):
                # 使用DeepseekOCRProcessor处理批量输入
                # 注意：DeepseekOCRProcessor可能不支持批量处理，需要逐个处理
                if len(images) > 0:
                    # 构建conversation，包含image token
                    image_token = "<image>"
                    conversation = prompts[0] if prompts else ""
                    
                    # 确保conversation中有正确数量的image token
                    if image_token not in conversation:
                        conversation = image_token + conversation
                    
                    # 调用tokenize_with_images方法
                    processed_data = self.processor.tokenize_with_images(
                        conversation=conversation,
                        images=images,  # 处理所有图像
                        bos=True,
                        eos=True,
                        cropping=True
                    )
                    
                    # tokenize_with_images返回的是[[input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes]]
                    # 我们需要提取这些值
                    if processed_data and len(processed_data) > 0 and len(processed_data[0]) >= 5:
                        # 提取处理后的数据
                        input_ids = processed_data[0][0]
                        pixel_values = processed_data[0][1]
                        images_crop = processed_data[0][2]
                        images_seq_mask = processed_data[0][3]
                        images_spatial_crop = processed_data[0][4]
                        
                        # 确保所有张量都在正确的设备上
                        inputs = {
                            "input_ids": input_ids.to(device),
                            "pixel_values": pixel_values.to(device),
                            "images_crop": images_crop.to(device),
                            "images_seq_mask": images_seq_mask.to(device),
                            "images_spatial_crop": images_spatial_crop.to(device)
                        }
                        
                        return inputs
                    else:
                        logger.warning(f"tokenize_with_images返回的数据格式不正确: {len(processed_data) if processed_data else 0}")
                        if processed_data and len(processed_data) > 0:
                            logger.warning(f"processed_data[0]长度: {len(processed_data[0]) if processed_data[0] else 0}")
            
            # 如果上面的方法失败，尝试使用通用处理器方法
            if self.processor is not None and hasattr(self.processor, '__call__'):
                try:
                    inputs = self.processor(prompts, images, return_tensors="pt", padding=True)
                    
                    # 确保所有输入都在正确的设备上
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(device)
                    
                    return inputs
                except Exception as e:
                    logger.warning(f"使用通用处理器方法失败: {str(e)}")
            
            # 如果没有处理器或处理器方法失败，使用tokenizer处理文本，手动处理图像
            text_inputs = self.tokenizer(prompts[0] if prompts else "", return_tensors="pt")
            
            # 简单的图像处理 - 处理所有图像
            if images:
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # 处理所有图像并堆叠
                image_tensors = [transform(img) for img in images]
                image_tensor = torch.stack(image_tensors)
                
                inputs = {
                    "input_ids": text_inputs["input_ids"].to(device),
                    "attention_mask": text_inputs["attention_mask"].to(device),
                    "pixel_values": image_tensor.to(device)
                }
                
                return inputs
            else:
                raise ValueError("没有可处理的图像")
                
        except Exception as e:
            logger.error(f"构建批量输入时发生错误: {str(e)}")
            # 如果发生错误，尝试使用简单的处理方式
            text_inputs = self.tokenizer(prompts[0] if prompts else "", return_tensors="pt")
            
            # 简单的图像处理 - 处理所有图像
            if images:
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # 处理所有图像并堆叠
                image_tensors = [transform(img) for img in images]
                image_tensor = torch.stack(image_tensors)
                
                inputs = {
                    "input_ids": text_inputs["input_ids"].to(device),
                    "attention_mask": text_inputs["attention_mask"].to(device),
                    "pixel_values": image_tensor.to(device)
                }
                
                return inputs
            else:
                raise ValueError("没有可处理的图像")
    
    def _decode_outputs(self, outputs: Any, inputs: Dict[str, Any], input_length: int = 0) -> List[str]:
        """
        解码模型输出为文本
        
        Args:
            outputs: 模型输出
            inputs: 输入数据
            input_length: 输入长度
            
        Returns:
            解码后的文本列表
        """
        try:
            # 检查输出类型
            if isinstance(outputs, list):
                # 如果是列表，直接解码每个元素
                results = []
                for i, output in enumerate(outputs):
                    try:
                        result = self._decode_single_output(output, inputs, input_length)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"解码输出 {i} 时出错: {str(e)}")
                        results.append("")
                return results
            elif isinstance(outputs, torch.Tensor):
                # 如果是张量，检查维度
                if outputs.dim() >= 2:
                    # 多维张量，可能是批量输出
                    batch_size = outputs.size(0)
                    results = []
                    for i in range(batch_size):
                        try:
                            # 提取单个输出
                            if outputs.dim() == 2:
                                # 2D输出，形状为[batch_size, seq_len]或[batch_size, vocab_size]
                                single_output = outputs[i]
                            elif outputs.dim() == 3:
                                # 3D输出，形状为[batch_size, seq_len, vocab_size]
                                single_output = outputs[i]
                            else:
                                # 其他维度，尝试取第一个维度
                                single_output = outputs[i]
                            
                            result = self._decode_single_output(single_output, inputs, input_length)
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"解码批次 {i} 时出错: {str(e)}")
                            results.append("")
                    return results
                else:
                    # 单个张量，直接解码
                    result = self._decode_single_output(outputs, inputs, input_length)
                    return [result]
            else:
                # 其他类型，尝试直接解码
                result = self._decode_single_output(outputs, inputs, input_length)
                return [result]
        except Exception as e:
            logger.error(f"解码输出时发生错误: {str(e)}")
            return [""]
    
    def _decode_single_output(self, output: Any, inputs: Dict[str, Any], input_length: int = 0) -> str:
        """
        解码单个输出
        
        Args:
            output: 单个输出
            inputs: 输入数据
            input_length: 输入长度
            
        Returns:
            解码后的文本
        """
        try:
            # 添加调试信息
            logger.info(f"单个输出类型: {type(output)}, 维度: {output.dim() if hasattr(output, 'dim') else 'N/A'}, 形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
            
            # 获取当前输入的ID
            current_input_ids = inputs.get('input_ids')
            if current_input_ids is not None and hasattr(current_input_ids, 'size'):
                if current_input_ids.dim() > 1:
                    # 如果是批量输入，取第一个
                    current_input_ids = current_input_ids[0]
                input_length = current_input_ids.size(0)
                logger.info(f"从输入获取的长度: {input_length}")
            
            # 如果是张量，进行解码
            if isinstance(output, torch.Tensor):
                # 获取tokenizer的词汇表大小
                vocab_size = None
                # 首先尝试从模型配置获取词表大小
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                    vocab_size = self.model.config.vocab_size
                    logger.info(f"从模型配置获取词汇表大小: {vocab_size}")
                # 其次尝试从tokenizer获取
                elif hasattr(self.tokenizer, 'vocab_size'):
                    vocab_size = self.tokenizer.vocab_size
                    logger.info(f"从tokenizer获取词汇表大小: {vocab_size}")
                # 最后尝试从tokenizer的词汇表获取
                elif hasattr(self.tokenizer, 'get_vocab'):
                    try:
                        vocab_size = len(self.tokenizer.get_vocab())
                        logger.info(f"从tokenizer.get_vocab()获取词汇表大小: {vocab_size}")
                    except:
                        vocab_size = 100000  # 默认值
                        logger.warning(f"无法获取词汇表大小，使用默认值: {vocab_size}")
                else:
                    vocab_size = 100000  # 默认值
                    logger.warning(f"无法获取词汇表大小，使用默认值: {vocab_size}")
                
                logger.info(f"Tokenizer词汇表大小: {vocab_size}")
                
                # 检查输出是否是logits还是token IDs
                # 对于2维输出，形状为[seq_len, vocab_size]或[batch_size, seq_len]
                # 对于3维输出，形状为[batch_size, seq_len, vocab_size]或[batch_size, num_sequences, seq_len]
                
                # 获取输出序列
                output_seq = output
                
                if output.dim() == 3 and output.size(-1) > 1000:
                    # 3维输出，最后一维是词汇表大小
                    logger.info("输出看起来像是logits（3维输出，词汇表大小 > 1000），使用argmax获取token IDs")
                    predicted_ids = torch.argmax(output, dim=-1)
                    logger.info(f"argmax后的token IDs: {predicted_ids.tolist()}")
                elif output.dim() == 2 and output.size(0) == 1 and output.size(1) > 1000:
                    # 2维输出，形状为[1, vocab_size]
                    logger.info("输出看起来像是logits（2维输出，词汇表大小 > 1000），使用argmax获取token IDs")
                    predicted_ids = torch.argmax(output, dim=-1)
                    logger.info(f"argmax后的token IDs: {predicted_ids.tolist()}")
                elif output.dim() == 2 and output.size(1) > 1000:
                    # 2维输出，形状为[seq_len, vocab_size]
                    logger.info("输出看起来像是logits（2维输出，词汇表大小 > 1000），使用argmax获取token IDs")
                    predicted_ids = torch.argmax(output, dim=-1)
                    logger.info(f"argmax后的token IDs: {predicted_ids.tolist()}")
                else:
                    # 输出可能是token IDs
                    logger.info("输出看起来像是token IDs")
                    predicted_ids = output
                    
                    # 检查输出中的最大值，如果值很大（比如>vocab_size），可能是logits
                    max_val = predicted_ids.max().item()
                    logger.info(f"token IDs中的最大值: {max_val}")
                    if max_val > vocab_size * 1.5:  # 使用词汇表大小的1.5倍作为阈值
                        logger.warning(f"token IDs中的最大值{max_val}超过词汇表大小{vocab_size}的1.5倍，可能是logits而不是token IDs")
                        # 这个输出实际上是logits，我们需要重新处理
                        # 回到原始输出并使用argmax
                        logger.info("重新处理原始输出作为logits")
                        predicted_ids = torch.argmax(output, dim=-1)
                        logger.info(f"使用argmax后的token IDs: {predicted_ids.tolist()}")
                        logger.info(f"token IDs长度: {len(predicted_ids)}")
                    
                    logger.info(f"处理后的token IDs: {predicted_ids.tolist()}")
                
                # 如果有输入长度，只取生成的部分
                if input_length > 0:
                    if predicted_ids.dim() == 1 and predicted_ids.size(0) > input_length:
                        # 1维张量，直接截取
                        generated_ids = predicted_ids[input_length:]
                        logger.info(f"1维张量，只取生成的部分，形状: {generated_ids.shape}")
                    elif predicted_ids.dim() == 2:
                        # 2维张量，可能是[batch_size, seq_len]或[seq_len, vocab_size]
                        if predicted_ids.size(0) == 1 and predicted_ids.size(1) > input_length:
                            # [1, seq_len]格式，取第二维
                            generated_ids = predicted_ids[0, input_length:]
                            logger.info(f"2维张量[1, seq_len]，只取生成的部分，形状: {generated_ids.shape}")
                        else:
                            # [seq_len, vocab_size]格式，这种情况不应该在这里处理
                            logger.warning("意外的2维张量格式，使用整个序列")
                            generated_ids = predicted_ids
                    else:
                        # 对于其他维度，尝试使用整个序列
                        generated_ids = predicted_ids
                        logger.info(f"序列长度不大于输入长度或维度不匹配，使用整个序列，形状: {predicted_ids.shape}")
                else:
                    generated_ids = predicted_ids
                    logger.info("没有输入长度，使用整个序列")
                
                # 解码为文本
                # 确保generated_ids是张量而不是列表
                if isinstance(generated_ids, list):
                    generated_ids = torch.tensor(generated_ids, device=self.device)
                
                # 如果是2D张量，取第一个序列
                if generated_ids.dim() == 2:
                    generated_ids = generated_ids[0]
                
                # 检查token IDs是否有效
                if generated_ids.numel() > 0:
                    max_id = generated_ids.max().item()
                    logger.info(f"生成token IDs中的最大值: {max_id}")
                    if max_id >= vocab_size:
                        logger.warning(f"生成token IDs中的最大值{max_id}超过词汇表大小{vocab_size}，可能产生乱码")
                
                decoded_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                logger.info(f"解码后的文本: '{decoded_text}'")
                
                # 如果解码结果为空或包含特殊字符，尝试使用整个序列
                if not decoded_text or len(decoded_text.strip()) == 0:
                    logger.warning("解码结果为空，尝试使用整个序列")
                    if predicted_ids.dim() == 2:
                        full_ids = predicted_ids[0]
                    else:
                        full_ids = predicted_ids
                    decoded_text = self.tokenizer.decode(
                        full_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    logger.info(f"使用整个序列解码后的文本: '{decoded_text}'")
                
                return decoded_text.strip()
            else:
                # 非张量输出，尝试直接转换为字符串
                logger.warning(f"输出不是张量类型: {type(output)}")
                return str(output)
                
        except Exception as e:
            logger.error(f"解码单个输出时发生错误: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return ""
    
    def set_prompt(self, prompt: str) -> None:
        """
        设置提示词
        
        Args:
            prompt: 提示词
        """
        self.prompt = prompt
        logger.info(f"提示词已更新: {prompt}")
    
    def get_prompt(self) -> str:
        """
        获取当前提示词
        
        Returns:
            当前提示词
        """
        return self.prompt
    
    def _get_device(self) -> torch.device:
        """
        获取当前设备
        
        Returns:
            当前设备
        """
        if hasattr(self, 'device'):
            return self.device
        else:
            # 使用统一的设备选择工具
            from src.core.utils.mps_utils import get_optimal_device
            return get_optimal_device()