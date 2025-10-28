#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformers引擎实现
"""

import os
import sys
import torch
import logging
from typing import List, Optional
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入配置
from src.core.config.settings import MODEL_PATH, DEFAULT_OCR_PROMPT
from src.cli.utils import get_compatible_device, get_appropriate_dtype, should_use_bfloat16
from src.core.base.ocr_engine import BaseOCREngine
from src.core.logging import debug_wrapper, debug_trace

logger = logging.getLogger(__name__)


class TransformersEngine(BaseOCREngine):
    """Transformers引擎实现"""
    
    def __init__(self, model_path: Optional[str] = None, prompt: Optional[str] = None,
                 base_size: int = 1024, image_size: int = 640, crop_mode: bool = True):
        """
        初始化Transformers引擎
        
        Args:
            model_path: 模型路径
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否启用裁剪模式
        """
        super().__init__(model_path, prompt, base_size, image_size, crop_mode)
        self.model = None
        self.tokenizer = None
        self.device = None
    
    @debug_wrapper
    def initialize(self) -> None:
        """初始化Transformers引擎"""
        debug_trace()
        try:
            # 设置模型路径，使用配置中的路径（已包含本地模型检查逻辑）
            model_name = self.model_path or MODEL_PATH
            logger.info(f"使用模型路径: {model_name}")
            
            # 使用统一的模型初始化器
            from src.core.model_initializer import ModelInitializer
            
            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = (
                model_name.startswith(("http://", "https://")) or
                model_name.startswith("deepseek-ai/") or
                model_name.startswith("huggingface.co/") or
                "/" not in model_name or  # 单个名称可能是远程仓库名
                (not os.path.exists(model_name) and not os.path.exists(os.path.expanduser(model_name)))
            )
            
            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo
            
            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用trust_remote_code=True
            trust_remote_code_for_model = is_remote_repo
            
            self.model, self.tokenizer = ModelInitializer.initialize_transformers_model_and_tokenizer(
                model_name, 
                trust_remote_code=trust_remote_code_for_model
            )
            
            # 检查可用的设备
            self.device = get_compatible_device()
            logger.info(f"使用 {self.device.type.upper()} 设备进行推理")
            
            # 将模型移到设备上并设置为评估模式
            from src.core.model_initializer import ModelInitializer
            self.model = ModelInitializer.move_model_to_device(self.model, self.device)
            
            # 根据设备类型选择合适的数据类型
            use_bfloat16 = should_use_bfloat16(self.device)
            if use_bfloat16 and self.device.type != "mps":
                # MPS设备上不使用bfloat16，保持默认的float32以确保兼容性
                # 检查模型是否有to方法
                if hasattr(self.model, 'to') and callable(getattr(self.model, 'to', None)):
                    self.model = self.model.to(torch.bfloat16)
            # MPS设备上不使用特殊的数据类型转换，保持默认的float32
            
            # 添加调试信息
            logger.debug(f"模型初始化完成，设备: {self.device}, 模型类型: {type(self.model)}")
            debug_trace()
        except Exception as e:
            logger.error(f"Transformers引擎初始化失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"Transformers引擎初始化失败: {str(e)}")
    
    @debug_wrapper
    def process(self, images: List[Image.Image], output_dir: str) -> None:
        """
        使用Transformers引擎处理图像
        
        Args:
            images: 图像列表
            output_dir: 输出目录路径
        """
        debug_trace()
        logger.debug(f"开始process方法，图像数量: {len(images) if images else 0}, 输出目录: {output_dir}")
        if self.model is None or self.tokenizer is None:
            logger.debug("模型或tokenizer未初始化，开始初始化")
            self.initialize()
            logger.debug("初始化完成")
        
        try:
            output_dir_path = Path(output_dir)
            # 确保输出目录存在，使用绝对路径
            output_dir_path = output_dir_path.resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录已创建: {output_dir_path}")
            
            # 处理每张图像
            results = []
            logger.info(f"开始处理 {len(images)} 张图像")
            for i, image in enumerate(images):
                logger.info(f"正在处理第 {i+1} 张图像，尺寸: {image.size}")
                debug_trace()
                # 保存临时图像文件
                temp_image_path = output_dir_path / f"temp_{i}.jpg"
                # 确保临时文件路径的目录存在
                temp_image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(temp_image_path, "JPEG")
                logger.info(f"临时图像已保存: {temp_image_path}")
                
                try:
                    # 使用原始模型的生成功能处理图像
                    prompt = self.prompt or DEFAULT_OCR_PROMPT
                    logger.info(f"开始OCR识别第 {i+1} 张图像")
                    
                    # 检查模型类型并使用相应的方法
                    if self.model is not None:
                        logger.debug("使用原始模型的生成功能")
                        
                        # 使用项目中的图像处理模块
                        from src.core.process.image_process import DeepseekOCRProcessor
                        processor = DeepseekOCRProcessor(tokenizer=self.tokenizer)
                        logger.debug("DeepseekOCRProcessor初始化完成")
                        
                        # 处理图像和提示词
                        processed_data = processor.tokenize_with_images(
                            conversation=prompt,
                            images=[image],
                            bos=True,
                            eos=True,
                            cropping=self.crop_mode
                        )
                        logger.debug(f"图像处理完成")
                        
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
                        device = self.device
                        input_ids = input_ids.to(device)
                        pixel_values = pixel_values.to(device)
                        images_crop = images_crop.to(device)
                        images_spatial_crop = images_spatial_crop.to(device)
                        
                        # 构造注意力掩码
                        attention_mask = torch.ones_like(input_ids)
                        
                        # 使用模型的生成功能
                        with torch.no_grad():
                            # 获取生成配置参数
                            from src.core.utils.generation_config import GenerationConfigManager
                            generation_config = GenerationConfigManager.get_generation_config(self.model)
                            
                            # 确保模型有generate方法
                            if not hasattr(self.model, 'generate'):
                                logger.error("模型没有generate方法，无法进行生成")
                                raise RuntimeError("模型没有generate方法，无法进行生成")
                            
                            # 生成结果
                            logger.debug("开始模型生成")
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=generation_config.get("max_new_tokens", 8192),
                                do_sample=generation_config.get("do_sample", False),
                                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer is not None else 0,
                                # 传递图像特征给模型
                                images=[[pixel_values, images_crop, images_spatial_crop]],
                                images_seq_mask=images_seq_mask,
                                images_spatial_crop=images_spatial_crop
                            )
                            logger.debug("模型生成完成")
                            
                            # 解码输出
                            logger.debug("开始解码输出")
                            if self.tokenizer is not None:
                                # 获取原始输入长度
                                input_length = input_ids.shape[1]
                                
                                # 只取生成的部分
                                if outputs.shape[1] > input_length:
                                    new_tokens = outputs[0, input_length:]
                                else:
                                    new_tokens = outputs[0]
                                
                                # 解码新生成的token
                                result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                logger.debug(f"解码完成，结果长度: {len(result)}")
                            else:
                                raise RuntimeError("解码失败：缺少tokenizer")
                        
                        logger.info(f"第 {i+1} 张图像OCR识别完成")
                        # 检查结果是否有效
                        if result and isinstance(result, str) and len(result.strip()) > 0:
                            results.append(result.strip())
                        else:
                            logger.warning(f"第 {i+1} 张图像OCR识别返回空结果或默认结果")
                            results.append(f"图像 {i+1} OCR识别未返回有效结果")
                    else:
                        raise RuntimeError("模型未正确初始化")
                except Exception as infer_error:
                    logger.warning(f"图像 {i+1} 处理失败 ({str(infer_error)})")
                    results.append(f"图像 {i+1} 处理失败: {str(infer_error)}")
                finally:
                    # 删除临时文件
                    if temp_image_path.exists():
                        temp_image_path.unlink()
                        logger.info(f"临时文件已删除: {temp_image_path}")
            
            # 保存结果
            logger.info("开始保存OCR结果")
            self._save_results(results, output_dir_path)
            logger.info("OCR处理完成")
        except Exception as e:
            logger.error(f"Transformers OCR执行失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"Transformers OCR执行失败: {str(e)}")
    
    @debug_wrapper
    def _save_results(self, results: list, output_dir: Path) -> None:
        """
        保存OCR结果
        
        Args:
            results: 结果列表
            output_dir: 输出目录路径
        """
        debug_trace()
        # 保存结果，只保存纯净的识别文本
        contents = ''
        successful_results = []
        for result in results:
            # 只添加成功处理的结果，过滤掉失败的提示和日志信息
            if result and isinstance(result, str):
                # 过滤掉错误信息和日志
                if not (result.startswith("图像") and "处理失败" in result) and \
                   result != "处理完成" and \
                   not result.startswith("2025-") and \
                   len(result.strip()) > 0:
                    if contents:  # 如果已有内容，添加换行符分隔
                        contents += '\n'
                    contents += result.strip()
                    successful_results.append(result.strip())
                else:
                    logger.debug(f"过滤掉无效结果: {result[:50]}...")
            else:
                logger.debug(f"跳过非字符串结果: {type(result)}")
        
        # 写入文件
        result_file = output_dir / "result.md"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        logger.info(f"OCR结果已保存到: {result_file}")
        logger.info(f"成功处理 {len(successful_results)} 张图像")
        debug_trace()
    
    def cleanup(self) -> None:
        """清理资源"""
        # Transformers引擎不需要特殊清理
        pass










