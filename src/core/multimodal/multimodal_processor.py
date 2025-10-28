#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的多模态处理器
整合所有多模态处理相关的功能，避免重复代码
"""

from collections.abc import Mapping, Sequence
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import math
from PIL import Image

# 第三方库导入
from transformers import BatchFeature, AutoTokenizer, AutoProcessor

# 项目内部导入
from src.core.process.image_process import (
    DeepseekOCRProcessor, count_tiles)

# 配置导入
from src.core.config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS, DEFAULT_OCR_PROMPT

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()

# vLLM相关导入（延迟导入，避免在不支持的平台上报错）
try:
    from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                        MultiModalKwargs, NestedTensors)
    from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                    ImageSize as VLLMImageSize, MultiModalDataItems)
    from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                            BaseProcessingInfo, PromptReplacement,
                                            PromptUpdate)
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
    VLLM_AVAILABLE = True
    # 创建别名以避免命名冲突
    ImageSize = VLLMImageSize
except ImportError:
    # 在不支持vLLM的平台上设置占位符
    MultiModalDataDict = object
    MultiModalFieldConfig = object
    MultiModalKwargs = object
    NestedTensors = object
    ImageEmbeddingItems = object
    ImageProcessorItems = object
    MultiModalDataItems = object
    BaseMultiModalProcessor = object
    BaseProcessingInfo = object
    PromptReplacement = object
    PromptUpdate = object
    BaseDummyInputsBuilder = object
    VLLM_AVAILABLE = False
    # 在不支持vLLM的平台上创建简单的ImageSize类
    class ImageSize:
        def __init__(self, width, height):
            self.width = width
            self.height = height


class MultimodalProcessor:
    """
    多模态处理器
    整合所有多模态处理相关的功能，避免重复代码
    """
    
    def __init__(self, model_path: str, device: str):
        """
        初始化统一多模态处理器
        
        Args:
            model_path: 模型路径
            device: 设备类型
        """
        self.model_path = model_path
        self.device = device
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        初始化处理器
        
        Returns:
            是否初始化成功
        """
        try:
            logger.info("初始化统一多模态处理器...")
            
            # 初始化处理器
            self.processor = DeepseekOCRProcessor.from_pretrained(self.model_path)
            
            # 初始化tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.is_initialized = True
            logger.info("统一多模态处理器初始化成功")
            
            return True
            
        except Exception as e:
            logger.error(f"初始化统一多模态处理器失败: {str(e)}")
            return False
    
    def process_inputs(self, images: List[Image.Image], prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        处理输入图像和提示词
        
        Args:
            images: 图像列表
            prompts: 提示词列表
            
        Returns:
            处理后的输入字典
        """
        if not self.is_initialized:
            raise RuntimeError("处理器未初始化")
            
        if len(images) != len(prompts):
            raise ValueError("图像数量和提示词数量不匹配")
            
        try:
            # 使用DeepseekOCRProcessor处理输入
            inputs = self.processor(
                images=images,
                text=prompts,
                return_tensors="pt"
            )
            
            # 将输入移动到指定设备
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            return inputs
            
        except Exception as e:
            logger.error(f"处理输入时发生错误: {str(e)}")
            raise
    
    def get_model(self) -> Any:
        """
        获取模型
        
        Returns:
            模型对象
        """
        return self.model
    
    def get_processor(self) -> Any:
        """
        获取处理器
        
        Returns:
            处理器对象
        """
        return self.processor
    
    def get_tokenizer(self) -> Any:
        """
        获取tokenizer
        
        Returns:
            tokenizer对象
        """
        return self.tokenizer


class DeepseekOCRProcessingInfo:
    """
    DeepSeek OCR处理信息类
    用于处理OCR相关的多模态信息
    """

    def get_hf_config(self):
        """
        获取HuggingFace配置
        
        Returns:
            HuggingFace配置对象
        """
        # 在不支持vLLM的平台上返回None
        if not VLLM_AVAILABLE:
            return None
        # 如果DeepseekVLV2Config未定义，使用默认配置
        try:
            # 尝试导入DeepseekVLV2Config，如果不存在则使用默认配置
            try:
                from transformers import AutoConfig
                return self.ctx.get_hf_config(AutoConfig)
            except ImportError:
                return self.ctx.get_hf_config()
        except NameError:
            # 如果DeepseekVLV2Config未定义，尝试使用默认配置
            return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        """
        获取HuggingFace处理器
        
        Args:
            **kwargs: 处理器参数
            
        Returns:
            HuggingFace处理器对象
        """
        return self.ctx.get_hf_processor(DeepseekOCRProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        """
        获取支持的多模态限制
        
        Returns:
            多模态限制映射
        """
        return {"image": None}

    def get_num_image_tokens(self,
                            *,
                            image_width: int,
                            image_height: int,
                            cropping: bool = True) -> int:
        """
        计算图像token数量
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            cropping: 是否进行裁剪
            
        Returns:
            图像token数量
        """
        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 640 and image_height <= 640:
                crop_ratio = [1, 1]
            else:
                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(image_width, image_height, image_size=IMAGE_SIZE)
                
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles >1 or num_height_tiles>1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self):
        """
        获取具有最多特征的图像尺寸
        
        Returns:
            图像尺寸对象
        """
        # 只在vLLM可用时创建ImageSize对象
        if VLLM_AVAILABLE:
            if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
                return ImageSize(width=1024*2, height=1024*2)
            return ImageSize(width=640*2, height=640*2)
        else:
            # 在不支持vLLM的平台上返回简单字典
            if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
                return {"width": 1024*2, "height": 1024*2}
            return {"width": 640*2, "height": 640*2}


class UnifiedDeepseekOCRDummyInputsBuilder:
    """
    统一的DeepSeek OCR虚拟输入构建器
    用于构建测试用的虚拟输入
    """
    
    def __init__(self, info):
        self.info = info
        
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        获取虚拟文本
        
        Args:
            mm_counts: 多模态计数映射
            
        Returns:
            虚拟文本字符串
        """
        num_images = mm_counts.get("image", 0)

        # 在不支持vLLM的平台上使用简化实现
        image_token = "<image>"

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ):
        """
        获取虚拟多模态数据
        
        Args:
            seq_len: 序列长度
            mm_counts: 多模态计数映射
            
        Returns:
            虚拟多模态数据字典
        """
        num_images = mm_counts.get("image", 0)

        # 在不支持vLLM的平台上返回空数据
        return {"image": []}


# 只在vLLM可用时定义相关类
if VLLM_AVAILABLE:
    class UnifiedDeepseekOCRMultiModalProcessor(
            BaseMultiModalProcessor[UnifiedDeepseekOCRProcessingInfo]):
        """
        统一的DeepSeek OCR多模态处理器
        处理OCR相关的多模态输入
        """

        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
        ) -> BatchFeature:
            """
            调用HuggingFace处理器
            
            Args:
                prompt: 提示文本
                mm_data: 多模态数据
                mm_kwargs: 多模态参数
                
            Returns:
                批处理特征对象
            """
            
            # logger.debug(mm_data)
            if mm_data:
                processed_outputs = self.info.ctx.call_hf_processor(
                    self.info.get_hf_processor(**mm_kwargs),
                    dict(prompt=prompt, **mm_data),
                    mm_kwargs,
                )

            else:
                tokenizer = self.info.get_tokenizer()
                processed_outputs = tokenizer(prompt,
                                        add_special_tokens=True,
                                        return_tensors="pt")

            return processed_outputs
else:
    class UnifiedDeepseekOCRMultiModalProcessor:
        """
        统一的DeepSeek OCR多模态处理器 (简化版)
        处理OCR相关的多模态输入
        """
        def __init__(self, *args, **kwargs):
            pass