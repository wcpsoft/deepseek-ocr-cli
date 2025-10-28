#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM引擎实现
符合统一的OCR引擎接口
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.logging import get_logger
from src.core.multimodal.ocr_engine_interface import BaseOCREngine
from src.core.config import get_config

# 获取日志记录器
logger = get_logger()


class VLLMEngine(BaseOCREngine):
    """
    vLLM引擎实现
    符合统一的OCR引擎接口
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化vLLM引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型
        """
        config = get_config()
        model_path = model_path or config.MODEL_PATH
        
        super().__init__(model_path, device)
        self.llm = None
        self.sampling_params = None
        self.processor = None
    
    def initialize(self) -> bool:
        """
        初始化vLLM引擎
        
        Returns:
            是否初始化成功
        """
        try:
            logger.info("初始化vLLM引擎...")
            
            # 延迟导入，避免在不需要时加载依赖
            from src.core.vllm.vllm_ocr_model import DeepseekOCRForCausalLM
            from vllm.model_executor.models.registry import ModelRegistry
            from vllm import LLM, SamplingParams
            from src.core.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
            from src.core.process.image_process import DeepseekOCRProcessor
            
            # 注册模型 - 使用正确的模型类型
            ModelRegistry.register_model("DeepseekVLV2ForCausalLM", DeepseekOCRForCausalLM)
            
            # 创建图像处理器
            self.processor = DeepseekOCRProcessor()
            
            # 创建LLM实例
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
            trust_remote_code_for_model = is_remote_repo
            
            self.llm = LLM(
                model=self.model_path,
                hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
                block_size=256,
                enforce_eager=False,
                trust_remote_code=trust_remote_code_for_model, 
                max_model_len=8192,
                swap_space=0,
                max_num_seqs=100,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                disable_mm_preprocessor_cache=True
            )
            
            # 设置采样参数
            logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, 
                                                            whitelist_token_ids={128821, 128822})]
            
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )
            
            self.is_initialized = True
            logger.info("vLLM引擎初始化成功")
            
            return True
            
        except Exception as e:
            logger.error(f"vLLM引擎初始化失败: {str(e)}")
            return False
    
    def _process_single_image(self, image: Union[Image.Image, torch.Tensor], prompt: str) -> str:
        """
        处理单个图像的具体实现
        
        Args:
            image: 图像对象
            prompt: 提示词
            
        Returns:
            OCR结果
        """
        # vLLM引擎主要用于批量处理，单个图像处理通过批量处理实现
        return self._process_batch_images([image], [prompt])[0]
    
    def _process_batch_images(self, images: List[Union[Image.Image, torch.Tensor]], prompts: List[str]) -> List[str]:
        """
        批量处理图像的具体实现
        
        Args:
            images: 图像列表
            prompts: 提示词列表
            
        Returns:
            OCR结果列表
        """
        if not self.is_available():
            raise RuntimeError("vLLM引擎未初始化或不可用")
        
        try:
            # 预处理图像
            processed_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    img = self._tensor_to_pil(img)
                processed_images.append(img)
            
            # 构建批量输入
            batch_inputs = []
            for i, (image, prompt) in enumerate(zip(processed_images, prompts)):
                # 使用processor处理图像
                processed_data = self.processor.tokenize_with_images(
                    images=[image], 
                    bos=True, 
                    eos=True, 
                    cropping=True
                )
                
                # 构造输入数据
                if processed_data and len(processed_data) > 0:
                    cache_item = {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": processed_data
                        },
                    }
                    batch_inputs.append(cache_item)
                else:
                    raise ValueError(f"图像 {i} 处理失败，未生成有效的输入数据")
            
            # 生成结果
            outputs_list = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
            
            # 解析结果
            results = []
            for output in outputs_list:
                content = output.outputs[0].text
                if '<｜end of sentence｜>' in content:
                    content = content.replace('<｜end of sentence｜>', '')
                results.append(content.strip())
            
            return results
            
        except Exception as e:
            logger.error(f"批量处理图像时发生错误: {str(e)}")
            raise
    
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
        import numpy as np
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
    
    def cleanup(self) -> None:
        """
        清理资源
        """
        try:
            if self.llm:
                del self.llm
                self.llm = None
            
            if self.processor:
                del self.processor
                self.processor = None
            
            self.is_initialized = False
            logger.info("vLLM引擎资源已清理")
            
        except Exception as e:
            logger.error(f"清理vLLM引擎资源时发生错误: {str(e)}")