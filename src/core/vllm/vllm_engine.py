#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM引擎实现
"""

import os
import sys
from typing import List, Any, Optional
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.base.ocr_engine import BaseOCREngine
from src.core.config import MODEL_PATH, PROMPT
from src.core.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from src.core.process.image_process import DeepseekOCRProcessor


class VLLMEngine(BaseOCREngine):
    """vLLM引擎实现"""
    
    def __init__(self, model_path: Optional[str] = None, prompt: Optional[str] = None,
                 base_size: int = 1024, image_size: int = 640, crop_mode: bool = True):
        """
        初始化vLLM引擎
        
        Args:
            model_path: 模型路径
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否启用裁剪模式
        """
        super().__init__(model_path, prompt, base_size, image_size, crop_mode)
        self.llm = None
        self.sampling_params = None
    
    def initialize(self) -> None:
        """初始化vLLM引擎"""
        try:
            # 延迟导入，避免在不需要时加载依赖
            from src.core.deepseek_ocr import DeepseekOCRForCausalLM
            from vllm.model_executor.models.registry import ModelRegistry
            from vllm import LLM, SamplingParams
            
            # 注册模型
            ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
            
            # 设置模型路径
            model_path = self.model_path or MODEL_PATH
            prompt = self.prompt or PROMPT
            
            self.llm = LLM(
                model=model_path,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
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
            
            logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, 
                                                            whitelist_token_ids={128821, 128822})]
            
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )
        except Exception as e:
            raise RuntimeError(f"vLLM引擎初始化失败: {str(e)}")
    
    def process(self, images: List[Image.Image], output_dir: str) -> None:
        """
        使用vLLM引擎处理图像
        
        Args:
            images: 图像列表
            output_dir: 输出目录路径
        """
        if self.llm is None:
            self.initialize()
        
        try:
            # 处理图像
            batch_inputs = []
            for image in images:
                # 使用processor处理图像
                processor = DeepseekOCRProcessor()
                
                # 处理图像和提示词
                prompt = self.prompt or "<image>\n<|grounding|>Convert the document to markdown."
                # 使用tokenize_with_images方法处理图像
                processed_data = processor.tokenize_with_images(
                    images=[image], 
                    bos=True, 
                    eos=True, 
                    cropping=self.crop_mode
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
                    raise ValueError("图像处理失败，未生成有效的输入数据")
            
            # 生成结果
            if self.llm is not None:
                outputs_list = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
            else:
                raise RuntimeError("vLLM引擎未正确初始化")
            
            # 保存结果
            self._save_results(outputs_list, Path(output_dir))
            
        except Exception as e:
            raise RuntimeError(f"vLLM OCR执行失败: {str(e)}")
    
    def _save_results(self, outputs_list: list, output_dir: Path) -> None:
        """
        保存OCR结果
        
        Args:
            outputs_list: 输出结果列表
            output_dir: 输出目录路径
        """
        # 保存原始结果
        contents = ''
        for output in outputs_list:
            content = output.outputs[0].text
            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
            contents += content + '\n'
        
        # 写入文件
        result_file = output_dir / "result.mmd"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        print(f"OCR结果已保存到: {result_file}")
    
    def cleanup(self) -> None:
        """清理资源"""
        # vLLM引擎不需要特殊清理
        pass