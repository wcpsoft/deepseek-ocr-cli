#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformers引擎实现
"""

import os
import sys
import torch
from typing import List, Optional
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 在导入其他模块之前注册配置类
try:
    from transformers import CONFIG_MAPPING
    from src.core.deepseek_ocr_config import DeepseekVLV2Config, DeepseekV2Config
    
    # 立即注册配置类（如果尚未注册）
    if "deepseek_vl_v2" not in CONFIG_MAPPING:
        CONFIG_MAPPING["deepseek_vl_v2"] = DeepseekVLV2Config
    if "deepseek_v2" not in CONFIG_MAPPING:
        CONFIG_MAPPING["deepseek_v2"] = DeepseekV2Config
except Exception as e:
    pass  # 在模块加载时忽略错误

from src.core.base.ocr_engine import BaseOCREngine
from src.cli.utils import get_compatible_device, should_use_bfloat16
from src.core.logging import get_logger, debug_trace, debug_wrapper

# 获取日志记录器
logger = get_logger()


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
            # 设置模型路径，优先使用本地模型
            model_name = self.model_path or 'deepseek-ai/DeepSeek-OCR'
            local_model_path = "./models/deepseek-ocr"
            
            # 检查本地模型是否存在
            if os.path.exists(local_model_path):
                logger.info(f"使用本地模型: {local_model_path}")
                model_name = local_model_path
            else:
                logger.info(f"本地模型不存在，将从远程下载: {model_name}")
            
            # 使用统一的模型初始化器
            from src.core.model_initializer import ModelInitializer
            self.model, self.tokenizer = ModelInitializer.initialize_transformers_model_and_tokenizer(model_name, trust_remote_code=True)
            
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
                    # 使用模型的infer方法处理图像
                    prompt = self.prompt or "<image>\n<|grounding|>Convert the document to markdown."
                    logger.info(f"开始OCR识别第 {i+1} 张图像")
                    # 确保传递正确的image_file参数
                    if self.model is not None:
                        logger.debug("调用模型的infer方法")
                        # 检查模型是否有infer方法
                        if hasattr(self.model, 'infer') and callable(getattr(self.model, 'infer', None)):
                            result = self.model.infer(
                                tokenizer=self.tokenizer,
                                prompt=prompt,
                                image_file=str(temp_image_path),  # 确保传递正确的文件路径
                                output_path=str(output_dir_path),      # 传递输出路径
                                base_size=self.base_size,
                                image_size=self.image_size,
                                crop_mode=self.crop_mode
                            )
                        else:
                            raise RuntimeError("模型没有实现infer方法")
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
        result_file = output_dir / "result.mmd"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        logger.info(f"OCR结果已保存到: {result_file}")
        logger.info(f"成功处理 {len(successful_results)} 张图像")
        debug_trace()
    
    def cleanup(self) -> None:
        """清理资源"""
        # Transformers引擎不需要特殊清理
        pass










