#!/usr/bin/env python3
"""
Transformers引擎实现
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.core.base.ocr_engine import BaseOCREngine

# 导入配置
from src.core.config.settings import DEFAULT_OCR_PROMPT, MODEL_PATH
from src.core.logging import debug_trace, debug_wrapper
from src.core.models.model_manager import ModelManager
from src.core.multimodal.enhanced_result_processor import EnhancedOCRResultProcessor
from src.core.process.image_handler import ImageHandler
from src.core.utils.generation_config import GenerationConfigManager
from src.core.utils.mps_utils import get_optimal_device

logger = logging.getLogger(__name__)


class TransformersEngine(BaseOCREngine):
    """Transformers引擎实现"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        prompt: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        device: Optional[str] = None,
        *,
        crop_mode: bool = True,
    ):
        """
        初始化Transformers引擎

        Args:
            model_path: 模型路径
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            device: 设备类型
            crop_mode: 是否启用裁剪模式
        """
        super().__init__(model_path, prompt, base_size, image_size, device, crop_mode=crop_mode)
        self.model_manager = ModelManager(model_path or MODEL_PATH)
        self.image_handler = None

    @debug_wrapper
    def initialize(self) -> bool:
        """初始化Transformers引擎"""
        debug_trace()
        try:
            # 加载模型和分词器
            self.model, self.tokenizer = self.model_manager.load_model_and_tokenizer()

            # 加载图像处理器
            self.processor = self.model_manager.load_processor()

            # 设置设备
            self.device = self.model_manager.setup_device()

            # 将模型移到设备上
            self.model_manager.move_model_to_device()

            # 初始化图像处理器
            if self.tokenizer:
                self.image_handler = ImageHandler(self.tokenizer)

            self.is_initialized = True

            # 添加调试信息
            logger.debug(f"模型初始化完成，设备: {self.device}, 模型类型: {type(self.model)}")
            debug_trace()
            return True
        except Exception as e:
            logger.error(f"Transformers引擎初始化失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False

    def process_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        处理单张图像并返回OCR结果

        Args:
            image: PIL图像对象
            prompt: 提示词

        Returns:
            OCR识别结果
        """
        debug_trace()
        logger.debug(f"开始process_image方法，图像尺寸: {image.size}")
        if not self.is_initialized:
            logger.debug("模型未初始化，开始初始化")
            if not self.initialize():
                raise RuntimeError("Transformers引擎初始化失败")
            logger.debug("初始化完成")

        # 确保image_handler已初始化
        if self.image_handler is None:
            raise RuntimeError("图像处理器未初始化")

        # 确保设备已设置
        device = self.device or get_optimal_device()

        try:
            # 使用图像处理器处理图像和提示词
            prompt_text = prompt or self.prompt or DEFAULT_OCR_PROMPT
            logger.info(f"开始OCR识别，提示词: {prompt_text}")

            # 处理图像和提示词
            processed_data = self.image_handler.process_image(image, prompt_text, crop_mode=self.crop_mode)
            logger.debug("图像处理完成")

            # 提取处理后的数据
            if (
                processed_data is None
                or len(processed_data) == 0
                or processed_data[0] is None
                or len(processed_data[0]) < 7
            ):
                raise ValueError("图像处理失败，未生成有效的输入数据")

            # 提取张量并移到设备
            (
                input_ids,
                pixel_values,
                images_crop,
                images_seq_mask,
                images_spatial_crop,
                _num_image_tokens,  # 未使用的变量，添加下划线前缀
                _image_shapes,  # 未使用的变量，添加下划线前缀
            ) = self.image_handler.extract_tensors(processed_data, device)

            # 构造注意力掩码
            attention_mask = torch.ones_like(input_ids)

            # 使用模型的生成功能
            with torch.no_grad():
                # 获取生成配置参数
                generation_config = GenerationConfigManager.get_generation_config(self.model)

                # 确保模型有generate方法
                if not hasattr(self.model, "generate"):
                    logger.error("模型没有generate方法，无法进行生成")
                    raise RuntimeError("模型没有generate方法，无法进行生成")

                # 生成结果 - 使用正确的参数格式
                logger.debug("开始模型生成")
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=generation_config.get("max_new_tokens", 8192),
                    do_sample=generation_config.get("do_sample", False),
                    pad_token_id=(self.tokenizer.eos_token_id if self.tokenizer is not None else 0),
                    # 传递图像特征给模型 - 使用正确的格式
                    images=[(images_crop, pixel_values)],
                    images_seq_mask=images_seq_mask.unsqueeze(0),
                    images_spatial_crop=images_spatial_crop,
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
                    logger.debug(f"解码结果: {result}")
                    return result.strip()
                else:
                    raise RuntimeError("解码失败：缺少tokenizer")

        except Exception as e:
            logger.error(f"图像处理失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"图像处理失败: {e!s}") from e

    @debug_wrapper
    def process(self, images: list[Image.Image], output_dir: str) -> None:  # noqa: C901
        """
        使用Transformers引擎处理图像

        Args:
            images: 图像列表
            output_dir: 输出目录路径
        """
        debug_trace()
        logger.debug(f"开始process方法，图像数量: {len(images) if images else 0}, 输出目录: {output_dir}")
        if not self.is_initialized:
            logger.debug("模型未初始化，开始初始化")
            if not self.initialize():
                raise RuntimeError("Transformers引擎初始化失败")
            logger.debug("初始化完成")

        # 确保image_handler已初始化
        if self.image_handler is None:
            raise RuntimeError("图像处理器未初始化")

        # 确保设备已设置
        device = self.device or get_optimal_device()

        try:
            output_dir_path = Path(output_dir)
            # 确保输出目录存在，使用绝对路径
            output_dir_path = output_dir_path.resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录已创建: {output_dir_path}")

            # 使用增强版的结果处理器
            result_processor = EnhancedOCRResultProcessor(str(output_dir_path))

            # 添加处理元数据
            result_processor.add_metadata("engine", "Transformers")
            result_processor.add_metadata("model_path", self.model_manager.model_path)
            result_processor.add_metadata("image_count", len(images))

            # 处理每张图像
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
                    # 使用process_image方法处理图像
                    result = self.process_image(image, self.prompt)
                    logger.info(f"第 {i+1} 张图像OCR识别完成")

                    # 检查结果是否有效
                    if result and isinstance(result, str) and len(result.strip()) > 0:
                        # 添加结果和元数据
                        metadata = {
                            "image_size": f"{image.size[0]}x{image.size[1]}",
                            "temp_file": str(temp_image_path.name),
                        }
                        result_processor.add_result(i, result.strip(), metadata)
                    else:
                        logger.warning(f"第 {i+1} 张图像OCR识别返回空结果或默认结果")
                        result_processor.add_error(
                            i,
                            "OCR识别未返回有效结果",
                            {
                                "image_size": f"{image.size[0]}x{image.size[1]}",
                                "temp_file": str(temp_image_path.name),
                            },
                        )
                except Exception as infer_error:
                    logger.warning(f"图像 {i+1} 处理失败 ({infer_error!s})")
                    result_processor.add_error(
                        i,
                        str(infer_error),
                        {
                            "image_size": (f"{image.size[0]}x{image.size[1]}" if image else None),
                            "temp_file": (str(temp_image_path.name) if temp_image_path.exists() else None),
                        },
                    )
                finally:
                    # 删除临时文件
                    if temp_image_path.exists():
                        temp_image_path.unlink()
                        logger.info(f"临时文件已删除: {temp_image_path}")

            # 保存结果
            logger.info("开始保存OCR结果")
            result_processor.save_results("result.md")
            result_processor.save_error_report()
            logger.info("OCR处理完成")
        except Exception as e:
            logger.error(f"Transformers OCR执行失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"Transformers OCR执行失败: {e!s}") from e

    def cleanup(self) -> None:
        """清理资源"""
        # Transformers引擎不需要特殊清理
        self.is_initialized = False


