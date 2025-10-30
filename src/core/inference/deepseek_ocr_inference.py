#!/usr/bin/env python3
"""
重构后的DeepSeek OCR推理类
符合单一职责原则，专注于OCR推理逻辑
"""


import torch
from PIL import Image

from src.core.config import get_config
from src.core.config.prompts import DEFAULT_OCR_PROMPT
from src.core.logging import get_logger
from src.core.models.model_manager import ModelManager
from src.core.multimodal.enhanced_result_processor import EnhancedOCRResultProcessor
from src.core.multimodal.ocr_engine_interface import BaseOCREngine
from src.core.process.image_handler import ImageHandler
from src.core.utils.mps_utils import get_optimal_device

logger = get_logger()


class DeepSeekOCRInference(BaseOCREngine):
    """
    DeepSeek OCR推理引擎
    专注于OCR推理逻辑，符合单一职责原则
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        """
        初始化DeepSeek OCR推理引擎

        Args:
            model_path: 模型路径
            device: 设备类型
        """
        config = get_config()
        model_path = model_path or config.MODEL_PATH

        super().__init__(model_path, device)
        self.model_manager = ModelManager(model_path)
        self.image_handler = None
        self.prompt = DEFAULT_OCR_PROMPT

    def initialize(self) -> bool:
        """
        初始化模型和处理器

        Returns:
            是否初始化成功
        """
        try:
            logger.info("初始化DeepSeek OCR推理引擎...")

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

            # 标记初始化成功
            self.is_initialized = True

            logger.info("DeepSeek OCR推理引擎初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化DeepSeek OCR推理引擎失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False

    def _process_single_image(self, image: Image.Image | torch.Tensor, prompt: str) -> str:
        """
        处理单个图像的具体实现

        Args:
            image: 图像对象或张量
            prompt: 提示词

        Returns:
            OCR结果
        """
        if not self.is_initialized or not self.image_handler:
            raise RuntimeError("OCR引擎未初始化或图像处理器不可用")

        try:
            # 如果传入的是张量，需要特殊处理
            if isinstance(image, torch.Tensor):
                logger.warning("直接传入张量输入，使用简化处理逻辑")
                # 对于张量输入，我们假设它已经是处理过的格式
                # 这里需要根据实际需求实现适当的处理逻辑
                # 暂时抛出异常，因为需要更复杂的实现
                raise NotImplementedError("直接处理张量输入尚未实现")

            # 处理图像（这里假设输入是图像对象）
            processed_data = self.image_handler.process_image(image, prompt)

            # 确保设备已设置
            device = self.model_manager.device or get_optimal_device()

            # 提取张量并移到设备
            (
                input_ids,
                pixel_values,
                images_crop,
                images_seq_mask,
                images_spatial_crop,
                _num_image_tokens,  # 未使用的变量
                _image_shapes,  # 未使用的变量
            ) = self.image_handler.extract_tensors(processed_data, device)

            # 构造注意力掩码
            attention_mask = torch.ones_like(input_ids)

            # 生成结果
            with torch.no_grad():
                # 获取生成配置参数
                from src.core.utils.generation_config import GenerationConfigManager

                generation_config = GenerationConfigManager.get_generation_config(self.model)

                # 确保模型有generate方法
                if not hasattr(self.model, "generate"):
                    logger.error("模型没有generate方法，无法进行生成")
                    raise RuntimeError("模型没有generate方法，无法进行生成")

                # 生成结果
                logger.debug("开始模型生成")
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=generation_config.get("max_new_tokens", 8192),
                    do_sample=generation_config.get("do_sample", False),
                    pad_token_id=(self.tokenizer.eos_token_id if self.tokenizer is not None else 0),
                    # 传递图像特征给模型
                    images=[[pixel_values, images_crop, images_spatial_crop]],
                    images_seq_mask=images_seq_mask,
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
                    return result
                else:
                    raise RuntimeError("解码失败：缺少tokenizer")
        except Exception as e:
            logger.error(f"处理图像时发生错误: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise

    def _process_batch_images(self, images: list[Image.Image | torch.Tensor], prompts: list[str]) -> list[str]:
        """
        批量处理图像的具体实现

        Args:
            images: 图像列表
            prompts: 提示词列表

        Returns:
            OCR结果列表
        """
        results = []
        for i, (image, prompt) in enumerate(zip(images, prompts, strict=False)):
            try:
                result = self._process_single_image(image, prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"处理第{i+1}张图像时发生错误: {e!s}")
                results.append("")
        return results

    def process_with_results(
        self,
        images: list[Image.Image | torch.Tensor],
        output_dir: str,
        prompts: list[str] | None = None,
    ) -> EnhancedOCRResultProcessor:
        """
        处理图像并返回增强版结果处理器

        Args:
            images: 图像列表
            output_dir: 输出目录
            prompts: 提示词列表

        Returns:
            增强版结果处理器
        """
        if prompts is None:
            prompts = [self.prompt] * len(images)
        elif len(prompts) != len(images):
            raise ValueError("提示词数量必须与图像数量相同")

        # 创建结果处理器
        result_processor = EnhancedOCRResultProcessor(output_dir)

        # 添加元数据
        result_processor.add_metadata("engine", "DeepSeekOCRInference")
        result_processor.add_metadata("model_path", self.model_manager.model_path)
        result_processor.add_metadata("image_count", len(images))

        # 处理每张图像
        for i, (image, prompt) in enumerate(zip(images, prompts, strict=False)):
            try:
                result = self._process_single_image(image, prompt)
                # 添加结果和元数据
                metadata = {"image_type": type(image).__name__}
                if isinstance(image, Image.Image):
                    metadata["image_size"] = f"{image.size[0]}x{image.size[1]}"
                result_processor.add_result(i, result, metadata)
            except Exception as e:
                logger.error(f"处理第{i+1}张图像时发生错误: {e!s}")
                result_processor.add_error(i, str(e), {"image_type": type(image).__name__})

        return result_processor

    def cleanup(self) -> None:
        """
        清理资源
        """
        super().cleanup()
        if hasattr(self, "model_manager"):
            # ModelManager会处理自己的资源清理
            pass
        if hasattr(self, "image_handler"):
            del self.image_handler
            self.image_handler = None
        logger.info("DeepSeek OCR推理引擎资源已清理")
