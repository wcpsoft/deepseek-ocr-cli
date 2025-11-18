#!/usr/bin/env python3
"""
图像处理器
专门负责图像的加载、预处理和特征提取
"""

from typing import Any

import torch
from PIL import Image

from src.core.logging import get_logger

logger = get_logger()


class ImageHandler:
    """专门负责图像处理的类"""

    def __init__(self, tokenizer: Any):
        """
        初始化图像处理器

        Args:
            tokenizer: 分词器
        """
        self.tokenizer = tokenizer
        try:
            from src.core.process.image_process import DeepseekOCRProcessor

            self.processor = DeepseekOCRProcessor(tokenizer=tokenizer)
        except ImportError as e:
            logger.error(f"无法导入DeepseekOCRProcessor: {e}")
            raise

    def load_image(self, image_path: str) -> Image.Image:
        """
        加载图像

        Args:
            image_path: 图像路径

        Returns:
            PIL图像对象
        """
        image = Image.open(image_path).convert("RGB")
        logger.debug(f"图像加载完成，尺寸: {image.size}")
        return image

    def process_image(
        self,
        image: Image.Image | torch.Tensor,
        prompt: str = "",
        *,
        crop_mode: bool = True,
    ) -> Any:
        """
        处理图像并提取特征

        Args:
            image: PIL图像对象或张量
            prompt: 提示词
            crop_mode: 是否启用裁剪模式

        Returns:
            处理后的数据
        """
        try:
            # 如果传入的是张量，需要特殊处理
            if isinstance(image, torch.Tensor):
                logger.warning("直接传入张量输入，使用简化处理逻辑")
                # 对于张量输入，我们假设它已经是处理过的格式
                # 这里需要根据实际需求实现适当的处理逻辑
                # 暂时抛出异常，因为需要更复杂的实现
                raise NotImplementedError("直接处理张量输入尚未实现")

            # 如果没有提供提示词，使用默认的图像标记
            if not prompt:
                # 使用image_token_id而不是image_token，确保一致性
                prompt = "<image>"

            # 直接调用processor的tokenize_with_images方法，传入正确的参数
            processed_data = self.processor.tokenize_with_images(
                prompt=prompt,
                images=[image],
                inference_mode=True,
                cropping=crop_mode,
            )

            # 验证处理后的数据
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

                return processed_data
            else:
                raise ValueError("图像处理失败，未生成有效的输入数据")
        except Exception as e:
            logger.error(f"处理图像时发生错误: {e!s}")
            raise

    def extract_tensors(
        self, processed_data: Any, device: torch.device, device_manager: Any
    ) -> tuple[torch.Tensor, ...]:
        """
        从处理后的数据中提取张量并移到指定设备

        Args:
            processed_data: 处理后的数据
            device: 目标设备
            device_manager: 设备管理器实例，用于统一张量移动操作

        Returns:
            提取的张量元组
        """
        try:
            # 根据原始仓库的数据结构正确提取元素（索引0-6）
            input_ids = processed_data[0][0]
            pixel_values = processed_data[0][1]
            images_crop = processed_data[0][2]
            images_seq_mask = processed_data[0][3]
            images_spatial_crop = processed_data[0][4]
            num_image_tokens = processed_data[0][5]
            image_shapes = processed_data[0][6]

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

            # 处理num_image_tokens和image_shapes可能为空列表的情况
            # 在某些情况下，这两个值可能为空列表，这是正常的
            if num_image_tokens is None:
                num_image_tokens = []
            if image_shapes is None:
                image_shapes = []

            # 使用 DeviceManager 统一移动张量到设备
            input_ids = device_manager.move_tensor_to_device(input_ids, device)
            pixel_values = device_manager.move_tensor_to_device(pixel_values, device)
            images_crop = device_manager.move_tensor_to_device(images_crop, device)
            images_spatial_crop = device_manager.move_tensor_to_device(images_spatial_crop, device)

            return (
                input_ids,
                pixel_values,
                images_crop,
                images_seq_mask,
                images_spatial_crop,
                num_image_tokens,
                image_shapes,
            )
        except Exception as e:
            logger.error(f"提取张量时发生错误: {e!s}")
            raise
