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
from src.core.config.app_config import get_device_config

# 导入配置
from src.core.config.settings import DEFAULT_OCR_PROMPT, MODEL_PATH
from src.core.logging import debug_trace, debug_wrapper
from src.core.models.model_manager import ModelManager
from src.core.multimodal.enhanced_result_processor import EnhancedOCRResultProcessor
from src.core.process.image_handler import ImageHandler
from src.core.utils.device_manager import get_optimal_device
from src.core.utils.generation_config import GenerationConfigManager

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
        model_manager: Optional[ModelManager] = None,
        image_handler: Optional[ImageHandler] = None,
        device_manager: Optional[Any] = None,
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
            model_manager: 注入的模型管理器
            image_handler: 注入的图像处理器
            device_manager: 注入的设备管理器
        """
        super().__init__(model_path, prompt, base_size, image_size, device, crop_mode=crop_mode)

        # 使用注入的依赖或创建默认实例
        self.model_manager = model_manager or ModelManager(model_path or MODEL_PATH)
        self.image_handler = image_handler
        self.device_manager = device_manager

    @debug_wrapper
    def initialize(self) -> bool:
        """初始化Transformers引擎"""
        debug_trace()
        try:
            # 加载模型和分词器
            self.model, self.tokenizer = self.model_manager.load_model_and_tokenizer()

            # 加载图像处理器
            self.processor = self.model_manager.load_processor()

            # 使用 DeviceManager 设置设备和移动模型
            if self.device_manager:
                self.device = self.device_manager.get_optimal_device()
                if self.model:
                    self.device_manager.move_model_to_device(self.model)
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
            else:
                # 回退到原来的设备设置方式
                from src.core.utils.device_manager import get_optimal_device
                self.device = get_optimal_device()
                if self.model and hasattr(self.model, "to"):
                    self.model = self.model.to(self.device)
                    if hasattr(self.model, "eval"):
                        self.model = self.model.eval()

            # 初始化图像处理器（使用依赖注入）
            if not self.image_handler and self.tokenizer:
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
        device = self.device or get_device_config()

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
            # 注意：这里创建的是2D注意力掩码，模型内部会将其转换为4D
            attention_mask = torch.ones_like(input_ids)

            # 确保attention_mask是正确的形状
            if attention_mask.dim() != 2:
                logger.warning(f"注意力掩码维度不正确，期望2D，实际为{attention_mask.dim()}D")
                # 如果不是2D，尝试重塑
                if attention_mask.numel() > 0:
                    attention_mask = attention_mask.view(1, -1)
                else:
                    # 如果是空张量，创建一个默认的
                    attention_mask = torch.ones(
                        1,
                        input_ids.shape[1] if input_ids.shape[1] > 0 else 1,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )

            # 使用模型的生成功能
            with torch.no_grad():
                # 获取生成配置参数
                generation_config = GenerationConfigManager.get_generation_config(self.model)

                # 确保模型有generate方法
                if not hasattr(self.model, "generate"):
                    logger.error("模型没有generate方法，无法进行生成")
                    raise RuntimeError("模型没有generate方法，无法进行生成")

                # 根据设备类型选择合适的上下文管理器和数据类型
                device_type = device.type if isinstance(device, torch.device) else "cpu"

                # 从app.yaml获取设备配置
                device_config = get_device_config(device_type)

                # 准备生成参数
                # 添加input_ids，因为模型的generate方法需要它
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "images": [(pixel_values, images_crop)],
                    "images_seq_mask": images_seq_mask.unsqueeze(0),
                    "images_spatial_crop": images_spatial_crop,
                    "max_new_tokens": generation_config["max_new_tokens"],
                    "temperature": generation_config["temperature"],
                    "do_sample": generation_config["temperature"] > 0,
                    "top_p": generation_config["top_p"],
                    "no_repeat_ngram_size": 30,  # 添加no_repeat_ngram_size参数，与原始代码一致
                    "use_cache": True,
                    "pad_token_id": (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    ),
                }

                # 根据设备类型选择执行方式
                if device_config.get("autocast") and device_config.get("dtype"):
                    # 获取数据类型
                    dtype_map = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32,
                    }
                    dtype = dtype_map.get(device_config["dtype"], torch.float32)

                    logger.debug(f"使用 autocast: {device_config['autocast']}, dtype: {dtype}")
                    with torch.autocast(device_config["autocast"], dtype=dtype):
                        outputs = self.model.generate(**generate_kwargs)
                else:
                    # 不使用autocast
                    logger.debug(f"不使用 autocast，设备类型: {device_type}")
                    outputs = self.model.generate(**generate_kwargs)
                logger.debug("模型生成完成")

                # 解码输出
                logger.debug("开始解码输出")
                if self.tokenizer is not None:
                    # 检查outputs是否有效
                    if outputs is None or outputs.numel() == 0:
                        logger.warning("模型输出为空，使用默认响应")
                        return "图像处理完成，但未生成有效文本。"

                    # 获取原始输入长度
                    input_length = input_ids.shape[1]
                    logger.debug(f"输入长度: {input_length}, 输出形状: {outputs.shape}")

                    # 检查outputs形状是否有效
                    if len(outputs.shape) < 2 or outputs.shape[1] <= input_length:
                        logger.warning(f"输出形状无效: {outputs.shape}, 输入长度: {input_length}")
                        return "图像处理完成，但输出格式不正确。"

                    # 只取生成的部分
                    new_tokens = outputs[0, input_length:]
                    logger.debug(f"新生成token数量: {new_tokens.shape[0] if new_tokens.numel() > 0 else 0}")

                    # 检查new_tokens是否为空
                    if new_tokens.numel() == 0:
                        logger.warning("没有生成新的token")
                        return "图像处理完成，但没有生成新的文本。"

                    # 解码新生成的token，不跳过特殊标记
                    result = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

                    # 直接使用strip()处理结果，与modeling_deepseekocr.py保持一致
                    clean_result = result.strip()

                    logger.debug(f"解码完成，结果长度: {len(clean_result)}")
                    logger.debug(f"解码结果: {clean_result}")
                    return clean_result
                else:
                    raise RuntimeError("解码失败：缺少tokenizer")

        except Exception as e:
            logger.error(f"图像处理失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"图像处理失败: {e!s}") from e

    @debug_wrapper
    def process_batch(self, images: list[Image.Image], prompts: list[str]) -> list[str]:
        """
        批量处理图像

        Args:
            images: 图像列表
            prompts: 提示词列表

        Returns:
            OCR结果列表
        """
        debug_trace()
        logger.debug(
            f"开始process_batch方法，图像数量: {len(images) if images else 0}, 提示词数量: {len(prompts) if prompts else 0}"
        )

        if not self.is_initialized:
            logger.debug("模型未初始化，开始初始化")
            if not self.initialize():
                raise RuntimeError("Transformers引擎初始化失败")
            logger.debug("初始化完成")

        if len(images) != len(prompts):
            raise ValueError("图像数量和提示词数量不匹配")

        results = []
        for i, (image, prompt) in enumerate(zip(images, prompts, strict=False)):
            logger.debug(f"处理第 {i+1} 张图像")
            try:
                result = self.process_image(image, prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"处理第 {i+1} 张图像时发生错误: {e!s}")
                results.append(f"处理失败: {e!s}")

        return results

    @debug_wrapper
    def process(self, images: list[Image.Image], output_dir: str) -> None:
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
                    logger.debug(f"OCR结果: {result}")

                    # 检查结果是否有效
                    if result and isinstance(result, str) and len(result.strip()) > 0:
                        logger.debug(f"结果有效，长度: {len(result.strip())}")
                        # 添加结果和元数据
                        metadata = {
                            "image_size": f"{image.size[0]}x{image.size[1]}",
                            "temp_file": str(temp_image_path.name),
                        }
                        result_processor.add_result(i, result.strip(), metadata)
                        logger.debug("结果已添加到处理器")
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
        # 清理设备缓存
        try:
            import torch

            # 使用self.device而不是get_optimal_device，确保使用正确的设备类型
            if self.device is not None:
                device_type = self.device.type
            else:
                # 如果self.device为None，则使用get_optimal_device
                from src.core.utils.device_manager import get_optimal_device

                optimal_device = get_optimal_device()
                device_type = optimal_device.type

            if device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device_type == "mps" and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif device_type == "dcu" and hasattr(torch, "dcu") and torch.dcu.is_available():
                torch.dcu.empty_cache()
            elif device_type == "amd" and hasattr(torch, "roc") and torch.roc.is_available():
                torch.roc.empty_cache()
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"清理设备缓存时出错: {e}")

        self.is_initialized = False
