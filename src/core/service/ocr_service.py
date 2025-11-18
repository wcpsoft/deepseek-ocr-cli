#!/usr/bin/env python3
"""
OCR服务实现
提供统一的OCR处理接口
"""

import os
from typing import Any

from PIL import Image
from torch import Tensor

from src.core.config.app_config import get_app_config
from src.core.utils.error_handling import handle_ocr_error, ImageProcessError, ConfigurationError

# 获取日志记录器
from src.core.logging import get_logger
from src.core.multimodal.ocr_engine_interface import OCREngineInterface
from src.core.multimodal.ocr_result_processor import BatchOCRProcessor

logger = get_logger()


class OCRService:
    """
    OCR服务类
    整合所有OCR功能，提供高级API
    """

    def __init__(self, engine: OCREngineInterface, output_dir: str | None = None):
        """
        初始化OCR服务

        Args:
            engine: OCR引擎
            output_dir: 输出目录
        """
        self.engine = engine
        config = get_app_config()
        self.output_dir = output_dir or config.output_path
        self.batch_processor = None

    @handle_ocr_error(default_return=False, context={"component": "OCRService", "operation": "initialize"})
    def initialize(self) -> bool:
        """
        初始化OCR服务

        Returns:
            是否初始化成功
        """
        logger.info("初始化OCR服务...")

        # 初始化引擎
        if not self.engine.initialize():
            logger.error("OCR引擎初始化失败")
            return False

        logger.info("OCR服务初始化成功")
        return True

    @handle_image_error(re_raise=True, context={"component": "OCRService", "operation": "process_image"})
    def process_image(self, image: Image.Image | str, prompt: str | None = None) -> str:
        """
        处理单个图像

        Args:
            image: 图像对象或图像路径
            prompt: 提示词

        Returns:
            OCR结果
        """
        # 如果是路径，加载图像
        if isinstance(image, str):
            image = self._load_image_from_path(image)

        # 使用默认提示词
        if prompt is None:
            config = get_app_config()
            prompt = config.prompt

        # 处理图像
        result = self.engine.process_image(image, prompt)

        return result

    @handle_image_error(re_raise=True, context={"component": "OCRService", "operation": "process_images"})
    def process_images(
        self,
        images: list[Image.Image | str],
        prompts: list[str] | None = None,
        output_filename: str = "result.mmd",
        *,
        stop_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        批量处理图像

        Args:
            images: 图像列表或图像路径列表
            prompts: 提示词列表
            output_filename: 输出文件名
            stop_on_error: 是否在遇到错误时停止处理

        Returns:
            处理结果字典
        """
        # 加载图像
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(self._load_image_from_path(img))
            else:
                loaded_images.append(img)

        # 使用默认提示词
        if prompts is None:
            config = get_app_config()
            prompts = [config.prompt] * len(loaded_images)

        # 创建批量处理器
        self.batch_processor = BatchOCRProcessor(self.output_dir, stop_on_error=stop_on_error)

        # 处理图像
        success = self._process_batch_with_engine(loaded_images, prompts, output_filename)

        # 获取结果摘要
        summary = self.batch_processor.result_processor.get_summary()

        return {
            "success": success,
            "summary": summary,
            "output_file": (
                os.path.join(self.output_dir, output_filename) if summary["has_valid_results"] else None
            ),
            "error_file": (os.path.join(self.output_dir, "error_report.txt") if summary["has_errors"] else None),
        }

    @handle_image_error(re_raise=True, context={"component": "OCRService", "operation": "process_document"})
    def process_document(
        self,
        document_path: str,
        output_filename: str = "result.mmd",
        prompt: str | None = None,
        *,
        stop_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        处理文档

        Args:
            document_path: 文档路径
            output_filename: 输出文件名
            prompt: 提示词
            stop_on_error: 是否在遇到错误时停止处理

        Returns:
            处理结果字典
        """
        # 将文档转换为图像
        images = self._convert_document_to_images(document_path)

        # 处理图像
        if prompt is not None:
            prompts = [prompt] * len(images)
            return self.process_images(
                images, prompts=prompts, output_filename=output_filename, stop_on_error=stop_on_error
            )
        else:
            return self.process_images(images, output_filename=output_filename, stop_on_error=stop_on_error)

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """
        从路径加载图像

        Args:
            image_path: 图像路径

        Returns:
            加载的图像
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            logger.error(f"加载图像失败: {image_path}, 错误: {e!s}")
            raise

    def _convert_document_to_images(self, document_path: str) -> list[Image.Image | str]:
        """
        将文档转换为图像

        Args:
            document_path: 文档路径

        Returns:
            图像列表
        """
        # 这里应该实现文档转换逻辑
        # 可以使用现有的文档处理逻辑
        from src.cli.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        images = processor.convert_to_images(document_path)
        # 转换为Image.Image | str类型以匹配process_images的签名
        return list(images)

    def _process_batch_with_engine(
        self, images: list[Image.Image | Tensor], prompts: list[str], output_filename: str
    ) -> bool:
        """
        使用引擎批量处理图像

        Args:
            images: 图像列表
            prompts: 提示词列表
            output_filename: 输出文件名

        Returns:
            是否成功处理所有图像
        """
        try:
            # 确保batch_processor已初始化
            if self.batch_processor is None:
                raise RuntimeError("批量处理器未初始化")

            # 批量处理图像
            results = self.engine.process_batch(images, prompts)

            # 添加结果到处理器
            for i, result in enumerate(results):
                if result and len(result.strip()) > 0:
                    self.batch_processor.result_processor.add_result(i, result)
                else:
                    self.batch_processor.result_processor.add_error(i, "OCR识别未返回有效结果")

            # 保存结果
            if self.batch_processor.result_processor.has_valid_results():
                self.batch_processor.result_processor.save_results(output_filename)

            if self.batch_processor.result_processor.has_errors():
                self.batch_processor.result_processor.save_error_report()

            # 打印摘要
            summary = self.batch_processor.result_processor.get_summary()
            logger.info(f"处理完成: {summary}")

            return not self.batch_processor.result_processor.has_errors()

        except Exception as e:
            logger.error(f"批量处理图像时发生错误: {e!s}")
            return False

    def cleanup(self) -> None:
        """
        清理资源
        """
        try:
            if self.engine:
                self.engine.cleanup()
            logger.info("OCR服务资源已清理")
        except Exception as e:
            logger.error(f"清理OCR服务资源时发生错误: {e!s}")

    def get_engine_info(self) -> dict[str, Any]:
        """
        获取引擎信息

        Returns:
            引擎信息字典
        """
        if self.engine:
            return self.engine.get_model_info()
        return {"error": "引擎未初始化"}
