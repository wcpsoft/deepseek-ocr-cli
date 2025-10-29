#!/usr/bin/env python3
"""
统一的OCR结果处理器
负责OCR结果的保存和处理，将处理和保存逻辑分离
"""

from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.core.utils.exception_handler import OCRException, SafeExecution

logger = get_logger()


class OCRResultProcessor:
    """
    OCR结果处理器
    负责OCR结果的保存和处理
    """

    def __init__(self, output_dir: str):
        """
        初始化结果处理器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.errors = []

    def add_result(self, image_index: int, result: str) -> None:
        """
        添加OCR结果

        Args:
            image_index: 图像索引
            result: OCR结果
        """
        if result and isinstance(result, str) and len(result.strip()) > 0:
            self.results.append((image_index, result.strip()))
        else:
            logger.warning(f"图像 {image_index} OCR识别返回空结果或默认结果")
            self.errors.append((image_index, "OCR识别未返回有效结果"))

    def add_error(self, image_index: int, error: str) -> None:
        """
        添加错误信息

        Args:
            image_index: 图像索引
            error: 错误信息
        """
        logger.error(f"图像 {image_index} 处理失败: {error}")
        self.errors.append((image_index, error))

    def has_valid_results(self) -> bool:
        """
        检查是否有有效的OCR结果

        Returns:
            是否有有效结果
        """
        return len(self.results) > 0

    def has_errors(self) -> bool:
        """
        检查是否有错误

        Returns:
            是否有错误
        """
        return len(self.errors) > 0

    def get_error_count(self) -> int:
        """
        获取错误数量

        Returns:
            错误数量
        """
        return len(self.errors)

    def get_result_count(self) -> int:
        """
        获取结果数量

        Returns:
            结果数量
        """
        return len(self.results)

    def save_results(self, filename: str = "result.mmd") -> str | None:
        """
        保存OCR结果到文件

        Args:
            filename: 输出文件名

        Returns:
            输出文件路径，如果没有有效结果则返回None
        """
        if not self.has_valid_results():
            logger.warning("没有有效的OCR结果，不保存文件")
            return None

        # 按图像索引排序结果
        sorted_results = sorted(self.results, key=lambda x: x[0])

        # 构建内容
        content = ""
        for _, result in sorted_results:
            content += result + "\n"

        # 写入文件
        result_file = self.output_dir / filename
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"OCR结果已保存到: {result_file}")
            return str(result_file)
        except Exception as e:
            logger.error(f"保存OCR结果失败: {e!s}")
            return None

    def save_error_report(self, filename: str = "error_report.txt") -> str | None:
        """
        保存错误报告到文件

        Args:
            filename: 错误报告文件名

        Returns:
            错误报告文件路径，如果没有错误则返回None
        """
        if not self.has_errors():
            return None

        # 按图像索引排序错误
        sorted_errors = sorted(self.errors, key=lambda x: x[0])

        # 构建错误报告
        content = "OCR处理错误报告\n"
        content += "=" * 50 + "\n"
        for image_index, error in sorted_errors:
            content += f"图像 {image_index}: {error}\n"

        # 写入文件
        error_file = self.output_dir / filename
        try:
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"错误报告已保存到: {error_file}")
            return str(error_file)
        except Exception as e:
            logger.error(f"保存错误报告失败: {e!s}")
            return None

    def get_summary(self) -> dict[str, Any]:
        """
        获取处理结果摘要

        Returns:
            处理结果摘要字典
        """
        return {
            "total_images": self.get_result_count() + self.get_error_count(),
            "successful_results": self.get_result_count(),
            "errors": self.get_error_count(),
            "output_dir": str(self.output_dir),
            "has_valid_results": self.has_valid_results(),
            "has_errors": self.has_errors(),
        }


class BatchOCRProcessor:
    """
    批量OCR处理器
    负责批量处理图像，并在遇到错误时停止处理
    """

    def __init__(self, output_dir: str, stop_on_error: bool = True):
        """
        初始化批量处理器

        Args:
            output_dir: 输出目录路径
            stop_on_error: 是否在遇到错误时停止处理
        """
        self.output_dir = output_dir
        self.stop_on_error = stop_on_error
        self.result_processor = OCRResultProcessor(output_dir)
        self.processing_stopped = False

    def process_images(self, images: list[Any], ocr_engine) -> bool:
        """
        批量处理图像

        Args:
            images: 图像列表
            ocr_engine: OCR引擎

        Returns:
            是否成功处理所有图像
        """
        try:
            for i, image in enumerate(images):
                if self.processing_stopped:
                    logger.info("处理已停止，跳过剩余图像")
                    break

                # 使用安全执行器处理单个图像
                def process_single(img=image, idx=i):
                    return self._process_single_image(img, idx, ocr_engine)

                result = SafeExecution.safe_execute(process_single, stop_on_error=self.stop_on_error)

                if result:
                    self.result_processor.add_result(i, result)

            # 保存结果
            if self.result_processor.has_valid_results():
                self.result_processor.save_results()

            if self.result_processor.has_errors():
                self.result_processor.save_error_report()

            # 打印摘要
            summary = self.result_processor.get_summary()
            logger.info(f"处理完成: {summary}")

            return not self.result_processor.has_errors()

        except OCRException as e:
            logger.error(f"批量处理过程中发生OCR异常: {e!s}")
            if e.details:
                logger.error(f"异常详情: {e.details}")
            return False
        except Exception as e:
            logger.error(f"批量处理过程中发生严重错误: {e!s}")
            return False

    def _process_single_image(self, image: Any, image_index: int, ocr_engine) -> str | None:
        """
        处理单个图像

        Args:
            image: 图像对象
            image_index: 图像索引
            ocr_engine: OCR引擎

        Returns:
            OCR结果，如果处理失败则返回None
        """
        # 这个方法应该由具体的OCR引擎实现
        raise NotImplementedError("子类必须实现_process_single_image方法")
