#!/usr/bin/env python3
"""
增强版OCR结果处理器
提供更强大的结果处理、过滤和保存功能
"""

import re
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

logger = get_logger()


class EnhancedOCRResultProcessor:
    """
    增强版OCR结果处理器
    提供更强大的结果处理、过滤和保存功能
    """

    def __init__(self, output_dir: str, *, filter_empty_results: bool = True) -> None:
        """
        初始化增强版结果处理器

        Args:
            output_dir: 输出目录路径
            filter_empty_results: 是否过滤空结果
        """
        self.output_dir = Path(output_dir)
        self.filter_empty_results = filter_empty_results
        self.results: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.image_counter = 0

    def add_result(self, image_index: int, result: str, metadata: dict[str, Any] | None = None) -> None:
        """
        添加OCR结果

        Args:
            image_index: 图像索引
            result: OCR结果
            metadata: 结果元数据
        """
        # 使用日志系统替代print语句
        logger.debug(f"添加结果: 图像索引={image_index}, 结果长度={len(result) if result else 0}")
        logger.debug(f"结果内容: {result}")

        # 过滤空结果
        if self.filter_empty_results and (not result or not isinstance(result, str) or len(result.strip()) == 0):
            logger.debug(f"图像 {image_index} OCR识别返回空结果，已过滤")
            self.errors.append({"index": image_index, "error": "OCR识别未返回有效结果"})
            return

        # 清理结果内容
        cleaned_result = self._clean_result(result)
        logger.debug(f"清理后结果长度: {len(cleaned_result)}")

        # 如果清理后结果为空且启用了过滤，则记录为错误
        if self.filter_empty_results and (not cleaned_result or len(cleaned_result.strip()) == 0):
            logger.debug(f"图像 {image_index} OCR识别返回空结果（清理后），已过滤")
            self.errors.append({"index": image_index, "error": "OCR识别未返回有效结果"})
            return

        self.results.append({"index": image_index, "result": cleaned_result, "metadata": metadata or {}})
        logger.debug(f"结果已添加到结果列表，当前结果数量: {len(self.results)}")

    def add_error(self, image_index: int, error: str, metadata: dict[str, Any] | None = None) -> None:
        """
        添加错误信息

        Args:
            image_index: 图像索引
            error: 错误信息
            metadata: 错误元数据
        """
        logger.error(f"图像 {image_index} 处理失败: {error}")
        self.errors.append({"index": image_index, "error": error, "metadata": metadata or {}})

    def _clean_result(self, result: str) -> str:
        """
        清理OCR结果，移除无关内容

        Args:
            result: 原始OCR结果

        Returns:
            清理后的结果
        """
        if not result or not isinstance(result, str):
            return ""

        # 移除多余的空白字符
        cleaned = re.sub(r"\s+", " ", result.strip())

        # 移除常见的无关内容（可以根据需要扩展）
        # 移除以特殊标记开头的行
        lines = cleaned.split("\n")
        filtered_lines = []
        for line in lines:
            # 跳过空行
            if not line.strip():
                continue
            # 跳过以特殊标记开头的行
            if line.strip().startswith(("=", "-", "*", "#", "---", "***")):
                continue
            filtered_lines.append(line)

        return "\n".join(filtered_lines)

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

    def filter_results(self, filter_func) -> list[dict[str, Any]]:
        """
        根据过滤函数过滤结果

        Args:
            filter_func: 过滤函数

        Returns:
            过滤后的结果列表
        """
        return [result for result in self.results if filter_func(result)]

    def get_results_by_index(self, start_index: int, end_index: int) -> list[dict[str, Any]]:
        """
        根据索引范围获取结果

        Args:
            start_index: 起始索引
            end_index: 结束索引

        Returns:
            指定范围内的结果列表
        """
        return [result for result in self.results if start_index <= result["index"] <= end_index]

    def save_results(self, filename: str = "result.md", format_type: str = "markdown") -> str | None:
        """
        保存OCR结果到文件

        Args:
            filename: 输出文件名
            format_type: 输出格式类型 (markdown, text, json)

        Returns:
            输出文件路径，如果没有有效结果则返回None
        """
        logger.debug(f"开始保存结果，当前结果数量: {len(self.results)}")
        if not self.has_valid_results():
            logger.debug("没有有效的OCR结果，不保存文件")
            return None

        # 按图像索引排序结果
        sorted_results = sorted(self.results, key=lambda x: x["index"])
        logger.debug(f"排序后结果数量: {len(sorted_results)}")

        # 根据格式类型生成内容
        if format_type.lower() == "json":
            content = self._generate_json_content(sorted_results)
        elif format_type.lower() == "text":
            content = self._generate_text_content(sorted_results)
        else:  # 默认为markdown
            content = self._generate_markdown_content(sorted_results)

        logger.debug(f"生成内容长度: {len(content)}")
        logger.debug(f"内容预览: {content[:200]}...")

        # 写入文件
        result_file = self.output_dir / filename
        logger.debug(f"准备写入文件: {result_file}")
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.debug(f"OCR结果已保存到: {result_file}")
            return str(result_file)
        except Exception as e:
            logger.debug(f"保存OCR结果失败: {e}")
            return None

    def _generate_markdown_content(self, results: list[dict[str, Any]]) -> str:
        """
        生成Markdown格式内容

        Args:
            results: 结果列表

        Returns:
            Markdown格式内容
        """
        content = ""
        for result in results:
            # 为每张图像的结果添加标题
            content += f"## 图像 {result['index'] + 1}\n\n"
            content += f"{result['result']}\n\n"
            content += "---\n\n"
        return content

    def _generate_text_content(self, results: list[dict[str, Any]]) -> str:
        """
        生成纯文本格式内容

        Args:
            results: 结果列表

        Returns:
            纯文本格式内容
        """
        content = ""
        for result in results:
            content += f"[图像 {result['index'] + 1}]\n"
            content += f"{result['result']}\n\n"
        return content

    def _generate_json_content(self, results: list[dict[str, Any]]) -> str:
        """
        生成JSON格式内容

        Args:
            results: 结果列表

        Returns:
            JSON格式内容
        """
        import json

        output_data = {
            "results": results,
            "errors": self.errors,
            "metadata": self.metadata,
        }
        return json.dumps(output_data, ensure_ascii=False, indent=2)

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
        sorted_errors = sorted(self.errors, key=lambda x: x["index"])

        # 构建错误报告
        content = "OCR处理错误报告\n"
        content += "=" * 50 + "\n"
        for error in sorted_errors:
            content += f"图像 {error['index'] + 1}: {error['error']}\n"
            # 如果有元数据，也添加到报告中
            if error.get("metadata"):
                content += f"  元数据: {error['metadata']}\n"
            content += "\n"

        # 写入文件
        error_file = self.output_dir / filename
        try:
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"错误报告已保存到: {error_file}")
            return str(error_file)
        except Exception as e:
            logger.error(f"保存错误报告失败: {e}")
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
            "metadata": self.metadata,
        }

    def add_metadata(self, key: str, value: Any) -> None:
        """
        添加元数据

        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value

    def merge_processor(self, other_processor: "EnhancedOCRResultProcessor") -> None:
        """
        合并另一个结果处理器的数据

        Args:
            other_processor: 另一个结果处理器
        """
        # 合并结果
        self.results.extend(other_processor.results)
        # 合并错误
        self.errors.extend(other_processor.errors)
        # 合并元数据
        self.metadata.update(other_processor.metadata)

    def clear(self) -> None:
        """
        清空所有结果和错误
        """
        self.results.clear()
        self.errors.clear()
        self.metadata.clear()


class BatchOCRResultProcessor:
    """
    批量OCR结果处理器
    负责处理批量OCR任务的结果
    """

    def __init__(self, output_dir: str):
        """
        初始化批量结果处理器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processors = {}

    def add_processor(self, batch_id: str, processor: EnhancedOCRResultProcessor) -> None:
        """
        添加结果处理器

        Args:
            batch_id: 批次ID
            processor: 结果处理器
        """
        self.processors[batch_id] = processor

    def save_all_results(self, filename_prefix: str = "batch_result") -> dict[str, str | None]:
        """
        保存所有批次的结果

        Args:
            filename_prefix: 文件名前缀

        Returns:
            保存结果的字典
        """
        results = {}
        for batch_id, processor in self.processors.items():
            filename = f"{filename_prefix}_{batch_id}.md"
            result_path = processor.save_results(filename)
            results[batch_id] = result_path
        return results

    def get_overall_summary(self) -> dict[str, Any]:
        """
        获取整体摘要

        Returns:
            整体摘要字典
        """
        total_results = 0
        total_errors = 0

        for processor in self.processors.values():
            total_results += processor.get_result_count()
            total_errors += processor.get_error_count()

        return {
            "total_batches": len(self.processors),
            "total_results": total_results,
            "total_errors": total_errors,
            "batch_ids": list(self.processors.keys()),
        }
