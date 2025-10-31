#!/usr/bin/env python3
"""
PDF转换器
将PDF文件转换为图像
"""

import logging
from pathlib import Path

# 忽略导入错误，因为这些包可能在运行时才安装
import fitz  # type: ignore # PyMuPDF
from PIL import Image

# 获取日志记录器
logger = logging.getLogger(__name__)


class PDFConverter:
    def __init__(self, dpi: int = 144):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: Path) -> list[Image.Image]:
        """将PDF转换为图像列表"""
        images = []

        # 打开PDF文档
        pdf_document = fitz.open(str(pdf_path))

        # 计算缩放比例
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # 遍历每一页
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # 获取页面的像素图
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            # 转换为PIL图像
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            images.append(image)

        # 关闭PDF文档
        pdf_document.close()

        return images

    def save_images(self, images: list[Image.Image], output_dir: Path, img_format: str = "JPEG"):
        """保存图像到指定目录"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            # 为每种格式设置适当的文件扩展名
            if img_format.upper() in ["JPEG", "JPG"]:
                ext = "jpg"
            elif img_format.upper() == "PNG":
                ext = "png"
            else:
                ext = img_format.lower()

            image_path = output_dir / f"page_{i+1:03d}.{ext}"
            image.save(image_path, format=img_format, quality=95 if img_format.upper() in ["JPEG", "JPG"] else None)
            logger.info(f"已保存图像: {image_path}")
