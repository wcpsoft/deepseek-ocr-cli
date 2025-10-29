#!/usr/bin/env python3
"""
PDF转换器
将PDF文件转换为图像
"""

from pathlib import Path

# 忽略导入错误，因为这些包可能在运行时才安装
import fitz  # type: ignore # PyMuPDF
from PIL import Image


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

    def save_images(self, images: list[Image.Image], output_dir: Path, format: str = "JPEG"):
        """保存图像到指定目录"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            # 确保图像模式正确
            if format.upper() == "JPEG" and image.mode in ("RGBA", "LA"):
                # 创建白色背景
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "RGBA":
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image)
                image = background

            # 保存图像
            image_path = output_dir / f"page_{i+1:03d}.{format.lower()}"
            image.save(
                image_path,
                format=format,
                quality=95 if format.upper() == "JPEG" else None,
            )
            print(f"已保存图像: {image_path}")


def main():
    """测试函数"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="PDF转图像工具")
    parser.add_argument("input", help="输入PDF文件路径")
    parser.add_argument("-o", "--output", help="输出目录路径", default="./output")
    parser.add_argument("--dpi", type=int, default=144, help="图像DPI (默认: 144)")
    parser.add_argument("--format", choices=["JPEG", "PNG"], default="JPEG", help="输出图像格式")

    args = parser.parse_args()

    # 创建转换器
    converter = PDFConverter(dpi=args.dpi)

    # 转换PDF
    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"错误: PDF文件不存在: {pdf_path}")
        sys.exit(1)

    print(f"正在转换PDF: {pdf_path.name}")
    images = converter.pdf_to_images(pdf_path)
    print(f"转换完成，共 {len(images)} 页")

    # 保存图像
    output_dir = Path(args.output)
    converter.save_images(images, output_dir, args.format)
    print(f"图像已保存到: {output_dir}")


if __name__ == "__main__":
    main()
