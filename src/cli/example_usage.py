#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR CLI使用示例
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def example_word_to_ocr():
    """Word文档转OCR示例"""
    print("示例1: 处理Word文档")
    print("-" * 20)
    
    # 模拟命令行调用
    print("命令: python -m src.cli.main example.docx -o output_dir --mode vllm")
    print("说明: 将Word文档转换为PDF，再转为图像，最后使用vLLM进行OCR识别")
    print()

def example_pdf_to_ocr():
    """PDF文档转OCR示例"""
    print("示例2: 处理PDF文档")
    print("-" * 20)
    
    # 模拟命令行调用
    print("命令: python -m src.cli.main example.pdf -o output_dir --mode transformers")
    print("说明: 将PDF文档转为图像，最后使用Transformers进行OCR识别")
    print()

def example_image_to_ocr():
    """图像OCR示例"""
    print("示例3: 处理图像文件")
    print("-" * 20)
    
    # 模拟命令行调用
    print("命令: python -m src.cli.main example.jpg -o output_dir --mode vllm")
    print("说明: 直接对图像文件使用vLLM进行OCR识别")
    print()

def example_model_download():
    """模型下载示例"""
    print("示例4: 下载模型")
    print("-" * 20)
    
    # 模拟命令行调用
    print("命令: python -m src.cli.download_models")
    print("说明: 下载DeepSeek OCR模型到本地models目录")
    print()

def main():
    """主函数"""
    print("DeepSeek OCR CLI使用示例")
    print("=" * 30)
    
    # 显示所有示例
    example_word_to_ocr()
    example_pdf_to_ocr()
    example_image_to_ocr()
    example_model_download()
    
    print("更多帮助信息:")
    print("命令: python -m src.cli.main --help")
    print("说明: 查看所有可用的命令行参数")

if __name__ == "__main__":
    main()