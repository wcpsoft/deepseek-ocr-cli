#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR CLI 主入口
支持多种文档格式转OCR识别
"""

import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="DeepSeek OCR CLI工具")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出目录路径", default="./output")
    parser.add_argument("-m", "--mode", choices=["auto", "vllm", "transformers"], 
                       help="推理模式 (auto: 自动选择, vllm: 使用vLLM引擎, transformers: 使用Transformers引擎)", default="auto")
    parser.add_argument("--model-path", help="模型路径", default=None)
    parser.add_argument("--prompt", help="OCR提示词", 
                       default="<image>\n<|grounding|>Convert the document to markdown.")
    parser.add_argument("--download-models", action="store_true", 
                       help="下载模型到本地")
    parser.add_argument("--base-size", type=int, default=1024, 
                       help="基础尺寸 (默认: 1024)")
    parser.add_argument("--image-size", type=int, default=640, 
                       help="图像尺寸 (默认: 640)")
    parser.add_argument("--crop-mode", action="store_true", 
                       help="是否启用裁剪模式")
    
    args = parser.parse_args()
    
    # 如果需要下载模型
    if args.download_models:
        # 延迟导入，避免在不需要时加载依赖
        from cli.model_manager import ModelManager
        model_manager = ModelManager()
        model_manager.download_models()
        return
    
    # 处理文档
    # 延迟导入，避免在不需要时加载依赖
    from cli.document_processor import DocumentProcessor
    processor = DocumentProcessor(
        mode=args.mode,
        model_path=args.model_path,
        prompt=args.prompt,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode
    )
    
    processor.process(args.input, args.output)


if __name__ == "__main__":
    main()