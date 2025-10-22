#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一调试脚本
包含CUDA警告修复和OCR测试功能
"""

import os
import sys
import warnings
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def suppress_cuda_warnings():
    """抑制CUDA相关警告"""
    # 抑制CUDA重复注册警告
    warnings.filterwarnings("ignore", message=".*Unable to register.*factory.*")
    warnings.filterwarnings("ignore", message=".*computation placer already registered.*")
    
    # 设置环境变量以减少警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow警告
    
    # 抑制absl日志消息
    os.environ['ABSL_LOG_TO_STDERR'] = '0'

def run_ocr_test():
    """运行OCR测试"""
    try:
        # 检查模型是否可用
        from cli.utils import ensure_model_available
        if not ensure_model_available(project_root):
            return 1
        
        # 导入文档处理器
        from cli.document_processor import DocumentProcessor
        
        # 检查示例文件
        samples_dir = project_root / "samples"
        if not samples_dir.exists():
            print("错误: samples目录不存在")
            return 1
        
        # 查找测试文件
        test_files = list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg"))
        if not test_files:
            print("错误: samples目录中没有找到图像文件")
            return 1
        
        # 创建输出目录
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        # 处理第一个图像文件
        print(f"正在处理文件: {test_files[0].name}")
        processor = DocumentProcessor(
            mode="transformers", 
            prompt="<image>\n<|grounding|>Convert the document to markdown."
        )
        processor.process(str(test_files[0]), str(output_dir))
        
        # 检查结果
        result_file = output_dir / "result.mmd"
        if result_file.exists():
            print(f"OCR处理完成，结果已保存到: {result_file}")
            
            # 显示结果内容
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("\n处理结果内容:")
                print("-" * 50)
                print(content[:1000] + "..." if len(content) > 1000 else content)
                print("-" * 50)
        else:
            print("错误: 未生成结果文件")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """主函数"""
    # 应用CUDA警告修复
    suppress_cuda_warnings()
    
    # 如果没有提供参数，运行默认的OCR测试
    if len(sys.argv) <= 1:
        print("正在运行OCR测试...")
        return run_ocr_test()
    
    # 根据参数执行不同的命令
    command = sys.argv[1]
    if command == "ocr":
        print("正在运行OCR测试...")
        return run_ocr_test()
    elif command == "download":
        # 运行模型下载
        try:
            from cli.download_models import main as download_main
            return download_main()
        except ImportError:
            print("错误: 无法导入download_models模块")
            return 1
    else:
        print(f"未知命令: {command}")
        print("可用命令: ocr, download")
        return 1

if __name__ == "__main__":
    sys.exit(main())