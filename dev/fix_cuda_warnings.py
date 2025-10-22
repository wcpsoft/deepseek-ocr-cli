#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA警告修复脚本
用于解决CUDA库重复注册警告问题
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

def main():
    """主函数"""
    print("正在应用CUDA警告修复...")
    suppress_cuda_warnings()
    print("CUDA警告修复已应用")
    
    # 如果提供了额外的参数，则执行相应的CLI命令
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "debug":
            # 运行调试OCR
            from dev.debug_ocr import main as debug_main
            sys.exit(debug_main())
        elif command == "download":
            # 运行模型下载
            from cli.download_models import main as download_main
            sys.exit(download_main())
        else:
            print(f"未知命令: {command}")
            sys.exit(1)

if __name__ == "__main__":
    main()