#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工具脚本
提供统一的调试入口和工具函数
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查是否安装了ipdb
try:
    import ipdb
    HAS_IPDB = True
except ImportError:
    HAS_IPDB = False
    print("警告: 未安装ipdb，将使用标准pdb进行调试")
    import pdb as ipdb

# 检查是否启用了调试模式
DEBUG_MODE = os.environ.get("DEBUG", "").upper() == "TRUE"


def debug_trace():
    """
    设置调试断点
    如果安装了ipdb则使用ipdb，否则使用标准pdb
    """
    if DEBUG_MODE:
        ipdb.set_trace()
    else:
        print("调试模式未启用，请设置环境变量 DEBUG=TRUE")


def debug_wrapper(func):
    """
    调试装饰器
    在函数执行前后设置断点
    
    Args:
        func: 要调试的函数
    """
    def wrapper(*args, **kwargs):
        if DEBUG_MODE:
            print(f"进入函数: {func.__name__}")
            ipdb.set_trace()
        
        try:
            result = func(*args, **kwargs)
            
            if DEBUG_MODE:
                print(f"函数 {func.__name__} 执行完成")
                ipdb.set_trace()
                
            return result
        except Exception as e:
            if DEBUG_MODE:
                print(f"函数 {func.__name__} 发生异常: {e}")
                ipdb.set_trace()
            raise
            
    return wrapper


def main():
    """主函数"""
    print("DeepSeek OCR 调试工具")
    print(f"调试模式: {'启用' if DEBUG_MODE else '未启用'}")
    print(f"IPDB支持: {'是' if HAS_IPDB else '否'}")
    
    if not DEBUG_MODE:
        print("\n使用方法:")
        print("1. 设置环境变量: export DEBUG=TRUE")
        print("2. 运行调试命令: python -m dev.debug")
        print("3. 或者直接运行: DEBUG=TRUE python -m dev.debug")
    
    # 示例调试代码
    if DEBUG_MODE:
        print("\n设置调试断点...")
        debug_trace()


if __name__ == "__main__":
    main()