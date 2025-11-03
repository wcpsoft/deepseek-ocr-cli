#!/usr/bin/env python3
"""
调试工具模块
提供统一的调试功能
"""

import os


def debug_trace():
    """
    调试跟踪函数
    在启用调试模式时设置断点
    """
    # 检查是否启用了调试模式
    if os.getenv("DEBUG", "").upper() in ("TRUE", "1", "YES", "ON"):
        set_debugger_trace()


def debug_wrapper(func):
    """
    调试装饰器
    在启用调试模式时自动设置断点

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """

    def wrapper(*args, **kwargs):
        # 检查是否启用了调试模式
        if os.getenv("DEBUG", "").upper() in ("TRUE", "1", "YES", "ON"):
            print(f"进入函数: {func.__name__}")
            set_debugger_trace()

        try:
            result = func(*args, **kwargs)

            # 检查是否启用了调试模式
            if os.getenv("DEBUG", "").upper() in ("TRUE", "1", "YES", "ON"):
                print(f"函数 {func.__name__} 执行完成")
                set_debugger_trace()

            return result
        except Exception as e:
            # 检查是否启用了调试模式
            if os.getenv("DEBUG", "").upper() in ("TRUE", "1", "YES", "ON"):
                print(f"函数 {func.__name__} 发生异常: {e}")
                set_debugger_trace()
            raise

    return wrapper


def set_debugger_trace():
    """
    设置调试器断点
    优先使用ipdb，回退到pdb
    只有在明确启用IPDB模式时才会进入调试器
    """
    # 检查是否明确启用了IPDB调试模式
    if os.getenv("IPDB", "").upper() in ("TRUE", "1", "YES", "ON"):
        try:
            # 优先使用ipdb
            import ipdb  # type: ignore[import-not-found]

            ipdb.set_trace()
        except ImportError:
            try:
                # 回退到pdb
                import pdb  # type: ignore[import-not-found]

                pdb.set_trace()
            except ImportError:
                # 如果都没有安装，只打印消息
                print("警告: 未安装ipdb或pdb，无法启动调试模式")
    # 如果没有启用IPDB模式，即使启用了DEBUG模式也不会自动进入调试器
    # DEBUG模式仅用于启用详细日志，不自动进入调试器