#!/usr/bin/env python3
"""
DeepSeek OCR CLI 调试脚本
提供统一的调试入口和功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查是否启用了调试模式
DEBUG_MODE = os.getenv("DEBUG", "").upper() in ("TRUE", "1", "YES", "ON")


def debug_trace():
    """
    调试跟踪函数
    在启用调试模式时设置断点
    """
    if DEBUG_MODE:
        # 检查是否安装了ipdb
        try:
            import ipdb

            ipdb.set_trace()
        except ImportError:
            try:
                import pdb

                pdb.set_trace()
            except ImportError:
                print("警告: 未安装ipdb或pdb，无法启动调试模式")
    else:
        print("调试模式未启用，请设置环境变量 DEBUG=TRUE")


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
        if DEBUG_MODE:
            print(f"进入函数: {func.__name__}")
            # 检查是否安装了ipdb
            try:
                import ipdb

                ipdb.set_trace()
            except ImportError:
                try:
                    import pdb

                    pdb.set_trace()
                except ImportError:
                    print("警告: 未安装ipdb或pdb，无法启动调试模式")

        try:
            result = func(*args, **kwargs)

            if DEBUG_MODE:
                print(f"函数 {func.__name__} 执行完成")
                # 检查是否安装了ipdb
                try:
                    import ipdb

                    ipdb.set_trace()
                except ImportError:
                    try:
                        import pdb

                        pdb.set_trace()
                    except ImportError:
                        print("警告: 未安装ipdb或pdb，无法启动调试模式")

            return result
        except Exception as e:
            if DEBUG_MODE:
                print(f"函数 {func.__name__} 发生异常: {e}")
                # 检查是否安装了ipdb
                try:
                    import ipdb

                    ipdb.set_trace()
                except ImportError:
                    try:
                        import pdb

                        pdb.set_trace()
                    except ImportError:
                        print("警告: 未安装ipdb或pdb，无法启动调试模式")
            raise

    return wrapper


# 主调试功能
def main():
    """主调试函数"""
    print("DeepSeek OCR CLI 调试模式")
    print(f"项目根目录: {project_root}")
    print(f"调试模式: {'启用' if DEBUG_MODE else '禁用'}")

    # 如果启用了调试模式，设置断点
    debug_trace()

    # 这里可以添加更多的调试功能
    print("调试脚本执行完成")


if __name__ == "__main__":
    main()
