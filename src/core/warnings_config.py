#!/usr/bin/env python3
"""
警告配置模块
用于配置和管理项目中的警告过滤器
"""

import warnings
import os


def configure_warnings():
    """
    配置项目警告过滤器
    
    在程序启动时调用此函数，设置所有需要的警告过滤器
    """
    # 过滤transformers库的get_max_cache弃用警告
    warnings.filterwarnings(
        "ignore",
        message="`get_max_cache\\(\\)` is deprecated for all Cache classes.*",
        category=FutureWarning,
        module="transformers.*"
    )
    
    # 过滤图像处理器未使用参数的警告
    warnings.filterwarnings(
        "ignore",
        message="Some kwargs in processor config are unused and will not have any effect.*",
        category=UserWarning,
        module="transformers.*"
    )
    
    # 过滤MPS后端不支持某些操作符的警告
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        warnings.filterwarnings(
            "ignore",
            message="The operator.*is not currently supported on the MPS backend.*",
            category=UserWarning,
            module="torch.*"
        )