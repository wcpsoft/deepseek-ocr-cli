#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR CLI 入口
支持多种文档格式转OCR识别
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 延迟导入，避免在不需要时加载依赖
def main():
    # 从cli目录导入主函数
    from cli.main import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()