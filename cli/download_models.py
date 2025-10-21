#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
用于下载DeepSeek OCR模型到本地models目录
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.model_manager import ModelManager

def main():
    """主函数"""
    print("DeepSeek OCR 模型下载工具")
    print("=" * 30)
    
    # 创建模型管理器
    model_manager = ModelManager("./models")
    
    # 下载默认模型
    try:
        model_manager.download_models()
        print("\n所有模型下载完成！")
        
        # 列出已下载的模型
        downloaded_models = model_manager.list_downloaded_models()
        if downloaded_models:
            print("\n已下载的模型:")
            for model in downloaded_models:
                print(f"  - {model}")
        else:
            print("\n暂无已下载的模型")
            
    except Exception as e:
        print(f"模型下载失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()