#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
用于下载DeepSeek OCR模型到本地models目录
"""

import sys
import os
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.model_manager import ModelManager

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepSeek OCR 模型下载工具")
    parser.add_argument("-d", "--dir", default="./models", help="模型存储目录 (默认: ./models)")
    parser.add_argument("-m", "--model", nargs="+", help="要下载的模型名称 (默认下载所有模型)")
    parser.add_argument("-f", "--force", action="store_true", help="强制重新下载已存在的模型")
    parser.add_argument("--add-custom", nargs=2, metavar=("NAME", "REPO_ID"), help="添加自定义模型")
    parser.add_argument("--list-custom", action="store_true", help="列出所有自定义模型")
    parser.add_argument("--list-downloaded", action="store_true", help="列出已下载的模型")
    
    args = parser.parse_args()
    
    print("DeepSeek OCR 模型下载工具")
    print("=" * 30)
    
    # 创建模型管理器
    model_manager = ModelManager(args.dir)
    
    # 处理自定义模型添加
    if args.add_custom:
        model_name, repo_id = args.add_custom
        model_manager.add_custom_model(model_name, repo_id)
        return
    
    # 列出自定义模型
    if args.list_custom:
        custom_models = model_manager.list_custom_models()
        if custom_models:
            print("\n自定义模型:")
            for name, repo_id in custom_models.items():
                print(f"  {name}: {repo_id}")
        else:
            print("\n暂无自定义模型")
        return
    
    # 列出已下载模型
    if args.list_downloaded:
        downloaded_models = model_manager.list_downloaded_models()
        if downloaded_models:
            print(f"\n已下载的模型 (存储在 {model_manager.get_model_dir()}):")
            for model in downloaded_models:
                print(f"  - {model}")
                # 显示模型详细信息
                info = model_manager.get_model_info(model)
                if info:
                    print(f"    路径: {info.get('path', 'N/A')}")
        else:
            print(f"\n在 {model_manager.get_model_dir()} 目录中暂无已下载的模型")
        return
    
    # 下载模型
    try:
        model_manager.download_models(args.model, args.force)
        print(f"\n模型下载完成！存储在: {model_manager.get_model_dir()}")
        
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