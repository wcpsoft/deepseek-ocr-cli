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
    parser.add_argument("--add-custom", nargs=3, metavar=("NAME", "REPO_ID", "SOURCE"), 
                       help="添加自定义模型 (名称 仓库ID 来源[huggingface|modelscope])")
    parser.add_argument("--list-custom", action="store_true", help="列出自定义模型")
    parser.add_argument("--list-downloaded", action="store_true", help="列出已下载的模型")
    parser.add_argument("--full-download", action="store_true", 
                       help="下载完整模型（包括文档、示例等文件），默认只下载运行必需的文件")
    
    args = parser.parse_args()
    
    print("DeepSeek OCR 模型下载工具")
    print("=" * 30)
    
    # 创建模型管理器
    model_manager = ModelManager(args.dir)
    
    # 处理自定义模型添加
    if args.add_custom:
        model_name, repo_id, source = args.add_custom
        if source not in ["huggingface", "modelscope"]:
            print(f"错误: 不支持的模型来源 '{source}'，仅支持 'huggingface' 或 'modelscope'")
            sys.exit(1)
        model_manager.add_custom_model(model_name, repo_id, source)
        return
    
    # 列出自定义模型
    if args.list_custom:
        custom_models = model_manager.list_custom_models()
        if custom_models:
            print("\n自定义模型:")
            for name, info in custom_models.items():
                if isinstance(info, dict):
                    repo_id = info.get("repo_id", "N/A")
                    source = info.get("source", "huggingface")
                    print(f"  {name}: {repo_id} (来源: {source})")
                else:
                    print(f"  {name}: {info} (来源: huggingface)")
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
                    repo_id = info.get('repo_id', 'N/A')
                    source = info.get('source', 'N/A')
                    path = info.get('path', 'N/A')
                    print(f"    仓库ID: {repo_id}")
                    print(f"    来源: {source}")
                    print(f"    路径: {path}")
        else:
            print(f"\n在 {model_manager.get_model_dir()} 目录中暂无已下载的模型")
        return
    
    # 下载模型
    try:
        # 设置过滤选项（这里我们通过模型管理器的实现来控制）
        if args.full_download:
            print("注意: 将下载完整模型文件（包括文档、示例等）")
        else:
            print("注意: 默认只下载运行必需的文件")
        
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