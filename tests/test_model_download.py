#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载测试脚本
测试默认模型 deepseek-ai/DeepSeek-OCR 的下载和验证功能
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_default_model_download():
    """测试默认模型下载"""
    try:
        from cli.model_manager import ModelManager
        
        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            manager = ModelManager(str(model_dir))
            
            print(f"测试默认模型下载到: {model_dir}")
            
            # 下载默认模型
            print("开始下载默认模型 deepseek-ai/DeepSeek-OCR...")
            manager.download_models(["deepseek-ocr"])
            
            # 检查模型是否下载成功
            downloaded_models = manager.list_downloaded_models()
            if "deepseek-ocr" in downloaded_models:
                print("✓ 默认模型下载成功")
                
                # 检查模型路径
                model_path = manager.get_model_path("deepseek-ocr")
                if model_path and Path(model_path).exists():
                    print(f"✓ 模型路径存在: {model_path}")
                    
                    # 检查模型信息
                    model_info = manager.get_model_info("deepseek-ocr")
                    if model_info:
                        print(f"✓ 模型信息获取成功:")
                        print(f"  仓库ID: {model_info.get('repo_id', 'N/A')}")
                        print(f"  来源: {model_info.get('source', 'N/A')}")
                        print(f"  路径: {model_info.get('path', 'N/A')}")
                        
                        # 验证模型完整性
                        if manager.verify_model("deepseek-ocr"):
                            print("✓ 模型完整性验证通过")
                            return True
                        else:
                            print("✗ 模型完整性验证失败")
                            return False
                    else:
                        print("✗ 无法获取模型信息")
                        return False
                else:
                    print("✗ 模型路径不存在")
                    return False
            else:
                print("✗ 默认模型未在下载列表中")
                return False
                
    except Exception as e:
        print(f"✗ 默认模型下载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_modelscope_model_download():
    """测试ModelScope模型下载（示例）"""
    try:
        from cli.model_manager import ModelManager
        
        # 检查是否安装了modelscope
        try:
            import modelscope
            print("检测到ModelScope支持")
        except ImportError:
            print("未安装ModelScope，跳过ModelScope模型下载测试")
            return True
        
        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            manager = ModelManager(str(model_dir))
            
            # 添加一个示例ModelScope模型（这里使用一个假的模型ID作为示例）
            # 在实际使用中，用户需要替换为真实的ModelScope模型ID
            manager.add_custom_model("test-ms-model", "AI-ModelScope/test-model", "modelscope")
            
            print("ModelScope模型下载测试（示例）准备完成")
            print("注意：此测试仅验证功能可用性，不实际下载模型")
            return True
                
    except Exception as e:
        print(f"✗ ModelScope模型下载测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("DeepSeek OCR 模型下载测试")
    print("=" * 30)
    
    tests = [
        test_default_model_download,
        test_modelscope_model_download
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n模型下载测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有模型下载测试通过")
        return 0
    else:
        print("✗ 部分模型下载测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())