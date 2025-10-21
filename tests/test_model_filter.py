#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载过滤测试脚本
测试模型下载时是否正确过滤了无关文件
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_download_filter():
    """测试模型下载过滤功能"""
    try:
        from cli.model_manager import ModelManager
        
        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            manager = ModelManager(str(model_dir))
            
            print(f"测试模型下载过滤功能: {model_dir}")
            
            # 检查默认模型信息
            assert "deepseek-ocr" in manager.default_models
            default_model = manager.default_models["deepseek-ocr"]
            assert default_model["repo_id"] == "deepseek-ai/DeepSeek-OCR"
            print("✓ 默认模型信息正确")
            
            # 检查模型下载方法是否存在
            assert hasattr(manager, '_download_from_huggingface')
            assert hasattr(manager, '_download_from_modelscope')
            print("✓ 模型下载方法存在")
            
            # 检查Hugging Face下载方法中的过滤配置
            import inspect
            download_method = manager._download_from_huggingface
            source_code = inspect.getsource(download_method)
            
            # 检查是否包含过滤模式
            assert "ignore_patterns" in source_code
            assert "allow_patterns" in source_code
            assert "*.md" in source_code
            assert "assets/*" in source_code
            assert "examples/*" in source_code
            print("✓ Hugging Face下载过滤配置正确")
            
            return True
                
    except Exception as e:
        print(f"✗ 模型下载过滤测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_verification():
    """测试模型验证功能"""
    try:
        from cli.model_manager import ModelManager
        
        manager = ModelManager("./test_models")
        
        # 检查模型验证方法
        assert hasattr(manager, 'verify_model')
        print("✓ 模型验证方法存在")
        
        # 检查验证逻辑
        import inspect
        verify_method = manager.verify_model
        source_code = inspect.getsource(verify_method)
        
        # 检查是否验证必需文件
        assert "config.json" in source_code
        assert "pytorch_model*.bin" in source_code or "pytorch_model" in source_code
        assert "*.safetensors" in source_code
        print("✓ 模型验证逻辑正确")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型验证测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("DeepSeek OCR 模型下载过滤测试")
    print("=" * 30)
    
    tests = [
        test_model_download_filter,
        test_model_verification
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n模型下载过滤测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有模型下载过滤测试通过")
        return 0
    else:
        print("✗ 部分模型下载过滤测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())