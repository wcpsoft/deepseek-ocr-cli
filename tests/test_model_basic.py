#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基本功能测试脚本
测试模型管理器的基本功能，不实际下载大模型文件
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_manager_basic():
    """测试模型管理器基本功能"""
    try:
        from cli.model_manager import ModelManager
        
        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            manager = ModelManager(str(model_dir))
            
            print(f"测试模型管理器基本功能: {model_dir}")
            
            # 测试获取模型目录
            assert Path(manager.get_model_dir()).resolve() == model_dir.resolve()
            print("✓ 获取模型目录功能正常")
            
            # 测试添加自定义模型
            manager.add_custom_model("test-model", "test/repo-id", "huggingface")
            custom_models = manager.list_custom_models()
            assert "test-model" in custom_models
            assert custom_models["test-model"]["repo_id"] == "test/repo-id"
            assert custom_models["test-model"]["source"] == "huggingface"
            print("✓ 添加自定义模型功能正常")
            
            # 测试添加ModelScope模型
            manager.add_custom_model("test-ms-model", "test/ms-model-id", "modelscope")
            custom_models = manager.list_custom_models()
            assert "test-ms-model" in custom_models
            assert custom_models["test-ms-model"]["source"] == "modelscope"
            print("✓ 添加ModelScope模型功能正常")
            
            # 测试移除自定义模型
            manager.remove_custom_model("test-model")
            custom_models = manager.list_custom_models()
            assert "test-model" not in custom_models
            print("✓ 移除自定义模型功能正常")
            
            return True
                
    except Exception as e:
        print(f"✗ 模型管理器基本功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_default_model_info():
    """测试默认模型信息"""
    try:
        from cli.model_manager import ModelManager
        
        manager = ModelManager("./test_models")
        
        # 检查默认模型信息
        assert "deepseek-ocr" in manager.default_models
        default_model = manager.default_models["deepseek-ocr"]
        assert isinstance(default_model, dict)
        assert default_model["repo_id"] == "deepseek-ai/DeepSeek-OCR"
        assert default_model["source"] == "huggingface"
        print("✓ 默认模型信息正确")
        
        return True
        
    except Exception as e:
        print(f"✗ 默认模型信息测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("DeepSeek OCR 模型基本功能测试")
    print("=" * 30)
    
    tests = [
        test_model_manager_basic,
        test_default_model_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n模型基本功能测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有模型基本功能测试通过")
        return 0
    else:
        print("✗ 部分模型基本功能测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())