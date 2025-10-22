#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS模式测试脚本
验证在Apple Silicon设备上只使用Transformers而不依赖vLLM和flash_attn
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def test_mps_environment():
    """测试MPS环境配置"""
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查MPS可用性
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("✓ MPS环境可用")
            device = torch.device("mps")
            print(f"  使用设备: {device}")
        else:
            print("✗ MPS环境不可用")
            device = torch.device("cpu")
            print(f"  使用设备: {device}")
            
        # 测试基本张量操作
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = x + y
        print("✓ 基本张量运算测试通过")
        
        return True
    except Exception as e:
        print(f"✗ MPS环境测试失败: {e}")
        return False


def test_transformers_import():
    """测试Transformers导入"""
    try:
        from transformers import AutoTokenizer, AutoModel
        print("✓ Transformers库导入成功")
        return True
    except Exception as e:
        print(f"✗ Transformers库导入失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=== MPS模式兼容性测试 ===\n")
    
    # 测试环境
    print("1. 环境检测:")
    test_mps_environment()
    print()
    
    # 测试依赖库
    print("2. 依赖库检测:")
    test_transformers_import()
    test_flash_attn_availability()
    test_vllm_availability()
    print()
    
    # 测试项目模块导入
    print("3. 项目模块导入测试:")
    try:
        from cli.document_processor import DocumentProcessor
        print("✓ DocumentProcessor导入成功")
        
        # 检查MPS环境下的模式选择
        processor = DocumentProcessor(mode="auto")
        mode = processor._determine_mode()
        print(f"  自动模式选择: {mode}")
        
        if mode == "transformers":
            print("✓ 在MPS环境下正确选择了Transformers模式")
        else:
            print(f"✗ 在MPS环境下选择了{mode}模式，应该选择Transformers模式")
            
    except Exception as e:
        print(f"✗ 项目模块导入测试失败: {e}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()