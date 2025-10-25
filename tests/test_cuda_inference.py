#!/usr/bin/env python3
"""
测试脚本：在CUDA设备上测试形状不匹配修复
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.model_initializer import TransformersOCRModel
from src.core.config import Config

def test_cuda_inference():
    """在CUDA设备上测试推理"""
    print("=== CUDA设备推理测试 ===")
    
    # 检查CUDA是否可用
    import torch
    if not torch.cuda.is_available():
        print("错误: CUDA设备不可用")
        return False
    
    print(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"设备内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载配置
    config = Config()
    
    # 初始化模型
    print("\n正在初始化模型...")
    model = TransformersOCRModel(config)
    
    # 测试推理
    print("\n开始测试推理...")
    try:
        # 使用samples/4.pdf进行测试
        result = model.infer("samples/4.pdf", "output/test_cuda")
        print("推理成功完成!")
        print(f"结果保存在: {result}")
        
        # 读取并显示结果
        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()
            print("\n=== OCR识别结果 ===")
            print(content)
        
        return True
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cuda_inference()
    sys.exit(0 if success else 1)