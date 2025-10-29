#!/usr/bin/env python3
"""
测试脚本：在CUDA设备上测试形状不匹配修复
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.core.transformers.transformers_engine import TransformersEngine


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

    # 初始化引擎
    print("\n正在初始化引擎...")
    engine = TransformersEngine()

    # 测试推理
    print("\n开始测试推理...")
    try:
        # 将PDF转换为图像
        from src.core.document_processor import DocumentProcessor

        doc_processor = DocumentProcessor()
        images = doc_processor.process_document("samples/4.pdf")

        # 使用引擎处理图像
        engine.process(images, "output/test_cuda")
        print("推理成功完成!")
        print("结果保存在: output/test_cuda")

        # 读取并显示结果
        result_file = os.path.join("output/test_cuda", "result.md")
        if os.path.exists(result_file):
            with open(result_file, encoding="utf-8") as f:
                content = f.read()
                print("\n=== OCR识别结果 ===")
                print(content)
        else:
            print("未找到结果文件")

        return True
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cuda_inference()
    sys.exit(0 if success else 1)
