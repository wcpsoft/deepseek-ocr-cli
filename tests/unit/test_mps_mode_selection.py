#!/usr/bin/env python3
"""
MPS模式选择单元测试 - 使用真实设备检测
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

from src.core.utils.device_manager import get_device_manager


def test_mps_mode_selection() -> None:
    """测试MPS模式选择"""
    device_manager = get_device_manager()

    # 检查MPS是否可用
    mps_available = device_manager.is_mps_available()
    print(f"MPS可用: {mps_available}")

    if mps_available:
        # 获取最优设备
        optimal_device = device_manager.get_optimal_device()
        print(f"最优设备: {optimal_device}")

        # 如果MPS可用，应该选择MPS设备
        assert optimal_device.type == "mps", f"MPS可用但未选择MPS设备，选择了: {optimal_device.type}"

        # 检查MPS设备配置
        dtype = device_manager.get_appropriate_dtype(optimal_device)
        should_use_bfloat16 = device_manager.should_use_bfloat16(optimal_device)

        # MPS设备应该使用float32，不支持bfloat16
        assert dtype == torch.float32, f"MPS设备应使用float32，但使用了: {dtype}"
        assert not should_use_bfloat16, "MPS设备不应使用bfloat16"

        print("MPS模式选择测试通过")
    else:
        print("MPS不可用，跳过MPS模式选择测试")


def test_cuda_mode_selection() -> None:
    """测试CUDA模式选择"""
    device_manager = get_device_manager()

    # 检查CUDA是否可用
    cuda_available = device_manager.is_cuda_available()
    print(f"CUDA可用: {cuda_available}")

    if cuda_available:
        # 获取最优设备
        optimal_device = device_manager.get_optimal_device()
        print(f"最优设备: {optimal_device}")

        # 如果CUDA可用，应该选择CUDA设备
        assert optimal_device.type == "cuda", f"CUDA可用但未选择CUDA设备，选择了: {optimal_device.type}"

        # 检查CUDA设备配置
        dtype = device_manager.get_appropriate_dtype(optimal_device)
        should_use_bfloat16 = device_manager.should_use_bfloat16(optimal_device)

        # CUDA设备可能使用bfloat16（如果支持）或float32
        assert dtype in [torch.float32, torch.bfloat16], f"CUDA设备使用了意外的数据类型: {dtype}"

        print("CUDA模式选择测试通过")
    else:
        print("CUDA不可用，跳过CUDA模式选择测试")


def test_cpu_mode_selection() -> None:
    """测试CPU模式选择"""
    device_manager = get_device_manager()

    # 获取最优设备
    optimal_device = device_manager.get_optimal_device()
    print(f"最优设备: {optimal_device}")

    # 如果没有GPU可用，应该选择CPU设备
    if not device_manager.is_cuda_available() and not device_manager.is_mps_available():
        assert optimal_device.type == "cpu", f"无GPU可用但未选择CPU设备，选择了: {optimal_device.type}"

        # 检查CPU设备配置
        dtype = device_manager.get_appropriate_dtype(optimal_device)
        should_use_bfloat16 = device_manager.should_use_bfloat16(optimal_device)

        # CPU设备应该使用float32，不支持bfloat16
        assert dtype == torch.float32, f"CPU设备应使用float32，但使用了: {dtype}"
        assert not should_use_bfloat16, "CPU设备不应使用bfloat16"

        print("CPU模式选择测试通过")
    else:
        print("有GPU可用，跳过CPU模式选择测试")


def test_device_priority() -> None:
    """测试设备优先级"""
    device_manager = get_device_manager()

    cuda_available = device_manager.is_cuda_available()
    mps_available = device_manager.is_mps_available()

    # 获取最优设备
    optimal_device = device_manager.get_optimal_device()
    print(f"最优设备: {optimal_device}")
    print(f"CUDA可用: {cuda_available}, MPS可用: {mps_available}")

    # 验证设备优先级
    if cuda_available:
        # CUDA优先级最高
        assert optimal_device.type == "cuda", "CUDA可用时应优先选择CUDA"
    elif mps_available:
        # MPS次之
        assert optimal_device.type == "mps", "MPS可用且CUDA不可用时应选择MPS"
    else:
        # CPU最后
        assert optimal_device.type == "cpu", "无GPU可用时应选择CPU"

    print("设备优先级测试通过")


def main():
    """主函数"""
    print("开始MPS模式选择测试...")

    # 运行测试
    tests = [
        ("MPS模式选择", test_mps_mode_selection),
        ("CUDA模式选择", test_cuda_mode_selection),
        ("CPU模式选择", test_cpu_mode_selection),
        ("设备优先级", test_device_priority),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            test_func()
            results.append(True)
            print(f"测试 {test_name}: 通过")
        except Exception as e:
            print(f"测试 {test_name} 失败: {e}")
            results.append(False)

    # 汇总结果
    passed = sum(results)
    total = len(results)
    print(f"\n测试结果: {passed}/{total} 通过")

    # 返回适当的退出码
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
