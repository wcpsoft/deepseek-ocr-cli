#!/usr/bin/env python3
"""
真实设备测试
使用真实设备进行测试，不使用MagicMock
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

from src.core.utils.device_manager import get_device_manager

# 获取日志记录器
logger = logging.getLogger(__name__)


def test_device_detection():
    """测试设备检测功能"""
    device_manager = get_device_manager()

    # 测试CUDA检测
    cuda_available = device_manager.is_cuda_available()
    print(f"CUDA可用: {cuda_available}")

    # 测试MPS检测
    mps_available = device_manager.is_mps_available()
    print(f"MPS可用: {mps_available}")

    # 获取最优设备
    optimal_device = device_manager.get_optimal_device()
    print(f"最优设备: {optimal_device}")

    # 获取适当的数据类型
    dtype = device_manager.get_appropriate_dtype()
    print(f"适当的数据类型: {dtype}")

    # 测试是否应该使用bfloat16
    should_use_bfloat16 = device_manager.should_use_bfloat16()
    print(f"是否应该使用bfloat16: {should_use_bfloat16}")

    return True


def test_device_tensor_operations():
    """测试设备张量操作"""
    device_manager = get_device_manager()
    device = device_manager.get_optimal_device()

    try:
        # 创建测试张量
        test_tensor = torch.zeros(10, 10)

        # 移动到设备
        device_tensor = device_manager.move_tensor_to_device(test_tensor)

        # 验证张量在正确的设备上
        assert device_tensor.device.type == device.type, f"张量未在正确的设备上: {device_tensor.device} != {device}"

        # 执行简单操作
        result = device_tensor + 1

        # 清理
        del test_tensor, device_tensor, result
        device_manager.clear_device_cache()

        print(f"设备 {device.type} 张量操作测试通过")
        return True

    except Exception as e:
        print(f"设备 {device.type} 张量操作测试失败: {e}")
        return False


def test_cuda_specific():
    """测试CUDA特定功能"""
    device_manager = get_device_manager()

    if not device_manager.is_cuda_available():
        print("CUDA不可用，跳过CUDA特定测试")
        return True

    try:
        # 测试CUDA设备数量
        device_count = torch.cuda.device_count()
        print(f"CUDA设备数量: {device_count}")

        # 测试当前设备
        current_device = torch.cuda.current_device()
        print(f"当前CUDA设备: {current_device}")

        # 测试设备名称
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA设备名称: {device_name}")

        # 测试设备内存
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_reserved = torch.cuda.memory_reserved(current_device)
        print(f"CUDA内存已分配: {memory_allocated / 1024**2:.2f} MB")
        print(f"CUDA内存已保留: {memory_reserved / 1024**2:.2f} MB")

        return True

    except Exception as e:
        print(f"CUDA特定测试失败: {e}")
        return False


def test_mps_specific():
    """测试MPS特定功能"""
    device_manager = get_device_manager()

    if not device_manager.is_mps_available():
        print("MPS不可用，跳过MPS特定测试")
        return True

    try:
        # 测试MPS设备
        device = torch.device("mps")

        # 创建测试张量
        test_tensor = torch.zeros(10, 10)
        mps_tensor = test_tensor.to(device)

        # 执行简单操作
        result = mps_tensor + 1

        # 清理
        del test_tensor, mps_tensor, result

        # 测试MPS缓存清理
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            print("MPS缓存清理完成")

        print("MPS特定测试通过")
        return True

    except Exception as e:
        print(f"MPS特定测试失败: {e}")
        return False


def main():
    """主函数"""
    print("开始真实设备测试...")

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    tests = [
        ("设备检测", test_device_detection),
        ("设备张量操作", test_device_tensor_operations),
        ("CUDA特定", test_cuda_specific),
        ("MPS特定", test_mps_specific),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            result = test_func()
            results.append(result)
            print(f"测试 {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            print(f"测试 {test_name} 异常: {e}")
            results.append(False)

    # 汇总结果
    passed = sum(results)
    total = len(results)
    print(f"\n测试结果: {passed}/{total} 通过")

    # 返回适当的退出码
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
