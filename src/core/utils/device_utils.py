#!/usr/bin/env python3
"""
设备工具函数

提供通用的设备操作和管理功能
"""

from typing import Any, Optional

import torch
import torch.nn as nn


def normalize_device_type(device: str | torch.device) -> torch.device:
    """
    规范化设备类型

    Args:
        device: 设备字符串或对象

    Returns:
        torch.device: 规范化的设备对象
    """
    if isinstance(device, torch.device):
        return device

    device_str = str(device).lower()

    # 映射常见设备别名
    device_map = {
        "gpu": "cuda",
        "cpu": "cpu",
        "mps": "mps",
        "cuda:0": "cuda",
        "cpu:0": "cpu",
    }

    # 如果是完整设备格式（如 cuda:0），直接使用
    if ":" in device_str and device_str.split(":")[0] in ["cuda", "cpu"]:
        return torch.device(device_str)

    # 使用映射表
    normalized = device_map.get(device_str, device_str)

    try:
        return torch.device(normalized)
    except Exception:
        return torch.device("cpu")


def get_device_dtype(device: str | torch.device, default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    获取设备默认数据类型

    Args:
        device: 设备对象
        default_dtype: 默认数据类型

    Returns:
        torch.dtype: 推荐的数据类型
    """
    device = normalize_device_type(device)

    if default_dtype is not None:
        return default_dtype

    # 根据设备类型选择最佳数据类型
    dtype_map = {
        "cuda": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        "mps": torch.float32,
        "cpu": torch.float32,
    }

    return dtype_map.get(device.type, torch.float32)


def get_device_memory_info(device: str | torch.device) -> dict[str, Any]:
    """
    获取设备内存信息

    Args:
        device: 设备对象

    Returns:
        Dict[str, Any]: 内存信息
    """
    device = normalize_device_type(device)

    if device.type == "cuda" and torch.cuda.is_available():
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_id)
        allocated_memory = torch.cuda.memory_allocated(device_id)

        return {
            "total_memory": total_memory,
            "reserved_memory": reserved_memory,
            "allocated_memory": allocated_memory,
            "free_memory": total_memory - reserved_memory,
            "total_memory_gb": total_memory / (1024**3),
            "reserved_memory_gb": reserved_memory / (1024**3),
            "allocated_memory_gb": allocated_memory / (1024**3),
            "free_memory_gb": (total_memory - reserved_memory) / (1024**3),
        }

    elif device.type == "mps":
        return {"device_type": "mps", "note": "MPS memory information not available through PyTorch"}

    else:
        return {"device_type": "cpu", "note": "CPU memory information not available through this function"}


def clear_device_cache(device: Optional[str | torch.device] = None) -> None:
    """
    清理设备缓存

    Args:
        device: 设备对象，None表示清理所有设备
    """
    if device is None:
        # 清理所有设备缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    else:
        device = normalize_device_type(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def is_device_available(device: str | torch.device) -> bool:
    """
    检查设备是否可用

    Args:
        device: 设备对象

    Returns:
        bool: 设备是否可用
    """
    device = normalize_device_type(device)

    try:
        if device.type == "cuda":
            return torch.cuda.is_available()
        elif device.type == "mps":
            return torch.backends.mps.is_available()
        elif device.type == "cpu":
            return True
        else:
            return False
    except Exception:
        return False


def get_optimal_device_for_size(memory_required: int, prefer_gpu: bool = True) -> torch.device:
    """
    根据内存需求获取最优设备

    Args:
        memory_required: 所需内存（字节）
        prefer_gpu: 是否优先使用GPU

    Returns:
        torch.device: 最优设备
    """
    if prefer_gpu and torch.cuda.is_available():
        # 尝试找到有足够内存的GPU
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            memory_info = get_device_memory_info(device)

            if memory_info.get("free_memory", 0) > memory_required:
                return device

    # 如果GPU不可用或内存不足，检查MPS
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # 最后使用CPU
    return torch.device("cpu")


def move_model_to_device_efficiently(
    model: nn.Module,
    device: str | torch.device,
    dtype: Optional[torch.dtype] = None,
    memory_efficient: bool = True,
) -> nn.Module:
    """
    高效地将模型移动到设备

    Args:
        model: PyTorch模型
        device: 目标设备
        dtype: 目标数据类型
        memory_efficient: 是否使用内存高效模式

    Returns:
        nn.Module: 移动后的模型
    """
    device = normalize_device_type(device)

    if dtype is None:
        dtype = get_device_dtype(device)

    try:
        if memory_efficient and device.type == "cuda":
            # 内存高效模式：先移动半精度模型
            if dtype in [torch.float16, torch.bfloat16]:
                model = model.half()
            model = model.to(device=device, dtype=dtype)
        else:
            model = model.to(device=device, dtype=dtype)

        model.eval()  # 设置为评估模式
        return model

    except RuntimeError as e:
        if "out of memory" in str(e):
            # 内存不足时的回退策略
            if device.type == "cuda":
                # 清理缓存并重试
                clear_device_cache(device)
                model = model.to(torch.device("cpu"))
                if dtype in [torch.float16, torch.bfloat16]:
                    model = model.half()
                model = model.to(device=device, dtype=torch.float32)
                model.eval()
                return model
        raise e


def get_device_utilization(device: str | torch.device) -> dict[str, float]:
    """
    获取设备利用率信息

    Args:
        device: 设备对象

    Returns:
        Dict[str, float]: 利用率信息
    """
    device = normalize_device_type(device)

    if device.type == "cuda" and torch.cuda.is_available():
        device_id = device.index if device.index is not None else torch.cuda.current_device()

        # 获取GPU使用率（需要安装 nvidia-ml-py）
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return {"gpu_utilization": util.gpu / 100.0, "memory_utilization": memory_info.used / memory_info.total}
        except (ImportError, Exception):
            pass

    return {"utilization": 0.0, "note": f"Cannot get utilization info for {device.type}"}


def benchmark_device_operation(
    operation: callable, device: str | torch.device, num_runs: int = 10, warmup_runs: int = 3
) -> dict[str, float]:
    """
    基准测试设备操作性能

    Args:
        operation: 要测试的操作函数
        device: 测试设备
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数

    Returns:
        Dict[str, float]: 性能指标
    """
    device = normalize_device_type(device)
    times = []

    # 预热
    for _ in range(warmup_runs):
        operation()

    # 正式测试
    import time

    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = operation()
        if hasattr(result, "item"):
            result.item()  # 确保操作完成
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "total_time": sum(times),
    }


def get_device_capability(device: str | torch.device) -> dict[str, Any]:
    """
    获取设备能力信息

    Args:
        device: 设备对象

    Returns:
        Dict[str, Any]: 设备能力信息
    """
    device = normalize_device_type(device)

    if device.type == "cuda" and torch.cuda.is_available():
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)

        return {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory": props.total_memory,
            "multiprocessor_count": props.multiprocessor_count,
            "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
            "max_shared_memory_per_block": props.max_shared_memory_per_block,
            "max_block_dims": list(props.max_block_dims),
            "max_grid_dims": list(props.max_grid_dims),
            "supports_bf16": props.major >= 7,  # Ampere及更高架构支持bfloat16
            "supports_tf32": props.major >= 8,  # Hopper架构支持TF32
        }

    elif device.type == "mps":
        return {
            "device_type": "mps",
            "supports_bf16": False,  # MPS 通常不支持bfloat16
            "supports_tf32": False,
        }

    else:
        return {
            "device_type": "cpu",
            "supports_bf16": True,  # CPU 通常支持
            "supports_tf32": False,
        }


def recommend_batch_size(
    model: nn.Module, device: str | torch.device, input_shape: tuple, memory_margin: float = 0.8
) -> int:
    """
    推荐适合的批次大小

    Args:
        model: 模型
        device: 设备
        input_shape: 单个样本的输入形状
        memory_margin: 内存使用安全边际

    Returns:
        int: 推荐的批次大小
    """
    device = normalize_device_type(device)

    if device.type != "cuda":
        # CPU/MPS 推荐较小的批次大小
        return 1

    try:
        memory_info = get_device_memory_info(device)
        available_memory = memory_info["free_memory"] * memory_margin

        # 估算单样本内存使用
        dummy_input = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            _ = model(dummy_input)

        current_usage = memory_info["allocated_memory"]
        single_sample_memory = current_usage

        if single_sample_memory > 0:
            estimated_batch_size = int(available_memory / single_sample_memory)
            return max(1, min(estimated_batch_size, 32))  # 限制最大批次大小
        else:
            return 1

    except Exception:
        return 1  # 保守估计
