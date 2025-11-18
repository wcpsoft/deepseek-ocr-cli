#!/usr/bin/env python3
"""
张量工具函数

提供通用的张量操作和验证功能
"""

from typing import Any, Optional

import torch


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """
    确保张量为2D

    Args:
        tensor: 输入张量

    Returns:
        torch.Tensor: 2D张量
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    elif tensor.dim() > 2:
        # 将多余维度展平
        return tensor.view(tensor.size(0), -1)
    return tensor


def ensure_shape(
    tensor: torch.Tensor, expected_shape: tuple | list, allow_extra_dims: bool = False
) -> torch.Tensor:
    """
    确保张量具有特定形状

    Args:
        tensor: 输入张量
        expected_shape: 期望的形状
        allow_extra_dims: 是否允许额外的维度

    Returns:
        torch.Tensor: 形状正确的张量

    Raises:
        ValueError: 形状不匹配
    """
    current_shape = tensor.shape

    if allow_extra_dims:
        if len(current_shape) >= len(expected_shape):
            if current_shape[: len(expected_shape)] != tuple(expected_shape):
                raise ValueError(f"张量形状 {current_shape} 与期望 {expected_shape} 不匹配")
        else:
            raise ValueError(f"张量维度 {len(current_shape)} 少于期望 {len(expected_shape)}")
    else:
        if current_shape != tuple(expected_shape):
            raise ValueError(f"张量形状 {current_shape} 与期望 {expected_shape} 不匹配")

    return tensor


def normalize_tensor(tensor: torch.Tensor, method: str = "minmax", eps: float = 1e-8) -> torch.Tensor:
    """
    标准化张量

    Args:
        tensor: 输入张量
        method: 标准化方法 ("minmax", "zscore", "l2")
        eps: 防止除零的小值

    Returns:
        torch.Tensor: 标准化后的张量
    """
    if method == "minmax":
        min_val, max_val = tensor.min(), tensor.max()
        if max_val - min_val < eps:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean_val, std_val = tensor.mean(), tensor.std()
        if std_val < eps:
            return torch.zeros_like(tensor)
        return (tensor - mean_val) / std_val

    elif method == "l2":
        norm = torch.norm(tensor)
        if norm < eps:
            return torch.zeros_like(tensor)
        return tensor / norm

    else:
        raise ValueError(f"不支持的标准化方法: {method}")


def tensor_memory_size(tensor: torch.Tensor) -> int:
    """
    计算张量的内存大小（字节）

    Args:
        tensor: 输入张量

    Returns:
        int: 内存大小（字节）
    """
    return tensor.numel() * tensor.element_size()


def get_tensor_info(tensor: torch.Tensor) -> dict:
    """
    获取张量的详细信息

    Args:
        tensor: 输入张量

    Returns:
        dict: 张量信息字典
    """
    return {
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "memory_bytes": tensor_memory_size(tensor),
        "memory_mb": tensor_memory_size(tensor) / (1024 * 1024),
        "requires_grad": tensor.requires_grad,
        "is_contiguous": tensor.is_contiguous(),
        "min": tensor.min().item() if tensor.numel() > 0 else None,
        "max": tensor.max().item() if tensor.numel() > 0 else None,
        "mean": tensor.float().mean().item() if tensor.numel() > 0 else None,
    }


def safe_tensor_operation(func):
    """
    安全张量操作装饰器

    Args:
        func: 张量操作函数

    Returns:
        装饰后的函数
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            # 处理CUDA内存错误
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError("GPU内存不足，请尝试减少batch size或使用CPU")
            raise
        except Exception as e:
            raise RuntimeError(f"张量操作失败: {e}")

    return wrapper


def split_tensor(tensor: torch.Tensor, split_size_or_sections: int | list, dim: int = 0) -> list[torch.Tensor]:
    """
    安全地分割张量

    Args:
        tensor: 输入张量
        split_size_or_sections: 分割大小或分割点列表
        dim: 分割维度

    Returns:
        List[torch.Tensor]: 分割后的张量列表
    """
    try:
        return torch.split(tensor, split_size_or_sections, dim=dim)
    except RuntimeError:
        # 如果分割失败，返回包含原张量的列表
        return [tensor]


def concatenate_tensors(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    安全地连接张量

    Args:
        tensors: 张量列表
        dim: 连接维度

    Returns:
        torch.Tensor: 连接后的张量
    """
    if not tensors:
        raise ValueError("张量列表不能为空")

    # 检查所有张量的设备是否一致
    first_device = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.device != first_device:
            raise ValueError("所有张量必须在同一设备上")

    return torch.cat(tensors, dim=dim)


def resize_tensor(
    tensor: torch.Tensor, target_size: int | tuple[int, ...], mode: str = "nearest"
) -> torch.Tensor:
    """
    调整张量大小

    Args:
        tensor: 输入张量
        target_size: 目标大小
        mode: 插值模式

    Returns:
        torch.Tensor: 调整大小后的张量
    """
    if tensor.dim() == 3:
        # 3D张量 (C, H, W)
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        resized = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=target_size, mode=mode)
        return resized.squeeze(0)

    elif tensor.dim() == 4:
        # 4D张量 (B, C, H, W)
        return torch.nn.functional.interpolate(tensor, size=target_size, mode=mode)

    else:
        raise ValueError(f"不支持的张量维度: {tensor.dim()}")


def tensor_to_numpy(tensor: torch.Tensor) -> Any:
    """
    将张量转换为numpy数组

    Args:
        tensor: PyTorch张量

    Returns:
        numpy.ndarray: numpy数组
    """
    if tensor.requires_grad:
        tensor = tensor.detach()

    if tensor.is_cuda:
        tensor = tensor.cpu()

    return tensor.numpy()


def batch_tensor_to_list(tensor: torch.Tensor, batch_dim: int = 0) -> list[torch.Tensor]:
    """
    将批次张量转换为张量列表

    Args:
        tensor: 批次张量
        batch_dim: 批次维度

    Returns:
        List[torch.Tensor]: 张量列表
    """
    return torch.split(tensor, 1, dim=batch_dim)


def stack_tensor_list(tensor_list: list[torch.Tensor], stack_dim: int = 0) -> torch.Tensor:
    """
    将张量列表堆叠为批次张量

    Args:
        tensor_list: 张量列表
        stack_dim: 堆叠维度

    Returns:
        torch.Tensor: 堆叠后的张量
    """
    if not tensor_list:
        raise ValueError("张量列表不能为空")

    # 确保所有张量的形状一致
    first_shape = tensor_list[0].shape
    for tensor in tensor_list[1:]:
        if tensor.shape != first_shape:
            raise ValueError("所有张量必须具有相同的形状")

    return torch.stack(tensor_list, dim=stack_dim)


def tensor_statistics(tensor: torch.Tensor) -> dict:
    """
    计算张量的统计信息

    Args:
        tensor: 输入张量

    Returns:
        dict: 统计信息
    """
    if tensor.numel() == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    flat_tensor = tensor.flatten()
    return {
        "count": flat_tensor.numel(),
        "mean": flat_tensor.float().mean().item(),
        "std": flat_tensor.float().std().item(),
        "min": flat_tensor.min().item(),
        "max": flat_tensor.max().item(),
        "median": flat_tensor.median().float().item(),
    }


def remove_empty_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    移除空的张量

    Args:
        tensors: 张量列表

    Returns:
        List[torch.Tensor]: 非空张量列表
    """
    return [tensor for tensor in tensors if tensor.numel() > 0]


def filter_tensors_by_size(
    tensors: list[torch.Tensor], min_elements: int = 1, max_elements: Optional[int] = None
) -> list[torch.Tensor]:
    """
    根据元素数量过滤张量

    Args:
        tensors: 张量列表
        min_elements: 最小元素数量
        max_elements: 最大元素数量，None表示无限制

    Returns:
        List[torch.Tensor]: 过滤后的张量列表
    """
    filtered = []
    for tensor in tensors:
        num_elements = tensor.numel()
        if num_elements >= min_elements:
            if max_elements is None or num_elements <= max_elements:
                filtered.append(tensor)

    return filtered


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    """
    比较两个张量

    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        rtol: 相对容差
        atol: 绝对容差

    Returns:
        dict: 比较结果
    """
    shape_match = tensor1.shape == tensor2.shape
    dtype_match = tensor1.dtype == tensor2.dtype
    device_match = tensor1.device == tensor2.device

    values_close = False
    if shape_match:
        try:
            values_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        except Exception:
            values_close = False

    return {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "device_match": device_match,
        "values_close": values_close,
        "identical": shape_match and dtype_match and device_match and values_close,
    }
