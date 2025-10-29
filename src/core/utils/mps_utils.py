"""
MPS设备检测和处理工具模块

该模块提供了检测MPS设备并进行特殊处理的工具函数。
"""

import torch

from src.core.logging import get_logger
from src.core.utils.device_manager import get_device_manager

# 获取日志记录器
logger = get_logger()


def is_mps_device() -> bool:
    """
    检测当前环境是否支持MPS设备

    Returns:
        bool: 如果支持MPS返回True，否则返回False
    """
    try:
        device_manager = get_device_manager()
        return device_manager.is_mps_available()
    except Exception as e:
        logger.warning(f"检测MPS设备时发生错误: {e!s}")
        return False


def get_optimal_device() -> torch.device:
    """
    获取当前环境下的最优设备

    Returns:
        torch.device: 最优设备对象
    """
    device_manager = get_device_manager()
    return device_manager.get_optimal_device()


def optimize_tensor_for_mps(
    tensor: torch.Tensor, max_sequence_length: int = 512, max_image_blocks: int = 10
) -> torch.Tensor:
    """
    优化张量以适应MPS设备的限制

    Args:
        tensor: 输入张量
        max_sequence_length: MPS设备上最大序列长度
        max_image_blocks: MPS设备上最大图像块数量

    Returns:
        torch.Tensor: 优化后的张量
    """
    if not is_mps_device() or tensor is None:
        return tensor

    logger.debug(f"优化张量以适应MPS设备，原始形状: {tensor.shape}")

    # 确保张量在MPS设备上
    if tensor.device.type != "mps":
        tensor = tensor.to("mps")

    # MPS设备上使用float32而不是float16，以提高兼容性
    if tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)

    # 对于序列张量，限制最大长度
    if tensor.dim() >= 2 and tensor.size(-1) > max_sequence_length:
        logger.debug(f"MPS设备上限制序列长度从{tensor.size(-1)}到{max_sequence_length}")
        # 保留最后max_sequence_length个元素
        tensor = tensor[..., -max_sequence_length:]

    # 对于图像张量，限制最大块数
    if tensor.dim() >= 4 and tensor.size(0) > max_image_blocks:
        logger.debug(f"MPS设备上限制图像块数量从{tensor.size(0)}到{max_image_blocks}")
        tensor = tensor[:max_image_blocks]

    logger.debug(f"优化后张量形状: {tensor.shape}")
    return tensor


def optimize_model_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    优化模型以适应MPS设备

    Args:
        model: 输入模型

    Returns:
        torch.nn.Module: 优化后的模型
    """
    if not is_mps_device():
        return model

    logger.info("优化模型以适应MPS设备")

    # 将模型移动到MPS设备
    model = model.to("mps")

    # 对于MPS设备，可能需要调整某些层的配置
    # 这里可以根据具体需求添加更多优化

    return model


def mps_safe_tensor_creation(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    在MPS设备上安全创建张量

    Args:
        shape: 张量形状
        dtype: 数据类型
        device: 目标设备，如果为None则自动选择
        **kwargs: 其他传递给torch.zeros的参数

    Returns:
        torch.Tensor: 创建的张量
    """
    if device is None:
        device = get_optimal_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # 在MPS设备上使用float32而不是float16
    if device.type == "mps" and dtype == torch.float16:
        dtype = torch.float32
        logger.debug("MPS设备上使用float32而不是float16创建张量")

    try:
        return torch.zeros(shape, dtype=dtype, device=device, **kwargs)
    except Exception as e:
        logger.error(f"在设备{device}上创建张量失败: {e!s}")
        # 回退到CPU
        logger.info("回退到CPU设备创建张量")
        # 确保回退时使用兼容的数据类型
        cpu_dtype = get_mps_compatible_dtype(dtype) if device.type == "mps" else dtype
        return torch.zeros(shape, dtype=cpu_dtype, device="cpu", **kwargs)


def mps_safe_model_load(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str | torch.device | None = None,
) -> torch.nn.Module:
    """
    在MPS设备上安全加载模型

    Args:
        model: 要加载权重的模型
        checkpoint_path: 检查点路径
        map_location: 映射位置，如果为None则自动选择

    Returns:
        torch.nn.Module: 加载权重后的模型
    """
    if map_location is None:
        map_location = get_optimal_device()
    elif isinstance(map_location, str):
        map_location = torch.device(map_location)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"成功在{map_location}设备上加载模型权重")
    except Exception as e:
        logger.error(f"在{map_location}设备上加载模型失败: {e!s}")
        # 尝试在CPU上加载，然后再移动到目标设备
        logger.info("尝试在CPU上加载模型权重，然后移动到目标设备")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # 对于MPS设备，使用优化函数来安全地移动模型
        if map_location.type == "mps":
            model = optimize_model_for_mps(model)
        else:
            model = model.to(map_location)
        logger.info(f"成功在CPU上加载模型权重并移动到{map_location}设备")

    return model


def get_mps_compatible_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    获取与MPS设备兼容的数据类型

    Args:
        dtype: 原始数据类型

    Returns:
        torch.dtype: MPS兼容的数据类型
    """
    # MPS设备不完全支持float16，使用float32替代
    if dtype == torch.float16:
        logger.debug("MPS设备上使用float32替代float16")
        return torch.float32

    # MPS设备不完全支持bfloat16，使用float32替代
    if dtype == torch.bfloat16:
        logger.debug("MPS设备上使用float32替代bfloat16")
        return torch.float32

    return dtype


def configure_mps_environment() -> None:
    """
    配置MPS环境设置
    """
    logger.info("配置MPS环境设置")

    # 设置MPS相关的环境变量
    import os

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 设置MPS内存分配策略（仅在MPS设备可用时）
    if is_mps_device() and hasattr(torch.mps, "empty_cache"):
        logger.debug("设置MPS内存缓存清理")
        torch.mps.empty_cache()

    logger.info("MPS环境配置完成")


def log_device_info() -> None:
    """
    记录当前设备信息
    """
    logger.info("设备信息:")
    logger.info(f"  PyTorch版本: {torch.__version__}")
    logger.info(f"  MPS可用: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        logger.info(f"  MPS构建: {torch.backends.mps.is_built()}")
    logger.info(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA设备数量: {torch.cuda.device_count()}")
        logger.info(f"  当前CUDA设备: {torch.cuda.current_device()}")

    device = get_optimal_device()
    logger.info(f"  选择的设备: {device}")


def clear_mps_cache() -> None:
    """
    清理MPS缓存
    """
    if is_mps_device() and hasattr(torch.mps, "empty_cache"):
        logger.debug("清理MPS缓存")
        torch.mps.empty_cache()


def get_mps_memory_info() -> dict:
    """
    获取MPS内存信息

    Returns:
        dict: 包含内存信息的字典
    """
    if not is_mps_device():
        return {"error": "MPS设备不可用"}

    try:
        # 尝试获取内存信息
        # 注意：PyTorch可能不直接提供MPS内存信息API
        # 这里返回一个基本的信息结构
        return {
            "device": "mps",
            "status": "available",
            "note": "PyTorch不直接提供MPS内存信息API",
        }
    except Exception as e:
        logger.error(f"获取MPS内存信息失败: {e!s}")
        return {"error": str(e)}
