"""
MPS设备检测和处理工具模块 (已弃用)

警告: 此模块已弃用。所有功能已迁移到 device_manager.py。
请更新导入语句:
    from src.core.utils.device_manager import get_optimal_device, get_device_manager, configure_device_environment, log_device_info

此文件仅为保持向后兼容性而保留，将在未来版本中删除。
"""

import warnings

import torch

from src.core.utils.device_manager import (
    configure_device_environment as _configure_device_environment,
)
from src.core.utils.device_manager import (
    get_compatible_dtype as _get_compatible_dtype,
)
from src.core.utils.device_manager import (
    get_device_manager,
)
from src.core.utils.device_manager import (
    get_optimal_device as _get_optimal_device,
)
from src.core.utils.device_manager import (
    log_device_info as _log_device_info,
)


def _deprecated_warning(old_name: str, new_location: str = "device_manager") -> None:
    """发出弃用警告"""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version. "
        f"Please use src.core.utils.{new_location} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def is_mps_device() -> bool:
    """
    检测当前环境是否支持MPS设备 (已弃用)

    .. deprecated::
        使用 device_manager.DeviceManager().is_mps_available() 替代
    """
    _deprecated_warning("is_mps_device()", "device_manager.get_device_manager().is_mps_available()")
    return get_device_manager().is_mps_available()


def get_optimal_device() -> torch.device:
    """
    获取当前环境下的最优设备 (已弃用)

    .. deprecated::
        使用 device_manager.get_optimal_device() 替代
    """
    _deprecated_warning("get_optimal_device()", "device_manager.get_optimal_device()")
    return _get_optimal_device()


def get_mps_compatible_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    获取与MPS设备兼容的数据类型 (已弃用)

    .. deprecated::
        使用 device_manager.get_compatible_dtype() 替代
    """
    _deprecated_warning("get_mps_compatible_dtype()", "device_manager.get_compatible_dtype()")
    return _get_compatible_dtype(dtype, torch.device("mps") if is_mps_device() else None)


def configure_mps_environment() -> None:
    """
    配置MPS环境 (已弃用)

    .. deprecated::
        使用 device_manager.configure_device_environment() 替代
    """
    _deprecated_warning("configure_mps_environment()", "device_manager.configure_device_environment()")
    _configure_device_environment()


def log_device_info() -> None:
    """
    记录当前设备信息 (已弃用)

    .. deprecated::
        使用 device_manager.log_device_info() 替代
    """
    _deprecated_warning("log_device_info()", "device_manager.log_device_info()")
    _log_device_info()


def clear_mps_cache() -> None:
    """
    清理MPS缓存 (已弃用)

    .. deprecated::
        使用 device_manager.DeviceManager().clear_device_cache() 替代
    """
    _deprecated_warning("clear_mps_cache()", "device_manager.get_device_manager().clear_device_cache()")
    get_device_manager().clear_device_cache()


def optimize_tensor_for_mps(
    tensor: torch.Tensor, max_sequence_length: int = 512, max_image_blocks: int = 10
) -> torch.Tensor:
    """
    优化张量以适应MPS设备的限制 (已弃用)

    .. deprecated::
        MPS 限制处理已内置到 device_manager.move_tensor_to_device() 中
    """
    _deprecated_warning("optimize_tensor_for_mps()", "device_manager.move_tensor_to_device()")

    if tensor is None:
        return tensor

    device_manager = get_device_manager()

    # 使用device_manager的张量移动功能
    if device_manager.is_mps_available():
        tensor = device_manager.move_tensor_to_device(tensor)

    return tensor


def optimize_model_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    优化模型以适应MPS设备 (已弃用)

    .. deprecated::
        使用 model.to(device_manager.get_optimal_device()) 替代
    """
    _deprecated_warning("optimize_model_for_mps()", "model.to(device_manager.get_optimal_device())")

    device_manager = get_device_manager()
    if device_manager.is_mps_available():
        model = model.to(device_manager.get_optimal_device())

    return model


def mps_safe_tensor_creation(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    在MPS设备上安全创建张量 (已弃用)

    .. deprecated::
        直接使用 torch.zeros() 并使用 device_manager.get_compatible_dtype() 处理dtype
    """
    _deprecated_warning("mps_safe_tensor_creation()", "torch.zeros() + device_manager.get_compatible_dtype()")

    if device is None:
        device = _get_optimal_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # 使用device_manager获取兼容的dtype
    dtype = _get_compatible_dtype(dtype, device)

    try:
        return torch.zeros(shape, dtype=dtype, device=device, **kwargs)
    except Exception:
        # 回退到CPU
        return torch.zeros(shape, dtype=dtype, device="cpu", **kwargs)


def mps_safe_model_load(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str | torch.device | None = None,
) -> torch.nn.Module:
    """
    在MPS设备上安全加载模型 (已弃用)

    .. deprecated::
        直接使用 torch.load() 并使用 device_manager.get_optimal_device()
    """
    _deprecated_warning("mps_safe_model_load()", "torch.load() + device_manager.get_optimal_device()")

    if map_location is None:
        map_location = _get_optimal_device()
    elif isinstance(map_location, str):
        map_location = torch.device(map_location)

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def get_mps_memory_info() -> dict:
    """
    获取MPS内存信息 (已弃用)

    .. deprecated::
        PyTorch不直接提供MPS内存信息API，此函数将被移除
    """
    _deprecated_warning("get_mps_memory_info()", "")

    device_manager = get_device_manager()
    if not device_manager.is_mps_available():
        return {"error": "MPS设备不可用"}

    return {
        "device": "mps",
        "status": "available",
        "note": "PyTorch不直接提供MPS内存信息API",
    }
