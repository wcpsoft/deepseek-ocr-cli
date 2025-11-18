#!/usr/bin/env python3
"""
设备管理器
统一管理设备检测、选择和配置
"""


import torch

from src.core.logging import get_logger

logger = get_logger()


class DeviceManager:
    """统一的设备管理器"""

    _instance = None
    _logged_messages = set()  # 防止重复日志

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化设备管理器"""
        # 使用单例模式，避免重复初始化
        if not hasattr(self, "initialized"):
            self.initialized = True

    def _log_once(self, message: str, level: str = "info") -> None:
        """
        只记录一次日志，防止重复

        Args:
            message: 日志消息
            level: 日志级别
        """
        if message not in self._logged_messages:
            getattr(logger, level)(message)
            self._logged_messages.add(message)

    def is_mps_available(self) -> bool:
        """
        检查MPS是否可用

        Returns:
            bool: MPS是否可用
        """
        try:
            return (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
            )
        except Exception:
            return False

    def is_cuda_available(self) -> bool:
        """
        检查CUDA是否可用

        Returns:
            bool: CUDA是否可用
        """
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    def get_optimal_device(self) -> torch.device:
        """
        获取最优设备

        Returns:
            torch.device: 最优设备
        """
        if self._optimal_device is not None:
            return self._optimal_device

        # 检查CUDA
        if self.is_cuda_available():
            try:
                # 验证CUDA设备是否真正可用
                device = torch.device("cuda")
                test_tensor = torch.zeros(1).to(device)
                del test_tensor
                torch.cuda.empty_cache()

                if not self._device_info_logged:
                    logger.info("检测到CUDA设备，将使用CUDA")
                    self._device_info_logged = True
                self._optimal_device = device
                return device
            except Exception as e:
                logger.warning(f"CUDA设备验证失败 ({e!s})，尝试其他设备...")

        # 检查MPS
        if self.is_mps_available():
            try:
                # 验证MPS设备是否真正可用
                device = torch.device("mps")
                test_tensor = torch.zeros(1).to(device)
                del test_tensor

                if not self._device_info_logged:
                    logger.info("检测到MPS设备，将使用MPS")
                    self._device_info_logged = True
                self._optimal_device = device
                return device
            except Exception as e:
                logger.warning(f"MPS设备验证失败 ({e!s})，使用CPU...")

        # 默认使用CPU
        if not self._device_info_logged:
            logger.info("未检测到GPU设备，将使用CPU")
            self._device_info_logged = True
        device = torch.device("cpu")
        self._optimal_device = device
        return device

    def get_appropriate_dtype(self, device: torch.device | None = None) -> torch.dtype:
        """
        根据设备获取适当的数据类型

        Args:
            device: 设备对象，如果为None则使用最优设备

        Returns:
            torch.dtype: 适当的数据类型
        """
        if device is None:
            device = self.get_optimal_device()

        # 数据类型选择映射表
        dtype_map = {
            "mps": torch.float32,
            "cuda": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        }

        return dtype_map.get(device.type, torch.float32)

    def should_use_bfloat16(self, device: torch.device | None = None) -> bool:
        """
        判断是否应该使用bfloat16数据类型

        Args:
            device: 设备对象，如果为None则使用最优设备

        Returns:
            bool: 是否应该使用bfloat16
        """
        if device is None:
            device = self.get_optimal_device()

        # bfloat16支持映射表
        bfloat16_support = {
            "mps": False,
            "cuda": torch.cuda.is_bf16_supported(),
        }

        return bfloat16_support.get(device.type, False)

    def move_tensor_to_device(self, tensor: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """
        将张量移动到指定设备

        Args:
            tensor: 张量
            device: 目标设备，如果为None则使用最优设备

        Returns:
            torch.Tensor: 移动后的张量
        """
        if device is None:
            device = self.get_optimal_device()

        # 对于MPS设备的特殊处理
        if device.type == "mps":
            # MPS设备上使用float32而不是float16，以提高兼容性
            if tensor.dtype == torch.float16:
                tensor = tensor.to(torch.float32)

        return tensor.to(device)

    def move_model_to_device(self, model: torch.nn.Module, device: torch.device | None = None) -> torch.nn.Module:
        """
        将模型移动到指定设备

        Args:
            model: PyTorch 模型
            device: 目标设备，如果为None则使用最优设备

        Returns:
            torch.nn.Module: 移动后的模型
        """
        if device is None:
            device = self.get_optimal_device()

        logger.info(f"将模型移动到设备: {device}")

        # 将模型移动到设备
        if hasattr(model, "to"):
            model = model.to(device)
        else:
            logger.warning("模型没有 to 方法，无法移动到设备")

        return model

    def clear_device_cache(self) -> None:
        """清理设备缓存"""
        device = self.get_optimal_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def get_compatible_dtype(self, dtype: torch.dtype, device: torch.device | None = None) -> torch.dtype:
        """
        获取与指定设备兼容的数据类型

        Args:
            dtype: 原始数据类型
            device: 目标设备，如果为None则使用最优设备

        Returns:
            torch.dtype: 兼容的数据类型
        """
        if device is None:
            device = self.get_optimal_device()

        # MPS设备不完全支持float16和bfloat16，使用float32替代
        if device.type == "mps" and dtype in (torch.float16, torch.bfloat16):
            logger.debug(f"MPS设备上使用float32替代{dtype}")
            return torch.float32

        return dtype

    def configure_device_environment(self) -> None:
        """
        配置设备环境，设置必要的环境变量和警告过滤
        应该在应用启动时调用
        """
        import os
        import warnings

        device = self.get_optimal_device()

        if device.type == "mps":
            # 设置PyTorch MPS fallback环境变量，使不支持的操作回退到CPU
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_ALLOW_NON_DETERMINISTIC_ALGO"] = "1"

            # 配置警告过滤
            warnings.filterwarnings(
                "ignore",
                message="The operator.*is not currently supported on the MPS backend.*",
                category=UserWarning,
                module="torch",
            )
            warnings.filterwarnings("ignore", message="Flash Attention is disabled.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*not supported on MPS.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*bfloat16.*", category=UserWarning)

            # 初始清理MPS缓存
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            logger.info("MPS环境配置完成")
        elif device.type == "cuda":
            logger.info("CUDA环境检测完成")

    def log_device_info(self) -> None:
        """记录当前设备信息"""
        logger.info("设备信息:")
        logger.info(f"  PyTorch版本: {torch.__version__}")
        logger.info(f"  MPS可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(f"  MPS构建: {torch.backends.mps.is_built()}")
        logger.info(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA设备数量: {torch.cuda.device_count()}")
            logger.info(f"  当前CUDA设备: {torch.cuda.current_device()}")


# 创建全局设备管理器实例
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """
    获取设备管理器实例

    Returns:
        DeviceManager: 设备管理器实例
    """
    return device_manager


def get_optimal_device() -> torch.device:
    """
    获取最优设备的便捷函数

    Returns:
        torch.device: 最优设备
    """
    return device_manager.get_optimal_device()


def get_appropriate_dtype(device: torch.device | None = None) -> torch.dtype:
    """
    获取适当数据类型的便捷函数

    Args:
        device: 设备对象，如果为None则使用最优设备

    Returns:
        torch.dtype: 适当的数据类型
    """
    return device_manager.get_appropriate_dtype(device)


def should_use_bfloat16(device: torch.device | None = None) -> bool:
    """
    判断是否应该使用bfloat16的便捷函数

    Args:
        device: 设备对象，如果为None则使用最优设备

    Returns:
        bool: 是否应该使用bfloat16
    """
    return device_manager.should_use_bfloat16(device)


def get_compatible_dtype(dtype: torch.dtype, device: torch.device | None = None) -> torch.dtype:
    """
    获取与设备兼容的数据类型

    Args:
        dtype: 原始数据类型
        device: 目标设备，如果为None则使用最优设备

    Returns:
        torch.dtype: 兼容的数据类型
    """
    return device_manager.get_compatible_dtype(dtype, device)


def configure_device_environment() -> None:
    """配置设备环境的便捷函数"""
    return device_manager.configure_device_environment()


def log_device_info() -> None:
    """记录设备信息的便捷函数"""
    return device_manager.log_device_info()
