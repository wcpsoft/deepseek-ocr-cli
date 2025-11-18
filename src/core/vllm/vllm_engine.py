#!/usr/bin/env python3
"""
vLLM引擎实现
"""

import logging
from typing import Any, Optional

from PIL import Image

from src.core.base.ocr_engine import BaseOCREngine
from src.core.utils.device_manager import get_optimal_device

# 获取日志记录器
logger = logging.getLogger(__name__)


class VLLMEngine(BaseOCREngine):
    """
    vLLM引擎实现类
    提供基于vLLM的OCR推理功能
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        prompt: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        *,
        crop_mode: bool = True,
        model_manager: Optional[Any] = None,
        device_manager: Optional[Any] = None,
    ) -> None:
        """
        初始化vLLM引擎

        Args:
            model_path: 模型路径
            device: 设备类型
            prompt: 提示词
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否启用裁剪模式
            model_manager: 注入的模型管理器
            device_manager: 注入的设备管理器
        """
        super().__init__(model_path, prompt, base_size, image_size, device, crop_mode=crop_mode)
        self.model: Any = None
        self.tokenizer: Any = None
        self._model_manager = model_manager
        self._device_manager = device_manager
        logger.info("vLLM引擎初始化完成")

    def initialize(self) -> bool:
        """初始化vLLM模型"""
        try:
            logger.info("初始化vLLM引擎...")

            # 使用 ModelPathResolver 统一处理路径解析和参数配置
            from src.core.utils.model_path_utils import ModelPathResolver

            # 获取统一的加载参数
            loading_params = ModelPathResolver.get_loading_params(
                self.model_path, trust_remote_code=True  # vLLM 需要 trust_remote_code
            )
            local_files_only = loading_params["local_files_only"]

            # vLLM相关导入（延迟导入，避免在不支持的平台上报错）
            try:
                from vllm import LLM  # type: ignore

                llm_class = LLM
            except ImportError:
                # 在不支持vLLM的平台上设置占位符
                llm_class = object
                logger.warning("vLLM未安装或不支持当前平台，使用占位符")

            # 使用注入的设备管理器或默认获取设备
            if self._device_manager:
                optimal_device = self._device_manager.get_optimal_device()
            else:
                from src.core.utils.device_manager import get_optimal_device

                optimal_device = get_optimal_device()
            device_type = optimal_device.type

            # 初始化模型
            vllm_kwargs = {
                "model": self.model_path or "",
                "trust_remote_code": True,
                "tokenizer_mode": "auto",
                "tensor_parallel_size": 1,
                "dtype": "auto",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": False,
                "disable_log_stats": True,
                "skip_tokenizer_init": False,
            }

            # 根据设备类型设置适当的参数
            if device_type == "cuda":
                # CUDA设备使用默认设置
                pass
            elif device_type == "mps":
                # MPS设备需要特殊处理
                logger.info("在MPS设备上运行，调整vLLM参数")
                vllm_kwargs["gpu_memory_utilization"] = 0.8
                # MPS上可能需要禁用某些CUDA特定功能
                vllm_kwargs["enforce_eager"] = True
            else:
                # CPU设备
                logger.info("在CPU设备上运行，调整vLLM参数")
                vllm_kwargs["gpu_memory_utilization"] = 0.5

            # 对于本地路径,确保local_files_only=True,这样就不会尝试从远程下载
            if local_files_only and self.model_path:
                vllm_kwargs["local_files_only"] = True

            self.model = llm_class(**vllm_kwargs)

            # 获取tokenizer
            if hasattr(self.model, "get_tokenizer") and callable(getattr(self.model, "get_tokenizer", None)):
                self.tokenizer = self.model.get_tokenizer()
            else:
                self.tokenizer = None

            self.is_initialized = True
            logger.info("vLLM引擎初始化成功")
            return True
        except Exception as e:
            logger.error(f"vLLM引擎初始化失败: {e}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False

    def process(self, images: list[Image.Image], output_dir: str) -> None:
        """
        处理图像列表

        Args:
            images: 图像列表
            output_dir: 输出目录路径
        """
        if not self.is_initialized:
            raise RuntimeError("模型未初始化")

        try:
            # 这里应该实现具体的文档处理逻辑
            # 由于这是示例代码,我们只返回一个占位符结果
            logger.info(f"处理 {len(images)} 张图像到目录: {output_dir}")
            # 实际实现应该处理每张图像并保存结果到output_dir
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise

    def cleanup(self) -> None:
        """清理资源"""
        if self.model is not None:
            # vLLM模型的清理
            self.model = None
        if self.tokenizer is not None:
            self.tokenizer = None

        # 清理设备缓存
        try:
            import torch

            optimal_device = get_optimal_device()
            device_type = optimal_device.type

            if device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device_type == "mps" and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            logger.warning(f"清理设备缓存时出错: {e}")

        self.is_initialized = False
        logger.info("vLLM引擎资源清理完成")
