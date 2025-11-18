#!/usr/bin/env python3
"""
统一配置管理器
整合所有配置项，避免重复定义和配置碎片化
"""

import logging
import os
from typing import Any, Optional

import torch
import yaml

# 使用延迟导入避免循环依赖
_transformers = None
_PROMPT = None
_ModelPathResolver = None

logger = logging.getLogger(__name__)


class AppConfig:
    """统一应用程序配置管理器"""

    _instance = None
    _config = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if self._config is None:
            self._config = self._load_config()
            # 初始化默认配置
            self._init_defaults()

    def _init_defaults(self):
        """初始化默认配置值"""
        # 延迟导入
        global _PROMPT
        if _PROMPT is None:
            from src.core.config.prompts import DEFAULT_OCR_PROMPT

            _PROMPT = DEFAULT_OCR_PROMPT

        # 图像处理配置默认值
        self._config.setdefault("image_processing", {})
        img_config = self._config["image_processing"]
        img_config.setdefault("base_size", 1024)
        img_config.setdefault("image_size", 640)
        img_config.setdefault("crop_mode", True)
        img_config.setdefault("min_crops", 2)
        img_config.setdefault("max_crops", 6)  # max:9; If your GPU memory is small, it is recommended to set it to 6.
        img_config.setdefault("max_concurrency", 100)  # If you have limited GPU memory, lower the concurrency count.
        img_config.setdefault("num_workers", 64)  # image pre-process (resize/padding) workers
        img_config.setdefault("print_num_vis_tokens", False)
        img_config.setdefault("skip_repeat", True)

        # 模型配置默认值
        self._config.setdefault("model", {})
        model_config = self._config["model"]
        model_config.setdefault("path", self._get_default_model_path())
        model_config.setdefault("default_ocr_prompt", _PROMPT)

        # 路径配置默认值
        self._config.setdefault("paths", {})
        paths_config = self._config["paths"]
        paths_config.setdefault("input_path", "")
        paths_config.setdefault("output_path", "")

    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        local_model_path = "./models/deepseek-ocr"
        if os.path.exists(local_model_path):
            return local_model_path
        return "./models/deepseek-ocr"  # change to your model path

    def _load_config(self) -> dict[str, Any]:
        """
        加载应用程序配置

        Returns:
            配置字典
        """
        try:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

            # 查找配置文件
            config_path = os.path.join(project_root, "app.yaml")

            # 如果配置文件不存在，返回空配置
            if not os.path.exists(config_path):
                logger.warning(f"配置文件不存在: {config_path}")
                return {}

            # 加载配置文件
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logger.info(f"成功加载配置文件: {config_path}")
            return config or {}

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    def get_device_config(self, device_type: str) -> dict[str, Any]:
        """
        获取指定设备类型的配置

        Args:
            device_type: 设备类型 (cuda, mps, cpu)

        Returns:
            设备配置字典
        """
        if not self._config:
            return {}

        device_configs = self._config.get("device_config", {})

        # 获取设备配置，如果不存在则返回CPU配置作为默认值
        device_config = device_configs.get(device_type, device_configs.get("cpu", {}))

        logger.debug(f"获取设备配置: {device_type} -> {device_config}")
        return device_config

    def get_model_config(self, model_name: str) -> Optional[dict[str, Any]]:
        """
        获取指定模型的配置

        Args:
            model_name: 模型名称

        Returns:
            模型配置字典
        """
        if not self._config:
            return None

        model_config = self._config.get("model_config", {})
        return model_config.get(model_name, {})

    def get_auto_map_config(self, model_name: str) -> Optional[dict[str, str]]:
        """
        获取指定模型的auto_map配置

        Args:
            model_name: 模型名称

        Returns:
            auto_map配置字典
        """
        model_config = self.get_model_config(model_name)
        if not model_config:
            return None

        return model_config.get("auto_map", {})

    def get_default_model(self) -> str:
        """
        获取默认模型名称

        Returns:
            默认模型名称
        """
        if not self._config:
            return "deepseek_vl_v2"

        return self._config.get("default_model", "deepseek_vl_v2")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键

        Args:
            key: 配置键，支持点号分隔的嵌套键如 "image_processing.base_size"
            default: 默认值

        Returns:
            配置值
        """
        if not self._config:
            return default

        # 处理嵌套键
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    # 图像处理配置
    @property
    def base_size(self) -> int:
        """图像基础尺寸"""
        return self.get("image_processing.base_size", 1024)

    @property
    def image_size(self) -> int:
        """图像尺寸"""
        return self.get("image_processing.image_size", 640)

    @property
    def crop_mode(self) -> bool:
        """裁剪模式"""
        return self.get("image_processing.crop_mode", True)

    @property
    def min_crops(self) -> int:
        """最小裁剪数"""
        return self.get("image_processing.min_crops", 2)

    @property
    def max_crops(self) -> int:
        """最大裁剪数"""
        return self.get("image_processing.max_crops", 6)

    @property
    def max_concurrency(self) -> int:
        """最大并发数"""
        return self.get("image_processing.max_concurrency", 100)

    @property
    def num_workers(self) -> int:
        """工作进程数"""
        return self.get("image_processing.num_workers", 64)

    @property
    def print_num_vis_tokens(self) -> bool:
        """是否打印视觉token数量"""
        return self.get("image_processing.print_num_vis_tokens", False)

    @property
    def skip_repeat(self) -> bool:
        """是否跳过重复"""
        return self.get("image_processing.skip_repeat", True)

    # 模型配置
    @property
    def model_path(self) -> str:
        """模型路径"""
        return self.get("model.path", "./models/deepseek-ocr")

    @property
    def MODEL_PATH(self) -> str:
        """模型路径（大写版本，保持向后兼容）"""
        return self.model_path

    @property
    def default_ocr_prompt(self) -> str:
        """默认OCR提示词"""
        return self.get("model.default_ocr_prompt", _PROMPT)

    @property
    def DEFAULT_OCR_PROMPT(self) -> str:
        """默认OCR提示词（大写版本，保持向后兼容）"""
        return self.default_ocr_prompt

    # 路径配置
    @property
    def input_path(self) -> str:
        """输入路径"""
        return self.get("paths.input_path", "")

    @property
    def INPUT_PATH(self) -> str:
        """输入路径（大写版本，保持向后兼容）"""
        return self.input_path

    @property
    def output_path(self) -> str:
        """输出路径"""
        return self.get("paths.output_path", "")

    @property
    def OUTPUT_PATH(self) -> str:
        """输出路径（大写版本，保持向后兼容）"""
        return self.output_path

    @property
    def tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer is None:
            # 延迟导入
            global _transformers, _ModelPathResolver
            if _transformers is None:
                from transformers import AutoTokenizer

                _transformers = AutoTokenizer
            if _ModelPathResolver is None:
                from src.core.utils.model_path_utils import ModelPathResolver

                _ModelPathResolver = ModelPathResolver

            # 使用 ModelPathResolver 统一处理路径解析和参数配置
            loading_params = _ModelPathResolver.get_loading_params(
                self.model_path, trust_remote_code=False  # 对于 tokenizer，不需要 trust_remote_code
            )
            local_files_only = loading_params["local_files_only"]
            trust_remote_code_for_tokenizer = loading_params["trust_remote_code"]

            self._tokenizer = _transformers.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code_for_tokenizer,
                local_files_only=local_files_only,
            )
        return self._tokenizer

    def get_tokenizer(self):
        """获取tokenizer（方法版本，保持向后兼容）"""
        return self.tokenizer

    def validate_config(self) -> list[str]:
        """
        验证配置的有效性

        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []

        # 验证图像处理配置
        img_config = self._config.get("image_processing", {})

        # 验证尺寸配置
        base_size = img_config.get("base_size", 1024)
        image_size = img_config.get("image_size", 640)

        if not isinstance(base_size, int) or base_size <= 0:
            errors.append("image_processing.base_size 必须是正整数")
        if not isinstance(image_size, int) or image_size <= 0:
            errors.append("image_processing.image_size 必须是正整数")
        if image_size > base_size:
            errors.append("image_processing.image_size 不能大于 base_size")

        # 验证裁剪配置
        min_crops = img_config.get("min_crops", 2)
        max_crops = img_config.get("max_crops", 6)

        if not isinstance(min_crops, int) or min_crops < 1:
            errors.append("image_processing.min_crops 必须是大于0的整数")
        if not isinstance(max_crops, int) or max_crops < min_crops:
            errors.append("image_processing.max_crops 必须大于等于 min_crops")
        if max_crops > 9:
            errors.append("image_processing.max_crops 不建议超过9")

        # 验证并发配置
        max_concurrency = img_config.get("max_concurrency", 100)
        num_workers = img_config.get("num_workers", 64)

        if not isinstance(max_concurrency, int) or max_concurrency <= 0:
            errors.append("image_processing.max_concurrency 必须是正整数")
        if not isinstance(num_workers, int) or num_workers <= 0:
            errors.append("image_processing.num_workers 必须是正整数")

        # 验证模型配置
        model_config = self._config.get("model", {})
        model_path = model_config.get("path", "")

        if not model_path:
            errors.append("model.path 不能为空")

        # 验证设备配置
        device_config = self._config.get("device_config", {})
        supported_devices = ["cuda", "mps", "cpu"]

        for device_type in device_config.keys():
            if device_type not in supported_devices:
                errors.append(f"不支持的设备类型: {device_type}")

        return errors

    def is_valid(self) -> bool:
        """
        检查配置是否有效

        Returns:
            配置是否有效
        """
        return len(self.validate_config()) == 0


# 全局配置实例
_app_config = AppConfig()


def get_app_config() -> AppConfig:
    """
    获取应用程序配置实例

    Returns:
        AppConfig: 应用程序配置实例
    """
    return _app_config


def get_device_config(device_type: str) -> dict[str, Any]:
    """
    获取指定设备类型的配置

    Args:
        device_type: 设备类型 (cuda, mps, cpu)

    Returns:
        设备配置字典
    """
    # 确保device_type是字符串
    if isinstance(device_type, torch.device):
        device_type = device_type.type

    return _app_config.get_device_config(device_type)


def get_model_auto_map(model_name: str) -> Optional[dict[str, str]]:
    """
    获取指定模型的auto_map配置

    Args:
        model_name: 模型名称

    Returns:
        auto_map配置字典
    """
    return _app_config.get_auto_map_config(model_name)


def get_default_model() -> str:
    """
    获取默认模型名称

    Returns:
        默认模型名称
    """
    return _app_config.get_default_model()
