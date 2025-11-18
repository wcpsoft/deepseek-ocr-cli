#!/usr/bin/env python3
"""
DeepSeek OCR 配置文件 (弃用包装器)
⚠️  此文件已被弃用，请使用 src.core.config.app_config 中的 AppConfig

此文件保持向后兼容性，将所有配置请求重定向到 AppConfig
"""

import logging
import warnings

# 延迟导入以避免循环依赖
_app_config = None


def _get_app_config():
    """延迟导入 AppConfig"""
    global _app_config
    if _app_config is None:
        from src.core.config.app_config import get_app_config

        _app_config = get_app_config()
    return _app_config


logger = logging.getLogger(__name__)

# 发出弃用警告
warnings.warn(
    "src.core.config.settings 已弃用，请使用 src.core.config.app_config.get_app_config()",
    DeprecationWarning,
    stacklevel=2,
)


# 向后兼容的常量导出（使用属性访问延迟加载）
def _get_config_value(attr_name):
    """获取配置值"""
    return getattr(_get_app_config(), attr_name)


BASE_SIZE = _get_config_value("base_size")
IMAGE_SIZE = _get_config_value("image_size")
CROP_MODE = _get_config_value("crop_mode")
MIN_CROPS = _get_config_value("min_crops")
MAX_CROPS = _get_config_value("max_crops")
MAX_CONCURRENCY = _get_config_value("max_concurrency")
NUM_WORKERS = _get_config_value("num_workers")
PRINT_NUM_VIS_TOKENS = _get_config_value("print_num_vis_tokens")
SKIP_REPEAT = _get_config_value("skip_repeat")
MODEL_PATH = _get_config_value("model_path")
INPUT_PATH = _get_config_value("input_path")
OUTPUT_PATH = _get_config_value("output_path")
DEFAULT_OCR_PROMPT = _get_config_value("default_ocr_prompt")

# 延迟加载的tokenizer（保持向后兼容）
TOKENIZER = None


def get_tokenizer():
    """
    获取tokenizer (已弃用)
    ⚠️  请使用 get_app_config().tokenizer 或 get_app_config().get_tokenizer()
    """
    warnings.warn("get_tokenizer() 已弃用，请使用 get_app_config().tokenizer", DeprecationWarning, stacklevel=2)
    return _get_app_config().get_tokenizer()


class Config:
    """
    配置类 (已弃用)
    ⚠️  请使用 src.core.config.app_config.AppConfig
    """

    def __init__(self):
        warnings.warn("Config 类已弃用，请使用 get_app_config()", DeprecationWarning, stacklevel=2)
        # 保留所有属性以保持向后兼容
        app_config = _get_app_config()
        self.base_size = app_config.base_size
        self.image_size = app_config.image_size
        self.crop_mode = app_config.crop_mode
        self.min_crops = app_config.min_crops
        self.max_crops = app_config.max_crops
        self.max_concurrency = app_config.max_concurrency
        self.num_workers = app_config.num_workers
        self.print_num_vis_tokens = app_config.print_num_vis_tokens
        self.skip_repeat = app_config.skip_repeat
        self.model_path = app_config.model_path
        self.MODEL_PATH = app_config.MODEL_PATH  # 保持兼容性
        self.input_path = app_config.input_path
        self.output_path = app_config.output_path
        self.prompt = app_config.default_ocr_prompt
        self._tokenizer = None

    @property
    def tokenizer(self):
        """延迟加载tokenizer (已弃用)"""
        warnings.warn("Config.tokenizer 已弃用，请使用 get_app_config().tokenizer", DeprecationWarning, stacklevel=2)
        return _get_app_config().tokenizer


# 创建全局配置实例 (保持向后兼容)
config = Config()


def get_config():
    """
    获取全局配置实例 (已弃用)
    ⚠️  请使用 get_app_config()
    """
    warnings.warn("get_config() 已弃用，请使用 get_app_config()", DeprecationWarning, stacklevel=2)
    return _get_app_config()
