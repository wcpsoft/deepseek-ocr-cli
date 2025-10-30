#!/usr/bin/env python3
"""
DeepSeek OCR统一配置文件
整合所有配置项，避免重复定义
"""

import os

from transformers import AutoTokenizer

# 导入提示词配置
from src.core.config.prompts import DEFAULT_OCR_PROMPT as PROMPT

# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100  # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True

# 检查本地模型路径
local_model_path = "./models/deepseek-ocr"
if os.path.exists(local_model_path):
    MODEL_PATH = local_model_path
else:
    MODEL_PATH = "./models/deepseek-ocr"  # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py;
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py;
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = ""
OUTPUT_PATH = ""

DEFAULT_OCR_PROMPT = PROMPT

# 延迟导入TOKENIZER，避免在不需要时加载依赖
TOKENIZER = None


def get_tokenizer():
    """延迟加载tokenizer"""
    global TOKENIZER
    if TOKENIZER is None:
        # 检查是否是本地路径，如果是则只使用本地文件
        # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
        is_remote_repo = (
            MODEL_PATH.startswith(("http://", "https://"))
            or MODEL_PATH.startswith("deepseek-ai/")
            or MODEL_PATH.startswith("huggingface.co/")
            or "/" not in MODEL_PATH  # 单个名称可能是远程仓库名
            or (not os.path.exists(MODEL_PATH) and not os.path.exists(os.path.expanduser(MODEL_PATH)))
        )

        # 对于本地路径，确保local_files_only=True
        # 对于远程仓库，确保local_files_only=False
        local_files_only = not is_remote_repo

        # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
        # 对于远程模型，使用trust_remote_code=True
        trust_remote_code_for_tokenizer = is_remote_repo

        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=trust_remote_code_for_tokenizer,
            local_files_only=local_files_only,
        )
    return TOKENIZER


# 配置类，用于面向对象的配置管理
class Config:
    """配置类，封装所有配置项"""

    def __init__(self):
        # 图像处理配置
        self.base_size = BASE_SIZE
        self.image_size = IMAGE_SIZE
        self.crop_mode = CROP_MODE
        self.min_crops = MIN_CROPS
        self.max_crops = MAX_CROPS
        self.max_concurrency = MAX_CONCURRENCY
        self.num_workers = NUM_WORKERS
        self.print_num_vis_tokens = PRINT_NUM_VIS_TOKENS
        self.skip_repeat = SKIP_REPEAT

        # 模型配置
        self.model_path = MODEL_PATH
        self.MODEL_PATH = MODEL_PATH  # 添加大写版本以保持兼容性

        # 路径配置
        self.input_path = INPUT_PATH
        self.output_path = OUTPUT_PATH

        # 提示词配置
        self.prompt = PROMPT

        # 延迟加载的tokenizer
        self._tokenizer = None

    @property
    def tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer is None:
            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = (
                self.model_path.startswith(("http://", "https://"))
                or self.model_path.startswith("deepseek-ai/")
                or self.model_path.startswith("huggingface.co/")
                or "/" not in self.model_path  # 单个名称可能是远程仓库名
                or (not os.path.exists(self.model_path) and not os.path.exists(os.path.expanduser(self.model_path)))
            )

            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo

            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用trust_remote_code=True
            trust_remote_code_for_tokenizer = is_remote_repo

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code_for_tokenizer,
                local_files_only=local_files_only,
            )
        return self._tokenizer


# 创建全局配置实例
config = Config()


def get_config():
    """获取全局配置实例"""
    return config
