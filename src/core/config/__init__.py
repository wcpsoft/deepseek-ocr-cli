# 导入新的统一配置
from .prompts import DEFAULT_OCR_PROMPT, get_prompt

# 明确导出常用常量
# 导出get_config函数
from .settings import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    MAX_CROPS,
    MIN_CROPS,
    PRINT_NUM_VIS_TOKENS,
    get_config,
    get_tokenizer,
)

# 为了向后兼容，导出PROMPT
PROMPT = DEFAULT_OCR_PROMPT

__all__ = [
    "BASE_SIZE",
    "CROP_MODE",
    "DEFAULT_OCR_PROMPT",
    "IMAGE_SIZE",
    "MAX_CROPS",
    "MIN_CROPS",
    "PRINT_NUM_VIS_TOKENS",
    "PROMPT",
    "get_config",
    "get_prompt",
    "get_tokenizer",
]
