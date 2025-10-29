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
