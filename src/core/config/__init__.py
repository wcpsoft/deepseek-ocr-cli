# 导入新的统一配置
from .settings import *
from .prompts import get_prompt, DEFAULT_OCR_PROMPT

# 导出get_config函数
from .settings import get_config

# 明确导出常用常量
from .settings import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS