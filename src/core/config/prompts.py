#!/usr/bin/env python3
"""
DeepSeek OCR提示词配置
统一管理所有OCR相关的提示词，确保与原始DeepSeek-OCR项目一致
"""

# 默认提示词，与原始DeepSeek-OCR项目保持一致
DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# 其他可用的提示词选项
FREE_OCR_PROMPT = "<image>\nFree OCR."
DOCUMENT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
IMAGE_OCR_PROMPT = "<image>\n<|grounding|>OCR this image."
FIGURE_PARSE_PROMPT = "<image>\nParse the figure."
GENERAL_DESCRIBE_PROMPT = "<image>\nDescribe this image in detail."
LOCALIZE_PROMPT_TEMPLATE = "<image>\nLocate <|ref|>{}<|/ref|> in the image."

# 提示词映射字典，便于管理和扩展
PROMPT_TEMPLATES = {
    "default": DEFAULT_OCR_PROMPT,
    "free": FREE_OCR_PROMPT,
    "document": DOCUMENT_OCR_PROMPT,
    "image": IMAGE_OCR_PROMPT,
    "figure": FIGURE_PARSE_PROMPT,
    "general": GENERAL_DESCRIBE_PROMPT,
}


def get_prompt(prompt_type: str = "default", custom_text: str | None = None) -> str:
    """
    获取指定类型的提示词

    Args:
        prompt_type: 提示词类型
        custom_text: 自定义文本

    Returns:
        提示词字符串
    """
    if custom_text:
        return custom_text

    return PROMPT_TEMPLATES.get(prompt_type, DEFAULT_OCR_PROMPT)
