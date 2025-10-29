"""
API 数据模型定义
"""

from pydantic import BaseModel


class OCRRequest(BaseModel):
    """OCR处理请求模型"""

    prompt: str | None = None


class OCRResponse(BaseModel):
    """OCR处理响应模型"""

    job_id: str
    mmd: str
    det_mmd: str
    layouts: str
    images: str
    all: str


class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""

    type: str
    progress: int | None = None
    message: str | None = None
