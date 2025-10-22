"""
API 数据模型定义
"""
from pydantic import BaseModel
from typing import Optional


class OCRRequest(BaseModel):
    """OCR处理请求模型"""
    prompt: Optional[str] = None


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
    progress: Optional[int] = None
    message: Optional[str] = None