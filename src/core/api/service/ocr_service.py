"""
OCR 服务实现
"""

from pathlib import Path

from fastapi import WebSocket

from cli.document_processor import DocumentProcessor
from cli.main import detect_mps_environment
from src.core.api.utils.helpers import generate_job_id
from src.core.config import MODEL_PATH, PROMPT


class OCRService:
    """OCR服务类"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.active_connections: dict[str, WebSocket] = {}

    def create_job(self) -> str:
        """创建新的OCR任务"""
        return generate_job_id()

    async def process_pdf(self, job_id: str, pdf_path: Path, output_dir: Path, prompt: str):
        """处理PDF文件"""
        try:
            if job_id in self.active_connections:
                await self.send_progress(job_id, 15, "初始化模型...")

            # 确定使用哪种模式
            mode = "auto"
            if detect_mps_environment():
                mode = "transformers"  # MPS环境下使用transformers
            else:
                # 检查vLLM是否可用
                try:
                    import vllm  # type: ignore # noqa: F401

                    mode = "vllm"
                except ImportError:
                    mode = "transformers"

            # 创建文档处理器
            processor = DocumentProcessor(mode=mode, model_path=MODEL_PATH, prompt=prompt or PROMPT)

            if job_id in self.active_connections:
                await self.send_progress(job_id, 30, "加载模型中...")

            # 处理PDF文件
            processor._process_pdf(str(pdf_path), str(output_dir))

            if job_id in self.active_connections:
                await self.send_complete(job_id)

        except Exception as e:
            if job_id in self.active_connections:
                await self.send_error(job_id, str(e))

    async def send_progress(self, job_id: str, progress: int, message: str):
        """通过WebSocket发送进度更新"""
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json(
                {"type": "progress", "progress": progress, "message": message}
            )

    async def send_log(self, job_id: str, message: str):
        """通过WebSocket发送日志消息"""
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json({"type": "log", "message": message})

    async def send_complete(self, job_id: str):
        """通过WebSocket发送完成消息"""
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json({"type": "complete"})

    async def send_error(self, job_id: str, error: str):
        """通过WebSocket发送错误消息"""
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json({"type": "error", "message": error})

    async def connect_websocket(self, job_id: str, websocket: WebSocket):
        """连接WebSocket"""
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect_websocket(self, job_id: str):
        """断开WebSocket连接"""
        if job_id in self.active_connections:
            del self.active_connections[job_id]
