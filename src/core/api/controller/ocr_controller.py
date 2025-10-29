"""
OCR 控制器实现
"""

import asyncio
import io
import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.core.api.model.ocr_model import OCRResponse
from src.core.api.service.ocr_service import OCRService
from src.core.api.utils.helpers import create_job_directories


class OCRController:
    """OCR控制器类"""

    def __init__(self, project_root: Path):
        self.router = APIRouter()
        self.project_root = project_root
        self.ocr_service = OCRService(project_root)
        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""
        self.router.post("/ocr/pdf")(self.ocr_pdf)
        self.router.websocket("/ws/{job_id}")(self.websocket_endpoint)
        self.router.get("/download/{job_id}/mmd")(self.download_mmd)
        self.router.get("/download/{job_id}/det_mmd")(self.download_det_mmd)
        self.router.get("/download/{job_id}/layouts")(self.download_layouts)
        self.router.get("/download/{job_id}/images")(self.download_images)
        self.router.get("/download/{job_id}/all")(self.download_all)

    async def ocr_pdf(self, file: UploadFile, prompt: str = Form("")):
        """处理PDF文件OCR识别"""
        if file is None:
            file = File(...)
        if file.content_type not in (
            "application/pdf",
            "application/x-pdf",
            "application/acrobat",
        ) and (not file.filename or not file.filename.lower().endswith(".pdf")):
            return JSONResponse({"error": "Please upload a PDF."}, status_code=400)

        job_id = self.ocr_service.create_job()
        uploads_dir, output_dir = create_job_directories(self.project_root, job_id)

        # 保存上传的文件
        pdf_path = uploads_dir / (file.filename or "uploaded_file.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 启动后台OCR任务
        asyncio.create_task(self.ocr_service.process_pdf(job_id, pdf_path, output_dir, prompt))

        # 立即返回任务信息
        return OCRResponse(
            job_id=job_id,
            mmd=f"/api/download/{job_id}/mmd",
            det_mmd=f"/api/download/{job_id}/det_mmd",
            layouts=f"/api/download/{job_id}/layouts",
            images=f"/api/download/{job_id}/images",
            all=f"/api/download/{job_id}/all",
        )

    async def websocket_endpoint(self, websocket: WebSocket, job_id: str):
        """WebSocket端点用于实时进度更新"""
        await self.ocr_service.connect_websocket(job_id, websocket)

        try:
            while True:
                # 保持连接活跃
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.ocr_service.disconnect_websocket(job_id)

    def download_mmd(self, job_id: str):
        """下载Markdown文件"""
        output_dir = self.project_root / "server" / "outputs" / job_id
        # 查找单个.mmd文件
        for p in output_dir.glob("*.mmd"):
            if not p.name.endswith("_det.mmd"):
                return FileResponse(str(p), filename=p.name, media_type="text/markdown")
        return JSONResponse({"error": "file not found"}, status_code=404)

    def download_det_mmd(self, job_id: str):
        """下载完整标注文件"""
        output_dir = self.project_root / "server" / "outputs" / job_id
        for p in output_dir.glob("*_det.mmd"):
            return FileResponse(str(p), filename=p.name, media_type="text/markdown")
        return JSONResponse({"error": "file not found"}, status_code=404)

    def download_layouts(self, job_id: str):
        """下载可视化PDF"""
        output_dir = self.project_root / "server" / "outputs" / job_id
        for p in output_dir.glob("*_layouts.pdf"):
            return FileResponse(str(p), filename=p.name, media_type="application/pdf")
        return JSONResponse({"error": "file not found"}, status_code=404)

    def download_images(self, job_id: str):
        """下载所有提取的图像为zip文件"""
        output_dir = self.project_root / "server" / "outputs" / job_id
        images_dir = output_dir / "images"

        if not images_dir.exists() or not any(images_dir.iterdir()):
            return JSONResponse({"error": "No images found"}, status_code=404)

        # 在内存中创建zip文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for img_file in images_dir.glob("*"):
                if img_file.is_file():
                    zip_file.write(img_file, arcname=f"images/{img_file.name}")

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="images_{job_id}.zip"'},
        )

    def download_all(self, job_id: str):
        """下载所有结果（markdown文件、图像和布局PDF）为zip文件"""
        output_dir = self.project_root / "server" / "outputs" / job_id

        if not output_dir.exists():
            return JSONResponse({"error": "Job not found"}, status_code=404)

        # 在内存中创建zip文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 添加输出目录中的所有文件
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    # 计算相对于output_dir的路径
                    rel_path = file_path.relative_to(output_dir)
                    zip_file.write(file_path, arcname=str(rel_path))

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="ocr_results_{job_id}.zip"'},
        )
