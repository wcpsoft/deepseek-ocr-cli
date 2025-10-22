"""
DeepSeek OCR API 服务核心实现
支持transformers和vllm两种推理模式
"""
import os
import shutil
import uuid
import asyncio
import zipfile
import io
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core.config import MODEL_PATH, PROMPT
from cli.document_processor import VLLMOCRProcessor, TransformersOCRProcessor, DocumentProcessor
from cli.main import detect_mps_environment

app = FastAPI(title="DeepSeek-OCR API Service")

# 存储活跃的WebSocket连接
active_connections: Dict[str, WebSocket] = {}

@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...), prompt: str = Form('')):
    """处理PDF文件OCR识别"""
    if file.content_type not in ("application/pdf", "application/x-pdf", "application/acrobat") and (not file.filename or not file.filename.lower().endswith('.pdf')):
        return JSONResponse({"error": "Please upload a PDF."}, status_code=400)

    job_id = uuid.uuid4().hex[:8]
    uploads_dir = project_root / 'server' / 'uploads' / job_id
    output_dir = project_root / 'server' / 'outputs' / job_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传的文件
    pdf_path = uploads_dir / (file.filename or "uploaded_file.pdf")
    with open(pdf_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # 启动后台OCR任务
    asyncio.create_task(run_ocr_task(job_id, pdf_path, output_dir, prompt))

    # 立即返回任务信息
    return {
        "job_id": job_id,
        "mmd": f"/api/download/{job_id}/mmd",
        "det_mmd": f"/api/download/{job_id}/det_mmd",
        "layouts": f"/api/download/{job_id}/layouts",
        "images": f"/api/download/{job_id}/images",
        "all": f"/api/download/{job_id}/all",
    }


async def run_ocr_task(job_id: str, pdf_path: Path, output_dir: Path, prompt: str):
    """运行OCR任务"""
    try:
        if job_id in active_connections:
            await send_progress(job_id, 15, "初始化模型...")

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
        processor = DocumentProcessor(
            mode=mode,
            model_path=MODEL_PATH,
            prompt=prompt or PROMPT
        )

        if job_id in active_connections:
            await send_progress(job_id, 30, "加载模型中...")

        # 处理PDF文件
        processor._process_pdf(str(pdf_path), str(output_dir))

        if job_id in active_connections:
            await send_complete(job_id)

    except Exception as e:
        if job_id in active_connections:
            await send_error(job_id, str(e))


async def send_progress(job_id: str, progress: int, message: str):
    """通过WebSocket发送进度更新"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "progress",
            "progress": progress,
            "message": message
        })


async def send_log(job_id: str, message: str):
    """通过WebSocket发送日志消息"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "log",
            "message": message
        })


async def send_complete(job_id: str):
    """通过WebSocket发送完成消息"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "complete"
        })


async def send_error(job_id: str, error: str):
    """通过WebSocket发送错误消息"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "error",
            "message": error
        })


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket端点用于实时进度更新"""
    await websocket.accept()
    active_connections[job_id] = websocket

    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
    except WebSocketDisconnect:
        if job_id in active_connections:
            del active_connections[job_id]


@app.get('/download/{job_id}/mmd')
def download_mmd(job_id: str):
    """下载Markdown文件"""
    output_dir = project_root / 'server' / 'outputs' / job_id
    # 查找单个.mmd文件
    for p in output_dir.glob('*.mmd'):
        if not p.name.endswith('_det.mmd'):
            return FileResponse(str(p), filename=p.name, media_type='text/markdown')
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get('/download/{job_id}/det_mmd')
def download_det_mmd(job_id: str):
    """下载完整标注文件"""
    output_dir = project_root / 'server' / 'outputs' / job_id
    for p in output_dir.glob('*_det.mmd'):
        return FileResponse(str(p), filename=p.name, media_type='text/markdown')
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get('/download/{job_id}/layouts')
def download_layouts(job_id: str):
    """下载可视化PDF"""
    output_dir = project_root / 'server' / 'outputs' / job_id
    for p in output_dir.glob('*_layouts.pdf'):
        return FileResponse(str(p), filename=p.name, media_type='application/pdf')
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get('/download/{job_id}/images')
def download_images(job_id: str):
    """下载所有提取的图像为zip文件"""
    output_dir = project_root / 'server' / 'outputs' / job_id
    images_dir = output_dir / 'images'

    if not images_dir.exists() or not any(images_dir.iterdir()):
        return JSONResponse({"error": "No images found"}, status_code=404)

    # 在内存中创建zip文件
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_file in images_dir.glob('*'):
            if img_file.is_file():
                zip_file.write(img_file, arcname=f'images/{img_file.name}')

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="images_{job_id}.zip"'}
    )


@app.get('/download/{job_id}/all')
def download_all(job_id: str):
    """下载所有结果（markdown文件、图像和布局PDF）为zip文件"""
    output_dir = project_root / 'server' / 'outputs' / job_id

    if not output_dir.exists():
        return JSONResponse({"error": "Job not found"}, status_code=404)

    # 在内存中创建zip文件
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 添加输出目录中的所有文件
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                # 计算相对于output_dir的路径
                rel_path = file_path.relative_to(output_dir)
                zip_file.write(file_path, arcname=str(rel_path))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="ocr_results_{job_id}.zip"'}
    )