"""
API 工具函数
"""

import uuid
from pathlib import Path


def generate_job_id() -> str:
    """生成唯一的任务ID"""
    return uuid.uuid4().hex[:8]


def create_job_directories(project_root: Path, job_id: str, *, is_web_api: bool = True) -> tuple[Path, Path]:
    """创建任务目录"""
    if is_web_api:
        uploads_dir = project_root / "server" / "uploads" / job_id
        output_dir = project_root / "server" / "outputs" / job_id
    else:
        # 命令行接口使用扁平目录结构
        uploads_dir = project_root / "uploads" / job_id
        output_dir = project_root / "outputs" / job_id

    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir, output_dir
