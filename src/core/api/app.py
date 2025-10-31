#!/usr/bin/env python3
"""
DeepSeek OCR Web API 服务入口
支持transformers和vllm两种推理模式
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 导入FastAPI应用

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.core.api.app:app", host="0.0.0.0", port=8000, reload=True)
