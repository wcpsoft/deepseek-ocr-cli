#!/usr/bin/env python3
"""
DeepSeek OCR Web API 服务入口
支持transformers和vllm两种推理模式
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 配置警告过滤器
from src.core.warnings_config import configure_warnings

configure_warnings()

# 从main.py导入应用实例
from src.core.api.main import app

# 导入配置管理器
from src.core.config.app_config import AppConfig

if __name__ == "__main__":
    import uvicorn

    # 从app.yaml加载配置
    config_manager = AppConfig()
    config = config_manager._config or {}
    server_config = config.get("server", {})

    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 8000)
    reload = server_config.get("reload", True)

    # 运行从src/core/api/main.py导入的应用实例
    uvicorn.run(app, host=host, port=port, reload=reload)
