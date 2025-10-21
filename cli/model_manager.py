#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
负责下载和管理OCR模型
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

# 忽略导入错误，因为这些包可能在运行时才安装
try:
    from huggingface_hub import snapshot_download
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

class ModelManager:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认模型信息
        self.default_models = {
            "deepseek-ocr": "deepseek-ai/DeepSeek-OCR"
        }
    
    def download_models(self, model_names: Optional[List[str]] = None):
        """下载指定的模型或所有默认模型"""
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("huggingface_hub 包未安装，请先安装: pip install huggingface_hub")
        
        if model_names is None:
            model_names = list(self.default_models.keys())
        
        for model_name in model_names:
            if model_name in self.default_models:
                self._download_model(model_name, self.default_models[model_name])
            else:
                print(f"未知模型: {model_name}")
    
    def _download_model(self, model_name: str, repo_id: str):
        """下载单个模型"""
        print(f"正在下载模型 {model_name} ({repo_id})...")
        
        model_path = self.model_dir / model_name
        
        try:
            # 使用huggingface_hub下载模型
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )
            print(f"模型 {model_name} 下载完成，保存至: {model_path}")
        except Exception as e:
            print(f"模型 {model_name} 下载失败: {str(e)}")
    
    def list_downloaded_models(self) -> List[str]:
        """列出已下载的模型"""
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir():
                models.append(item.name)
        return models
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """获取模型路径"""
        model_path = self.model_dir / model_name
        if model_path.exists():
            return str(model_path)
        return None