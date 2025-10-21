#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
负责下载和管理OCR模型
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

class ModelManager:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认模型信息
        self.default_models = {
            "deepseek-ocr": "deepseek-ai/DeepSeek-OCR"
        }
        
        # 模型配置文件路径
        self.config_file = self.model_dir / "model_config.json"
        self.model_configs = self._load_model_configs()
    
    def _load_model_configs(self) -> Dict[str, Any]:
        """加载模型配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_model_configs(self):
        """保存模型配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_configs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 无法保存模型配置: {e}")
    
    def set_model_dir(self, model_dir: str):
        """设置模型目录"""
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.model_dir / "model_config.json"
        self.model_configs = self._load_model_configs()
    
    def get_model_dir(self) -> str:
        """获取当前模型目录"""
        return str(self.model_dir)
    
    def download_models(self, model_names: Optional[List[str]] = None, force_redownload: bool = False):
        """下载指定的模型或所有默认模型"""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError("huggingface_hub 包未安装，请先安装: pip install huggingface_hub")
        
        if model_names is None:
            model_names = list(self.default_models.keys())
        
        for model_name in model_names:
            if model_name in self.default_models:
                self._download_model(model_name, self.default_models[model_name], force_redownload)
            else:
                # 检查是否是自定义模型
                if model_name in self.model_configs.get("custom_models", {}):
                    repo_id = self.model_configs["custom_models"][model_name]
                    self._download_model(model_name, repo_id, force_redownload)
                else:
                    print(f"未知模型: {model_name}")
    
    def _download_model(self, model_name: str, repo_id: str, force_redownload: bool = False):
        """下载单个模型"""
        print(f"正在下载模型 {model_name} ({repo_id})...")
        
        model_path = self.model_dir / model_name
        
        # 检查是否已存在且不强制重新下载
        if model_path.exists() and not force_redownload:
            print(f"模型 {model_name} 已存在，跳过下载。如需重新下载请使用 --force 参数")
            return
        
        try:
            # 如果目录存在且强制重新下载，则删除旧目录
            if model_path.exists() and force_redownload:
                print(f"删除已存在的模型目录: {model_path}")
                shutil.rmtree(model_path)
            
            # 使用huggingface_hub下载模型
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )
            print(f"模型 {model_name} 下载完成，保存至: {model_path}")
            
            # 更新模型配置
            if "downloaded_models" not in self.model_configs:
                self.model_configs["downloaded_models"] = {}
            
            self.model_configs["downloaded_models"][model_name] = {
                "repo_id": repo_id,
                "path": str(model_path),
                "downloaded_at": __import__('datetime').datetime.now().isoformat()
            }
            
            self._save_model_configs()
        except Exception as e:
            print(f"模型 {model_name} 下载失败: {str(e)}")
    
    def add_custom_model(self, model_name: str, repo_id: str):
        """添加自定义模型"""
        if "custom_models" not in self.model_configs:
            self.model_configs["custom_models"] = {}
        
        self.model_configs["custom_models"][model_name] = repo_id
        self._save_model_configs()
        print(f"已添加自定义模型: {model_name} -> {repo_id}")
    
    def remove_custom_model(self, model_name: str):
        """移除自定义模型"""
        if "custom_models" in self.model_configs and model_name in self.model_configs["custom_models"]:
            del self.model_configs["custom_models"][model_name]
            self._save_model_configs()
            print(f"已移除自定义模型: {model_name}")
        else:
            print(f"未找到自定义模型: {model_name}")
    
    def list_downloaded_models(self) -> List[str]:
        """列出已下载的模型"""
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir() and item.name != ".git":
                models.append(item.name)
        return models
    
    def list_custom_models(self) -> Dict[str, str]:
        """列出自定义模型"""
        return self.model_configs.get("custom_models", {})
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """获取模型路径"""
        model_path = self.model_dir / model_name
        if model_path.exists():
            return str(model_path)
        return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型详细信息"""
        if "downloaded_models" in self.model_configs:
            return self.model_configs["downloaded_models"].get(model_name)
        return None
    
    def verify_model(self, model_name: str) -> bool:
        """验证模型完整性"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            return False
        
        # 检查是否存在基本文件
        model_path = Path(model_path)
        required_files = ["config.json", "pytorch_model.bin"]
        
        for file in required_files:
            if not (model_path / file).exists():
                # 检查是否存在分片模型文件
                if file == "pytorch_model.bin":
                    shard_pattern = "pytorch_model-*.bin"
                    shards = list(model_path.glob(shard_pattern))
                    if not shards:
                        return False
                else:
                    return False
        
        return True