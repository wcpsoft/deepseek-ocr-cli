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

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()

# 定义需要忽略的文件和文件夹模式
# 这些文件通常与模型运行无关，仅用于开发、测试或文档目的
IGNORE_PATTERNS = [
    "*.md",                    # 忽略Markdown文档
    "README*",                 # 忽略README文件
    "LICENSE*",                # 忽略许可证文件
    "assets/*",                # 忽略assets文件夹（通常包含示例图片等）
    "examples/*",              # 忽略examples文件夹
    "scripts/*",               # 忽略scripts文件夹
    "tests/*",                 # 忽略测试文件夹
    ".git*",                   # 忽略Git相关文件
    "*.ipynb_checkpoints",     # 忽略Jupyter Notebook检查点
    ".ipynb_checkpoints/*",    # 忽略Jupyter Notebook检查点文件夹
    ".msc",                    # 忽略.msc文件
    ".mv",                     # 忽略.mv文件
    "._____temp",              # 忽略临时文件
    # 保留必要的模型实现文件，只忽略不需要的Python文件
    # 注意：我们保留模型实现文件，因为它们是运行模型所必需的
    "convert_*.py",            # 忽略转换脚本
    "flax_*.py",               # 忽略Flax相关文件
    "tf_*.py",                 # 忽略TensorFlow相关文件
    "torch_*.py",              # 忽略PyTorch相关文件
]

class ModelManager:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认模型信息 - 支持多个源
        self.default_models = {
            "deepseek-ocr": {
                "repo_id": "deepseek-ai/DeepSeek-OCR",
                "source": "huggingface",  # 添加缺失的source字段
                "sources": {
                    "huggingface": "deepseek-ai/DeepSeek-OCR",
                    "modelscope": "deepseek-ai/DeepSeek-OCR"  # 如果ModelScope上有相同的模型ID
                }
            }
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
            logger.warning(f"无法保存模型配置: {e}")
    
    def set_model_dir(self, model_dir: str):
        """设置模型目录"""
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.model_dir / "model_config.json"
        self.model_configs = self._load_model_configs()
    
    def get_model_dir(self) -> str:
        """获取当前模型目录"""
        return str(self.model_dir)
    
    def download_models(self, model_names: Optional[List[str]] = None, force_redownload: bool = False, source: str = "huggingface"):
        """下载指定的模型或所有默认模型"""
        if model_names is None:
            model_names = list(self.default_models.keys())
        
        for model_name in model_names:
            if model_name in self.default_models:
                model_info = self.default_models[model_name]
                repo_id = model_info["repo_id"]  # 默认使用repo_id
                # 如果指定了源且该模型支持该源，则使用对应源的repo_id
                if isinstance(model_info, dict) and "sources" in model_info:
                    sources = model_info["sources"]
                    if source in sources:
                        repo_id = sources[source]
                self._download_model(model_name, repo_id, source, force_redownload)
            else:
                # 检查是否是自定义模型
                if model_name in self.model_configs.get("custom_models", {}):
                    model_info = self.model_configs["custom_models"][model_name]
                    if isinstance(model_info, dict):
                        repo_id = model_info["repo_id"]
                        source = model_info.get("source", "huggingface")
                    else:
                        repo_id = model_info
                        source = "huggingface"
                    self._download_model(model_name, repo_id, source, force_redownload)
                else:
                    logger.error(f"未知模型: {model_name}")
    
    def _download_model(self, model_name: str, repo_id: str, source: str = "huggingface", force_redownload: bool = False):
        """下载单个模型"""
        logger.info(f"正在从 {source} 下载模型 {model_name} ({repo_id})...")
        
        model_path = self.model_dir / model_name
        
        # 检查是否已存在且不强制重新下载
        if model_path.exists() and not force_redownload:
            logger.info(f"模型 {model_name} 已存在，跳过下载。如需重新下载请使用 --force 参数")
            return
        
        try:
            # 如果目录存在且强制重新下载，则删除旧目录
            if model_path.exists() and force_redownload:
                logger.info(f"删除已存在的模型目录: {model_path}")
                shutil.rmtree(model_path)
            
            # 根据来源下载模型
            if source == "huggingface":
                self._download_from_huggingface(repo_id, model_path)
            elif source == "modelscope":
                self._download_from_modelscope(repo_id, model_path)
            else:
                logger.error(f"不支持的模型来源: {source}")
                return
            
            logger.info(f"模型 {model_name} 下载完成，保存至: {model_path}")
            
            # 更新模型配置
            if "downloaded_models" not in self.model_configs:
                self.model_configs["downloaded_models"] = {}
            
            self.model_configs["downloaded_models"][model_name] = {
                "repo_id": repo_id,
                "source": source,
                "path": str(model_path),
                "downloaded_at": __import__('datetime').datetime.now().isoformat()
            }
            
            self._save_model_configs()
        except Exception as e:
            logger.error(f"模型 {model_name} 下载失败: {str(e)}")
    
    def _download_from_huggingface(self, repo_id: str, model_path: Path):
        """从Hugging Face下载模型（仅下载运行必需的文件）"""
        try:
            from huggingface_hub import snapshot_download
            
            # 添加允许模式以提高下载效率
            allow_patterns = [
                "*",              # 允许所有文件
                "**/*",           # 允许所有子目录文件
            ]
            
            logger.info(f"正在从Hugging Face下载模型 {repo_id} 到 {model_path}")
            
            # 下载模型文件
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                ignore_patterns=IGNORE_PATTERNS,
                allow_patterns=allow_patterns,  # 添加allow_patterns参数
                resume_download=True  # 允许断点续传
            )
        except ImportError:
            raise RuntimeError("huggingface_hub 包未安装，请先安装: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"从Hugging Face下载模型失败: {str(e)}")
    
    def _download_from_modelscope(self, model_id: str, model_path: Path):
        """从ModelScope下载模型（仅下载运行必需的文件）"""
        try:
            # 动态导入，避免在不使用ModelScope时出错
            import importlib
            modelscope_module = importlib.import_module("modelscope")
            ms_snapshot_download = getattr(modelscope_module, "snapshot_download")
            
            logger.info(f"正在从ModelScope下载模型 {model_id} 到 {model_path}")
            
            # ModelScope的下载函数
            ms_snapshot_download(
                model_id=model_id,
                local_dir=str(model_path),
                ignore_patterns=IGNORE_PATTERNS
            )
        except ImportError:
            raise RuntimeError("modelscope 包未安装，请先安装: pip install modelscope")
        except Exception as e:
            raise RuntimeError(f"从ModelScope下载模型失败: {str(e)}")
    
    def add_custom_model(self, model_name: str, repo_id: str, source: str = "huggingface"):
        """添加自定义模型"""
        if "custom_models" not in self.model_configs:
            self.model_configs["custom_models"] = {}
        
        self.model_configs["custom_models"][model_name] = {
            "repo_id": repo_id,
            "source": source
        }
        self._save_model_configs()
        logger.info(f"已添加自定义模型: {model_name} -> {repo_id} (来源: {source})")
    
    def remove_custom_model(self, model_name: str):
        """移除自定义模型"""
        if "custom_models" in self.model_configs and model_name in self.model_configs["custom_models"]:
            del self.model_configs["custom_models"][model_name]
            self._save_model_configs()
            logger.info(f"已移除自定义模型: {model_name}")
        else:
            logger.warning(f"未找到自定义模型: {model_name}")
    
    def list_downloaded_models(self) -> List[str]:
        """列出已下载的模型"""
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir() and item.name != ".git":
                models.append(item.name)
        return models
    
    def list_custom_models(self) -> Dict[str, Dict[str, str]]:
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
        required_files = ["config.json"]
        
        # 检查是否存在模型文件（支持多种格式）
        model_files = list(model_path.glob("pytorch_model*.bin")) + list(model_path.glob("*.safetensors"))
        
        # 如果没有找到标准模型文件，检查是否存在model.safetensors.index.json
        # 这个文件包含了模型权重文件的索引信息
        if not model_files:
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                try:
                    import json
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                        if "weight_map" in index_data:
                            # 从索引文件中获取权重文件名
                            weight_files = set()
                            for weight_file in index_data["weight_map"].values():
                                weight_files.add(weight_file)
                            
                            # 检查这些权重文件是否存在
                            for weight_file in weight_files:
                                if (model_path / weight_file).exists():
                                    model_files.append(model_path / weight_file)
                except Exception:
                    pass
        
        for file in required_files:
            if not (model_path / file).exists():
                return False
        
        # 检查是否存在模型权重文件
        if not model_files:
            # 如果没有找到模型权重文件，但存在索引文件，也认为模型有效
            index_files = list(model_path.glob("*.index.json"))
            if not index_files:
                return False
        
        return True