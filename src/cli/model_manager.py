#!/usr/bin/env python3
"""
模型管理器
负责下载和管理OCR模型
"""

import json
import shutil
from pathlib import Path
from typing import Any

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()

# 定义模型下载的文件模式，确保只下载运行必需的文件
MODEL_DOWNLOAD_PATTERNS = [
    "*.json",  # 包括config.json, tokenizer_config.json等
    "*.txt",  # 包括special_tokens_map.txt等
    "*.model",  # 包括tokenizer.model等
    "*.bin",  # 包括PyTorch模型权重文件
    "*.safetensors",  # 包括Safetensors模型权重文件
    "*.index.json",  # 包括模型索引文件
    "*.tiktoken",  # 包括tokenizer文件
    "tokenizer.json",  # 特别包含tokenizer.json
    "!configuration_*.py",  # 排除配置文件，使用项目内部实现
    "!modeling_*.py",  # 排除建模文件，使用项目内部实现
    "!*.md",  # 排除Markdown文档
    "!README*",  # 排除README文件
    "!LICENSE*",  # 排除许可证文件
    "!assets/*",  # 排除assets文件夹
    "!examples/*",  # 排除examples文件夹
    "!scripts/*",  # 排除scripts文件夹
    "!tests/*",  # 排除测试文件夹
    "!*.ipynb_checkpoints",  # 排除Jupyter Notebook检查点
    "!*.ipynb_checkpoints/*",  # 排除Jupyter Notebook检查点文件夹
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
                    "modelscope": "deepseek-ai/DeepSeek-OCR",  # 如果ModelScope上有相同的模型ID
                },
            }
        }

        # 模型配置文件路径
        self.config_file = self.model_dir / "model_config.json"
        self.model_configs = self._load_model_configs()

    def _load_model_configs(self) -> dict[str, Any]:
        """加载模型配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_model_configs(self):
        """保存模型配置"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
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

    def download_models(
        self,
        model_names: list[str] | None = None,
        force_redownload: bool = False,
        source: str = "huggingface",
    ):
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

    def _download_model(
        self,
        model_name: str,
        repo_id: str,
        source: str = "huggingface",
        force_redownload: bool = False,
    ):
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

            # 清理下载的Python文件
            self.clean_model_py_files(model_name)

            # 更新模型配置
            if "downloaded_models" not in self.model_configs:
                self.model_configs["downloaded_models"] = {}

            self.model_configs["downloaded_models"][model_name] = {
                "repo_id": repo_id,
                "source": source,
                "path": str(model_path),
                "downloaded_at": __import__("datetime").datetime.now().isoformat(),
            }

            self._save_model_configs()
        except Exception as e:
            logger.error(f"模型 {model_name} 下载失败: {e!s}")

    def _download_from_huggingface(self, repo_id: str, model_path: Path):
        """从Hugging Face下载模型（仅下载运行必需的文件）"""
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"正在从Hugging Face下载模型 {repo_id} 到 {model_path}")

            # 下载模型文件，添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(model_path),
                        allow_patterns=MODEL_DOWNLOAD_PATTERNS,
                        resume_download=True,  # 允许断点续传
                    )
                    break  # 成功下载则退出循环
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"下载尝试 {attempt + 1} 失败: {e!s}，正在重试...")
                        import time

                        time.sleep(2**attempt)  # 指数退避
                    else:
                        raise e  # 最后一次尝试失败则抛出异常
        except ImportError as import_error:
            raise RuntimeError("huggingface_hub 包未安装，请先安装: pip install huggingface_hub") from import_error
        except Exception as e:
            raise RuntimeError(f"从Hugging Face下载模型失败: {e!s}") from e

    def _download_from_modelscope(self, model_id: str, model_path: Path):
        """从ModelScope下载模型（仅下载运行必需的文件）"""
        try:
            # 动态导入，避免在不使用ModelScope时出错
            import importlib

            modelscope_module = importlib.import_module("modelscope")
            ms_snapshot_download = modelscope_module.snapshot_download

            logger.info(f"正在从ModelScope下载模型 {model_id} 到 {model_path}")

            # ModelScope的下载函数，添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ms_snapshot_download(
                        model_id=model_id,
                        local_dir=str(model_path),
                        allow_patterns=MODEL_DOWNLOAD_PATTERNS,
                    )
                    break  # 成功下载则退出循环
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"下载尝试 {attempt + 1} 失败: {e!s}，正在重试...")
                        import time

                        time.sleep(2**attempt)  # 指数退避
                    else:
                        raise e  # 最后一次尝试失败则抛出异常
        except ImportError as import_error:
            raise RuntimeError("modelscope 包未安装，请先安装: pip install modelscope") from import_error
        except Exception as e:
            raise RuntimeError(f"从ModelScope下载模型失败: {e!s}") from e

    def add_custom_model(self, model_name: str, repo_id: str, source: str = "huggingface"):
        """添加自定义模型"""
        if "custom_models" not in self.model_configs:
            self.model_configs["custom_models"] = {}

        self.model_configs["custom_models"][model_name] = {
            "repo_id": repo_id,
            "source": source,
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

    def list_downloaded_models(self) -> list[str]:
        """列出已下载的模型"""
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir() and item.name != ".git":
                models.append(item.name)
        return models

    def list_custom_models(self) -> dict[str, dict[str, str]]:
        """列出自定义模型"""
        return self.model_configs.get("custom_models", {})

    def get_model_path(self, model_name: str) -> str | None:
        """获取模型路径"""
        model_path = self.model_dir / model_name
        if model_path.exists():
            return str(model_path)
        return None

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """获取模型详细信息"""
        if "downloaded_models" in self.model_configs:
            return self.model_configs["downloaded_models"].get(model_name)
        return None

    def verify_model(self, model_name: str) -> bool:
        """验证模型完整性"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            return False

        model_path = Path(model_path)

        # 检查基本文件
        if not self._check_required_files(model_path):
            return False

        # 检查模型权重文件
        if not self._check_model_weights(model_path):
            return False

        # 验证模型目录不包含Python文件
        if not self._check_no_py_files(model_path):
            return False

        return True

    def _check_required_files(self, model_path: Path) -> bool:
        """检查必需的配置文件"""
        required_files = ["config.json"]
        for file in required_files:
            if not (model_path / file).exists():
                return False
        return True

    def _check_model_weights(self, model_path: Path) -> bool:
        """检查模型权重文件"""
        # 检查是否存在模型文件（支持多种格式）
        model_files = list(model_path.glob("pytorch_model*.bin")) + list(model_path.glob("*.safetensors"))

        # 如果没有找到标准模型文件，检查是否存在model.safetensors.index.json
        if not model_files:
            if self._check_index_file(model_path, model_files):
                return True

        # 检查是否存在模型权重文件
        if not model_files:
            # 如果没有找到模型权重文件，但存在索引文件，也认为模型有效
            index_files = list(model_path.glob("*.index.json"))
            if not index_files:
                return False

        return True

    def _check_index_file(self, model_path: Path, model_files: list) -> bool:
        """检查索引文件"""
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            try:
                import json

                with open(index_file) as f:
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
                        return True
            except Exception:
                pass
        return False

    def _check_no_py_files(self, model_path: Path) -> bool:
        """检查模型目录不包含Python文件"""
        py_files = list(model_path.glob("*.py"))
        if py_files:
            logger.warning(f"模型目录 {model_path.name} 包含Python文件: {[f.name for f in py_files]}")
            return False
        return True

    def clean_model_py_files(self, model_name: str) -> bool:
        """清理模型目录中的Python文件"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            return False

        model_path = Path(model_path)
        py_files = list(model_path.glob("*.py"))

        if py_files:
            logger.info(f"正在清理模型 {model_name} 中的Python文件...")
            for py_file in py_files:
                try:
                    py_file.unlink()
                    logger.info(f"已删除: {py_file.name}")
                except Exception as e:
                    logger.error(f"删除 {py_file.name} 失败: {e!s}")
                    return False
            logger.info(f"已清理 {len(py_files)} 个Python文件")

        return True
