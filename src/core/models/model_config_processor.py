#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型配置处理器
用于动态修改模型配置，确保正确加载源代码中的模型实现
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from transformers import AutoConfig

# 导入应用程序配置
from src.core.config.app_config import get_app_config, get_model_auto_map

logger = logging.getLogger(__name__)


class ModelConfigProcessor:
    """
    模型配置处理器

    用于动态修改模型配置，确保正确加载源代码中的模型实现
    """

    def __init__(self, model_path: str, source_code_path: Optional[str] = None):
        """
        初始化模型配置处理器

        Args:
            model_path: 模型路径
            source_code_path: 源代码路径，默认为项目根目录下的src/core/models
        """
        self.model_path = model_path
        if source_code_path is None:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            source_code_path = os.path.join(project_root, "src", "core", "models")
        self.source_code_path = source_code_path

        # 检查是否是本地路径
        self.is_local_path = os.path.exists(model_path)

        # 获取应用程序配置
        self.app_config = get_app_config()

        logger.info(
            f"模型配置处理器初始化: model_path={model_path}, source_code_path={source_code_path}, is_local={self.is_local_path}"
        )

    def get_modified_config(self, **kwargs) -> Dict[str, Any]:
        """
        获取修改后的配置

        Args:
            **kwargs: 传递给AutoConfig.from_pretrained的参数

        Returns:
            修改后的配置字典
        """
        try:
            # 加载原始配置
            config = AutoConfig.from_pretrained(self.model_path, **kwargs)
            config_dict = config.to_dict()

            # 如果是本地路径，修改auto_map以指向源代码
            if self.is_local_path and "auto_map" in config_dict:
                original_auto_map = config_dict["auto_map"].copy()
                modified_auto_map = {}

                for key, value in original_auto_map.items():
                    if "." in value:
                        # 提取模块名和类名
                        module_name, class_name = value.rsplit(".", 1)

                        # 构建新的模块路径，指向源代码
                        source_module_path = os.path.join(self.source_code_path, f"{module_name}.py")
                        if os.path.exists(source_module_path):
                            # 使用相对路径
                            relative_path = os.path.relpath(self.source_code_path, self.model_path)
                            if relative_path == ".":
                                new_module_path = module_name
                            else:
                                new_module_path = os.path.join(relative_path, module_name).replace(os.sep, ".")

                            modified_auto_map[key] = f"{new_module_path}.{class_name}"
                            logger.info(f"修改auto_map[{key}]: {value} -> {modified_auto_map[key]}")
                        else:
                            # 如果源代码文件不存在，保持原样
                            modified_auto_map[key] = value
                            logger.warning(f"源代码文件不存在: {source_module_path}，保持原样: {value}")
                    else:
                        # 如果没有点号，保持原样
                        modified_auto_map[key] = value

                # 更新配置
                config_dict["auto_map"] = modified_auto_map

            return config_dict

        except Exception as e:
            logger.error(f"获取修改后的配置失败: {e}")
            raise

    def load_config_with_modification(self, trust_remote_code: bool = True, local_files_only: bool = False, **kwargs):
        """
        加载配置并应用修改

        Args:
            trust_remote_code: 是否信任远程代码
            local_files_only: 是否仅使用本地文件
            **kwargs: 其他参数

        Returns:
            修改后的配置对象
        """
        try:
            # 如果是本地路径，修改配置
            if self.is_local_path:
                # 读取配置文件
                config_path = os.path.join(self.model_path, "config.json")
                if not os.path.exists(config_path):
                    logger.warning(f"配置文件不存在: {config_path}")
                    return None

                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)

                # 根据app.yaml配置动态替换auto_map
                self._apply_app_config_auto_map(config_dict)

                # 创建临时配置文件
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
                    json.dump(config_dict, temp_file, indent=2)
                    temp_config_path = temp_file.name

                try:
                    # 从临时配置文件加载配置
                    config = AutoConfig.from_pretrained(
                        self.model_path,
                        config=temp_config_path,
                        trust_remote_code=trust_remote_code,
                        local_files_only=local_files_only,
                        **kwargs,
                    )

                    # 应用auto_map配置
                    self._apply_config_auto_map(config)

                    return config
                finally:
                    # 删除临时文件
                    os.unlink(temp_config_path)
            else:
                # 远程模型，直接加载
                return AutoConfig.from_pretrained(
                    self.model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only, **kwargs
                )
        except Exception as e:
            logger.error(f"加载配置并应用修改失败: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _apply_app_config_auto_map(self, config_dict: Dict[str, Any]) -> None:
        """
        根据app.yaml配置应用auto_map替换

        Args:
            config_dict: 配置字典
        """
        try:
            # 获取默认模型名称
            default_model = self.app_config.get_default_model()

            # 获取app.yaml中的auto_map配置
            app_auto_map = get_model_auto_map(default_model)

            # 如果app.yaml中有配置且不为空，则使用app.yaml的配置
            if app_auto_map:
                logger.info(f"使用app.yaml中的auto_map配置: {app_auto_map}")
                config_dict["auto_map"] = app_auto_map
            else:
                # 如果app.yaml中没有配置或为空，则使用默认配置
                logger.info("app.yaml中没有auto_map配置，使用默认配置")
                if "auto_map" not in config_dict:
                    config_dict["auto_map"] = {
                        "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig",
                        "AutoModelForCausalLM": "modeling_deepseekocr.DeepseekOCRForCausalLM",
                    }
        except Exception as e:
            logger.warning(f"应用app.yaml配置时出错: {e}")
            # 出错时使用默认配置
            if "auto_map" not in config_dict:
                config_dict["auto_map"] = {
                    "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig",
                    "AutoModelForCausalLM": "modeling_deepseekocr.DeepseekOCRForCausalLM",
                }

    def _apply_config_auto_map(self, config) -> None:
        """
        应用auto_map配置到配置对象

        Args:
            config: 配置对象
        """
        try:
            # 获取默认模型名称
            default_model = self.app_config.get_default_model()

            # 获取app.yaml中的auto_map配置
            app_auto_map = get_model_auto_map(default_model)

            # 如果app.yaml中有配置且不为空，则使用app.yaml的配置
            if app_auto_map:
                logger.info(f"应用auto_map配置到配置对象: {app_auto_map}")
                config.auto_map = app_auto_map
            else:
                # 如果app.yaml中没有配置或为空，则使用默认配置
                logger.info("应用默认auto_map配置到配置对象")
                config.auto_map = {
                    "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig",
                    "AutoModelForCausalLM": "modeling_deepseekocr.DeepseekOCRForCausalLM",
                }
        except Exception as e:
            logger.warning(f"应用auto_map配置到配置对象时出错: {e}")
            # 出错时使用默认配置
            config.auto_map = {
                "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig",
                "AutoModelForCausalLM": "modeling_deepseekocr.DeepseekOCRForCausalLM",
            }


def get_model_config_processor(model_path: str, source_code_path: Optional[str] = None) -> ModelConfigProcessor:
    """
    获取模型配置处理器

    Args:
        model_path: 模型路径
        source_code_path: 源代码路径

    Returns:
        模型配置处理器实例
    """
    return ModelConfigProcessor(model_path, source_code_path)
