#!/usr/bin/env python3
"""
应用程序配置加载器
用于加载和管理app.yaml配置文件
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class AppConfig:
    """应用程序配置管理器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载应用程序配置
        
        Returns:
            配置字典
        """
        try:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            
            # 查找配置文件
            config_path = os.path.join(project_root, "app.yaml")
            
            # 如果配置文件不存在，返回空配置
            if not os.path.exists(config_path):
                logger.warning(f"配置文件不存在: {config_path}")
                return {}
            
            # 加载配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"成功加载配置文件: {config_path}")
            return config or {}
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定模型的配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置字典
        """
        if not self._config:
            return None
            
        model_config = self._config.get("model_config", {})
        return model_config.get(model_name, {})
    
    def get_auto_map_config(self, model_name: str) -> Optional[Dict[str, str]]:
        """
        获取指定模型的auto_map配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            auto_map配置字典
        """
        model_config = self.get_model_config(model_name)
        if not model_config:
            return None
            
        return model_config.get("auto_map", {})
    
    def get_default_model(self) -> str:
        """
        获取默认模型名称
        
        Returns:
            默认模型名称
        """
        if not self._config:
            return "deepseek_vl_v2"
            
        return self._config.get("default_model", "deepseek_vl_v2")


# 全局配置实例
_app_config = AppConfig()


def get_app_config() -> AppConfig:
    """
    获取应用程序配置实例
    
    Returns:
        AppConfig: 应用程序配置实例
    """
    return _app_config


def get_model_auto_map(model_name: str) -> Optional[Dict[str, str]]:
    """
    获取指定模型的auto_map配置
    
    Args:
        model_name: 模型名称
        
    Returns:
        auto_map配置字典
    """
    return _app_config.get_auto_map_config(model_name)


def get_default_model() -> str:
    """
    获取默认模型名称
    
    Returns:
        默认模型名称
    """
    return _app_config.get_default_model()