"""
模型路径解析工具

统一处理模型路径的检测、验证和参数生成。
消除项目中 7 处重复的远程仓库检测逻辑。
"""

import os
from pathlib import Path


class ModelPathResolver:
    """模型路径解析器 - 统一的远程仓库检测和参数生成"""

    # 远程仓库前缀
    REMOTE_PREFIXES = (
        "http://",
        "https://",
        "deepseek-ai/",
        "huggingface.co/",
        "hf.co/",
    )

    @staticmethod
    def is_remote_repo(model_path: str) -> bool:
        """
        检查模型路径是否为远程仓库

        Args:
            model_path: 模型路径字符串

        Returns:
            bool: 如果是远程仓库返回 True，否则返回 False

        Examples:
            >>> ModelPathResolver.is_remote_repo("deepseek-ai/deepseek-vl")
            True
            >>> ModelPathResolver.is_remote_repo("/path/to/local/model")
            False
            >>> ModelPathResolver.is_remote_repo("https://huggingface.co/model")
            True
        """
        # 简化的远程仓库检查逻辑
        is_invalid = not model_path or not isinstance(model_path, str)
        if is_invalid:
            return True

        # 远程标识符检查
        has_remote_prefix = model_path.startswith(ModelPathResolver.REMOTE_PREFIXES)
        has_no_path_sep = "/" not in model_path and "\\" not in model_path
        if has_remote_prefix or has_no_path_sep:
            return True

        # 本地路径存在性检查
        expanded_path = os.path.expanduser(model_path)
        return not os.path.exists(expanded_path)

    @staticmethod
    def get_loading_params(model_path: str, trust_remote_code: bool = True) -> dict[str, bool]:
        """
        根据模型路径生成加载参数

        Args:
            model_path: 模型路径
            trust_remote_code: 是否信任远程代码

        Returns:
            Dict[str, bool]: 包含 local_files_only 和 trust_remote_code 的字典

        Examples:
            >>> ModelPathResolver.get_loading_params("deepseek-ai/deepseek-vl")
            {'local_files_only': False, 'trust_remote_code': True}
            >>> ModelPathResolver.get_loading_params("/local/model", trust_remote_code=False)
            {'local_files_only': True, 'trust_remote_code': False}
        """
        is_remote = ModelPathResolver.is_remote_repo(model_path)

        return {
            "local_files_only": not is_remote,
            "trust_remote_code": is_remote and trust_remote_code,
        }

    @staticmethod
    def resolve_path(model_path: str) -> str:
        """
        解析模型路径，展开用户路径和相对路径

        Args:
            model_path: 原始模型路径

        Returns:
            str: 解析后的绝对路径（对于本地路径）或原始路径（对于远程路径）

        Examples:
            >>> ModelPathResolver.resolve_path("~/models/deepseek")
            '/home/user/models/deepseek'
            >>> ModelPathResolver.resolve_path("deepseek-ai/deepseek-vl")
            'deepseek-ai/deepseek-vl'
        """
        if not model_path:
            return model_path

        # 如果是远程路径，直接返回
        if ModelPathResolver.is_remote_repo(model_path):
            return model_path

        # 展开用户路径
        expanded = os.path.expanduser(model_path)

        # 转换为绝对路径
        absolute = os.path.abspath(expanded)

        return absolute

    @staticmethod
    def validate_path(model_path: str, must_exist: bool = False) -> bool:
        """
        验证模型路径的有效性

        Args:
            model_path: 模型路径
            must_exist: 本地路径是否必须存在

        Returns:
            bool: 路径是否有效

        Examples:
            >>> ModelPathResolver.validate_path("deepseek-ai/deepseek-vl")
            True
            >>> ModelPathResolver.validate_path("/nonexistent/path", must_exist=True)
            False
        """
        if not model_path:
            return False

        # 远程路径始终有效（由 HuggingFace 验证）
        if ModelPathResolver.is_remote_repo(model_path):
            return True

        # 本地路径验证
        resolved = ModelPathResolver.resolve_path(model_path)

        if must_exist:
            return os.path.exists(resolved)

        return True

    @staticmethod
    def get_model_type(model_path: str) -> str:
        """
        推断模型类型（本地或远程）

        Args:
            model_path: 模型路径

        Returns:
            str: "remote" 或 "local"

        Examples:
            >>> ModelPathResolver.get_model_type("deepseek-ai/deepseek-vl")
            'remote'
            >>> ModelPathResolver.get_model_type("/path/to/model")
            'local'
        """
        return "remote" if ModelPathResolver.is_remote_repo(model_path) else "local"

    @staticmethod
    def is_huggingface_url(model_path: str) -> bool:
        """
        检查是否为 HuggingFace URL

        Args:
            model_path: 模型路径

        Returns:
            bool: 是否为 HuggingFace URL
        """
        if not model_path:
            return False

        hf_patterns = ("huggingface.co/", "hf.co/")
        return any(pattern in model_path for pattern in hf_patterns)

    @staticmethod
    def get_cache_dir(model_path: str, cache_root: str | None = None) -> str | None:
        """
        获取模型缓存目录

        Args:
            model_path: 模型路径
            cache_root: 缓存根目录

        Returns:
            str | None: 缓存目录路径，远程模型返回 None 使用默认缓存
        """
        if ModelPathResolver.is_remote_repo(model_path):
            # 远程模型使用 HuggingFace 默认缓存
            return None

        if cache_root:
            # 使用指定的缓存根目录
            model_name = Path(model_path).name
            return os.path.join(cache_root, model_name)

        return None
