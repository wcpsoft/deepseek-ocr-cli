"""ModelPathResolver 单元测试"""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.core.utils.model_path_utils import ModelPathResolver, get_loading_params, is_remote_repo


class TestModelPathResolver(unittest.TestCase):
    """测试 ModelPathResolver 工具类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_model_path = os.path.join(self.temp_dir, "model")
        os.makedirs(self.temp_model_path, exist_ok=True)

    def tearDown(self):
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_remote_repo_http(self):
        """测试 HTTP/HTTPS 远程仓库检测"""
        self.assertTrue(ModelPathResolver.is_remote_repo("http://example.com/model"))
        self.assertTrue(ModelPathResolver.is_remote_repo("https://example.com/model"))
        self.assertTrue(ModelPathResolver.is_remote_repo("http://huggingface.co/model"))

    def test_is_remote_repo_deepseek(self):
        """测试 DeepSeek 远程仓库检测"""
        self.assertTrue(ModelPathResolver.is_remote_repo("deepseek-ai/deepseek-ocr"))
        self.assertTrue(ModelPathResolver.is_remote_repo("deepseek-ai/deepseek-vl"))

    def test_is_remote_repo_huggingface(self):
        """测试 HuggingFace 远程仓库检测"""
        self.assertTrue(ModelPathResolver.is_remote_repo("huggingface.co/model/name"))

    def test_is_remote_repo_single_name(self):
        """测试单个名称（可能是远程仓库名）"""
        self.assertTrue(ModelPathResolver.is_remote_repo("bert-base-uncased"))
        self.assertTrue(ModelPathResolver.is_remote_repo("gpt2"))
        self.assertTrue(ModelPathResolver.is_remote_repo("model-name"))

    def test_is_remote_repo_local_path(self):
        """测试本地路径检测"""
        # 存在的路径
        self.assertTrue(ModelPathResolver.is_remote_repo(self.temp_model_path))

        # 不存在但格式像本地路径
        self.assertTrue(ModelPathResolver.is_remote_repo("/path/to/local/model"))
        self.assertTrue(ModelPathResolver.is_remote_repo("./local/model"))
        self.assertTrue(ModelPathResolver.is_remote_repo("../local/model"))
        self.assertTrue(ModelPathResolver.is_remote_repo("C:\\model\\path"))

    def test_get_loading_params_remote(self):
        """测试远程仓库的加载参数"""
        params = ModelPathResolver.get_loading_params("https://example.com/model")
        self.assertFalse(params["local_files_only"])
        self.assertTrue(params["trust_remote_code"])

        params = ModelPathResolver.get_loading_params("deepseek-ai/model", trust_remote_code=False)
        self.assertFalse(params["local_files_only"])
        self.assertFalse(params["trust_remote_code"])

    def test_get_loading_params_local(self):
        """测试本地路径的加载参数"""
        # 测试存在的本地路径
        with patch("os.path.exists", return_value=True):
            params = ModelPathResolver.get_loading_params(self.temp_model_path)
            self.assertTrue(params["local_files_only"])
            self.assertTrue(params["trust_remote_code"])

        # 测试不存在的本地路径格式
        with patch("os.path.exists", return_value=False):
            params = ModelPathResolver.get_loading_params("/some/local/path")
            self.assertTrue(params["local_files_only"])
            self.assertTrue(params["trust_remote_code"])

    def test_get_model_type(self):
        """测试模型类型检测"""
        self.assertEqual(ModelPathResolver.get_model_type("https://example.com/model"), "remote")
        self.assertEqual(ModelPathResolver.get_model_type("deepseek-ai/model"), "remote")
        self.assertEqual(ModelPathResolver.get_model_type("bert-base-uncased"), "remote")

        with patch("os.path.exists", return_value=True):
            self.assertEqual(ModelPathResolver.get_model_type("/local/path"), "local")

    def test_validate_path(self):
        """测试路径验证"""
        # 测试存在的路径
        with patch("os.path.exists", return_value=True):
            self.assertTrue(ModelPathResolver.validate_path(self.temp_model_path))

        # 测试不存在的路径
        with patch("os.path.exists", return_value=False):
            self.assertFalse(ModelPathResolver.validate_path("/nonexistent/path"))

        # 测试远程 URL（总是返回 True）
        self.assertTrue(ModelPathResolver.validate_path("https://example.com/model"))
        self.assertTrue(ModelPathResolver.validate_path("deepseek-ai/model"))

    def test_resolve_path(self):
        """测试路径解析"""
        # 测试远程 URL（不修改）
        url = "https://example.com/model"
        self.assertEqual(ModelPathResolver.resolve_path(url), url)

        # 测试远程仓库（不修改）
        repo = "deepseek-ai/model"
        self.assertEqual(ModelPathResolver.resolve_path(repo), repo)

        # 测试本地路径（扩展用户路径）
        with patch("os.path.expanduser") as mock_expand:
            mock_expand.return_value = "/expanded/path"
            path = "~/model"
            self.assertEqual(ModelPathResolver.resolve_path(path), "/expanded/path")

    def test_deprecated_functions(self):
        """测试弃用的便捷函数"""
        # 测试 is_remote_repo 函数
        self.assertTrue(is_remote_repo("https://example.com/model"))

        # 测试 get_loading_params 函数
        params = get_loading_params("https://example.com/model")
        self.assertFalse(params["local_files_only"])
        self.assertTrue(params["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
