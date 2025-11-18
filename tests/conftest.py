#!/usr/bin/env python3
"""
pytest配置文件
"""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入测试工具
from tests.utils import TestUtils


@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def samples_dir(project_root: Path) -> Path:
    """示例文件目录"""
    return project_root / "samples"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """临时目录"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="function")
def test_files(temp_dir: Path) -> dict:
    """测试文件字典"""
    return TestUtils.create_test_files_in_dir(temp_dir)


@pytest.fixture(scope="function")
def mock_torch_cpu():
    """模拟torch CPU模块"""
    with TestUtils.patch_torch("cpu"):
        yield TestUtils.create_mock_torch("cpu")


@pytest.fixture(scope="function")
def mock_torch_cuda():
    """模拟torch CUDA模块"""
    with TestUtils.patch_torch("cuda"):
        yield TestUtils.create_mock_torch("cuda")


@pytest.fixture(scope="function")
def mock_torch_mps():
    """模拟torch MPS模块"""
    with TestUtils.patch_torch("mps"):
        yield TestUtils.create_mock_torch("mps")


@pytest.fixture(scope="function")
def mock_transformers():
    """模拟transformers模块"""
    with TestUtils.patch_transformers():
        yield TestUtils.create_mock_transformers()


@pytest.fixture(scope="function")
def mock_vllm():
    """模拟vllm模块"""
    with TestUtils.patch_vllm():
        yield TestUtils.create_mock_vllm()


@pytest.fixture(scope="function")
def mock_pil():
    """模拟PIL模块"""
    with TestUtils.patch_pil():
        yield TestUtils.create_mock_pil()


@pytest.fixture(scope="function")
def mock_cv2():
    """模拟cv2模块"""
    with TestUtils.patch_cv2():
        yield TestUtils.create_mock_cv2()


def is_model_available() -> bool:
    """检查模型文件是否可用"""
    model_dir = Path(__file__).parent.parent / "models" / "deepseek-ocr"
    if not model_dir.exists():
        return False

    # 检查关键模型文件是否存在
    model_files = ["model-00001-of-000001.safetensors", "config.json", "tokenizer.json"]

    return all((model_dir / file).exists() for file in model_files)
