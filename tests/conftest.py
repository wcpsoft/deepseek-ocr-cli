#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest配置文件
"""

import sys
import os
from pathlib import Path
import pytest
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def samples_dir(project_root):
    """示例文件目录"""
    return project_root / "samples"

@pytest.fixture(scope="function")
def temp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)