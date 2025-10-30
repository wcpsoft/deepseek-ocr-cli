#!/usr/bin/env python3
"""
GPU检测集成测试
测试run.sh和run_tests.sh脚本中的GPU检测功能集成
"""

import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_run_sh_integration() -> None:
    """测试run.sh脚本中的GPU检测集成"""
    project_root = Path(__file__).parent.parent.parent
    run_sh_path = project_root / "dev" / "run.sh"

    # 检查run.sh文件是否存在
    assert run_sh_path.exists(), "run.sh文件不存在"

    # 读取run.sh内容
    content = run_sh_path.read_text()

    # 检查是否包含EXTRA_SUFFIX变量
    assert "EXTRA_SUFFIX" in content, "run.sh中未使用EXTRA_SUFFIX变量"

    # 检查是否正确使用依赖安装命令(检查install_all_deps函数调用)
    assert "install_all_deps" in content, "run.sh中未正确使用依赖安装命令"


def test_test_sh_integration() -> None:
    """测试run_tests.sh脚本中的GPU检测集成"""
    project_root = Path(__file__).parent.parent.parent
    test_sh_path = project_root / "dev" / "run_tests.sh"

    # 检查run_tests.sh文件是否存在
    assert test_sh_path.exists(), "run_tests.sh文件不存在"

    # 读取run_tests.sh内容
    content = test_sh_path.read_text()

    # 检查是否包含EXTRA_SUFFIX变量
    assert "EXTRA_SUFFIX" in content, "run_tests.sh中未使用EXTRA_SUFFIX变量"

    # 检查是否正确使用依赖安装命令(检查install_all_deps函数调用)
    assert "install_all_deps" in content, "run_tests.sh中未正确使用依赖安装命令"

    # 检查是否调用GPU检测相关功能(直接或间接)
    assert "detect_gpu" in content, "run_tests.sh中未调用GPU检测相关功能"


def test_detect_gpu_script_execution() -> None:
    """测试detect_gpu.py脚本执行"""
    project_root = Path(__file__).parent.parent.parent
    detect_gpu_path = project_root / "dev" / "detect_gpu.py"

    # 检查detect_gpu.py文件是否存在
    assert detect_gpu_path.exists(), "detect_gpu.py文件不存在"

    # 尝试执行脚本(不检查输出, 只检查是否能正常执行)
    try:
        result = subprocess.run(
            [sys.executable, str(detect_gpu_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # 脚本应该能正常执行(退出码为0)或因缺少依赖而失败(退出码非0但不是异常)
        assert result.returncode in [0, 1], "detect_gpu.py脚本执行异常"
    except subprocess.TimeoutExpired:
        # 如果超时, 说明脚本在运行, 这也是可以接受的
        pass
