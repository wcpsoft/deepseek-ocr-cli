#!/usr/bin/env python3
"""
DeepSeek OCR CLI 代码质量检查与修复工具
提供统一的代码质量检查和修复功能
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent

    # 构建脚本路径
    script_path = script_dir / "auto_check_code_style.sh"

    # 确保脚本存在且可执行
    if not script_path.exists():
        print(f"错误: 找不到脚本文件 {script_path}")
        sys.exit(1)

    # 添加执行权限（如果需要）
    if not os.access(script_path, os.X_OK):
        try:
            script_path.chmod(0o755)
        except Exception as e:
            print(f"警告: 无法添加执行权限: {e}")

    # 执行脚本并传递所有参数
    try:
        result = subprocess.run([str(script_path)] + sys.argv[1:], check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except Exception as e:
        print(f"错误: 执行脚本时发生异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
