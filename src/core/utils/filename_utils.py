#!/usr/bin/env python3
"""
文件名生成工具
提供统一的输出文件名生成函数
"""

from pathlib import Path


def generate_output_filename(
    input_path: str | Path, default_filename: str = "result.mmd", extension: str = ".mmd"
) -> str:
    """
    根据输入路径生成输出文件名

    Args:
        input_path: 输入路径（文件或目录）
        default_filename: 默认文件名，当无法从输入路径生成时使用
        extension: 输出文件扩展名

    Returns:
        生成的输出文件名
    """
    try:
        input_path = Path(input_path)

        # 如果输入是文件，使用文件名（不含扩展名）作为基础
        if input_path.is_file():
            return f"{input_path.stem}{extension}"

        # 如果输入是目录，使用目录名加上"_results"作为基础
        elif input_path.is_dir():
            return f"{input_path.name}_results{extension}"

        # 如果路径不存在，尝试从路径字符串中提取最后部分
        elif input_path.name:
            return f"{input_path.stem}{extension}"

        # 其他情况使用默认文件名
        else:
            return default_filename

    except Exception:
        # 出现任何异常时返回默认文件名
        return default_filename


def generate_output_filename_from_paths(
    input_paths: list[str | Path], default_filename: str = "result.mmd", extension: str = ".mmd"
) -> str:
    """
    从输入路径列表中生成输出文件名

    Args:
        input_paths: 输入路径列表
        default_filename: 默认文件名，当无法从输入路径生成时使用
        extension: 输出文件扩展名

    Returns:
        生成的输出文件名
    """
    # 尝试从第一个有效路径生成文件名
    for path in input_paths:
        if path:  # 跳过None值或空字符串
            filename = generate_output_filename(path, default_filename, extension)
            if filename != default_filename:
                return filename

    # 如果所有路径都无法生成有效文件名，使用默认文件名
    return default_filename


def generate_unique_output_filename(output_dir: str | Path, base_filename: str, extension: str = ".mmd") -> str:
    """
    生成唯一的输出文件名，避免覆盖已存在的文件

    Args:
        output_dir: 输出目录
        base_filename: 基础文件名
        extension: 输出文件扩展名

    Returns:
        唯一的输出文件名
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确保基础文件名包含扩展名
    if not base_filename.endswith(extension):
        base_filename = f"{base_filename}{extension}"

    # 检查文件是否已存在
    output_path = output_dir / base_filename
    if not output_path.exists():
        return base_filename

    # 如果文件已存在，添加数字后缀
    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix

    counter = 1
    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_output_path = output_dir / new_filename
        if not new_output_path.exists():
            return new_filename
        counter += 1
