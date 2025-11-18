#!/usr/bin/env python3
"""
路径工具函数

提供通用的路径处理和验证功能
"""

import os
from pathlib import Path
from typing import Optional


def validate_path(path: str | Path, must_exist: bool = False) -> Path:
    """
    验证并规范化路径

    Args:
        path: 路径字符串或Path对象
        must_exist: 是否要求路径必须存在

    Returns:
        Path: 规范化的Path对象

    Raises:
        ValueError: 路径无效或不存在
    """
    if isinstance(path, str):
        path = Path(path).expanduser()

    if must_exist and not path.exists():
        raise ValueError(f"路径不存在: {path}")

    return path.resolve()


def ensure_dir(path: str | Path, mode: int = 0o755) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径
        mode: 目录权限模式

    Returns:
        Path: 目录路径对象
    """
    path = validate_path(path)
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path


def get_safe_filename(filename: str, max_length: int = 255) -> str:
    """
    获取安全的文件名

    Args:
        filename: 原始文件名
        max_length: 最大长度

    Returns:
        str: 安全的文件名
    """
    # 移除或替换不安全的字符
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename

    for char in unsafe_chars:
        safe_name = safe_name.replace(char, "_")

    # 移除首尾的空格和点
    safe_name = safe_name.strip(" .")

    # 确保不超过最大长度
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[: max_length - len(ext)] + ext

    return safe_name or "unnamed"


def get_file_size_human(size_bytes: int) -> str:
    """
    将字节大小转换为人类可读格式

    Args:
        size_bytes: 字节数

    Returns:
        str: 人类可读的大小格式
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0

    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1

    return f"{size_bytes:.1f} {units[unit_index]}"


def is_image_file(file_path: str | Path) -> bool:
    """
    判断文件是否为支持的图像格式

    Args:
        file_path: 文件路径

    Returns:
        bool: 是否为图像文件
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif", ".ico", ".svg"}

    path = validate_path(file_path)
    return path.suffix.lower() in image_extensions


def is_document_file(file_path: str | Path) -> bool:
    """
    判断文件是否为支持的文档格式

    Args:
        file_path: 文件路径

    Returns:
        bool: 是否为文档文件
    """
    document_extensions = {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".xls", ".xlsx", ".ppt", ".pptx"}

    path = validate_path(file_path)
    return path.suffix.lower() in document_extensions


def get_relative_path(path: str | Path, base: str | Path) -> Path:
    """
    获取相对于基目录的相对路径

    Args:
        path: 目标路径
        base: 基目录路径

    Returns:
        Path: 相对路径
    """
    path = validate_path(path)
    base = validate_path(base)

    try:
        return path.relative_to(base)
    except ValueError:
        # 如果路径不在基目录下，返回原路径
        return path


def find_files_by_extension(directory: str | Path, extensions: set[str], recursive: bool = True) -> list[Path]:
    """
    在目录中查找指定扩展名的文件

    Args:
        directory: 搜索目录
        extensions: 文件扩展名集合（带点号，如 {'.jpg', '.png'}）
        recursive: 是否递归搜索子目录

    Returns:
        list[Path]: 找到的文件路径列表
    """
    directory = validate_path(directory, must_exist=True)
    extensions = {ext.lower() for ext in extensions}
    found_files = []

    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            found_files.append(file_path)

    return sorted(found_files)


def create_unique_filename(directory: str | Path, base_name: str, extension: str) -> Path:
    """
    在指定目录中创建唯一文件名

    Args:
        directory: 目标目录
        base_name: 基础文件名
        extension: 文件扩展名（带点号）

    Returns:
        Path: 唯一的文件路径
    """
    directory = ensure_dir(directory)
    extension = extension.lower() if not extension.startswith(".") else extension

    safe_name = get_safe_filename(base_name)
    counter = 1

    filename = f"{safe_name}{extension}"
    file_path = directory / filename

    while file_path.exists():
        filename = f"{safe_name}_{counter}{extension}"
        file_path = directory / filename
        counter += 1

    return file_path


def copy_file_with_backup(
    source: str | Path, destination: str | Path, backup_suffix: str = ".backup"
) -> Path:
    """
    复制文件，如果目标存在则创建备份

    Args:
        source: 源文件路径
        destination: 目标文件路径
        backup_suffix: 备份文件后缀

    Returns:
        Path: 目标文件路径
    """
    import shutil

    source = validate_path(source, must_exist=True)
    destination = validate_path(destination)
    ensure_dir(destination.parent)

    # 如果目标文件存在，创建备份
    if destination.exists():
        backup_path = destination.with_suffix(f"{destination.suffix}{backup_suffix}")
        shutil.move(str(destination), str(backup_path))

    shutil.copy2(str(source), str(destination))
    return destination


def get_project_root(start_path: Optional[str | Path] = None) -> Path:
    """
    获取项目根目录（向上查找 .git 或 pyproject.toml 文件）

    Args:
        start_path: 搜索起始路径，默认为当前文件

    Returns:
        Path: 项目根目录
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    start_path = validate_path(start_path)

    current = start_path
    while current.parent != current:  # 到达文件系统根目录
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # 如果找不到项目根目录，返回起始路径的父目录
    return start_path.parent


def format_file_size(size: int) -> str:
    """
    格式化文件大小显示

    Args:
        size: 文件大小（字节）

    Returns:
        str: 格式化的大小字符串
    """
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"
