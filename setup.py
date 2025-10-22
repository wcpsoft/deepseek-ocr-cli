#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR CLI setup script
"""

from setuptools import setup
import toml
import os


def read_requirements(filename):
    """Read requirements from file"""
    filepath = os.path.join(os.path.dirname(__file__), "requirements", filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []


# 读取pyproject.toml配置
with open("pyproject.toml", "r", encoding="utf-8") as fh:
    pyproject_data = toml.load(fh)

# 提取项目配置
project_config = pyproject_data["project"]

# 使用pyproject.toml中定义的依赖
install_requires = project_config.get("dependencies", [])

# 构建入口点
entry_points = {}
if "scripts" in project_config:
    entry_points["console_scripts"] = [
        f"{name}={value}" for name, value in project_config["scripts"].items()
    ]

setup(
    name=project_config["name"],
    version=project_config["version"],
    description=project_config["description"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author=project_config["authors"][0]["name"] if project_config.get("authors") else "",
    author_email=project_config["authors"][0]["email"] if project_config.get("authors") else "",
    url=project_config["urls"]["Homepage"] if project_config.get("urls") else "",
    classifiers=project_config.get("classifiers", []),
    python_requires=project_config["requires-python"],
    install_requires=install_requires,
    entry_points=entry_points,
    packages=["cli", "src", "dev"],
    package_data={
        "cli": ["*.py"],
        "src": ["*.py"],
        "dev": ["*.py"],
    },
    project_urls=project_config.get("urls", {}),
)