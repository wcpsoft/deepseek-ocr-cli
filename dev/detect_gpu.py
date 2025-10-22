#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU检测脚本
用于检测系统中的GPU类型并推荐合适的推理模式
"""

import sys
import os
import platform
import subprocess

def detect_gpu():
    """检测GPU类型"""
    system = platform.system()
    
    # 检查NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return "nvidia"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 检查AMD GPU (ROCm)
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return "amd"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 检查Apple Silicon (MPS)
    if system == "Darwin" and platform.processor() == "arm":
        # 检查是否支持MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    
    # 检查DCU
    # 这里只是一个示例，实际检测DCU需要特定的工具
    # if os.path.exists("/dev/dcu"): 
    #     return "dcu"
    
    # 默认返回CPU
    return "cpu"

def get_extra_suffix(gpu_type):
    """获取额外依赖后缀"""
    suffix_map = {
        "nvidia": "[nvidia]",
        "amd": "[amd]",
        "mps": "[mps]",
        "dcu": "[dcu]",
        "cpu": "[cpu]"
    }
    return suffix_map.get(gpu_type, "[cpu]")

def get_recommended_mode(gpu_type):
    """根据GPU类型推荐推理模式"""
    # MPS环境下推荐使用Transformers模式
    if gpu_type == "mps":
        return "transformers"
    # 其他环境推荐使用vLLM模式（如果可用）
    else:
        return "vllm"

# 为测试添加的兼容性函数
def get_gpu_type():
    """获取GPU类型（为测试兼容性保留）"""
    return detect_gpu()

def get_extra_require_suffix(gpu_type):
    """获取额外依赖后缀（为测试兼容性保留）"""
    # 根据测试期望，CPU类型和未知类型应该返回空字符串
    if gpu_type == "cpu" or gpu_type == "unknown":
        return ""
    else:
        return get_extra_suffix(gpu_type)

def get_pytorch_install_cmd(gpu_type):
    """获取PyTorch安装命令（为测试兼容性保留）"""
    if gpu_type == "cpu":
        return "uv pip install -e ."
    else:
        return f"uv pip install -e .[{gpu_type}]"

def main():
    """主函数"""
    gpu_type = detect_gpu()
    extra_suffix = get_extra_suffix(gpu_type)
    recommended_mode = get_recommended_mode(gpu_type)
    
    # 输出结果供shell脚本使用
    print(f"GPU_TYPE={gpu_type}")
    print(f"EXTRA_SUFFIX={extra_suffix}")
    print(f"推荐的推理模式: {recommended_mode}")

if __name__ == "__main__":
    main()