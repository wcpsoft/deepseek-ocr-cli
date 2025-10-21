#!/usr/bin/env python3
"""
GPU检测脚本，用于自动识别系统中的GPU类型
支持NVIDIA、AMD、Apple Silicon等硬件平台
"""

import subprocess
import sys
import platform
import os


def detect_nvidia_gpu():
    """检测NVIDIA GPU"""
    try:
        # 尝试运行nvidia-smi命令
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            # 解析nvidia-smi输出获取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA-SMI' in line:
                    # 提取CUDA版本
                    cuda_version = None
                    if 'CUDA' in line:
                        parts = line.split('CUDA:')
                        if len(parts) > 1:
                            cuda_version = parts[1].strip().split()[0]
                    return True, cuda_version
            return True, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False, None


def detect_amd_gpu():
    """检测AMD GPU"""
    try:
        # 在Linux上尝试使用rocm-smi
        if platform.system().lower() == 'linux':
            result = subprocess.run(['rocm-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 检查是否存在ROCm相关环境变量
    if 'ROCM_PATH' in os.environ or 'ROCM_HOME' in os.environ:
        return True
    
    return False


def detect_apple_silicon():
    """检测Apple Silicon"""
    if platform.system().lower() == 'darwin':
        # 检查是否为ARM架构（Apple Silicon）
        if platform.machine().lower() in ['arm64', 'aarch64']:
            return True
        # 检查系统信息
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if 'Apple' in result.stdout and 'Chip' in result.stdout:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return False


def detect_dcu():
    """检测DCU（Direct Compute Unit）"""
    # 检查环境变量
    dcu_env_vars = ['DCU_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES']
    for var in dcu_env_vars:
        if os.environ.get(var):
            return True
    
    # 检查是否存在DCU相关库或工具
    try:
        # 尝试查找DCU相关工具
        result = subprocess.run(['which', 'dcu-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return False


def get_gpu_type():
    """获取GPU类型"""
    # 检测各种GPU类型
    if detect_nvidia_gpu()[0]:
        return "nvidia"
    elif detect_amd_gpu():
        return "amd"
    elif detect_apple_silicon():
        return "mps"
    elif detect_dcu():
        return "dcu"
    else:
        return "cpu"


def get_pytorch_install_cmd(gpu_type):
    """根据GPU类型获取PyTorch安装命令"""
    if gpu_type == "nvidia":
        return "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118"
    elif gpu_type == "amd":
        return "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1"
    elif gpu_type == "mps":
        return "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu"
    elif gpu_type == "dcu":
        return "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu"
    else:
        return "uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu"


def get_extra_require_suffix(gpu_type):
    """根据GPU类型获取额外依赖后缀"""
    if gpu_type == "nvidia":
        return "[nvidia]"
    elif gpu_type == "amd":
        return "[amd]"
    elif gpu_type == "mps":
        return "[mps]"
    elif gpu_type == "dcu":
        return "[dcu]"
    else:
        return ""


def main():
    """主函数，检测GPU类型并输出相应信息"""
    print("正在检测系统GPU环境...")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    
    gpu_type = get_gpu_type()
    cuda_version = None
    
    # 检测各种GPU类型
    if gpu_type == "nvidia":
        _, cuda_version = detect_nvidia_gpu()
        print("✓ 检测到NVIDIA GPU")
        if cuda_version:
            print(f"  CUDA版本: {cuda_version}")
    elif gpu_type == "amd":
        print("✓ 检测到AMD GPU (ROCm)")
    elif gpu_type == "mps":
        print("✓ 检测到Apple Silicon (MPS)")
    elif gpu_type == "dcu":
        print("✓ 检测到DCU (Direct Compute Unit)")
    else:
        print("⚠ 未检测到专用GPU，将使用CPU运行")
    
    # 输出推荐的PyTorch安装命令
    print("\n推荐的PyTorch安装命令:")
    print(f"  {get_pytorch_install_cmd(gpu_type)}")
    
    # 输出GPU类型供其他脚本使用
    print(f"\nGPU_TYPE={gpu_type}")
    
    # 输出额外依赖后缀
    extra_suffix = get_extra_require_suffix(gpu_type)
    if extra_suffix:
        print(f"EXTRA_SUFFIX={extra_suffix}")
    
    return gpu_type


if __name__ == "__main__":
    main()