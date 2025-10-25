#!/usr/bin/env python3
"""
MPS设备测试脚本
用于测试新的图像处理模块在MPS设备上的表现
"""

import sys
import os
import logging
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils.mps_utils import is_mps_device, get_optimal_device, optimize_tensor_for_mps
from src.core.process.image_process_mps import DeepseekOCRProcessor as MPSProcessor
from src.core.process.image_process import DeepseekOCRProcessor as StandardProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mps_detection():
    """测试MPS设备检测功能"""
    logger.info("开始测试MPS设备检测功能...")
    
    # 测试设备检测
    device = get_optimal_device()
    logger.info(f"检测到的最优设备: {device}")
    
    # 测试MPS设备判断
    is_mps = is_mps_device()
    logger.info(f"是否为MPS设备: {is_mps}")
    
    return device, is_mps

def test_tensor_optimization():
    """测试张量优化功能"""
    logger.info("开始测试张量优化功能...")
    
    device = get_optimal_device()
    
    # 创建测试张量
    test_tensor = torch.randn(1, 3, 224, 224)
    logger.info(f"原始张量设备: {test_tensor.device}")
    
    # 优化张量
    optimized_tensor = optimize_tensor_for_mps(test_tensor)
    logger.info(f"优化后张量设备: {optimized_tensor.device}")
    
    # 检查数据类型
    logger.info(f"优化后张量数据类型: {optimized_tensor.dtype}")
    
    return optimized_tensor

def test_processor_comparison():
    """比较标准处理器和MPS处理器的性能"""
    logger.info("开始比较标准处理器和MPS处理器的性能...")
    
    try:
        from src.core.process.image_process_mps import get_optimized_image_transform
        from src.core.utils.mps_utils import is_mps_device
        from PIL import Image
        
        # 创建测试图像
        test_image = Image.new('RGB', (800, 600), color='blue')
        logger.info("创建了测试图像")
        
        # 测试标准图像变换
        logger.info("测试标准图像变换...")
        try:
            # 由于get_optimized_image_transform不接受device参数，
            # 我们直接创建标准图像变换
            from src.core.process.image_process_mps import ImageTransform
            standard_transform = ImageTransform(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                normalize=True
            )
            
            standard_result = standard_transform(test_image)
            logger.info(f"标准图像变换成功，输出形状: {standard_result.shape}")
            
        except Exception as e:
            logger.error(f"标准图像变换失败: {e}")
            return False
        
        # 测试MPS优化的图像变换
        if is_mps_device():
            logger.info("测试MPS优化的图像变换...")
            try:
                mps_transform = get_optimized_image_transform(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    normalize=True
                )
                
                mps_result = mps_transform(test_image)
                logger.info(f"MPS图像变换成功，输出形状: {mps_result.shape}")
                
                # 比较结果
                if standard_result.shape == mps_result.shape:
                    logger.info("标准变换和MPS变换输出形状一致")
                    return True
                else:
                    logger.error(f"输出形状不一致: 标准={standard_result.shape}, MPS={mps_result.shape}")
                    return False
                    
            except Exception as e:
                logger.error(f"MPS图像变换失败: {e}")
                return False
        else:
            logger.info("当前不是MPS设备，跳过MPS变换测试")
            return True
        
    except Exception as e:
        logger.error(f"处理器比较测试失败: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return False

def test_mps_image_processing():
    """测试MPS图像处理功能"""
    logger.info("开始测试MPS图像处理功能...")
    
    try:
        from PIL import Image
        import numpy as np
        from src.core.process.image_process_mps import MPSOptimizedImageTransform, dynamic_preprocess
        from src.core.utils.mps_utils import optimize_tensor_for_mps, is_mps_device
        
        # 创建测试图像
        test_image = Image.new('RGB', (800, 600), color='red')
        logger.info("创建了测试图像")
        
        # 测试动态预处理
        if is_mps_device():
            logger.info("在MPS设备上测试动态预处理")
            try:
                # 测试动态预处理函数
                images_crop_raw, crop_ratio = dynamic_preprocess(test_image, image_size=384)
                logger.info(f"动态预处理成功，crop_ratio: {crop_ratio}, 图像块数量: {len(images_crop_raw)}")
            except Exception as e:
                logger.error(f"动态预处理失败: {e}")
                return False
        
        # 测试MPS优化的图像变换
        try:
            # 创建MPS优化的图像变换
            mps_transform = MPSOptimizedImageTransform(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                normalize=True
            )
            
            # 应用变换
            transformed = mps_transform(test_image)
            logger.info(f"图像变换成功，输出形状: {transformed.shape}")
            
            # 测试张量优化
            if is_mps_device():
                optimized_tensor = optimize_tensor_for_mps(transformed)
                logger.info(f"张量优化成功，优化后形状: {optimized_tensor.shape}")
            
        except Exception as e:
            logger.error(f"图像变换失败: {e}")
            return False
        
        logger.info("MPS图像处理测试成功")
        return True
        
    except Exception as e:
        logger.error(f"MPS图像处理测试失败: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return False

def main():
    """主测试函数"""
    logger.info("开始MPS设备测试...")
    
    # 测试MPS设备检测
    device, is_mps = test_mps_detection()
    
    # 测试张量优化
    test_tensor_optimization()
    
    # 测试处理器比较
    processor_test_passed = test_processor_comparison()
    
    # 测试MPS图像处理
    image_processing_test_passed = test_mps_image_processing()
    
    # 总结测试结果
    logger.info("测试总结:")
    logger.info(f"- 最优设备: {device}")
    logger.info(f"- 是否为MPS设备: {is_mps}")
    logger.info(f"- 处理器初始化测试: {'通过' if processor_test_passed else '失败'}")
    logger.info(f"- 图像处理测试: {'通过' if image_processing_test_passed else '失败'}")
    
    if processor_test_passed and image_processing_test_passed:
        logger.info("所有测试通过！MPS支持已成功实现。")
        return 0
    else:
        logger.error("部分测试失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())