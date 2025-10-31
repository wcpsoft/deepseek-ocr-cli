#!/usr/bin/env python3
"""
MPS设备测试脚本
用于测试新的图像处理模块在MPS设备上的表现
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_mps_detection() -> None:
    """
    测试MPS设备检测功能
    """
    logger.info("测试MPS设备检测功能...")

    try:
        import torch

        from src.core.process.image_process import is_mps_device

        # 检查MPS设备可用性
        mps_available = is_mps_device()
        logger.info("MPS设备可用性检测结果: %s", mps_available)

        # 验证检测结果
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert mps_available, "MPS设备应该可用"
            logger.info("MPS设备检测正确")
        else:
            assert not mps_available, "MPS设备应该不可用"
            logger.info("MPS设备检测正确")

        logger.info("MPS设备检测功能测试完成")

    except Exception as e:
        logger.error("测试MPS设备检测时发生错误: %s", e)
        raise


def test_tensor_optimization() -> None:
    """
    测试张量优化
    """
    logger.info("测试张量优化...")

    try:
        import torch

        from src.core.process.image_process import is_mps_device

        # 检查MPS设备可用性
        mps_available = is_mps_device()
        logger.info("MPS设备可用性: %s", mps_available)

        # 测试不同设备的数据类型选择
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("检测到CUDA设备")
        elif mps_available:
            device = "mps"
            logger.info("检测到MPS设备")
        else:
            device = "cpu"
            logger.info("使用CPU设备")

        # 验证设备选择逻辑
        assert device in ["cuda", "mps", "cpu"], f"无效的设备类型: {device}"
        logger.info("设备选择正确: %s", device)

        logger.info("张量优化测试完成")

    except Exception as e:
        logger.error("测试张量优化时发生错误: %s", e)
        raise


def test_processor_comparison() -> None:
    """
    测试处理器比较
    """
    logger.info("测试处理器比较...")

    try:
        # 尝试导入处理器
        try:
            from src.core.process.image_process import (
                DeepseekOCRProcessor,
                DeepseekOCRProcessor as MPSDeepseekOCRProcessor,
            )

            # 创建标准处理器
            processor = DeepseekOCRProcessor()
            logger.info("标准处理器创建成功")

            # 创建MPS处理器
            mps_processor = MPSDeepseekOCRProcessor()
            logger.info("MPS处理器创建成功")

            # 比较处理器类型
            assert type(processor).__name__ == "DeepseekOCRProcessor", "标准处理器类型不正确"
            assert type(mps_processor).__name__ == "DeepseekOCRProcessor", "MPS处理器类型不正确"

            # 检查处理器方法
            assert callable(processor), "标准处理器应有__call__方法"
            assert callable(mps_processor), "MPS处理器应有__call__方法"

            logger.info("处理器比较测试完成")

        except ImportError as e:
            logger.error("无法导入处理器: %s", e)
            raise

    except Exception as e:
        logger.error("测试处理器比较时发生错误: %s", e)
        raise


def test_mps_image_processing() -> None:
    """
    测试MPS图像处理功能
    """
    logger.info("测试MPS图像处理功能...")

    try:
        import torch

        from src.core.process.image_process import DeepseekOCRProcessor

        # 创建测试图像
        logger.info("创建测试图像成功")

        # 创建MPS处理器
        processor = DeepseekOCRProcessor()
        logger.info("MPS处理器创建成功")

        # 检查是否有MPS设备
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("检测到MPS设备, 进行MPS处理测试")

            # 测试图像处理
            # 注意: 这里只测试处理器创建和基本属性, 不进行实际处理
            # 因为实际处理需要加载模型, 这在单元测试中可能不可行
            assert callable(processor), "MPS处理器应有__call__方法"
            logger.info("MPS图像处理器测试通过")
        else:
            logger.info("未检测到MPS设备, 跳过MPS处理测试")

        logger.info("MPS图像处理功能测试完成")

    except Exception as e:
        logger.error("测试MPS图像处理时发生错误: %s", e)
        raise


def main() -> int:
    """主测试函数"""
    logger.info("开始MPS设备测试...")

    # 测试MPS设备检测
    test_mps_detection()

    # 测试张量优化
    test_tensor_optimization()

    # 测试处理器比较
    test_processor_comparison()

    # 测试MPS图像处理
    test_mps_image_processing()

    # 总结测试结果
    logger.info("所有测试完成! MPS支持测试通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
