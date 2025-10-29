#!/usr/bin/env python3
"""
测试重构后的OCR功能
"""

import os
import sys

from src.core.factory.ocr_engine_factory import OCREngineFactory
from src.core.logging import get_logger, setup_logging
from src.core.utils.exception_handler import ExceptionHandler

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

logger = get_logger()


def test_engine_factory():
    """
    测试引擎工厂
    """
    logger.info("测试引擎工厂...")

    # 获取可用引擎
    available_engines = OCREngineFactory.get_available_engines()
    logger.info(f"可用引擎: {available_engines}")

    # 检查transformers引擎是否可用
    if OCREngineFactory.is_engine_available("transformers"):
        logger.info("Transformers引擎可用")
    else:
        logger.warning("Transformers引擎不可用")

    # 检查vLLM引擎是否可用
    if OCREngineFactory.is_engine_available("vllm"):
        logger.info("vLLM引擎可用")
    else:
        logger.warning("vLLM引擎不可用")


def test_ocr_service():
    """
    测试OCR服务
    """
    logger.info("测试OCR服务...")

    try:
        from src.core.service.ocr_service import OCRService

        # 创建引擎
        factory = OCREngineFactory()
        engine = factory.create_engine("transformers")

        # 创建服务
        service = OCRService(engine)
        logger.info("OCR服务创建成功")

        # 初始化服务
        assert service.initialize(), "服务初始化失败"
        logger.info("服务初始化成功")

        # 获取引擎信息
        info = service.get_engine_info()
        assert isinstance(info, dict), "引擎信息应该是字典"
        logger.info(f"引擎信息: {info}")

        # 清理资源
        service.cleanup()
        logger.info("OCR服务测试完成")

    except Exception as e:
        logger.error(f"测试OCR服务时发生错误: {e!s}")
        raise


def test_exception_handling():
    """
    测试异常处理
    """
    logger.info("测试异常处理...")

    try:
        # 测试内存检查
        has_enough_memory = ExceptionHandler.check_memory_availability(8.0)
        logger.info(f"内存检查结果: {has_enough_memory}")

        # 获取内存信息
        memory_info = ExceptionHandler._get_memory_info()
        logger.info(f"内存信息: {memory_info}")

    except Exception as e:
        logger.error(f"测试异常处理失败: {e!s}")
        raise


def test_image_processing():
    """
    测试图像处理功能
    """
    logger.info("测试图像处理功能...")

    try:

        from src.core.service.ocr_service import OCRService

        # 检查测试图像是否存在
        test_image_path = "samples/4.pdf"
        assert os.path.exists(test_image_path), f"测试图像不存在: {test_image_path}"
        logger.info(f"找到测试图像: {test_image_path}")

        # 创建引擎
        factory = OCREngineFactory()
        engine = factory.create_engine("transformers")

        # 创建OCR服务
        service = OCRService(engine)
        logger.info("OCR服务创建成功")

        # 初始化服务
        assert service.initialize(), "服务初始化失败"
        logger.info("服务初始化成功")

        # 处理文档
        result = service.process_document(test_image_path)
        assert isinstance(result, dict), "处理结果应该是字典"
        assert "success" in result, "结果应包含success字段"
        logger.info(f"文档处理结果: {result}")

        # 清理资源
        service.cleanup()
        logger.info("图像处理测试完成")

    except Exception as e:
        logger.error(f"测试图像处理时发生错误: {e!s}")
        raise


def main():
    """
    主函数
    """
    # 设置日志
    setup_logging(level="INFO")

    logger.info("开始测试重构后的OCR功能...")

    # 运行测试
    tests = [
        ("引擎工厂", test_engine_factory),
        ("OCR服务", test_ocr_service),
        ("异常处理", test_exception_handling),
        ("图像处理", test_image_processing),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"运行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"测试 {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            logger.error(f"测试 {test_name} 发生异常: {e!s}")
            results.append((test_name, False))

    # 输出测试结果
    logger.info("=" * 50)
    logger.info("测试结果汇总:")
    for test_name, result in results:
        logger.info(f"  {test_name}: {'通过' if result else '失败'}")

    # 计算通过率
    passed = sum(1 for _, result in results if result)
    total = len(results)
    pass_rate = passed / total * 100

    logger.info(f"通过率: {passed}/{total} ({pass_rate:.1f}%)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
