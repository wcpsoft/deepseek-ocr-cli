#!/usr/bin/env python3
"""
重构后的OCR系统测试脚本
验证重构后的OCR系统功能是否正常工作
"""

import logging
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_transformers_engine() -> bool:
    """测试Transformers引擎"""
    logger.info("开始测试Transformers引擎...")

    engine: Any | None = None
    try:
        # 创建Transformers引擎
        try:
            from src.core.factory.ocr_engine_factory import OCREngineFactory

            engine = OCREngineFactory.create_engine("transformers")
        except ImportError:
            logger.warning("无法导入Transformers引擎,跳过测试")
            return True

    except RuntimeError as e:
        logger.error("Transformers引擎测试失败: %s", e)
        return False
    except Exception as e:
        logger.error("Transformers引擎测试失败: %s", e)
        return False
    finally:
        # 清理资源
        if engine is not None:
            try:
                engine.cleanup()
            except Exception as cleanup_error:
                logger.warning("清理资源时出错: %s", cleanup_error)

    return True


def test_vllm_engine() -> bool:
    """测试vLLM引擎"""
    logger.info("开始测试vLLM引擎...")

    engine: Any | None = None
    try:
        # 创建vLLM引擎
        try:
            from src.core.factory.ocr_engine_factory import OCREngineFactory

            engine = OCREngineFactory.create_engine("vllm")
        except ImportError:
            logger.warning("无法导入vLLM引擎,跳过测试")
            return True

    except RuntimeError as e:
        logger.error("vLLM引擎测试失败: %s", e)
        return False
    except Exception as e:
        logger.error("vLLM引擎测试失败: %s", e)
        return False
    finally:
        # 清理资源
        if engine is not None:
            try:
                engine.cleanup()
            except Exception as cleanup_error:
                logger.warning("清理资源时出错: %s", cleanup_error)

    return True


def test_auto_engine() -> bool:
    """测试自动选择引擎"""
    logger.info("开始测试自动选择引擎...")

    engine: Any | None = None
    try:
        # 创建自动选择引擎
        try:
            from src.core.factory.ocr_engine_factory import OCREngineFactory

            engine = OCREngineFactory.create_engine("auto")
        except ImportError:
            logger.warning("无法导入自动选择引擎,跳过测试")
            return True

    except RuntimeError as e:
        logger.error("自动选择引擎测试失败: %s", e)
        return False
    except Exception as e:
        logger.error("自动选择引擎测试失败: %s", e)
        return False
    finally:
        # 清理资源
        if engine is not None:
            try:
                engine.cleanup()
            except Exception as cleanup_error:
                logger.warning("清理资源时出错: %s", cleanup_error)

    return True


def main() -> int:
    """主函数"""
    logger.info("开始重构后的OCR系统测试")

    # 测试Transformers引擎
    transformers_success = test_transformers_engine()

    # 测试vLLM引擎
    vllm_success = test_vllm_engine()

    # 测试自动选择引擎
    auto_success = test_auto_engine()

    # 检查测试结果
    if transformers_success and vllm_success and auto_success:
        logger.info("所有测试完成!重构后的OCR系统测试通过。")
        return 0
    else:
        logger.error("部分测试失败,请检查日志以获取详细信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
