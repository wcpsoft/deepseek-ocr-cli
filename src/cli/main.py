#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR CLI 主入口
支持多种文档格式转OCR识别
使用策略模式，提供统一的命令行接口
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.logging import get_logger, setup_logging
from src.core.factory.ocr_engine_factory import OCREngineFactory
from src.core.service.ocr_service import OCRService
from src.core.config import get_config
from src.core.utils.exception_handler import ExceptionHandler, OCRException

logger = get_logger()


def detect_mps_environment():
    """检测是否在MPS环境下"""
    try:
        import torch
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except ImportError:
        return False


class OCRStrategy:
    """
    OCR处理策略基类
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        初始化策略
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.config = get_config()
        self.engine = None
        self.service = None
    
    def execute(self) -> int:
        """
        执行策略
        
        Returns:
            退出码
        """
        raise NotImplementedError("子类必须实现execute方法")
    
    def initialize_service(self) -> bool:
        """
        初始化OCR服务
        
        Returns:
            是否初始化成功
        """
        try:
            # 创建OCR引擎
            self.engine = OCREngineFactory.create_engine(
                engine_type=self.args.mode,
                model_path=getattr(self.args, 'model_path', None)
            )
            
            # 创建OCR服务
            self.service = OCRService(
                engine=self.engine,
                output_dir=self.args.output
            )
            
            # 初始化服务
            if not self.service.initialize():
                logger.error("OCR服务初始化失败")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"初始化OCR服务失败: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """
        清理资源
        """
        try:
            if self.service:
                self.service.cleanup()
        except Exception as e:
            logger.error(f"清理资源时发生错误: {str(e)}")


class ImageOCRStrategy(OCRStrategy):
    """
    图像OCR策略
    """
    
    def execute(self) -> int:
        """
        执行图像OCR
        
        Returns:
            退出码
        """
        try:
            # 初始化服务
            if not self.initialize_service():
                return 1
            
            # 处理图像
            result = self.service.process_image(self.args.input)
            
            # 输出结果
            if result:
                print(result)
                return 0
            else:
                logger.error("图像处理失败")
                return 1
                
        except OCRException as e:
            logger.error(f"OCR处理异常: {str(e)}")
            if e.details:
                logger.error(f"异常详情: {e.details}")
            return 1
        except Exception as e:
            logger.error(f"处理图像时发生未知错误: {str(e)}")
            return 1
        finally:
            self.cleanup()


class DocumentOCRStrategy(OCRStrategy):
    """
    文档OCR策略
    """
    
    def execute(self) -> int:
        """
        执行文档OCR
        
        Returns:
            退出码
        """
        try:
            # 初始化服务
            if not self.initialize_service():
                return 1
            
            # 处理文档
            result = self.service.process_document(
                self.args.input,
                output_filename="result.mmd",
                stop_on_error=True
            )
            
            # 输出结果
            if result["success"]:
                logger.info("文档处理成功")
                if result["output_file"]:
                    logger.info(f"结果已保存到: {result['output_file']}")
                return 0
            else:
                logger.error("文档处理失败")
                if result["error_file"]:
                    logger.error(f"错误报告已保存到: {result['error_file']}")
                return 1
                
        except OCRException as e:
            logger.error(f"OCR处理异常: {str(e)}")
            if e.details:
                logger.error(f"异常详情: {e.details}")
            return 1
        except Exception as e:
            logger.error(f"处理文档时发生未知错误: {str(e)}")
            return 1
        finally:
            self.cleanup()


class BatchOCRStrategy(OCRStrategy):
    """
    批量OCR策略
    """
    
    def execute(self) -> int:
        """
        执行批量OCR
        
        Returns:
            退出码
        """
        try:
            # 初始化服务
            if not self.initialize_service():
                return 1
            
            # 获取图像列表
            images = self._get_image_list()
            
            # 处理图像
            result = self.service.process_images(
                images,
                output_filename="result.mmd",
                stop_on_error=True
            )
            
            # 输出结果
            if result["success"]:
                logger.info("批量处理成功")
                if result["output_file"]:
                    logger.info(f"结果已保存到: {result['output_file']}")
                return 0
            else:
                logger.error("批量处理失败")
                if result["error_file"]:
                    logger.error(f"错误报告已保存到: {result['error_file']}")
                return 1
                
        except OCRException as e:
            logger.error(f"OCR处理异常: {str(e)}")
            if e.details:
                logger.error(f"异常详情: {e.details}")
            return 1
        except Exception as e:
            logger.error(f"批量处理时发生未知错误: {str(e)}")
            return 1
        finally:
            self.cleanup()
    
    def _get_image_list(self) -> List[str]:
        """
        获取图像列表
        
        Returns:
            图像路径列表
        """
        if os.path.isfile(self.args.input):
            # 单个文件
            return [self.args.input]
        elif os.path.isdir(self.args.input):
            # 目录，获取所有支持的图像文件
            supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_list = []
            
            for root, _, files in os.walk(self.args.input):
                for file in files:
                    if Path(file).suffix.lower() in supported_extensions:
                        image_list.append(os.path.join(root, file))
            
            return sorted(image_list)
        else:
            raise FileNotFoundError(f"输入路径不存在: {self.args.input}")


class OCRContext:
    """
    OCR上下文
    负责选择和执行适当的策略
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        初始化上下文
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.strategy = None
    
    def set_strategy(self, strategy: OCRStrategy) -> None:
        """
        设置策略
        
        Args:
            strategy: OCR策略
        """
        self.strategy = strategy
    
    def execute_strategy(self) -> int:
        """
        执行策略
        
        Returns:
            退出码
        """
        if not self.strategy:
            logger.error("未设置OCR策略")
            return 1
        
        return self.strategy.execute()


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="DeepSeek OCR命令行工具")
    
    # 基本参数
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("-o", "--output", help="输出目录路径", default="./output")
    parser.add_argument("-m", "--mode", choices=["auto", "vllm", "transformers"], 
                       help="推理模式 (auto: 自动选择, vllm: 使用vLLM引擎, transformers: 使用Transformers引擎)", default="auto")
    parser.add_argument("--model-path", help="模型路径", default=None)
    parser.add_argument("--prompt", help="OCR提示词", 
                       default="<image>\n<|grounding|>Convert the document to markdown.")
    parser.add_argument("--download-models", action="store_true", 
                       help="下载模型到本地")
    parser.add_argument("--base-size", type=int, default=1024, 
                       help="基础尺寸 (默认: 1024)")
    parser.add_argument("--image-size", type=int, default=640, 
                       help="图像尺寸 (默认: 640)")
    parser.add_argument("--crop-mode", action="store_true", 
                       help="是否启用裁剪模式")
    parser.add_argument("--debug", action="store_true", 
                       help="启用调试模式，输出详细日志")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="日志级别")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    try:
        # 解析参数
        args = parse_arguments()
        
        # 如果在MPS环境下且模式设置为vLLM，则提示用户并自动切换到transformers
        if detect_mps_environment() and args.mode == "vllm":
            print("警告: MPS环境不支持vLLM引擎，自动切换到Transformers引擎")
            args.mode = "transformers"
        
        # 如果是auto模式，自动选择合适的引擎
        if args.mode == "auto":
            if detect_mps_environment():
                args.mode = "transformers"
            else:
                # 默认使用transformers
                args.mode = "transformers"
        
        # 如果需要下载模型
        if args.download_models:
            # 延迟导入，避免在不需要时加载依赖
            from cli.model_manager import ModelManager
            model_manager = ModelManager()
            model_manager.download_models()
            return
        
        # 设置日志
        if args.debug:
            args.log_level = "DEBUG"
        setup_logging(level=args.log_level)
        
        # 创建上下文
        context = OCRContext(args)
        
        # 选择策略
        if args.batch:
            strategy = BatchOCRStrategy(args)
        elif os.path.isfile(args.input):
            # 根据文件扩展名选择策略
            file_ext = Path(args.input).suffix.lower()
            if file_ext in {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}:
                strategy = DocumentOCRStrategy(args)
            else:
                strategy = ImageOCRStrategy(args)
        else:
            # 默认使用批量策略
            strategy = BatchOCRStrategy(args)
        
        # 设置并执行策略
        context.set_strategy(strategy)
        exit_code = context.execute_strategy()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()