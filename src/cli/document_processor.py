#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理器
支持多种文档格式转图像并进行OCR识别
"""

import os
import subprocess
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

# 设置MPS回退环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# PyMuPDF用于PDF处理
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        raise ImportError("请安装PyMuPDF包: pip install PyMuPDF")

import img2pdf  # type: ignore
from PIL import Image

# 导入utils模块中的统一函数
from src.cli.utils import get_compatible_device, get_appropriate_dtype, should_use_bfloat16
# 导入新的引擎工厂
from src.core.factory.ocr_engine_factory import OCREngineFactory
# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()

class DocumentProcessor:
    def __init__(self, mode="auto", model_path=None, prompt=None, 
                 base_size=1024, image_size=640, crop_mode=True, debug=False):
        self.mode = mode
        self.model_path = model_path
        self.prompt = prompt
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.debug = debug
        
        # 如果启用调试模式，设置日志级别为DEBUG
        if self.debug:
            import logging
            # 设置根日志记录器级别
            logging.getLogger().setLevel(logging.DEBUG)
            # 设置OCR日志记录器级别
            logging.getLogger("deepseek_ocr").setLevel(logging.DEBUG)
            logger.debug("调试模式已启用")
        
        # 支持的文档格式
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.doc': self._process_document,
            '.docx': self._process_document,
            '.ppt': self._process_document,
            '.pptx': self._process_document,
            '.xls': self._process_document,
            '.xlsx': self._process_document,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
        }

    def process(self, input_path: str, output_dir: str):
        """主处理函数"""
        input_path_obj = Path(input_path)
        output_dir_obj = Path(output_dir)
        
        if not input_path_obj.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # 获取文件扩展名
        ext = input_path_obj.suffix.lower()
        
        if ext in self.supported_formats:
            self.supported_formats[ext](str(input_path_obj), str(output_dir_obj))
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def _process_document(self, input_path: str, output_dir: str):
        """处理文档文件（Word, PPT, Excel等）"""
        logger.info(f"正在将 {Path(input_path).name} 转换为PDF...")
        
        # 使用LibreOffice将文档转换为PDF
        pdf_path = self._convert_to_pdf(Path(input_path), Path(output_dir))
        
        # 处理PDF
        self._process_pdf(str(pdf_path), output_dir)
    
    def _process_pdf(self, input_path: str, output_dir: str):
        """处理PDF文件"""
        logger.info(f"正在处理PDF文件: {Path(input_path).name}")
        
        # 将PDF转换为图像
        images = self._pdf_to_images(Path(input_path))
        
        # 进行OCR识别
        self._perform_ocr(images, Path(output_dir))
    
    def _process_image(self, input_path: str, output_dir: str):
        """处理图像文件"""
        logger.info(f"正在处理图像文件: {Path(input_path).name}")
        
        # 打开图像
        image = Image.open(input_path)
        
        # 进行OCR识别
        self._perform_ocr([image], Path(output_dir))
    
    def _convert_to_pdf(self, input_path: Path, output_dir: Path) -> Path:
        """使用LibreOffice将文档转换为PDF"""
        try:
            # 检查LibreOffice是否可用
            if not shutil.which("libreoffice"):
                raise RuntimeError("未找到LibreOffice，请确保已安装并添加到PATH环境变量中")
            
            # 创建临时目录用于转换
            temp_dir = tempfile.mkdtemp()
            
            # 构建LibreOffice命令
            cmd = [
                'libreoffice',
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', temp_dir,
                str(input_path)
            ]
            
            # 执行转换
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"LibreOffice转换失败: {result.stderr}")
            
            # 查找生成的PDF文件
            pdf_files = list(Path(temp_dir).glob("*.pdf"))
            if not pdf_files:
                raise RuntimeError("未找到转换后的PDF文件")
            
            # 移动PDF文件到输出目录
            output_pdf = output_dir / f"{input_path.stem}.pdf"
            shutil.move(str(pdf_files[0]), str(output_pdf))
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
            return output_pdf
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("LibreOffice转换超时")
        except Exception as e:
            raise RuntimeError(f"转换过程中发生错误: {str(e)}")
    
    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """将PDF转换为图像列表"""
        logger.info(f"开始将PDF转换为图像: {pdf_path}")
        images = []
        
        pdf_document = fitz.open(str(pdf_path))  # type: ignore
        logger.info(f"PDF文档已打开，共 {pdf_document.page_count} 页")
        zoom = 144 / 72.0  # 144 DPI
        matrix = fitz.Matrix(zoom, zoom)  # type: ignore
        
        for page_num in range(pdf_document.page_count):
            logger.info(f"正在处理第 {page_num + 1} 页")
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)  # type: ignore
            
            # 转换为PIL图像
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            images.append(image)
            logger.info(f"第 {page_num + 1} 页处理完成，尺寸: {pixmap.width}x{pixmap.height}")
        
        pdf_document.close()  # type: ignore
        logger.info(f"PDF转换完成，共生成 {len(images)} 张图像")
        return images
    
    def _perform_ocr(self, images: List[Image.Image], output_dir: Path):
        """执行OCR识别"""
        logger.info(f"正在对 {len(images)} 张图像进行OCR识别...")
        
        # 智能模式选择
        actual_mode = self._determine_mode()
        
        # 使用工厂方法创建相应的OCR引擎
        ocr_engine = OCREngineFactory.create_engine(
            engine_type=actual_mode,
            model_path=self.model_path,
            prompt=self.prompt,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode
        )
        
        # 初始化引擎
        ocr_engine.initialize()
        
        # 执行OCR处理
        ocr_engine.process(images, str(output_dir))
        
        # 清理资源
        ocr_engine.cleanup()
    
    def _determine_mode(self):
        """确定实际使用的模式"""
        if self.mode == "auto":
            # 自动检测可用的引擎
            if self._is_vllm_available() and not self._is_mps_environment():
                return "vllm"
            else:
                return "transformers"
        elif self.mode == "vllm" and self._is_mps_environment():
            logger.warning("MPS环境不支持vLLM，自动回退到Transformers模式")
            return "transformers"
        else:
            # 使用指定的模式
            return self.mode
    
    def _is_vllm_available(self):
        """检查vLLM是否可用"""
        try:
            import vllm  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False
    
    def _is_mps_environment(self):
        """检查是否在MPS环境下"""
        try:
            import torch
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except ImportError:
            return False
    
    def convert_to_images(self, document_path: str) -> List[Image.Image]:
        """
        将文档转换为图像列表
        
        Args:
            document_path: 文档路径
            
        Returns:
            图像列表
        """
        document_path_obj = Path(document_path)
        
        if not document_path_obj.exists():
            raise FileNotFoundError(f"输入文件不存在: {document_path}")
        
        # 获取文件扩展名
        ext = document_path_obj.suffix.lower()
        
        if ext == '.pdf':
            # 直接处理PDF
            return self._pdf_to_images(document_path_obj)
        elif ext in ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
            # 先转换为PDF，再处理
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = self._convert_to_pdf(document_path_obj, Path(temp_dir))
                return self._pdf_to_images(pdf_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            # 直接加载图像
            image = Image.open(document_path)
            return [image]
        else:
            raise ValueError(f"不支持的文件格式: {ext}")