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
from pathlib import Path
from typing import Optional, List

# PyMuPDF用于PDF处理
import fitz  # type: ignore
import img2pdf  # type: ignore
from PIL import Image


class DocumentProcessor:
    def __init__(self, mode="auto", model_path=None, prompt=None, 
                 base_size=1024, image_size=640, crop_mode=True):
        self.mode = mode
        self.model_path = model_path
        self.prompt = prompt
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        
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
        print(f"正在将 {Path(input_path).name} 转换为PDF...")
        
        # 使用LibreOffice将文档转换为PDF
        pdf_path = self._convert_to_pdf(Path(input_path), Path(output_dir))
        
        # 处理PDF
        self._process_pdf(str(pdf_path), output_dir)
    
    def _process_pdf(self, input_path: str, output_dir: str):
        """处理PDF文件"""
        print(f"正在处理PDF文件: {Path(input_path).name}")
        
        # 将PDF转换为图像
        images = self._pdf_to_images(Path(input_path))
        
        # 进行OCR识别
        self._perform_ocr(images, Path(output_dir))
    
    def _process_image(self, input_path: str, output_dir: str):
        """处理图像文件"""
        print(f"正在处理图像文件: {Path(input_path).name}")
        
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
        images = []
        
        pdf_document = fitz.open(str(pdf_path))
        zoom = 144 / 72.0  # 144 DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # 转换为PIL图像
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            images.append(image)
        
        pdf_document.close()
        return images
    
    def _perform_ocr(self, images: List[Image.Image], output_dir: Path):
        """执行OCR识别"""
        print(f"正在对 {len(images)} 张图像进行OCR识别...")
        
        # 智能模式选择
        actual_mode = self._determine_mode()
        
        if actual_mode == "vllm":
            print("使用 vLLM 引擎进行OCR识别...")
            self._perform_ocr_vllm(images, output_dir)
        elif actual_mode == "transformers":
            print("使用 Transformers 引擎进行OCR识别...")
            self._perform_ocr_transformers(images, output_dir)
        else:
            raise ValueError(f"不支持的模式: {actual_mode}")
    
    def _determine_mode(self):
        """确定实际使用的模式"""
        if self.mode == "auto":
            # 自动检测可用的引擎
            if self._is_vllm_available():
                return "vllm"
            else:
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
    
    def _perform_ocr_vllm(self, images: List[Image.Image], output_dir: Path):
        """使用vLLM执行OCR"""
        # 这里需要导入vLLM相关模块
        try:
            # 添加项目根目录到路径
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            # 延迟导入，避免在不需要时加载依赖
            from src.core.config import MODEL_PATH, PROMPT
            from src.core.deepseek_ocr import DeepseekOCRForCausalLM
            from vllm.model_executor.models.registry import ModelRegistry  # type: ignore
            from vllm import LLM, SamplingParams  # type: ignore
            from src.core.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
            from src.core.process.image_process import DeepseekOCRProcessor
            
            # 设置模型路径
            model_path = self.model_path or MODEL_PATH
            prompt = self.prompt or PROMPT
            
            ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
            
            llm = LLM(
                model=model_path,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                enforce_eager=False,
                trust_remote_code=True, 
                max_model_len=8192,
                swap_space=0,
                max_num_seqs=100,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                disable_mm_preprocessor_cache=True
            )
            
            logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, 
                                                            whitelist_token_ids={128821, 128822})]
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )
            
            # 处理图像
            batch_inputs = []
            for image in images:
                cache_item = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": DeepseekOCRProcessor().tokenize_with_images(
                            images=[image], 
                            bos=True, 
                            eos=True, 
                            cropping=self.crop_mode
                        )
                    },
                }
                batch_inputs.append(cache_item)
            
            # 生成结果
            outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
            
            # 保存结果
            self._save_ocr_results(outputs_list, images, output_dir)
            
        except Exception as e:
            raise RuntimeError(f"vLLM OCR执行失败: {str(e)}")
    
    def _perform_ocr_transformers(self, images: List[Image.Image], output_dir: Path):
        """使用Transformers执行OCR"""
        try:
            # 添加项目根目录到路径
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            # 延迟导入，避免在不需要时加载依赖
            from transformers.models.auto.modeling_auto import AutoModel
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            import torch
            
            # 设置模型路径
            model_name = self.model_path or 'deepseek-ai/DeepSeek-OCR'
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # 直接使用默认实现，避免flash_attention_2相关问题
            model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                use_safetensors=True
            )
            
            # 检查可用的设备
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            
            model = model.eval().to(device).to(torch.bfloat16)
            
            # 处理每张图像
            results = []
            for i, image in enumerate(images):
                # 保存临时图像文件
                temp_image_path = output_dir / f"temp_{i}.jpg"
                image.save(temp_image_path, "JPEG")
                
                # 执行推理
                result = model.infer(
                    tokenizer, 
                    prompt=self.prompt,
                    image_file=str(temp_image_path),
                    output_path=str(output_dir),
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=self.crop_mode,
                    save_results=False,
                    test_compress=True
                )
                
                results.append(result)
                
                # 删除临时文件
                temp_image_path.unlink()
            
            # 保存结果
            self._save_ocr_results_transformers(results, output_dir)
            
        except Exception as e:
            raise RuntimeError(f"Transformers OCR执行失败: {str(e)}")
    
    def _save_ocr_results(self, outputs_list: list, images: List[Image.Image], output_dir: Path):
        """保存vLLM OCR结果"""
        # 保存原始结果
        contents = ''
        for output in outputs_list:
            content = output.outputs[0].text
            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
            contents += content + '\n'
        
        # 写入文件
        result_file = output_dir / "result.mmd"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        print(f"OCR结果已保存到: {result_file}")
    
    def _save_ocr_results_transformers(self, results: list, output_dir: Path):
        """保存Transformers OCR结果"""
        # 保存结果
        contents = ''
        for result in results:
            contents += str(result) + '\n'
        
        # 写入文件
        result_file = output_dir / "result.mmd"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        print(f"OCR结果已保存到: {result_file}")