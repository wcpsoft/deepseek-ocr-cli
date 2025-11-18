#!/usr/bin/env python3
"""
测试output_processor.py中的函数
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

# 直接导入output_processor模块，避免循环依赖
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "core", "utils"))
from output_processor import (
    load_image,
    re_match,
    extract_coordinates_and_label,
    draw_bounding_boxes,
    process_image_with_refs,
    save_ocr_results,
    clean_output_text,
)


class TestOutputProcessor:
    """测试output_processor中的函数"""

    def test_load_image_success(self):
        """测试成功加载图像"""
        # 创建临时图像文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # 创建测试图像
            test_image = Image.new("RGB", (100, 100), color="red")
            test_image.save(tmp_path)
            
            # 测试加载图像
            loaded_image = load_image(tmp_path)
            
            # 验证结果
            assert loaded_image is not None
            assert loaded_image.size == (100, 100)
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_image_file_not_found(self):
        """测试加载不存在的图像文件"""
        result = load_image("nonexistent_image.png")
        assert result is None

    def test_re_match(self):
        """测试正则匹配函数"""
        test_text = "<|ref|>image<|/ref|><|det|>title<|/det|><|ref|>text<|/ref|><|det|>content<|/det|>"
        matches, matches_image, matches_other = re_match(test_text)
        
        # 验证结果
        assert len(matches) == 2
        assert len(matches_image) == 1
        assert len(matches_other) == 1
        # matches是元组列表，每个元组包含完整匹配、ref内容和det内容
        assert matches[0] == ("<|ref|>image<|/ref|><|det|>title<|/det|>", "image", "title")
        assert matches[1] == ("<|ref|>text<|/ref|><|det|>content<|/det|>", "text", "content")
        # matches_image和matches_other是完整匹配字符串
        assert matches_image[0] == "<|ref|>image<|/ref|><|det|>title<|/det|>"
        assert matches_other[0] == "<|ref|>text<|/ref|><|det|>content<|/det|>"

    def test_extract_coordinates_and_label(self):
        """测试提取坐标和标签函数"""
        ref_text = ["", "title", "[[100, 100, 200, 200], [300, 300, 400, 400]]"]
        result = extract_coordinates_and_label(ref_text, 1000, 1000)
        
        # 验证结果
        assert result is not None
        label_type, cor_list = result
        assert label_type == "title"
        assert len(cor_list) == 2
        assert cor_list[0] == [100, 100, 200, 200]
        assert cor_list[1] == [300, 300, 400, 400]

    def test_extract_coordinates_and_label_invalid_format(self):
        """测试提取坐标和标签函数 - 无效格式"""
        ref_text = ["", "title", "invalid_coordinates"]
        result = extract_coordinates_and_label(ref_text, 1000, 1000)
        
        # 验证结果
        assert result is None

    def test_draw_bounding_boxes(self):
        """测试绘制边界框函数"""
        # 创建测试图像
        test_image = Image.new("RGB", (500, 500), color="white")
        cor_list = [[100, 100, 200, 200], [300, 300, 400, 400]]
        label_type = "title"
        
        # 测试绘制边界框
        result_image = draw_bounding_boxes(test_image, cor_list, label_type)
        
        # 验证结果
        assert result_image is not None
        assert result_image.size == (500, 500)

    def test_process_image_with_refs(self):
        """测试处理图像函数"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建临时图像文件
            tmp_image_path = os.path.join(tmp_dir, "test_image.png")
            test_image = Image.new("RGB", (500, 500), color="white")
            test_image.save(tmp_image_path)
            
            refs = [
                ["", "title", "[[100, 100, 200, 200], [300, 300, 400, 400]]"],
                ["", "content", "[[50, 50, 150, 150]]"]
            ]
            output_path = os.path.join(tmp_dir, "output")
            
            # 测试处理图像
            result_image = process_image_with_refs(tmp_image_path, refs, output_path)
            
            # 验证结果
            assert result_image is not None
            assert result_image.size == (500, 500)

    def test_save_ocr_results(self):
        """测试保存OCR结果函数"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建测试图像
            test_image = Image.new("RGB", (500, 500), color="white")
            output_path = tmp_dir
            test_text = "测试OCR结果文本"
            
            # 测试保存结果
            save_ocr_results(test_text, test_image, output_path)
            
            # 验证结果 - 文件名格式为 目录名_results.mmd
            dir_name = os.path.basename(output_path)
            expected_filename = f"{dir_name}_results.mmd"
            output_file = os.path.join(output_path, expected_filename)
            
            assert os.path.exists(output_file)
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert content == test_text

