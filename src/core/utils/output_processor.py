"""
输出处理工具函数
用于统一管理OCR结果的保存和处理
"""

import os
import re
import sys
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.core.utils.filename_utils import generate_output_filename


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    加载图像并处理EXIF方向
    
    Args:
        image_path: 图像路径
        
    Returns:
        处理后的PIL图像对象，如果加载失败则返回None
    """
    try:
        image = Image.open(image_path)
        # 处理EXIF方向信息
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"加载图像时出错: {e}")
        try:
            # 如果EXIF处理失败，尝试直接加载
            return Image.open(image_path)
        except:
            return None


def re_match(text: str) -> Tuple[List, List, List]:
    """
    使用正则表达式匹配文本中的特定模式
    
    Args:
        text: 待匹配的文本
        
    Returns:
        Tuple[List, List, List]: 匹配结果、图像匹配项和其他匹配项
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text: List, image_width: int, image_height: int) -> Optional[Tuple]:
    """
    从引用文本中提取坐标和标签
    
    Args:
        ref_text: 引用文本
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        Optional[Tuple]: 标签类型和坐标列表
    """
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"提取坐标和标签时出错: {e}")
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image: Image.Image, cor_list: List, label_type: str, output_path: Optional[str] = None) -> Image.Image:
    """
    在图像上绘制边界框
    
    Args:
        image: 输入图像
        cor_list: 坐标列表
        label_type: 标签类型
        output_path: 输出路径，如果为None则不保存
        
    Returns:
        Image.Image: 处理后的图像
    """
    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()

    img_idx = 0
    
    # 确保images目录存在
    if output_path:
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    
    color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
    color_a = color + (20, )
    
    for points in cor_list:
        x1, y1, x2, y2 = points

        x1 = int(x1 / 999 * image_width)
        y1 = int(y1 / 999 * image_height)
        x2 = int(x2 / 999 * image_width)
        y2 = int(y2 / 999 * image_height)

        if label_type == 'image':
            try:
                if output_path:
                    cropped = image.crop((x1, y1, x2, y2))
                    cropped.save(f"{output_path}/images/{img_idx}.jpg")
            except Exception as e:
                print(f"保存裁剪图像时出错: {e}")
                pass
            img_idx += 1
            
        try:
            if label_type == 'title':
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
            
            text_x = x1
            text_y = max(0, y1 - 15)
                
            text_bbox = draw.textbbox((0, 0), label_type, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                        fill=(255, 255, 255, 30))
            
            draw.text((text_x, text_y), label_type, font=font, fill=color)
        except Exception as e:
            print(f"绘制边界框时出错: {e}")
            pass
            
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image_path: str, refs: List, output_path: Optional[str] = None) -> Image.Image:
    """
    处理图像并绘制引用的边界框
    
    Args:
        image_path: 图像路径
        refs: 引用列表
        output_path: 输出路径，如果为None则不保存
        
    Returns:
        Image.Image: 处理后的图像
    """
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()

    img_idx = 0
    
    # 确保images目录存在
    if output_path:
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    
    for ref in refs:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            if output_path:
                                cropped = image.crop((x1, y1, x2, y2))
                                cropped.save(f"{output_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(f"保存裁剪图像时出错: {e}")
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        
                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception as e:
                        print(f"绘制边界框时出错: {e}")
                        pass
        except Exception as e:
            print(f"处理引用时出错: {e}")
            continue
            
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def save_ocr_results(
    outputs: str, 
    image_draw: Image.Image, 
    output_path: str, 
    default_filename: str = "result.mmd"
) -> None:
    """
    保存OCR结果到文件，处理图像引用和其他内容
    
    Args:
        outputs: OCR输出文本
        image_draw: 用于绘制边界框的图像
        output_path: 输出路径
        default_filename: 默认文件名
    """
    # 去除可能的结束标记
    stop_str = '<｜end▁of▁sentence｜>'
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    # 使用正则表达式匹配内容
    matches_ref, matches_images, matches_other = re_match(outputs)
    
    # 处理图像引用
    for idx, a_match_image in enumerate(tqdm(matches_images, desc="处理图像引用")):
        outputs = outputs.replace(a_match_image, f'![](images/{idx}.jpg)\n')
    
    # 处理其他匹配项
    for idx, a_match_other in enumerate(tqdm(matches_other, desc="处理其他内容")):
        outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    # 生成输出文件名
    try:
        output_filename = generate_output_filename(output_path, default_filename)
    except Exception as e:
        print(f"生成输出文件名时出错: {e}，使用默认文件名")
        output_filename = default_filename
    
    # 保存处理后的文本
    output_file_path = os.path.join(output_path, output_filename)
    with open(output_file_path, 'w', encoding='utf-8') as afile:
        afile.write(outputs)
    
    # 处理图像并保存带边界框的结果
    if matches_ref:
        result_image = process_image_with_refs(image_draw, matches_ref, output_path)
        result_image.save(f"{output_path}/result_with_boxes.jpg")
    
    print(f"OCR结果已保存到: {output_file_path}")


def clean_output_text(text: str) -> str:
    """
    清理输出文本，去除重复内容和格式问题
    
    Args:
        text: 原始文本
        
    Returns:
        str: 清理后的文本
    """
    # 去除重复的单词或短语
    # 例如：将"Background Background Background"替换为"Background"
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text)
    
    # 去除多余的标点符号
    text = re.sub(r'([.,;:!?])\1+', r'\1', text)
    
    # 去除多余的数字前缀
    text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
    
    # 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text