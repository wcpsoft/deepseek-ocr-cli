# DeepSeek-OCR 形状不匹配修复说明

## 问题描述

在MPS设备上运行DeepSeek-OCR时，遇到了以下错误：
1. `RuntimeError: shape '[1, 1, 256, 1280]' is invalid for input of size 327680`
2. `RuntimeError: Invalid buffer size: 182.01 GB`

## 修复内容

### 1. 修复形状不匹配问题

在 `src/core/model_initializer.py` 中修复了图像特征处理中的形状不匹配问题：

```python
# 修复前
image_embeds = self.projector(pixel_values)
image_embeds = image_embeds.view(bs, -1, self.config.hidden_size)

# 修复后
image_embeds = self.projector(pixel_values)
# 检查实际形状并调整
if image_embeds.dim() == 4:
    # 如果是4D张量 [batch, channels, height, width]，需要转换为3D
    bs, channels, height, width = image_embeds.shape
    image_embeds = image_embeds.view(bs, channels, -1).transpose(1, 2)
elif image_embeds.dim() == 3 and image_embeds.shape[1] != self.config.hidden_size:
    # 如果是3D张量但中间维度不是hidden_size，需要调整
    bs, seq_len, features = image_embeds.shape
    image_embeds = image_embeds.transpose(1, 2)  # 转换维度
```

### 2. 添加输入数据结构兼容性

修复了处理数据时的索引问题，确保与原始仓库的数据结构兼容：

```python
# 根据原始仓库的数据结构正确提取元素（索引0-6）
input_ids = processed_data[0][0]
pixel_values = processed_data[0][1]
images_crop = processed_data[0][2]
images_seq_mask = processed_data[0][3]
images_spatial_crop = processed_data[0][4]
num_image_tokens = processed_data[0][5]
image_shapes = processed_data[0][6]
```

## 测试建议

### 在MPS设备上测试

由于MPS设备内存限制，建议使用以下环境变量：

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 ./dev/run.sh samples/4.pdf output --mode transformers
```

### 在CUDA设备上测试

在有足够内存的NVIDIA GPU上，代码应该可以正常运行：

```bash
./dev/run.sh samples/4.pdf output --mode transformers
```

或者使用测试脚本：

```bash
python tests/test_cuda_inference.py
```

## 预期结果

在CUDA设备上，应该能够正确识别PDF内容并生成完整的Markdown文件，例如：

```
以下的图片输出了什么？
Test Image[特别说明：这部分PDF源文件是图片]

答案：Test Image
这是一个公式x1 + x2 = 3
```

## 注意事项

1. MPS设备由于内存限制，可能无法处理大型文档
2. 在CUDA设备上，确保有足够的GPU内存（建议8GB以上）
3. 如果仍有内存问题，可以考虑降低max_new_tokens参数
4. 确保已安装所有必要的依赖项，特别是对于CUDA设备