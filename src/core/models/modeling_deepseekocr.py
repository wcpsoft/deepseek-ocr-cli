import copy
import logging
import math
import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# 导入配置类
from .deepencoder import build_clip_l, build_sam_vit_b
from .modeling_deepseekv2 import DeepseekV2ForCausalLM, DeepseekV2Model

# 设置日志
logger = logging.getLogger(__name__)

# =================== 工具函数 ===================


def dynamic_preprocess(
    image: Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 640
) -> tuple[list[Image.Image], list[int]]:
    """动态预处理图像，将大图像分割成多个小块

    Args:
        image: 输入图像
        min_num: 最小块数
        max_num: 最大块数
        image_size: 目标图像尺寸

    Returns:
        Tuple[List[Image.Image], List[int]]: 处理后的图像列表和裁剪比例
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算最佳块数
    if aspect_ratio > 1:  # 宽图
        target_width = image_size
        target_height = int(image_size / aspect_ratio)
        width_ratio = orig_width / target_width
        height_ratio = orig_height / target_height
        num_blocks = min(max(int(width_ratio * height_ratio), min_num), max_num)
        width_blocks = min(num_blocks, int(width_ratio))
        height_blocks = num_blocks // width_blocks
    else:  # 高图或正方形
        target_height = image_size
        target_width = int(image_size * aspect_ratio)
        width_ratio = orig_width / target_width
        height_ratio = orig_height / target_height
        num_blocks = min(max(int(width_ratio * height_ratio), min_num), max_num)
        height_blocks = min(num_blocks, int(height_ratio))
        width_blocks = num_blocks // height_blocks

    # 确保至少有一个块
    width_blocks = max(1, width_blocks)
    height_blocks = max(1, height_blocks)

    # 调整图像大小
    image = image.resize((target_width, target_height))

    # 如果不需要分割
    if width_blocks == 1 and height_blocks == 1:
        return [image], [1, 1]

    # 计算每个块的尺寸
    block_width = target_width // width_blocks
    block_height = target_height // height_blocks

    # 分割图像
    blocks = []
    for i in range(height_blocks):
        for j in range(width_blocks):
            left = j * block_width
            top = i * block_height
            right = left + block_width
            bottom = top + block_height

            # 处理边缘情况
            if j == width_blocks - 1:
                right = target_width
            if i == height_blocks - 1:
                bottom = target_height

            block = image.crop((left, top, right, bottom))
            blocks.append(block)

    return blocks, [width_blocks, height_blocks]


def load_image(image_path: str) -> Image.Image:
    """加载图像

    Args:
        image_path: 图像路径

    Returns:
        Image.Image: 加载的图像
    """
    return Image.open(image_path).convert("RGB")


def draw_bounding_boxes(image: Image.Image, boxes: list[tuple[int, int, int, int]], color: str = "red") -> Image.Image:
    """在图像上绘制边界框

    Args:
        image: 输入图像
        boxes: 边界框列表，每个框为(x1, y1, x2, y2)
        color: 边界框颜色

    Returns:
        Image.Image: 绘制了边界框的图像
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=color, width=2)
    return image


def extract_text_from_output(output: str) -> str:
    """从模型输出中提取文本内容

    Args:
        output: 模型原始输出

    Returns:
        str: 提取的文本内容
    """
    # 移除特殊标记
    text = re.sub(r"<\|.*?\|>", "", output)
    # 移除多余的空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def re_match(text: str) -> tuple[list[str], list[str], list[str]]:
    """使用正则表达式匹配文本中的引用、图像和其他元素

    Args:
        text: 输入文本

    Returns:
        Tuple[List[str], List[str], List[str]]: 引用列表、图像列表和其他元素列表
    """
    # 匹配引用格式 【ref】
    refs = re.findall(r"【.*?】", text)

    # 匹配图像格式 ![...](...)
    images = re.findall(r"!\[.*?\]\(.*?\)", text)

    # 匹配其他特殊格式
    others = re.findall(r"\\.*?\{.*?\}", text)

    return refs, images, others


def process_image_with_refs(image: Image.Image, refs: list[str], output_path: str) -> Image.Image:
    """处理图像中的引用

    Args:
        image: 输入图像
        refs: 引用列表
        output_path: 输出路径

    Returns:
        Image.Image: 处理后的图像
    """
    # 这里应该实现引用处理逻辑
    # 暂时返回原图像
    return image


# =================== 图像变换类 ===================


class BasicImageTransform:
    """基础图像变换类"""

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        """初始化图像变换

        Args:
            mean: 均值
            std: 标准差
            normalize: 是否归一化
        """
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """应用图像变换

        Args:
            image: 输入图像

        Returns:
            torch.Tensor: 变换后的图像张量
        """
        # 转换为张量
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW

        # 归一化
        if self.normalize:
            mean = torch.tensor(self.mean).view(3, 1, 1)
            std = torch.tensor(self.std).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor


# =================== 文本流类 ===================


class NoEOSTextStreamer:
    """不包含EOS标记的文本流"""

    def __init__(self, tokenizer, skip_prompt: bool = False, skip_special_tokens: bool = False):
        """初始化文本流

        Args:
            tokenizer: 分词器
            skip_prompt: 是否跳过提示
            skip_special_tokens: 是否跳过特殊标记
        """
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []

    def put(self, token_ids):
        """添加标记

        Args:
            token_ids: 标记ID
        """
        self.tokens.extend(token_ids)

    def end(self):
        """结束流"""

    def decode(self, token_ids):
        """解码标记

        Args:
            token_ids: 标记ID

        Returns:
            str: 解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=self.skip_special_tokens)


# =================== 配置类 ===================


class DeepseekOCRConfig(PretrainedConfig):
    """DeepseekOCR模型配置类"""

    model_type = "deepseek_ocr"

    def __init__(
        self,
        # DeepseekV2Config需要的属性
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="gready",
        n_group=None,
        topk_group=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_mla=True,
        # OCR特定属性
        vision_config=None,
        projector_config=None,
        **kwargs,
    ):
        # DeepseekV2Config需要的属性
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = float(rms_norm_eps)
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_mla = use_mla

        # OCR特定属性
        self.vision_config = vision_config or {}
        self.projector_config = projector_config or {}

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """从字典创建配置对象，处理language_config中的vocab_size

        Args:
            config_dict: 配置字典
            **kwargs: 其他参数

        Returns:
            DeepseekOCRConfig: 配置对象
        """
        # 检查是否有language_config，并从中提取vocab_size
        if "language_config" in config_dict and "vocab_size" in config_dict["language_config"]:
            # 如果language_config中有vocab_size，使用它
            config_dict["vocab_size"] = config_dict["language_config"]["vocab_size"]
            logger.info(f"从language_config中提取vocab_size: {config_dict['vocab_size']}")

        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """从预训练模型加载配置，处理language_config中的vocab_size

        Args:
            model_path: 模型路径
            **kwargs: 其他参数

        Returns:
            DeepseekOCRConfig: 配置对象
        """
        # 先调用父类方法加载配置
        config = super().from_pretrained(model_path, **kwargs)

        # 检查是否有language_config，并从中提取vocab_size
        if hasattr(config, "language_config") and config.language_config and "vocab_size" in config.language_config:
            # 如果language_config中有vocab_size，使用它
            old_vocab_size = config.vocab_size
            config.vocab_size = config.language_config["vocab_size"]
            logger.info(f"从language_config中更新vocab_size: {old_vocab_size} -> {config.vocab_size}")

        return config


# =================== 模型类 ===================


class DeepseekOCRModel(DeepseekV2Model):
    """DeepSeek OCR模型类"""

    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekOCRConfig):
        """初始化模型

        Args:
            config: 配置对象
        """
        super().__init__(config)

        # 初始化视觉编码器
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        # 初始化投影器
        n_embed = 1280
        self.projector = MlpProjector(dict(projector_type="linear", input_dim=2048, n_embed=n_embed))

        # 初始化图像特殊标记
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple | BaseModelOutputWithPast:
        """前向传播

        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_values: 过去的键值对
            inputs_embeds: 输入嵌入
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力
            output_hidden_states: 是否输出隐藏状态
            images: 图像张量
            images_seq_mask: 图像序列掩码
            images_spatial_crop: 图像空间裁剪信息
            return_dict: 是否返回字典

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: 输出结果
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        sam_model = getattr(self, "sam_model", None)
        vision_model = getattr(self, "vision_model", None)

        if (
            sam_model is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
            and images[0] is not None
            and torch.sum(images[0][1]).item() != 0
        ):
            idx = 0

            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]
                image_ori = image[1]

                with torch.no_grad():
                    if torch.sum(patches).item() != 0:
                        # 处理局部特征
                        local_features_1 = sam_model(patches)
                        local_features_2 = vision_model(patches, local_features_1)
                        local_features = torch.cat(
                            (local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1
                        )
                        local_features = self.projector(local_features)

                        # 处理全局特征
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1
                        )
                        global_features = self.projector(global_features)

                        logger.debug("图像特征提取完成:")
                        logger.debug(f"  - 全局特征张量形状: {global_features.shape}")
                        logger.debug(f"  - 局部特征张量形状: {local_features.shape}")

                        _, hw, n_dim = global_features.shape
                        h = w = int(hw**0.5)

                        _, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2**0.5)

                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat(
                            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )
                        global_features = global_features.view(-1, n_dim)

                        local_features = (
                            local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                            .permute(0, 2, 1, 3, 4)
                            .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                        )
                        local_features = torch.cat(
                            [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)],
                            dim=1,
                        )
                        local_features = local_features.view(-1, n_dim2)

                        global_local_features = torch.cat(
                            [local_features, global_features, self.view_seperator[None, :]], dim=0
                        )
                    else:
                        # 只有全局特征
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1
                        )
                        global_features = self.projector(global_features)

                        logger.debug("图像特征提取完成:")
                        logger.debug(f"  - 全局特征张量形状: {global_features.shape}")
                        logger.debug("  - 无局部特征")

                        _, hw, n_dim = global_features.shape
                        h = w = int(hw**0.5)

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat(
                            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )
                        global_features = global_features.view(-1, n_dim)

                        global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                    images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    # 确保掩码在正确的设备上
                    device = inputs_embeds.device
                    images_seq_mask_device = images_seq_mask[idx].unsqueeze(-1).to(device)
                    inputs_embeds[idx].masked_scatter_(images_seq_mask_device, images_in_this_batch)

                idx += 1

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class DeepseekOCRForCausalLM(DeepseekV2ForCausalLM):
    """DeepSeek OCR用于因果语言建模的类"""

    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekOCRConfig):
        """初始化模型

        Args:
            config: 配置对象
        """
        super().__init__(config)
        self.model = DeepseekOCRModel(config)

        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple | CausalLMOutputWithPast:
        """前向传播

        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_values: 过去的键值对
            inputs_embeds: 输入嵌入
            labels: 标签
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力
            output_hidden_states: 是否输出隐藏状态
            images: 图像张量
            images_seq_mask: 图像序列掩码
            images_spatial_crop: 图像空间裁剪信息
            return_dict: 是否返回字典

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: 输出结果
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 执行模型前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[list] = None,
        images_seq_mask: Optional[torch.Tensor] = None,
        images_spatial_crop: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ):
        """生成文本

        Args:
            input_ids: 输入token ID
            images: 图像列表
            images_seq_mask: 图像序列掩码
            images_spatial_crop: 图像空间裁剪信息
            max_new_tokens: 最大新token数量
            **kwargs: 其他生成参数

        Returns:
            torch.Tensor: 生成的token ID
        """
        # 直接调用父类的generate方法
        return super().generate(
            input_ids=input_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def _get_image_embeds(self, pixel_values, images_crop):
        """获取图像嵌入

        Args:
            pixel_values: 图像像素值列表
            images_crop: 图像裁剪信息列表

        Returns:
            torch.Tensor: 图像嵌入
        """
        # 如果没有图像，返回None
        if not pixel_values:
            return None

        # 处理每个图像
        all_image_features = []

        for i, (pixel_value, crop_info) in enumerate(zip(pixel_values, images_crop)):
            try:
                # 确保pixel_value是4D张量 [batch_size, channels, height, width]
                if pixel_value.dim() == 3:
                    # 如果是3D，添加batch维度
                    pixel_value = pixel_value.unsqueeze(0)

                # 获取视觉特征
                try:
                    # 使用self.model.visual_encoder而不是self.model.vision_model
                    # VitModel.forward()需要两个参数：x和patch_embeds
                    # 对于patch_embeds，我们可以传入None，因为CLIPVisionEmbeddings会处理它
                    image_features = self.model.visual_encoder(pixel_value, None)
                    print(f"image_features shape: {image_features.shape if image_features is not None else None}")
                except Exception as e:
                    print(f"获取视觉特征时出错: {e}")
                    image_features = None

                # 如果没有获取到特征，跳过此图像
                if image_features is None:
                    print("无法获取视觉特征，跳过此图像")
                    continue

                # 处理高分辨率特征
                if hasattr(self.model, "high_visual_encoder") and self.model.high_visual_encoder is not None:
                    try:
                        # 确保高分辨率图像也是4D张量
                        hr_pixel_value = pixel_value
                        if hr_pixel_value.dim() == 3:
                            hr_pixel_value = hr_pixel_value.unsqueeze(0)

                        hr_image_features = self.model.high_visual_encoder(hr_pixel_value)
                        print(
                            f"hr_image_features shape: {hr_image_features.shape if hr_image_features is not None else None}"
                        )
                    except Exception as e:
                        print(f"获取高分辨率特征时出错: {e}")
                        hr_image_features = None

                    # 确保两个张量的维度匹配
                    if (
                        hr_image_features is not None
                        and isinstance(image_features, torch.Tensor)
                        and isinstance(hr_image_features, torch.Tensor)
                    ):
                        try:
                            # 如果维度不匹配，尝试调整
                            if image_features.dim() != hr_image_features.dim():
                                # 如果hr_image_features是4D而image_features是3D，将hr_image_features展平
                                if hr_image_features.dim() == 4 and image_features.dim() == 3:
                                    # 将4D张量[batch, channels, height, width]展平为[batch, height*width, channels]
                                    batch_size, channels, height, width = hr_image_features.shape
                                    hr_image_features = hr_image_features.view(batch_size, channels, -1).permute(
                                        0, 2, 1
                                    )
                                    # 然后与image_features在序列长度维度上连接
                                    if hr_image_features.shape[1] > 0 and image_features.shape[1] > 0:
                                        # 确保特征维度匹配
                                        if image_features.shape[-1] != hr_image_features.shape[-1]:
                                            # 如果维度不匹配，只使用image_features
                                            print(
                                                f"特征维度不匹配: image_features.shape[-1]={image_features.shape[-1]}, hr_image_features.shape[-1]={hr_image_features.shape[-1]}"
                                            )
                                        else:
                                            image_features = torch.cat([image_features, hr_image_features], dim=1)
                                # 如果image_features是4D而hr_image_features是3D，将image_features展平
                                elif image_features.dim() == 4 and hr_image_features.dim() == 3:
                                    # 将4D张量[batch, channels, height, width]展平为[batch, height*width, channels]
                                    batch_size, channels, height, width = image_features.shape
                                    image_features = image_features.view(batch_size, channels, -1).permute(0, 2, 1)
                                    # 然后与hr_image_features在序列长度维度上连接
                                    if image_features.shape[1] > 0 and hr_image_features.shape[1] > 0:
                                        # 确保特征维度匹配
                                        if image_features.shape[-1] != hr_image_features.shape[-1]:
                                            # 如果维度不匹配，只使用hr_image_features
                                            print(
                                                f"特征维度不匹配: image_features.shape[-1]={image_features.shape[-1]}, hr_image_features.shape[-1]={hr_image_features.shape[-1]}"
                                            )
                                            image_features = hr_image_features
                                        else:
                                            image_features = torch.cat([image_features, hr_image_features], dim=1)
                                else:
                                    # 如果无法处理维度不匹配，只使用image_features
                                    print(
                                        f"无法处理维度不匹配: image_features.dim()={image_features.dim()}, hr_image_features.dim()={hr_image_features.dim()}"
                                    )
                            else:
                                # 如果维度匹配，检查特征维度是否一致
                                if image_features.shape[-1] != hr_image_features.shape[-1]:
                                    # 如果特征维度不匹配，只使用image_features
                                    print(
                                        f"特征维度不匹配: image_features.shape[-1]={image_features.shape[-1]}, hr_image_features.shape[-1]={hr_image_features.shape[-1]}"
                                    )
                                else:
                                    # 直接连接
                                    image_features = torch.cat([image_features, hr_image_features], dim=1)
                        except Exception as e:
                            print(f"合并特征时出错: {e}")
                            # 如果合并失败，只使用image_features

                # 应用投影器
                if hasattr(self.model, "projector") and self.model.projector is not None:
                    try:
                        # 检查投影器期望的输入维度
                        if hasattr(self.model.projector, "cfg") and hasattr(self.model.projector.cfg, "input_dim"):
                            expected_dim = self.model.projector.cfg.input_dim
                            actual_dim = image_features.shape[-1]

                            if actual_dim != expected_dim:
                                print(f"投影器输入维度不匹配: 期望={expected_dim}, 实际={actual_dim}")

                                # 如果实际维度大于期望维度，截断
                                if actual_dim > expected_dim:
                                    image_features = image_features[..., :expected_dim]
                                # 如果实际维度小于期望维度，填充
                                else:
                                    pad_size = expected_dim - actual_dim
                                    padding = torch.zeros(
                                        (*image_features.shape[:-1], pad_size),
                                        dtype=image_features.dtype,
                                        device=image_features.device,
                                    )
                                    image_features = torch.cat([image_features, padding], dim=-1)

                        # 应用投影器
                        image_features = self.model.projector(image_features)
                    except Exception as e:
                        print(f"应用投影器时出错: {e}")
                        # 如果投影器失败，跳过此图像
                        continue

                # 调整形状
                if isinstance(image_features, torch.Tensor):
                    if image_features.dim() == 3:
                        # [batch_size, seq_len, hidden_size] -> [seq_len, hidden_size]
                        image_features = image_features.squeeze(0)
                    all_image_features.append(image_features)
            except Exception as e:
                # 如果处理单个图像失败，记录错误并继续
                print(f"处理图像 {i} 时出错: {e}")
                continue

        # 合并所有图像特征
        if all_image_features:
            try:
                # 检查所有特征张量的形状是否兼容
                if len(all_image_features) > 1:
                    # 获取第一个特征的形状作为参考
                    ref_shape = all_image_features[0].shape
                    for i, feat in enumerate(all_image_features[1:], 1):
                        if feat.shape != ref_shape:
                            print(f"警告: 图像特征 {i} 的形状 {feat.shape} 与参考形状 {ref_shape} 不匹配")
                            # 尝试调整形状或跳过不匹配的特征
                            if feat.dim() != ref_shape[0] or feat.shape[-1] != ref_shape[-1]:
                                print(f"跳过不匹配的特征 {i}")
                                all_image_features[i] = None

                # 过滤掉None值
                valid_features = [f for f in all_image_features if f is not None]

                if valid_features:
                    return torch.cat(valid_features, dim=0)
                else:
                    print("没有有效的图像特征可以合并")
                    return None
            except Exception as e:
                print(f"合并图像特征时出错: {e}")
                return None
        else:
            return None


class MlpProjector(nn.Module):
    """MLP投影器"""

    def __init__(self, cfg):
        """初始化投影器

        Args:
            cfg: 配置对象或字典
        """
        super().__init__()
        self.cfg = cfg

        # 如果cfg是字典，创建一个简单的对象来访问属性
        if isinstance(cfg, dict):

            class SimpleConfig:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)

                def get(self, key, default=None):
                    return getattr(self, key, default)

            cfg = SimpleConfig(cfg)

        # 初始化modules变量
        modules = None

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "normlayer_downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.LayerNorm(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio),
                nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio),
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            channel_div = cfg.get("channel_div", 0.5)
            self.high_up_proj = nn.Linear(cfg.input_dim[0], int(cfg.n_embed * channel_div))
            self.low_up_proj = nn.Linear(cfg.input_dim[1], cfg.n_embed - int(cfg.n_embed * channel_div))

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed // 2, cfg.n_embed // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)
            self.low_layers = copy.deepcopy(modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.get("token_pooling", False):
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        if cfg.get("conv_fusion_high_low_features", False):
            self.fusion_layer = nn.Linear(cfg.input_dim, cfg.input_dim)

        # 只有在需要时才设置self.layers
        if cfg.projector_type != "low_high_split_mlp_gelu":
            self.layers = modules

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量
            *args: 其他位置参数
            **kwargs: 其他关键字参数

        Returns:
            torch.Tensor: 投影后的张量
        """
        # 如果cfg是字典，创建一个简单的对象来访问属性
        if isinstance(self.cfg, dict):

            class SimpleConfig:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)

                def get(self, key, default=None):
                    return getattr(self, key, default)

            cfg = SimpleConfig(self.cfg)
        else:
            cfg = self.cfg

        if cfg.projector_type == "identity":
            return x

        elif cfg.projector_type == "linear":
            return self.layers(x)

        elif cfg.projector_type == "mlp_gelu":
            return self.layers(x)

        elif cfg.projector_type in ["normlayer_downsample_mlp_gelu", "downsample_mlp_gelu"]:
            return self.layers(x)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            # 处理高低分辨率特征
            if isinstance(x, (list, tuple)) and len(x) == 2:
                high_features, low_features = x
                high_proj = self.high_up_proj(high_features)
                low_proj = self.low_up_proj(low_features)
                features = torch.cat([high_proj, low_proj], dim=-1)
                return self.layers(features)
            else:
                return self.layers(x)

        elif cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            # 处理混合特征
            if isinstance(x, (list, tuple)) and len(x) == 2:
                high_features, low_features = x
                high_proj = self.high_up_proj(high_features)
                low_proj = self.low_up_proj(low_features)
                features = torch.cat([high_proj, low_proj], dim=-1)
                return self.layers(features)
            else:
                return self.layers(x)

        elif cfg.projector_type == "low_high_split_mlp_gelu":
            # 处理低高分离特征
            if isinstance(x, (list, tuple)) and len(x) == 2:
                high_features, low_features = x
                high_proj = self.high_up_proj(high_features)
                low_proj = self.low_up_proj(low_features)
                high_out = self.high_layers(high_proj)
                low_out = self.low_layers(low_proj)
                return torch.cat([high_out, low_out], dim=-1)
            else:
                return self.layers(x)

        else:
            return self.layers(x)


# =================== 推理函数 ===================


def infer(
    model: DeepseekOCRForCausalLM,
    tokenizer: PreTrainedTokenizer,
    image_path: str,
    prompt: str = "请识别图像中的文字",
    output_path: str = "./output",
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
    no_repeat_ngram_size: int = 3,
    use_cache: bool = True,
    eval_mode: bool = True,
    save_results: bool = True,
    test_compress: bool = False,
    crop_mode: bool = True,
    image_file: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """执行OCR推理

    Args:
        model: OCR模型
        tokenizer: 分词器
        image_path: 图像路径
        prompt: 提示文本
        output_path: 输出路径
        temperature: 采样温度
        max_new_tokens: 最大新token数
        no_repeat_ngram_size: 不重复n-gram大小
        use_cache: 是否使用缓存
        eval_mode: 是否为评估模式
        save_results: 是否保存结果
        test_compress: 是否测试压缩
        crop_mode: 是否使用裁剪模式
        image_file: 图像文件名
        device: 计算设备

    Returns:
        Optional[str]: 识别结果文本
    """
    # 设置设备 - 强制使用CPU以避免MPS内存不足问题
    device = torch.device("cpu")

    # 确保模型在正确的设备上
    model = model.to(device)

    # 加载图像
    image = load_image(image_path)

    # 获取模型配置
    image_size = model.config.image_size
    base_size = model.config.base_size
    patch_size = model.config.patch_size
    downsample_ratio = model.config.downsample_ratio

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 准备对话
    if image_file:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{prompt}",
                "images": [f"{image_file}"],
            },
            {"role": "ffffffffffffffff", "content": ""},
        ]
    elif prompt:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{prompt}",
            },
            {"role": "ffffffffffffffff", "content": ""},
        ]
    else:
        raise ValueError("prompt is none!")

    # 准备输入
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

    # CPU设备使用float32而不是bfloat16
    dtype = torch.float32

    # 处理图像
    if crop_mode and (image.size[0] > 640 or image.size[1] > 640):
        # 动态预处理大图像
        images_crop_raw, crop_ratio = dynamic_preprocess(image)
        width_crop_num, height_crop_num = crop_ratio
    else:
        images_crop_raw = []
        width_crop_num, height_crop_num = 1, 1

    # 处理全局视图
    global_view = image.copy()
    if global_view.size[0] < base_size or global_view.size[1] < base_size:
        # 填充到目标大小
        ratio = base_size / max(global_view.size[0], global_view.size[1])
        new_size = (int(global_view.size[0] * ratio), int(global_view.size[1] * ratio))
        global_view = global_view.resize(new_size)

    # 创建填充图像
    padded_image = Image.new("RGB", (base_size, base_size), (128, 128, 128))
    offset = ((base_size - global_view.size[0]) // 2, (base_size - global_view.size[1]) // 2)
    padded_image.paste(global_view, offset)

    # 转换为张量
    global_tensor = image_transform(padded_image).to(dtype)

    # 处理裁剪图像
    crop_tensors = []
    for crop_img in images_crop_raw:
        crop_tensor = image_transform(crop_img).to(dtype)
        crop_tensors.append(crop_tensor)

    # 如果没有裁剪图像，创建一个空的张量
    if not crop_tensors:
        crop_tensors = [torch.zeros_like(global_tensor)]

    # 合并裁剪图像
    if crop_tensors:
        crops_tensor = torch.stack(crop_tensors, dim=0)
    else:
        crops_tensor = torch.zeros((1, 3, base_size, base_size), dtype=dtype)

    # 准备输入ID
    # 根据template，image_token_id应该固定为128815
    image_token = "<image>"
    image_token_id = 128815  # 使用与template一致的固定值

    # 检查token是否在词汇表中，以及ID是否超出范围
    if image_token_id >= tokenizer.vocab_size:
        logger.warning(f"图像token ID {image_token_id} 超出词汇表范围 {tokenizer.vocab_size}")
        logger.warning("需要在模型适配器中扩展嵌入层大小")

    logger.info(f"使用固定的图像token ID: {image_token_id}, 词汇表大小: {tokenizer.vocab_size}")

    # 简化的文本编码 - 实际应该使用tokenizer
    prompt_tokens = [1]  # 假设的BOS token

    # 添加图像token
    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

    # 添加全局图像token
    image_tokens = [image_token_id] * (num_queries_base * num_queries_base + 1)
    prompt_tokens.extend(image_tokens)

    # 添加裁剪图像token
    if width_crop_num > 1 or height_crop_num > 1:
        crop_image_tokens = [image_token_id] * (num_queries * num_queries * width_crop_num * height_crop_num + 1)
        prompt_tokens.extend(crop_image_tokens)

    # 转换为张量
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # 创建图像序列掩码
    images_seq_mask = torch.zeros(len(prompt_tokens), dtype=torch.bool).to(device)
    images_seq_mask[1 : len(image_tokens) + 1] = True  # 标记图像token

    # 创建图像空间裁剪信息
    images_spatial_crop = torch.tensor([[width_crop_num, height_crop_num]], dtype=torch.long).to(device)

    # 确保所有张量都在正确的设备上
    global_tensor = global_tensor.unsqueeze(0).to(device)
    crops_tensor = crops_tensor.to(device)

    # 准备图像对
    images = [(crops_tensor, global_tensor)]

    # 生成结果
    with torch.no_grad():
        # CPU设备使用float32，不支持bfloat16
        with torch.autocast(device_type="cpu", dtype=torch.float32):
            output_ids = model.generate(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask.unsqueeze(0),
                images_spatial_crop=images_spatial_crop,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 2,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                use_cache=use_cache,
                do_sample=True,
            )

    # 解码输出
    if eval_mode:
        # 跳过输入部分，只取生成的部分
        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)

        # 提取文本内容
        result_text = extract_text_from_output(output_text)

        # 保存结果
        if save_results:
            with open(f"{output_path}/result.mmd", "w", encoding="utf-8") as f:
                f.write(result_text)

            # 保存带边界框的图像
            result_image = draw_bounding_boxes(image.copy(), [])
            result_image.save(f"{output_path}/result_with_boxes.jpg")

        return result_text

    return None
