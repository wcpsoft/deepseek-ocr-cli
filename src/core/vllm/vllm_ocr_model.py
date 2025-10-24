#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek OCR模型vLLM实现
"""

import math
from typing import List, Optional, Tuple, Mapping, Union, Sequence, Set, Iterable
import torch
import torch.nn as nn

# 第三方库导入
from transformers import BatchFeature
from addict import Dict

# vLLM相关导入（延迟导入，避免在不支持的平台上报错）
try:
    from vllm.config import VllmConfig
    from vllm.model_executor import SamplingMetadata
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype, cached_tokenizer_from_config
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                        MultiModalKwargs, NestedTensors)
    from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                    ImageSize, MultiModalDataItems)
    from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                            BaseProcessingInfo, PromptReplacement,
                                            PromptUpdate)
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
    from vllm.sequence import IntermediateTensors
    from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
    from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                        init_vllm_registered_model, maybe_prefix,
                        merge_multimodal_embeddings)
    VLLM_AVAILABLE = True
except ImportError:
    # 在不支持vLLM的平台上设置占位符
    VllmConfig = object
    SamplingMetadata = object
    QuantizationConfig = object
    set_default_torch_dtype = None
    cached_tokenizer_from_config = None
    MULTIMODAL_REGISTRY = None
    MultiModalDataDict = object
    MultiModalFieldConfig = object
    MultiModalKwargs = object
    NestedTensors = object
    ImageEmbeddingItems = object
    ImageProcessorItems = object
    ImageSize = object
    MultiModalDataItems = object
    BaseMultiModalProcessor = object
    BaseProcessingInfo = object
    PromptReplacement = object
    PromptUpdate = object
    BaseDummyInputsBuilder = object
    IntermediateTensors = object
    MultiModalEmbeddings = object
    SupportsMultiModal = object
    SupportsPP = object
    AutoWeightsLoader = object
    WeightsMapper = object
    flatten_bn = lambda x: x
    init_vllm_registered_model = lambda *args, **kwargs: None
    maybe_prefix = lambda prefix, name: name
    merge_multimodal_embeddings = lambda *args, **kwargs: None
    VLLM_AVAILABLE = False

# 项目内部导入
from src.core.process.image_process import (
    DeepseekOCRProcessor, count_tiles)
from src.core.deepencoder.sam_vary_sdpa import build_sam_vit_b
from src.core.deepencoder.clip_sdpa import build_clip_l
from src.core.deepencoder.build_linear import MlpProjector
from src.core.models.base_ocr_model import BaseDeepseekOCRForCausalLM
from src.core.deepseek_ocr_config import DeepseekVLV2Config

# 配置导入
from src.core.config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS, PROMPT

# 常量定义
_IMAGE_TOKEN = "<image>"


class DeepseekOCRProcessingInfo(BaseProcessingInfo):
    """
    DeepSeek OCR处理信息类
    用于处理OCR相关的多模态信息
    """

    def get_hf_config(self):
        """
        获取HuggingFace配置
        
        Returns:
            HuggingFace配置对象
        """
        # 延迟导入配置类以避免循环导入
        from src.core.deepseek_ocr_config import DeepseekVLV2Config
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        """
        获取HuggingFace处理器
        
        Args:
            **kwargs: 处理器参数
            
        Returns:
            HuggingFace处理器对象
        """
        return self.ctx.get_hf_processor(DeepseekOCRProcessor, **kwargs)

    def get_supported_mm_limits(self) -> dict:
        """
        获取支持的多模态限制
        
        Returns:
            多模态限制映射
        """
        return {"image": None}

    def get_num_image_tokens(self,
                            *,
                            image_width: int,
                            image_height: int,
                            cropping: bool = True) -> int:
        """
        计算图像token数量
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            cropping: 是否进行裁剪
            
        Returns:
            图像token数量
        """
        # image_size = hf_processor.image_size
        # patch_size = hf_processor.patch_size
        # downsample_ratio = hf_processor.downsample_ratio

        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 640 and image_height <= 640:
                crop_ratio = [1, 1]
            else:
                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(image_width, image_height, image_size=IMAGE_SIZE)
                
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles >1 or num_height_tiles>1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0


        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        """
        获取具有最多特征的图像尺寸
        
        Returns:
            图像尺寸对象
        """

        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024*2, height=1024*2)
        return ImageSize(width=640*2, height=640*2)


class DeepseekOCRDummyInputsBuilder(
        BaseDummyInputsBuilder[DeepseekOCRProcessingInfo]):
    """
    DeepSeek OCR虚拟输入构建器
    用于构建测试用的虚拟输入
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        获取虚拟文本
        
        Args:
            mm_counts: 多模态计数映射
            
        Returns:
            虚拟文本字符串
        """
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        """
        获取虚拟多模态数据
        
        Args:
            seq_len: 序列长度
            mm_counts: 多模态计数映射
            
        Returns:
            虚拟多模态数据字典
        """
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        if '<image>' in PROMPT:
            return {
                "image":
                DeepseekOCRProcessor().tokenize_with_images(images = self._get_dummy_images(width=max_image_size.width,
                                    height=max_image_size.height,
                                    num_images=num_images), bos=True, eos=True, cropping=CROP_MODE)
            }
        else:
            return {
                "image": []
            }


class DeepseekOCRMultiModalProcessor(
        BaseMultiModalProcessor[DeepseekOCRProcessingInfo]):
    """
    DeepSeek OCR多模态处理器
    处理OCR相关的多模态输入
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        调用HuggingFace处理器
        
        Args:
            prompt: 提示文本
            mm_data: 多模态数据
            mm_kwargs: 多模态参数
            
        Returns:
            批处理特征对象
        """
        
        # print(mm_data)
        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(prompt,
                                        add_special_tokens=True,
                                        return_tensors="pt")

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """
        获取多模态字段配置
        
        Args:
            hf_inputs: HuggingFace输入
            hf_processor_mm_kwargs: HuggingFace处理器多模态参数
            
        Returns:
            多模态字段配置映射
        """
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            images_spatial_crop=MultiModalFieldConfig.batched("image"),
            # image_embeds=MultiModalFieldConfig.batched("image2"),
            images_crop=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        """
        获取提示更新序列
        
        Args:
            mm_items: 多模态数据项
            hf_processor_mm_kwargs: HuggingFace处理器多模态参数
            out_mm_kwargs: 输出多模态参数
            
        Returns:
            提示更新序列
        """
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))



            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:

                
                width = images[0][-1][0][0]
                height = images[0][-1][0][1]

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=width,
                    image_height=height,
                    # flag = True,
                    cropping=CROP_MODE,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        """
        缓存应用HuggingFace处理器
        
        Args:
            prompt: 提示文本或token列表
            mm_data_items: 多模态数据项
            hf_processor_mm_kwargs: HuggingFace处理器多模态参数
            
        Returns:
            处理结果元组
        """
        # The processor logic is different for len(images) <= 2 vs > 2
        # Since the processing cache assumes that the processor output is
        # invariant of how many images are passed per prompt, we only
        # perform caching for the most common case
        if mm_data_items.get_count("image", strict=False) > 2:
            # This code path corresponds to the cache being disabled
            return self._apply_hf_processor_main(
                prompt=prompt,
                mm_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                enable_hf_prompt_update=True,
            )

        return super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder)
class DeepseekOCRForCausalLM(BaseDeepseekOCRForCausalLM):
    """
    DeepSeek OCR因果语言模型
    支持多模态输入的因果语言模型实现
    """

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "language.": "language_model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """
        初始化DeepSeek OCR因果语言模型
        
        Args:
            vllm_config: vLLM配置对象
            prefix: 模型前缀
        """
        super().__init__()

        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config


        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
    
        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")




        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if self.text_config.topk_method == "noaux_tc":
            architectures = ["DeepseekV3ForCausalLM"]
        elif not self.text_config.use_mla:
            architectures = ["DeepseekForCausalLM"]
        else:
            architectures = ["DeepseekV2ForCausalLM"]

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.text_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=architectures,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)



    def _parse_and_validate_image_input(
            self, **kwargs: object):
        """
        解析和验证图像输入
        
        Args:
            **kwargs: 输入参数
            
        Returns:
            解析后的图像输入或None
        """
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)


        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                f"Got type: {type(pixel_values)}")

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                f"Got type: {type(images_spatial_crop)}")
            
            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image crop. "
                                f"Got type: {type(images_crop)}")

            return [pixel_values, images_crop, images_spatial_crop]


        raise AssertionError("This line should be unreachable.")
    


    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        """
        将像素值转换为嵌入向量
        
        Args:
            pixel_values: 像素值张量
            images_crop: 图像裁剪张量
            images_spatial_crop: 图像空间裁剪张量
            
        Returns:
            嵌入向量张量
        """

        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch = []


        # print(type(images_crop))

        # print(pixel_values.shape)


        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # with torch.set_grad_enabled(False):
                # 根据设备类型决定是否使用bfloat16
                patches = images_crop[jdx][0]
                image_ori = pixel_values[jdx]
                
                # 检查设备类型，MPS上避免使用bfloat16以确保兼容性
                if patches.device.type != "mps":
                    patches = patches.to(torch.bfloat16)
                if image_ori.device.type != "mps":
                    image_ori = image_ori.to(torch.bfloat16)

                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:  # if all values = 0, no crop
                    # P, C, H, W = patches.shape
                    # crop_flag = 1
                    local_features_1 = self.sam_model(patches)
                    #TODO del patches 
                    # torch.compiler.cudagraph_mark_step_begin()
                    local_features_2 = self.vision_model(patches, local_features_1)  


                    local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    local_features = self.projector(local_features)


                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('PATCHES: ', local_features.shape)
                        print('=====================')

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)

                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )

                    global_features = global_features.view(-1, n_dim)


                    local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                    local_features = torch.cat(
                        [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                    )
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                
                else:
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('NO PATCHES')
                        print('=====================')

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )

                    global_features = global_features.view(-1, n_dim)

                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(
            self, image_input) -> torch.Tensor:
        """
        处理图像输入
        
        Args:
            image_input: 图像输入数据
            
        Returns:
            处理后的图像张量
        """

        # image_input: [pixel_values, images_crop, images_spatial_crop]
    
        pixel_values = image_input[0]
        images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # 检查设备类型，MPS上避免使用bfloat16以确保兼容性
        if pixel_values.device.type != "mps":
            pixel_values = pixel_values.to(torch.bfloat16)
        if images_crop.device.type != "mps":
            images_crop = images_crop.to(torch.bfloat16)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values, images_crop = images_crop,  images_spatial_crop=images_spatial_crop)

        # local_total_time = time.time() - local_start

        # print('encoder_time: ', local_total_time)
        # exit()
        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        """
        获取语言模型
        
        Returns:
            语言模型模块
        """
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        """
        获取多模态嵌入向量
        
        Args:
            **kwargs: 输入参数
            
        Returns:
            多模态嵌入向量或None
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """
        获取输入嵌入向量
        
        Args:
            input_ids: 输入ID张量
            multimodal_embeddings: 多模态嵌入向量
            
        Returns:
            输入嵌入向量张量
        """


        inputs_embeds = self.language_model.get_input_embeddings(input_ids)


        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.image_token_id)
            # print(len(multimodal_embeddings))
            # print(input_ids.shape)
            # print(type(inputs_embeds))
            # print(inputs_embeds.shape)
            
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object):
        """
        前向传播函数
        
        Args:
            input_ids: 输入ID张量
            positions: 位置张量
            intermediate_tensors: 中间张量
            inputs_embeds: 输入嵌入向量
            **kwargs: 其他参数
            
        Returns:
            前向传播结果
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                    vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """
        计算logits
        
        Args:
            hidden_states: 隐藏状态张量
            sampling_metadata: 采样元数据
            
        Returns:
            logits张量或None
        """
        return self.language_model.compute_logits(hidden_states,
                                                sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """
        加载模型权重
        
        Args:
            weights: 权重元组迭代器
        
        Returns:
            已加载的权重名称集合
        """
        processed_weights = []
        
        for name, tensor in weights:
            if 'sam_model' in name or 'vision_model' in name or 'projector' in name or 'image_newline' in name or 'view_seperator' in name:
                new_name = name.replace('model.', '', 1)
            else:
                new_name = 'language.' + name

            processed_weights.append((new_name, tensor))
        
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(processed_weights, mapper=self.hf_to_vllm_mapper)