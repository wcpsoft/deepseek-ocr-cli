#!/usr/bin/env python3
"""
DeepSeek OCR模型vLLM处理器实现
"""

import math
from collections.abc import Mapping, Sequence

# 第三方库导入
from transformers import BatchFeature

# vLLM相关导入（延迟导入，避免在不支持的平台上报错）
try:
    from vllm.config import VllmConfig
    from vllm.model_executor import SamplingMetadata
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.model_executor.models.interfaces import (
        MultiModalEmbeddings,
        SupportsMultiModal,
        SupportsPP,
    )
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        WeightsMapper,
        flatten_bn,
        init_vllm_registered_model,
        maybe_prefix,
        merge_multimodal_embeddings,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import (
        MultiModalDataDict,
        MultiModalFieldConfig,
        MultiModalKwargs,
        NestedTensors,
    )
    from vllm.multimodal.parse import (
        ImageEmbeddingItems,
        ImageProcessorItems,
        ImageSize,
        MultiModalDataItems,
    )
    from vllm.multimodal.processing import (
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
    )
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
    from vllm.sequence import IntermediateTensors

    VLLM_AVAILABLE = True
except ImportError:
    # 在不支持vLLM的平台上设置占位符
    VllmConfig = object
    SamplingMetadata = object
    QuantizationConfig = object
    set_default_torch_dtype = None
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

    def flatten_bn(x):
        return x

    def init_vllm_registered_model(*args: object, **kwargs: object) -> None:
        return None

    def maybe_prefix(prefix: str, name: str) -> str:
        return name

    def merge_multimodal_embeddings(*args: object, **kwargs: object) -> None:
        return None

    VLLM_AVAILABLE = False

from src.core.config import BASE_SIZE, CROP_MODE, DEFAULT_OCR_PROMPT, IMAGE_SIZE

# 项目内部导入
from src.core.process.image_process import DeepseekOCRProcessor, count_tiles

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

    def get_num_image_tokens(self, *, image_width: int, image_height: int, cropping: bool = True) -> int:
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
        if num_width_tiles > 1 or num_height_tiles > 1:
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
            return ImageSize(width=1024 * 2, height=1024 * 2)
        return ImageSize(width=640 * 2, height=640 * 2)


class DeepseekOCRDummyInputsBuilder(BaseDummyInputsBuilder[DeepseekOCRProcessingInfo]):
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

        if "<image>" in DEFAULT_OCR_PROMPT:
            return {
                "image": DeepseekOCRProcessor().tokenize_with_images(
                    images=self._get_dummy_images(
                        width=max_image_size.width,
                        height=max_image_size.height,
                        num_images=num_images,
                    ),
                    bos=True,
                    eos=True,
                    cropping=CROP_MODE,
                )
            }
        else:
            return {"image": []}


class DeepseekOCRMultiModalProcessor(BaseMultiModalProcessor[DeepseekOCRProcessingInfo]):
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
            processed_outputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

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
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "images_spatial_crop": MultiModalFieldConfig.batched("image"),
            # image_embeds=MultiModalFieldConfig.batched("image2"),
            "images_crop": MultiModalFieldConfig.batched("image"),
        }

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
            images = mm_items.get_items("image", (ImageEmbeddingItems, ImageProcessorItems))

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
        prompt: str | list[int],
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
