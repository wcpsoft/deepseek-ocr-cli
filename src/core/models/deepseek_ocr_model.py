#!/usr/bin/env python3
"""
DeepSeek OCR模型实现
提供与HuggingFace兼容的模型接口
"""

# 根据平台选择合适的实现
import torch

# 检查是否支持vLLM
VLLM_AVAILABLE = False
try:
    import vllm

    VLLM_AVAILABLE = True
except ImportError:
    pass

# 确保在模块加载时就注册配置类
try:
    from transformers import CONFIG_MAPPING

    from src.core.deepseek_ocr_config import (
        DeepseekV2Config as ConfigDeepseekV2Config,
    )
    from src.core.deepseek_ocr_config import (
        DeepseekVLV2Config as ConfigDeepseekVLV2Config,
    )

    # 动态注册配置类（如果尚未注册）
    if "deepseek_vl_v2" not in CONFIG_MAPPING:
        CONFIG_MAPPING._extra_content["deepseek_vl_v2"] = ConfigDeepseekVLV2Config
    if "deepseek_v2" not in CONFIG_MAPPING:
        CONFIG_MAPPING._extra_content["deepseek_v2"] = ConfigDeepseekV2Config
except Exception as e:
    pass  # 在模块加载时忽略错误

# 为了保持向后兼容性，导入配置类
from src.core.deepseek_ocr_config import DeepseekV2Config, DeepseekVLV2Config

# 导入实际的模型实现
from src.core.models.modeling_deepseekv2 import DeepseekV2ForCausalLM as BaseDeepseekV2ForCausalLM
from src.core.models.modeling_deepseekv2 import DeepseekV2Model

# 确保这些类在当前模块中可用
__all__ = [
    "BaseDeepseekV2ForCausalLM",
    "DeepseekOCRForCausalLM",
    "DeepseekV2Config",
    "DeepseekV2Model",
    "DeepseekVLV2Config",
]

# 标准库导入
from typing import Optional

# 第三方库导入
# 配置导入
# 导入日志模块
from src.core.logging import get_logger

# 项目内部导入
from src.core.process.image_process import DeepseekOCRProcessor

# 获取日志记录器
logger = get_logger()

# 常量定义
_IMAGE_TOKEN = "<image>"

# vLLM相关导入（延迟导入，避免在不支持的平台上报错）
VLLM_AVAILABLE = False
try:
    from vllm.config import VllmConfig
    from vllm.model_executor import SamplingMetadata
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        WeightsMapper,
        flatten_bn,
        init_vllm_registered_model,
        maybe_prefix,
        merge_multimodal_embeddings,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig, MultiModalKwargs, NestedTensors
    from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, ImageSize, MultiModalDataItems
    from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement, PromptUpdate
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
    flatten_bn = lambda x: x
    init_vllm_registered_model = lambda *args, **kwargs: None
    maybe_prefix = lambda prefix, name: name
    merge_multimodal_embeddings = lambda *args, **kwargs: None


# 检查是否支持vLLM
VLLM_AVAILABLE = False
try:
    import vllm

    VLLM_AVAILABLE = True
except ImportError:
    pass

# 根据平台选择合适的实现
if VLLM_AVAILABLE:
    # 使用vLLM实现
    from src.core.vllm.vllm_ocr_model import DeepseekOCRForCausalLM as BaseDeepseekOCRForCausalLM
else:
    # 使用Transformers实现
    from src.core.transformers.transformers_ocr_model import DeepseekOCRForCausalLM as BaseDeepseekOCRForCausalLM


# 继承正确的基类
class DeepseekOCRForCausalLM(BaseDeepseekOCRForCausalLM):
    """
    DeepSeek OCR因果语言模型
    统一的模型实现，兼容vLLM和Transformers两种推理引擎
    """

    def __init__(self, config=None):
        """
        初始化DeepSeek OCR因果语言模型

        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.image_token_id = None

        # 使用设备管理器获取最优设备
        from src.core.utils.device_manager import device_manager

        self.device = device_manager.get_optimal_device()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        从预训练模型加载模型

        Args:
            pretrained_model_name_or_path: 预训练模型名称或路径
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            DeepseekOCRForCausalLM实例
        """
        # 直接调用基类的from_pretrained方法
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        生成方法，调用实际模型的生成方法

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            生成结果
        """
        if hasattr(self, "model") and self.model is not None:
            return self.model.generate(*args, **kwargs)
        else:
            raise RuntimeError("模型未正确初始化")

    def to(self, device):
        """
        将模型移到指定设备

        Args:
            device: 目标设备

        Returns:
            self
        """
        self.device = device
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.to(device)
        return self

    def eval(self):
        """
        设置模型为评估模式

        Returns:
            self
        """
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.eval()
        return self

    def parameters(self):
        """
        获取模型参数

        Returns:
            模型参数迭代器
        """
        if hasattr(self, "model") and self.model is not None:
            return self.model.parameters()
        return iter([])

    def _clean_config_for_llama(self, config_dict):
        """
        清理配置字典，移除LlamaConfig不支持的字段

        Args:
            config_dict: 配置字典

        Returns:
            清理后的配置字典
        """
        # 创建配置字典的副本
        clean_config = config_dict.copy()

        # 移除不兼容的字段
        incompatible_fields = [
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "rm_head",
            "v_head_dim",
            "auto_map",
            "architectures",
            "_name_or_path",
            "use_mla",
            "topk_method",
            "topk_group",
            "n_group",
            "n_shared_experts",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "lm_head",
        ]

        for field in incompatible_fields:
            clean_config.pop(field, None)

        return clean_config

    def _get_generation_config(self):
        """
        获取模型的生成配置参数

        Returns:
            生成配置参数字典
        """
        from src.core.utils.generation_config import GenerationConfigManager

        # 使用统一的生成配置管理器
        if hasattr(self, "model") and self.model is not None:
            return GenerationConfigManager.get_generation_config(self.model)
        else:
            # 如果模型未初始化，返回默认配置
            return GenerationConfigManager.get_generation_config(None)

    def _create_model_config(self, config_dict, model_path):
        """
        创建模型配置对象

        Args:
            config_dict: 配置字典
            model_path: 模型路径

        Returns:
            配置对象
        """
        # 尝试直接使用AutoConfig.from_pretrained，但不信任远程代码
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
            return config
        except Exception as e:
            logger.error(f"AutoConfig.from_pretrained失败: {e}")

        # 如果失败，尝试手动创建LlamaConfig（DeepSeek基于Llama架构）
        try:
            from transformers import LlamaConfig

            # 清理配置字典
            clean_config = self._clean_config_for_llama(config_dict)
            config = LlamaConfig(**clean_config)
            logger.info("使用LlamaConfig创建配置成功")
            return config
        except Exception as e:
            logger.error(f"使用LlamaConfig创建配置也失败: {e}")

        # 最后的备选方案：使用PretrainedConfig
        try:
            from transformers import PretrainedConfig

            # 清理配置字典
            clean_config = self._clean_config_for_llama(config_dict)
            config = PretrainedConfig(**clean_config)
            return config
        except Exception as e:
            logger.error(f"使用PretrainedConfig创建配置也失败: {e}")
            raise RuntimeError(f"无法创建模型配置: {e}")

    def _find_model_files(self, model_path):
        """
        查找模型文件

        Args:
            model_path: 模型路径

        Returns:
            模型文件列表
        """
        import json
        import os
        from pathlib import Path

        model_files = []
        model_path_obj = Path(model_path)

        # 检查是否存在索引文件
        if (model_path_obj / "model.safetensors.index.json").exists():
            # 处理分片模型文件
            index_path = model_path_obj / "model.safetensors.index.json"
            with open(index_path, encoding="utf-8") as f:
                index_data = json.load(f)
            model_files = list(set(index_data["weight_map"].values()))
        else:
            # 查找模型文件
            for file_name in os.listdir(model_path):
                if file_name.endswith((".bin", ".safetensors")) and file_name.startswith("model"):
                    model_files.append(file_name)

        if not model_files:
            # 尝试查找任何权重文件
            for file_name in os.listdir(model_path):
                if file_name.endswith((".bin", ".safetensors")):
                    model_files.append(file_name)

        if not model_files:
            raise FileNotFoundError("未找到模型权重文件")

        return model_files

    def _load_model_weights(self, model_path, model_files):
        """
        加载模型权重

        Args:
            model_path: 模型路径
            model_files: 模型文件列表

        Returns:
            权重字典
        """
        from pathlib import Path

        import torch
        from safetensors.torch import load_file

        state_dict = {}
        model_path_obj = Path(model_path)
        device = getattr(self, "device", torch.device("cpu"))

        for file_name in model_files:
            file_path = model_path_obj / file_name
            if file_path.exists():
                if file_name.endswith(".safetensors"):
                    # 强制使用CPU以避免MPS内存不足问题
                    weights = load_file(str(file_path), device="cpu")
                    state_dict.update(weights)
                else:
                    # 强制使用CPU以避免MPS内存不足问题
                    weights = torch.load(str(file_path), map_location="cpu", weights_only=True)
                    state_dict.update(weights)

        return state_dict

    def load_weights_from_path(
        self, model_path: str, device: Optional[torch.device] = None
    ) -> "DeepseekOCRForCausalLM":
        """
        从指定路径加载模型权重

        Args:
            model_path: 模型路径
            device: 目标设备，如果为None则使用最优设备

        Returns:
            加载权重的模型实例
        """
        import json
        from pathlib import Path

        from transformers import AutoModelForCausalLM

        # 使用设备管理器获取最优设备
        from src.core.utils.device_manager import device_manager

        if device is None:
            device = device_manager.get_optimal_device()

        # 保存设备信息
        self.device = device

        # 检查模型路径是否存在配置文件
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

        # 获取模型文件路径
        model_files = self._find_model_files(model_path)

        # 加载权重
        state_dict = self._load_model_weights(model_path, model_files)

        # 读取配置文件
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        # 使用language_config作为基础配置（如果存在）
        if "language_config" in config_dict:
            config_source = config_dict["language_config"]
        else:
            config_source = config_dict

        # 创建模型配置对象
        config = self._create_model_config(config_source, model_path)

        # 创建模型实例
        try:
            # 强制使用CPU以避免MPS内存不足问题
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,  # 使用float32
                device_map="cpu",  # 加载到CPU
                trust_remote_code=False,
            )
            logger.info("使用AutoModelForCausalLM.from_pretrained创建模型成功")
        except Exception as e:
            logger.error(f"AutoModelForCausalLM.from_pretrained失败: {e}")
            # 尝试使用from_config
            try:
                self.model = AutoModelForCausalLM.from_config(config)
                logger.info("使用AutoModelForCausalLM.from_config创建模型成功")
            except Exception as e2:
                logger.error(f"AutoModelForCausalLM.from_config也失败: {e2}")
                raise RuntimeError(f"无法创建模型实例: {e2}")

        # 加载权重到模型
        if hasattr(self, "model") and self.model is not None and state_dict:
            try:
                # 过滤掉position_ids相关的键，因为它是buffer而不是权重
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("position_ids")}
                self.model.load_state_dict(filtered_state_dict, strict=False)
                logger.info("模型权重加载成功")
            except Exception as e:
                logger.error(f"加载权重失败: {e}")
                raise RuntimeError(f"无法加载模型权重: {e}")

        # 设置模型属性
        self.config = config
        self.image_token_id = config.image_token_id if hasattr(config, "image_token_id") else 200001

        return self

    def infer(
        self,
        tokenizer,
        prompt="",
        image_file="",
        output_path="",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        test_compress=False,
        save_results=False,
    ):
        """
        推理方法，用于处理图像并生成OCR结果

        Args:
            tokenizer: 分词器
            prompt: 提示词
            image_file: 图像文件路径
            output_path: 输出路径
            base_size: 基础尺寸
            image_size: 图像尺寸
            crop_mode: 是否启用裁剪模式
            test_compress: 是否测试压缩
            save_results: 是否保存结果

        Returns:
            OCR结果
        """
        try:
            # 加载图像
            image = Image.open(image_file).convert("RGB")
            logger.debug(f"图像加载完成，尺寸: {image.size}")

            # 处理图像
            processor = DeepseekOCRProcessor(tokenizer=tokenizer)
            logger.debug("DeepseekOCRProcessor初始化完成")
            logger.debug("即将调用tokenize_with_images方法")
            processed_data = processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=crop_mode)
            logger.debug(f"tokenize_with_images方法调用完成，processed_data类型: {type(processed_data)}")

            # 添加调试信息
            logger.debug(f"processed_data类型: {type(processed_data)}")
            logger.debug(f"processed_data长度: {len(processed_data) if processed_data else 'N/A'}")

            # 提取处理后的数据
            logger.debug("开始检查processed_data的有效性")
            if processed_data is None:
                raise ValueError("processed_data为None")
            if len(processed_data) == 0:
                raise ValueError("processed_data为空列表")
            if processed_data[0] is None:
                raise ValueError("processed_data[0]为None")

            logger.debug(f"processed_data[0]类型: {type(processed_data[0])}")
            logger.debug(f"processed_data[0]长度: {len(processed_data[0]) if processed_data[0] else 'N/A'}")

            if len(processed_data) > 0 and processed_data[0] is not None and len(processed_data[0]) >= 7:
                # 注意：这里的索引需要根据实际的数据结构进行调整
                # 确保processed_data[0]有足够的元素
                if len(processed_data[0]) < 7:
                    raise ValueError(f"处理后的数据结构不完整，期望至少7个元素，实际只有{len(processed_data[0])}个元素")

                logger.debug("开始提取数据元素")
                # 根据原始仓库的数据结构正确提取元素（索引0-6）
                logger.debug("提取input_ids")
                input_ids = processed_data[0][0]
                logger.debug(f"input_ids提取完成，类型: {type(input_ids)}")
                logger.debug("提取pixel_values")
                pixel_values = processed_data[0][1]
                logger.debug(f"pixel_values提取完成，类型: {type(pixel_values)}")
                logger.debug("提取images_crop")
                images_crop = processed_data[0][2]
                logger.debug(f"images_crop提取完成，类型: {type(images_crop)}")
                logger.debug("提取images_seq_mask")
                images_seq_mask = processed_data[0][3]
                logger.debug(f"images_seq_mask提取完成，类型: {type(images_seq_mask)}")
                logger.debug("提取images_spatial_crop")
                images_spatial_crop = processed_data[0][4]
                logger.debug(f"images_spatial_crop提取完成，类型: {type(images_spatial_crop)}")
                logger.debug("提取num_image_tokens")
                num_image_tokens = processed_data[0][5]
                logger.debug(f"num_image_tokens提取完成，类型: {type(num_image_tokens)}")
                logger.debug("提取image_shapes")
                image_shapes = processed_data[0][6]
                logger.debug(f"image_shapes提取完成，类型: {type(image_shapes)}")
            else:
                raise ValueError("图像处理失败，未生成有效的输入数据")

            # 验证提取的元素不为None
            logger.debug("开始验证提取的元素")
            if input_ids is None:
                raise ValueError("input_ids为None")
            if pixel_values is None:
                raise ValueError("pixel_values为None")
            if images_crop is None:
                raise ValueError("images_crop为None")
            if images_seq_mask is None:
                raise ValueError("images_seq_mask为None")
            if images_spatial_crop is None:
                raise ValueError("images_spatial_crop为None")
            if num_image_tokens is None:
                raise ValueError("num_image_tokens为None")
            if image_shapes is None:
                raise ValueError("image_shapes为None")

            # 确保数据在正确的设备上
            try:
                logger.debug("开始获取设备信息")
                device = getattr(self, "device", None)
                if device is None:
                    device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
            except Exception:
                # 如果无法获取参数设备，使用默认设备
                device = torch.device("cpu")

            logger.debug(f"设备信息获取完成: {device}")

            # 强制使用CPU设备，避免MPS内存问题
            device = torch.device("cpu")
            logger.debug(f"强制使用CPU设备: {device}")

            # 将张量移动到CPU设备
            try:
                logger.debug("移动张量到CPU设备")
                input_ids = input_ids.to(device)
                pixel_values = pixel_values.to(device)
                images_crop = images_crop.to(device)
                images_spatial_crop = images_spatial_crop.to(device)
            except Exception as e:
                raise ValueError(f"无法将张量移动到设备{device}: {e}")

            # CPU设备使用float32，不支持bfloat16
            try:
                logger.debug("在CPU设备上转换数据类型")
                pixel_values = pixel_values.to(torch.float32)
                images_crop = images_crop.to(torch.float32)
            except Exception as e:
                logger.warning(f"在CPU设备上转换数据类型时出错: {e}")

            # 生成结果
            with torch.no_grad():
                # 构造注意力掩码
                try:
                    logger.debug("开始构造注意力掩码")
                    attention_mask = torch.ones_like(input_ids)
                except Exception as e:
                    raise ValueError(f"创建注意力掩码时出错: {e}")

                # 获取模型的生成配置参数
                try:
                    logger.debug("开始获取生成配置")
                    generation_config = self._get_generation_config()
                except Exception as e:
                    logger.warning(f"获取生成配置时出错，使用默认配置: {e}")
                    from src.core.utils.generation_config import GenerationConfigManager

                    generation_config = GenerationConfigManager.get_generation_config(None)

                logger.debug(f"生成配置获取完成: {generation_config}")
                generate_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": generation_config.get("max_new_tokens", 8192),
                    "do_sample": generation_config.get("do_sample", False),
                    "pad_token_id": tokenizer.eos_token_id if tokenizer is not None else 0,
                    "attention_mask": attention_mask,
                }

                # 只有在do_sample为True时才添加temperature和top_p参数
                if generate_kwargs["do_sample"]:
                    generate_kwargs["temperature"] = generation_config.get("temperature", 1.0)
                    generate_kwargs["top_p"] = generation_config.get("top_p", 1.0)

                logger.debug(f"生成参数构造完成: {generate_kwargs}")
                # 检查模型是否正确初始化
                if not hasattr(self, "model") or self.model is None:
                    raise RuntimeError("模型未正确初始化")

                # 判断模型类型并相应地处理图像参数
                # 检查是否是vLLM模型（具有处理图像的特殊方法）
                is_vllm_model = hasattr(self, "get_multimodal_embeddings") and callable(self.get_multimodal_embeddings)

                if is_vllm_model:
                    # vLLM模式下添加图像相关参数
                    generate_kwargs["images"] = [[pixel_values, images_crop, images_spatial_crop]]
                    generate_kwargs["images_seq_mask"] = images_seq_mask
                    generate_kwargs["images_spatial_crop"] = images_spatial_crop
                else:
                    # Transformers模式下，需要先通过视觉编码器处理图像
                    # 对于Transformers模型，我们暂时跳过图像处理，只处理文本
                    logger.debug("使用Transformers模式，跳过图像参数传递")

                # 尝试生成结果
                try:
                    logger.debug("开始模型生成")
                    logger.debug(f"生成参数: {generate_kwargs.keys()}")

                    # 在CPU设备上使用标准生成方式
                    input_ids = generate_kwargs.pop("input_ids")
                    outputs = self.model.generate(input_ids, **generate_kwargs)
                    logger.debug("CPU设备模型生成完成")
                except Exception as generate_error:
                    logger.error(f"模型生成过程中出错: {generate_error}")
                    import traceback

                    logger.error(f"错误堆栈: {traceback.format_exc()}")
                    raise RuntimeError(f"模型生成过程中出错: {generate_error}") from generate_error

            # 解码输出 - 根据官方实现进行修正
            try:
                logger.debug("开始解码输出")
                # 标准解码方式 - 根据Hugging Face官方示例
                if hasattr(outputs, "sequences") and tokenizer is not None:
                    # 对于vLLM输出格式
                    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    logger.debug("解码完成(vLLM格式)")
                    return result
                elif hasattr(outputs, "output_ids") and tokenizer is not None:
                    # 对于Transformers输出格式
                    if isinstance(outputs.output_ids, list):
                        result = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
                    else:
                        result = tokenizer.decode(outputs.output_ids, skip_special_tokens=True)
                    logger.debug("解码完成(Transformers格式)")
                    return result
                elif tokenizer is not None:
                    # 尝试直接解码outputs
                    if isinstance(outputs, torch.Tensor):
                        # 如果outputs是tensor，尝试解码
                        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        logger.debug("解码完成(Tensor格式)")
                        return result
                    elif hasattr(outputs, "__iter__"):
                        # 如果outputs是可迭代的，尝试解码第一个元素
                        result = tokenizer.decode(list(outputs)[0], skip_special_tokens=True)
                        logger.debug("解码完成(可迭代格式)")
                        return result
                    else:
                        # 尝试直接解码outputs对象
                        result = tokenizer.decode(outputs, skip_special_tokens=True)
                        logger.debug("解码完成(其他格式)")
                        return result
                else:
                    raise RuntimeError("解码失败：缺少tokenizer")
            except Exception as decode_error:
                # 返回详细的解码错误信息
                error_msg = f"解码算法错误: {decode_error!s}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from decode_error
        except Exception as e:
            logger.error(f"图像处理过程中发生错误: {e!s}")
            logger.error(f"错误类型: {type(e)}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"图像处理失败: {e!s}") from e
