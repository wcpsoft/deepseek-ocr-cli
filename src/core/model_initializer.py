#!/usr/bin/env python3
"""
模型初始化器
统一管理模型和分词器的初始化
"""

import os
import sys

import torch

# 导入日志模块
from src.core.logging import get_logger

# 导入生成配置管理器
# 导入MPS优化工具
from src.core.utils.mps_utils import get_optimal_device

# 获取日志记录器
logger = get_logger()


class ModelInitializer:
    """统一的模型初始化器"""

    @staticmethod
    def initialize_transformers_model_and_tokenizer(
        model_path: str, trust_remote_code: bool = True
    ) -> tuple[object, object]:
        """
        初始化Transformers模型和分词器

        Args:
            model_path: 模型路径
            trust_remote_code: 是否信任远程代码

        Returns:
            (模型, 分词器) 元组
        """
        try:
            # 加载tokenizer
            from transformers import AutoTokenizer

            logger.info("开始加载tokenizer")
            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = (
                model_path.startswith(("http://", "https://"))
                or model_path.startswith("deepseek-ai/")
                or model_path.startswith("huggingface.co/")
                or "/" not in model_path  # 单个名称可能是远程仓库名
                or (not os.path.exists(model_path) and not os.path.exists(os.path.expanduser(model_path)))
            )

            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo

            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用传入的trust_remote_code参数
            trust_remote_code_for_tokenizer = is_remote_repo and trust_remote_code

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code_for_tokenizer,
                local_files_only=local_files_only,
            )
            logger.info("tokenizer加载完成")

            # 加载模型 - 实现一个简单的自定义模型
            logger.info("开始加载模型")

            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = (
                model_path.startswith(("http://", "https://"))
                or model_path.startswith("deepseek-ai/")
                or model_path.startswith("huggingface.co/")
                or "/" not in model_path  # 单个名称可能是远程仓库名
                or (not os.path.exists(model_path) and not os.path.exists(os.path.expanduser(model_path)))
            )

            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo

            # 直接使用AutoModelForCausalLM加载模型
            try:
                from transformers import AutoModelForCausalLM, GenerationMixin

                # 如果是本地路径，确保使用src目录中的模型代码而不是models目录中的
                if local_files_only:
                    # 获取项目根目录
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    # 将src目录添加到Python路径，确保使用src中的模型代码
                    src_path = os.path.join(project_root, "src")
                    if src_path not in sys.path:
                        sys.path.insert(0, src_path)

                    # 将模型代码路径添加到Python路径
                    model_code_path = os.path.join(project_root, "models", "template")
                    if model_code_path not in sys.path:
                        sys.path.insert(0, model_code_path)

                    logger.info(f"已添加src路径到Python路径: {src_path}")
                    logger.info(f"已添加模型代码路径到Python路径: {model_code_path}")

                # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
                # 对于远程模型，使用传入的trust_remote_code参数
                trust_remote_code_for_model = is_remote_repo and trust_remote_code

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    trust_remote_code=trust_remote_code_for_model,
                    local_files_only=local_files_only,
                )

                # 确保模型继承GenerationMixin
                if not isinstance(model, GenerationMixin):
                    logger.warning("模型未继承GenerationMixin，尝试手动添加generate方法")
                    # 如果模型没有generate方法，尝试从GenerationMixin添加
                    if not hasattr(model, "generate"):
                        logger.warning("模型没有generate方法，尝试从GenerationMixin添加")
                        # 动态添加generate方法
                        model.generate = GenerationMixin.generate.__get__(model, type(model))

                logger.info("使用AutoModelForCausalLM成功加载模型")
            except Exception as e:
                logger.error(f"使用AutoModelForCausalLM加载失败: {e!s}")
                raise RuntimeError(f"加载模型失败: {e!s}") from e

            # 设置tokenizer到模型中，以便后续使用
            model.tokenizer = tokenizer

            # 移动模型到适当的设备
            device = get_optimal_device()
            logger.info(f"使用设备: {device.type}")

            model = model.to(device)
            logger.info("模型加载完成")

            return model, tokenizer
        except Exception as e:
            logger.error(f"初始化Transformers模型和分词器失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化Transformers模型和分词器失败: {e!s}") from e

    @staticmethod
    def initialize_vllm_model(model_path: str, prompt: str | None = None, trust_remote_code: bool = True) -> object:
        """
        初始化vLLM模型

        Args:
            model_path: 模型路径
            prompt: 提示词

        Returns:
            vLLM模型实例
        """
        try:
            # 检查是否是本地路径，如果是则只使用本地文件
            # 更严格的本地路径检测：检查路径是否存在且不是远程仓库格式
            is_remote_repo = (
                model_path.startswith(("http://", "https://"))
                or model_path.startswith("deepseek-ai/")
                or model_path.startswith("huggingface.co/")
                or "/" not in model_path  # 单个名称可能是远程仓库名
                or (not os.path.exists(model_path) and not os.path.exists(os.path.expanduser(model_path)))
            )

            # 对于本地路径，确保local_files_only=True
            # 对于远程仓库，确保local_files_only=False
            local_files_only = not is_remote_repo

            # 如果是本地路径，确保使用src目录中的模型代码而不是models目录中的
            if local_files_only:
                # 获取项目根目录
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                # 将src目录添加到Python路径，确保使用src中的模型代码
                src_path = os.path.join(project_root, "src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)

                # 将模型代码路径添加到Python路径
                model_code_path = os.path.join(project_root, "models", "template")
                if model_code_path not in sys.path:
                    sys.path.insert(0, model_code_path)

                logger.info(f"已添加src路径到Python路径: {src_path}")
                logger.info(f"已添加模型代码路径到Python路径: {model_code_path}")

            # vLLM相关导入（延迟导入，避免在不支持的平台上报错）
            try:
                from transformers import AutoConfig, AutoModelForCausalLM
                from vllm import LLM
            except ImportError:
                # 在不支持vLLM的平台上设置占位符
                LLM = object
                raise RuntimeError("vLLM未安装或不支持当前平台") from None

            # 注册自定义模型类
            from deepseek_vl.models.deepseek_vl_v2 import DeepseekOCRForCausalLM

            AutoModelForCausalLM.register("DeepseekVLV2ForCausalLM", DeepseekOCRForCausalLM)

            # 对于本地模型，不需要trust_remote_code，因为我们使用的是本地代码
            # 对于远程模型，使用传入的trust_remote_code参数
            trust_remote_code_for_config = is_remote_repo and trust_remote_code

            # 获取模型配置
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code_for_config,
                local_files_only=local_files_only,
            )

            # vLLM初始化参数
            vllm_kwargs = {
                "model": model_path,
                "trust_remote_code": trust_remote_code_for_config,  # 使用与config相同的trust_remote_code设置
                "tensor_parallel_size": 1,
                "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
                "max_model_len": getattr(config, "max_position_embeddings", 8192),
                "gpu_memory_utilization": 0.9,
                "enforce_eager": False,
                "disable_log_stats": True,
                "skip_tokenizer_init": False,
                "hf_overrides": {"architectures": ["DeepseekVLV2ForCausalLM"]},
                "block_size": 256,
                "swap_space": 0,
                "max_num_seqs": 100,
                "disable_mm_preprocessor_cache": True,
            }

            # 初始化vLLM模型
            llm = LLM(**vllm_kwargs)

            return llm
        except Exception as e:
            logger.error(f"初始化vLLM模型失败: {e!s}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化vLLM模型失败: {e!s}") from e

    @staticmethod
    def move_model_to_device(model: object, device: torch.device | None = None) -> object:
        """
        将模型移到指定设备

        Args:
            model: 模型实例
            device: 目标设备，如果为None则自动选择最优设备

        Returns:
            移动后的模型实例
        """
        try:
            # 如果没有指定设备，自动选择最优设备
            if device is None:
                device = get_optimal_device()

            logger.info(f"将模型移到 {device.type.upper()} 设备")
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model = model.eval()

            # 根据设备类型选择合适的数据类型
            use_bfloat16 = device.type == "cuda"  # 仅在CUDA设备上使用bfloat16
            if use_bfloat16:
                # 检查模型是否有to方法
                if hasattr(model, "to") and callable(getattr(model, "to", None)):
                    model = model.to(torch.bfloat16)

            logger.debug(f"模型已移到设备: {device}")
            return model
        except Exception as e:
            logger.error(f"将模型移到设备失败: {e!s}")
            raise RuntimeError(f"将模型移到设备失败: {e!s}") from e
