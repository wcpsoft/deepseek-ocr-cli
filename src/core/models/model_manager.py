#!/usr/bin/env python3
"""
模型管理器
专门负责模型的加载、初始化和词汇表管理
设备管理职责已移交给 DeviceManager
"""

from typing import Any

from src.core.logging import get_logger
from src.core.utils.error_handling import handle_model_error
from src.core.utils.model_path_utils import ModelPathResolver

logger = get_logger()


class ModelManager:
    """专门负责模型管理的类，设备管理已移交给 DeviceManager"""

    def __init__(self, model_path: str):
        """
        初始化模型管理器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        # 移除了 self.device，设备管理交给 DeviceManager

    @handle_model_error(re_raise=True, context={"component": "ModelManager", "operation": "load_model_and_tokenizer"})
    def load_model_and_tokenizer(self, trust_remote_code: bool = True) -> tuple[Any, Any]:
        """
        加载模型和分词器

        Args:
            trust_remote_code: 是否信任远程代码

        Returns:
            (模型, 分词器) 元组
        """
        # 使用 ModelPathResolver 获取加载参数
        loading_params = ModelPathResolver.get_loading_params(self.model_path, trust_remote_code)

        # 加载tokenizer
        logger.info(f"加载tokenizer: {self.model_path}")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=loading_params["trust_remote_code"],
            local_files_only=loading_params["local_files_only"],
        )

        # 加载模型
        logger.info(f"加载模型: {self.model_path}")
        self.model = self._load_model(trust_remote_code, loading_params)

        logger.info("模型和分词器加载完成")
        return self.model, self.tokenizer

    @handle_model_error(re_raise=True, context={"component": "ModelManager", "operation": "_load_model"})
    def _load_model(self, trust_remote_code: bool, loading_params: dict) -> Any:
        """加载模型的具体实现"""
        # 使用模型工厂创建模型实例
        from src.core.models.model_factory import create_ocr_model

        model = create_ocr_model(
            model_type="transformers",
            model_path=self.model_path,
            trust_remote_code=loading_params["trust_remote_code"],
            torch_dtype="auto",  # 设备管理交给 DeviceManager，这里使用 auto
            local_files_only=loading_params["local_files_only"],
        )
        logger.info("成功加载OCR模型")
        return model

    def load_processor(self) -> Any:
        """加载图像处理器"""
        try:
            logger.info(f"加载图像处理器: {self.model_path}")
            from src.core.process.image_process import DeepseekOCRProcessor

            self.processor = DeepseekOCRProcessor.from_pretrained(self.model_path)
            logger.info("成功加载DeepseekOCRProcessor")
            return self.processor
        except Exception as e:
            logger.error(f"无法加载图像处理器: {e!s}")
            self.processor = None
            return None

    # 设备管理方法已移至 DeviceManager
    # 以下方法已移除：
    # - setup_device(): 使用 DeviceManager.get_optimal_device()
    # - move_model_to_device(): 使用 DeviceManager.move_model_to_device()

    def adjust_vocab_size(self, model: Any, tokenizer: Any, image_token_id: int) -> None:
        """
        统一调整词汇表大小和嵌入层

        Args:
            model: 要调整的模型
            tokenizer: 分词器
            image_token_id: 图像token ID
        """
        try:
            # 计算所需的词汇表大小
            required_vocab_size = max(
                tokenizer.vocab_size,
                image_token_id + 1,
                model.config.vocab_size if hasattr(model, "config") else tokenizer.vocab_size,
            )

            # 更新模型配置中的词汇表大小
            if hasattr(model, "config"):
                model.config.vocab_size = required_vocab_size

            logger.info(
                f"更新配置词汇表大小为: {required_vocab_size} "
                f"(tokenizer: {tokenizer.vocab_size}, image_token_id: {image_token_id})"
            )

            # 调整嵌入层大小
            self._resize_embeddings(model, required_vocab_size)

        except Exception as e:
            logger.error(f"调整词汇表大小失败: {e!s}")
            # 不抛出异常，允许继续运行
            logger.warning("词汇表大小调整失败，使用原始配置")

    def _resize_embeddings(self, model: Any, new_size: int) -> None:
        """
        调整模型的嵌入层大小

        Args:
            model: 要调整的模型
            new_size: 新的嵌入层大小
        """
        if not hasattr(model, "get_input_embeddings"):
            logger.warning("模型没有get_input_embeddings方法，跳过嵌入层调整")
            return

        try:
            embeddings = model.get_input_embeddings()
            if not hasattr(embeddings, "num_embeddings"):
                logger.warning("嵌入层没有num_embeddings属性，跳过调整")
                return

            current_size = embeddings.num_embeddings
            if current_size >= new_size:
                logger.debug(f"嵌入层大小已足够: {current_size} >= {new_size}")
                return

            logger.info(f"扩展嵌入层大小从 {current_size} 到 {new_size}")

            # 使用transformers的内置方法调整嵌入层
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(new_size)
                logger.info("使用transformers内置方法调整嵌入层成功")
                return

            # 手动调整嵌入层（备用方案）
            self._manual_resize_embeddings(embeddings, new_size, current_size)

            # 确保模型的词汇表大小也更新
            if hasattr(model, "config"):
                model.config.vocab_size = new_size

        except Exception as e:
            logger.error(f"调整嵌入层失败: {e!s}")
            raise

    def _manual_resize_embeddings(self, embeddings: Any, new_size: int, current_size: int) -> None:
        """
        手动调整嵌入层大小

        Args:
            embeddings: 嵌入层
            new_size: 新大小
            current_size: 当前大小
        """
        if not hasattr(embeddings, "weight"):
            logger.warning("嵌入层没有weight属性，无法手动调整")
            return

        # 获取当前权重
        current_weight = embeddings.weight.data

        # 检查嵌入维度
        embedding_dim = current_weight.size(1) if current_weight.dim() > 1 else None

        if embedding_dim is None:
            logger.warning("无法确定嵌入维度，跳过手动调整")
            return

        # 创建新的权重张量
        import torch

        new_weight = torch.zeros(new_size, embedding_dim, dtype=current_weight.dtype)

        # 复制原有权重
        new_weight[:current_size] = current_weight

        # 应用新权重
        embeddings.weight.data = new_weight

        # 更新嵌入层的大小
        if hasattr(embeddings, "num_embeddings"):
            embeddings.num_embeddings = new_size

        logger.info("手动调整嵌入层大小成功")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            # 设备信息已移除，由 DeviceManager 管理
            "has_model": self.model is not None,
            "has_tokenizer": self.tokenizer is not None,
            "has_processor": self.processor is not None,
        }
