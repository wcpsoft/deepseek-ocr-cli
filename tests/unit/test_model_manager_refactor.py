"""测试重构后的 ModelManager 功能"""

import unittest
from unittest.mock import MagicMock, patch

from src.core.models.model_manager import ModelManager


class TestModelManagerRefactor(unittest.TestCase):
    """测试 ModelManager 的词汇表大小调整功能"""

    def setUp(self):
        """测试前准备"""
        self.model_path = "test_model_path"
        self.model_manager = ModelManager(self.model_path)

    def test_adjust_vocab_size_basic(self):
        """测试基本的词汇表大小调整"""
        # 创建模拟模型和tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        image_token_id = 32001

        # 配置模拟模型的配置
        mock_model.config = MagicMock()
        mock_model.config.vocab_size = 32000
        mock_model.get_input_embeddings.return_value = MagicMock()
        mock_model.get_input_embeddings.return_value.num_embeddings = 32000

        # 测试词汇表调整
        self.model_manager.adjust_vocab_size(mock_model, mock_tokenizer, image_token_id)

        # 验证配置更新
        self.assertEqual(mock_model.config.vocab_size, 32002)  # max(32000, 32001+1, 32000)

    def test_adjust_vocab_size_with_resize_token_embeddings(self):
        """测试使用内置resize_token_embeddings方法"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        image_token_id = 32002

        mock_model.config = MagicMock()
        mock_model.config.vocab_size = 32000
        mock_model.get_input_embeddings.return_value = MagicMock()
        mock_model.get_input_embeddings.return_value.num_embeddings = 32000
        mock_model.resize_token_embeddings = MagicMock()

        self.model_manager.adjust_vocab_size(mock_model, mock_tokenizer, image_token_id)

        # 验证调用resize_token_embeddings
        mock_model.resize_token_embeddings.assert_called_once_with(32003)

    def test_adjust_vocab_size_manual_resize(self):
        """测试手动调整嵌入层大小"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        image_token_id = 32002

        mock_model.config = MagicMock()
        mock_model.config.vocab_size = 32000
        mock_model.get_input_embeddings.return_value = MagicMock()
        mock_model.get_input_embeddings.return_value.num_embeddings = 32000
        mock_model.resize_token_embeddings = MagicMock(side_effect=AttributeError("No method"))

        # 模拟嵌入层
        mock_embeddings = MagicMock()
        mock_embeddings.weight.data = MagicMock()
        mock_embeddings.weight.data.size.return_value = [32000, 768]  # [vocab_size, embedding_dim]
        mock_embeddings.weight.data.dtype = "float32"
        mock_embeddings.num_embeddings = 32000
        mock_model.get_input_embeddings.return_value = mock_embeddings

        with patch("torch.zeros") as mock_zeros:
            mock_zeros.return_value = MagicMock()
            self.model_manager.adjust_vocab_size(mock_model, mock_tokenizer, image_token_id)

            # 验证手动调整调用
            mock_zeros.assert_called_once_with(32003, 768, dtype="float32")

    def test_adjust_vocab_size_no_embeddings_method(self):
        """测试模型没有get_input_embeddings方法的情况"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        image_token_id = 32001

        # 模拟没有get_input_embeddings方法
        del mock_model.get_input_embeddings

        # 这不应该抛出异常
        try:
            self.model_manager.adjust_vocab_size(mock_model, mock_tokenizer, image_token_id)
        except AttributeError:
            self.fail("adjust_vocab_size 不应该因为缺少get_input_embeddings而抛出异常")

    def test_adjust_vocab_size_already_sufficient(self):
        """测试词汇表大小已经足够的情况"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        image_token_id = 100  # 很小的值

        mock_model.config = MagicMock()
        mock_model.config.vocab_size = 32000
        mock_model.get_input_embeddings.return_value = MagicMock()
        mock_model.get_input_embeddings.return_value.num_embeddings = 32000
        mock_model.resize_token_embeddings = MagicMock()

        self.model_manager.adjust_vocab_size(mock_model, mock_tokenizer, image_token_id)

        # 不应该调用resize_token_embeddings，因为已经足够
        mock_model.resize_token_embeddings.assert_not_called()

    def test_manual_resize_embeddings_no_weight(self):
        """测试嵌入层没有weight属性的情况"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 32000
        image_token_id = 32002

        mock_model.config = MagicMock()
        mock_model.config.vocab_size = 32000
        mock_model.get_input_embeddings.return_value = MagicMock()
        mock_model.get_input_embeddings.return_value.num_embeddings = 32000
        mock_model.resize_token_embeddings = MagicMock(side_effect=AttributeError("No method"))

        # 模拟没有weight属性
        mock_embeddings = MagicMock()
        del mock_embeddings.weight
        mock_model.get_input_embeddings.return_value = mock_embeddings

        # 这不应该抛出异常
        try:
            self.model_manager.adjust_vocab_size(mock_model, mock_tokenizer, image_token_id)
        except AttributeError:
            self.fail("manual_resize_embeddings 不应该因为缺少weight而抛出异常")


class TestEngineDependencyInjection(unittest.TestCase):
    """测试引擎的依赖注入功能"""

    def test_vllm_engine_dependency_injection(self):
        """测试VLLM引擎的依赖注入"""
        from src.core.vllm.vllm_engine import VLLMEngine

        mock_model_manager = MagicMock()
        mock_device_manager = MagicMock()

        engine = VLLMEngine(
            model_path="test_path", model_manager=mock_model_manager, device_manager=mock_device_manager
        )

        self.assertEqual(engine._model_manager, mock_model_manager)
        self.assertEqual(engine._device_manager, mock_device_manager)

    def test_transformers_engine_dependency_injection(self):
        """测试Transformers引擎的依赖注入"""
        from src.core.models.model_manager import ModelManager
        from src.core.transformers.transformers_engine import TransformersEngine

        mock_model_manager = ModelManager("test_path")
        mock_image_handler = MagicMock()
        mock_device_manager = MagicMock()

        engine = TransformersEngine(
            model_path="test_path",
            model_manager=mock_model_manager,
            image_handler=mock_image_handler,
            device_manager=mock_device_manager,
        )

        self.assertEqual(engine.model_manager, mock_model_manager)
        self.assertEqual(engine.image_handler, mock_image_handler)
        self.assertEqual(engine.device_manager, mock_device_manager)

    def test_factory_dependency_injection(self):
        """测试工厂的依赖注入"""
        from src.core.factory.ocr_engine_factory import OCREngineFactory

        mock_model_manager = MagicMock()
        mock_image_handler = MagicMock()
        mock_device_manager = MagicMock()

        # 注册测试引擎类
        class MockEngine:
            def __init__(
                self,
                model_path=None,
                device=None,
                prompt=None,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                model_manager=None,
                image_handler=None,
                device_manager=None,
            ):
                self.model_manager = model_manager
                self.image_handler = image_handler
                self.device_manager = device_manager

        OCREngineFactory.register_engine("test", MockEngine)

        # 测试依赖注入
        engine = OCREngineFactory.create_engine(
            engine_type="test",
            model_manager=mock_model_manager,
            image_handler=mock_image_handler,
            device_manager=mock_device_manager,
        )

        self.assertEqual(engine.model_manager, mock_model_manager)
        self.assertEqual(engine.image_handler, mock_image_handler)
        self.assertEqual(engine.device_manager, mock_device_manager)


if __name__ == "__main__":
    unittest.main()
