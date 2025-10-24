import math
from typing import List, Tuple, Optional, Union

import torch
# 延迟导入torchvision，避免在不需要时加载
try:
    import torchvision.transforms as T
except ImportError:
    T = None

from PIL import Image, ImageOps
from transformers import AutoProcessor, BatchFeature, LlamaTokenizerFast, ProcessorMixin

# 导入日志模块
from src.core.logging import get_logger

# 获取日志记录器
logger = get_logger()

# 修复配置导入问题
try:
    from src.core.config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, MIN_CROPS, MAX_CROPS, PROMPT, get_tokenizer
    logger.debug("成功从src.core.config导入配置")
except ImportError:
    # 如果直接运行此文件，使用默认值
    logger.warning("无法从src.core.config导入配置，使用默认值")
    IMAGE_SIZE = 640
    BASE_SIZE = 1024
    CROP_MODE = True
    MIN_CROPS = 2
    MAX_CROPS = 6
    PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
    MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
    
    # 延迟导入TOKENIZER，避免在不需要时加载依赖
    TOKENIZER = None

    def get_tokenizer():
        """延迟加载tokenizer"""
        global TOKENIZER
        if TOKENIZER is None:
            from transformers import AutoTokenizer
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return TOKENIZER

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # logger.debug(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def count_tiles(orig_width, orig_height, min_num=MIN_CROPS, max_num=MAX_CROPS, image_size=640, use_thumbnail=False):
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # logger.debug(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    return target_aspect_ratio


def dynamic_preprocess(image, min_num=MIN_CROPS, max_num=MAX_CROPS, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # logger.debug(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # logger.debug(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


class ImageTransform:

    def __init__(self,
                 mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 normalize: bool = True):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        if T is not None:
            transform_pipelines: List = [T.ToTensor()]

            if normalize and T is not None:
                transform_pipelines.append(T.Normalize(mean, std))

            self.transform = T.Compose(transform_pipelines) if T is not None else None
        else:
            self.transform = None

    def __call__(self, pil_img: Image.Image):
        if self.transform is not None:
            x = self.transform(pil_img)
            return x
        else:
            # 如果没有torchvision，返回原始图像
            return pil_img


class DeepseekOCRProcessor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer = None,
        candidate_resolutions: Tuple[Tuple[int, int], ...] = ((1024, 1024),),
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        logger.debug("开始初始化DeepseekOCRProcessor")

        # self.candidate_resolutions = candidate_resolutions # placeholder no use
        self.image_size = IMAGE_SIZE
        self.base_size = BASE_SIZE
        # self.patch_size = patch_size
        self.patch_size = 16 
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        # self.downsample_ratio = downsample_ratio
        self.downsample_ratio = 4

        self.image_transform = ImageTransform(mean=image_mean, std=image_std, normalize=normalize)
        logger.debug("ImageTransform初始化完成")

        # 使用延迟加载的tokenizer
        self.tokenizer = tokenizer or get_tokenizer()
        logger.debug(f"tokenizer初始化完成，类型: {type(self.tokenizer)}")
        # self.tokenizer = add_special_token(tokenizer)
        self.tokenizer.padding_side = 'left'  # must set this，padding side with make a difference in batch inference

        # add the pad_token as special token to use 'tokenizer.pad_token' and 'tokenizer.pad_token_id'
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})

        # add image token
        # image_token_id = self.tokenizer.vocab.get(image_token)
        # if image_token_id is None:
        #     special_tokens = [image_token]
        #     special_tokens_dict = {"additional_special_tokens": special_tokens}
        #     self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        logger.debug(f"image_token_id: {self.image_token_id}")

        # add five special tokens for grounding-related tasks
        # <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>
        # special_tokens = ['<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>']
        # special_tokens_dict = {"additional_special_tokens": special_tokens}

        # special_tokens = ['<image>','<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>', '<td>', '</td>', '<tr>', '</tr>']
        # special_tokens_dict = {"additional_special_tokens": special_tokens}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        # # add special tokens for SFT data
        # special_tokens = ["<|User|>", "<|Assistant|>"]
        # special_tokens_dict = {"additional_special_tokens": special_tokens}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(
            self.tokenizer,
            **kwargs,
        )
        logger.debug("DeepseekOCRProcessor初始化完成")

    # def select_best_resolution(self, image_size):
    #     # used for cropping
    #     original_width, original_height = image_size
    #     best_fit = None
    #     max_effective_resolution = 0
    #     min_wasted_resolution = float("inf")

    #     for width, height in self.candidate_resolutions:
    #         scale = min(width / original_width, height / original_height)
    #         downscaled_width, downscaled_height = int(
    #             original_width * scale), int(original_height * scale)
    #         effective_resolution = min(downscaled_width * downscaled_height,
    #                                    original_width * original_height)
    #         wasted_resolution = (width * height) - effective_resolution

    #         if effective_resolution > max_effective_resolution or (
    #                 effective_resolution == max_effective_resolution
    #                 and wasted_resolution < min_wasted_resolution):
    #             max_effective_resolution = effective_resolution
    #             min_wasted_resolution = wasted_resolution
    #             best_fit = (width, height)

    #     return best_fit

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
        self,
        prompt: str,
        images: List,
        inference_mode: bool = True,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - pixel_values (torch.FloatTensor): [n_patches, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (prompt is not None and images is not None
                ), "prompt and images must be used at the same time."

        sft_format = prompt

        input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, _ = images[0]


        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "images_crop": images_crop,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "num_image_tokens": num_image_tokens,
        }


        # prepare = BatchFeature(
        #     data=dict(
        #         input_ids=input_ids,
        #         pixel_values=pixel_values,
        #         images_crop = images_crop,
        #         images_seq_mask=images_seq_mask,
        #         images_spatial_crop=images_spatial_crop,
        #         num_image_tokens=num_image_tokens,
        #     ),
        #     tensor_type="pt",
        # )
        # return prepare

    # 简化__call__方法以避免签名不匹配问题
    def __call__(self, *args, **kwargs):
        """
        兼容ProcessorMixin的__call__方法
        
        Returns:
            outputs (BaseProcessorOutput): the output of the processor
        """
        
        # 处理参数
        text = kwargs.get("text", None)
        images = kwargs.get("images", None)
        prompt = kwargs.get("prompt", text)
        image_list = kwargs.get("image", images)
        inference_mode = kwargs.get("inference_mode", True)
        
        # 如果传入的是单个图像，转换为列表
        if image_list is not None and not isinstance(image_list, list):
            image_list = [image_list]
            
        # 确保参数有效
        if prompt is None or image_list is None:
            # 直接返回空的BatchFeature而不是调用父类方法
            return BatchFeature(data={}, tensor_type="pt")
            
        prepare = self.process_one(
            prompt=prompt,
            images=image_list,
            inference_mode=inference_mode,
        )

        # 转换为BatchFeature以保持兼容性
        return BatchFeature(data=prepare, tensor_type="pt")

    def tokenize_with_images(
        self,
        # conversation: str,
        images: List[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        """Tokenize text with <image> tags."""
        try:
            logger.debug("开始tokenize_with_images方法")
            logger.debug(f"输入参数: images数量={len(images) if images else 0}, bos={bos}, eos={eos}, cropping={cropping}")
            # print(conversation)
            conversation = PROMPT
            logger.debug(f"conversation: {conversation}")
            logger.debug(f"images长度: {len(images)}")
            assert conversation.count(self.image_token) == len(images)
            text_splits = conversation.split(self.image_token)
            logger.debug(f"text_splits: {text_splits}")
            images_list, images_crop_list, images_seq_mask, images_spatial_crop = [], [], [], []
            image_shapes = []
            num_image_tokens = []
            tokenized_str = []
            # print('image: ', len(images))
            for i, (text_sep, image) in enumerate(zip(text_splits, images)):
                logger.debug(f"处理第{i}个图像，text_sep: '{text_sep}', image尺寸: {image.size if image else 'None'}")
                if image is None:
                    logger.error(f"第{i}个图像为None")
                    raise ValueError(f"第{i}个图像为None")
                """encode text_sep"""
                tokenized_sep = self.encode(text_sep, bos=False, eos=False)
                logger.debug(f"tokenized_sep: {tokenized_sep}")
                tokenized_str += tokenized_sep
                images_seq_mask += [False] * len(tokenized_sep)

                """select best resolution for anyres"""
                # if cropping:
                #     best_width, best_height = self.select_best_resolution(image.size)
                # else:
                #     best_width, best_height = self.image_size, self.image_size

                image_shapes.append(image.size)

                # 初始化images_crop_raw
                images_crop_raw = []
                
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                else:
                    if cropping:
                        # print('image-size: ', image.size)
                        # print('open_size:', image.size)
                        logger.debug(f"开始动态预处理图像，尺寸: {image.size}")
                        images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=IMAGE_SIZE)
                        logger.debug(f"动态预处理完成，crop_ratio: {crop_ratio}, images_crop_raw数量: {len(images_crop_raw) if images_crop_raw else 0}")
                        # print('crop_ratio: ', crop_ratio)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                # print(crop_ratio)
                """process the global view"""

                # if cropping
                if self.image_size <= 640 and not cropping:
                    # print('directly resize')
                    # 使用高质量的重采样方法
                    logger.debug("直接调整图像大小")
                    image = image.resize((self.image_size, self.image_size), resample=Image.Resampling.LANCZOS)

                logger.debug("开始处理全局视图")
                global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                        color=tuple(int(x * 255) for x in self.image_transform.mean))
                images_list.append(self.image_transform(global_view))
                logger.debug("全局视图处理完成")

                """record height / width crop num"""
                # width_crop_num, height_crop_num = best_width // self.image_size, best_height // self.image_size
                num_width_tiles, num_height_tiles = crop_ratio
                images_spatial_crop.append([num_width_tiles, num_height_tiles])




                if num_width_tiles > 1 or num_height_tiles > 1:
                    """process the local views"""
                    # 确保images_crop_raw已定义且不为空
                    if images_crop_raw:
                        logger.debug(f"处理局部视图，数量: {len(images_crop_raw)}")
                        for i in range(len(images_crop_raw)):
                            images_crop_list.append(self.image_transform(images_crop_raw[i]))
                        logger.debug("局部视图处理完成")

                # """process the global view"""
                # global_view = ImageOps.pad(image, (self.image_size, self.image_size),
                #                            color=tuple(int(x * 255) for x in self.image_transform.mean))
                # images_list.append(self.image_transform(global_view))

                # """process the local views"""
                # local_view = ImageOps.pad(image, (best_width, best_height),
                #                           color=tuple(int(x * 255) for x in self.image_transform.mean))
                # for i in range(0, best_height, self.image_size):
                #     for j in range(0, best_width, self.image_size):
                #         images_list.append(
                #             self.image_transform(local_view.crop((j, i, j + self.image_size, i + self.image_size))))

                # """add image tokens"""
                """add image tokens"""
                logger.debug("开始添加图像tokens")
                num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
                num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)


                tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
                tokenized_image += [self.image_token_id]
                if num_width_tiles > 1 or num_height_tiles > 1:
                    tokenized_image += ([self.image_token_id] * (num_queries * num_width_tiles) + [self.image_token_id]) * (
                                num_queries * num_height_tiles)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                num_image_tokens.append(len(tokenized_image))
                logger.debug("图像tokens添加完成")

            """process the last text split"""
            logger.debug("处理最后一个文本分割")
            tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            """add the bos and eos tokens"""
            logger.debug("添加bos和eos tokens")
            if bos:
                tokenized_str = [self.bos_id] + tokenized_str
                images_seq_mask = [False] + images_seq_mask
            if eos:
                tokenized_str = tokenized_str + [self.eos_id]
                images_seq_mask = images_seq_mask + [False]

            assert len(tokenized_str) == len(
                images_seq_mask), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"
        


            masked_tokenized_str = []
            for token_index in tokenized_str:
                if token_index != self.image_token_id:
                    masked_tokenized_str.append(token_index)
                else:
                    masked_tokenized_str.append(self.ignore_id)

            assert len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str), \
                (f"tokenized_str's length {len(tokenized_str)}, input_ids' length {len(masked_tokenized_str)}, "
                 f"imags_seq_mask's length {len(images_seq_mask)}, are not equal")

            logger.debug("开始创建张量")
            input_ids = torch.LongTensor(tokenized_str)
            target_ids = torch.LongTensor(masked_tokenized_str)
            images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

            # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
            target_ids[(input_ids < 0) |
                       (input_ids == self.image_token_id)] = self.ignore_id
            input_ids[input_ids < 0] = self.pad_id

            inference_mode = True

            if inference_mode:
                # Remove the ending eos token
                assert input_ids[-1] == self.eos_id
                input_ids = input_ids[:-1]
                target_ids = target_ids[:-1]
                images_seq_mask = images_seq_mask[:-1]

            if len(images_list) == 0:
                logger.debug("images_list为空，创建默认张量")
                pixel_values = torch.zeros((1, 3, self.base_size, self.base_size))
                images_spatial_crop = torch.zeros((1, 1), dtype=torch.long)
                images_crop = torch.zeros((1, 3, self.image_size, self.image_size)).unsqueeze(0)
            else:
                logger.debug(f"images_list不为空，数量: {len(images_list)}")
                pixel_values = torch.stack(images_list, dim=0)
                images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
                if images_crop_list:
                    logger.debug(f"images_crop_list不为空，数量: {len(images_crop_list)}")
                    images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)
                else:
                    logger.debug("images_crop_list为空，创建默认张量")
                    images_crop = torch.zeros((1, 3, self.image_size, self.image_size)).unsqueeze(0)

            input_ids = input_ids.unsqueeze(0)
            logger.debug("张量创建完成")

            # 调试信息：打印各个张量的形状
            logger.debug(f"input_ids shape: {input_ids.shape}")
            logger.debug(f"pixel_values shape: {pixel_values.shape}")
            logger.debug(f"images_crop shape: {images_crop.shape}")
            logger.debug(f"images_spatial_crop shape: {images_spatial_crop.shape}")
            
            # 确保所有返回值都不是None
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
            
            # 返回与原始仓库一致的数据结构
            result = [[input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes]]
            
            # 添加详细的调试信息
            logger.debug(f"tokenize_with_images方法返回结果，result长度: {len(result)}")
            if result and len(result) > 0 and result[0]:
                logger.debug(f"result[0]长度: {len(result[0])}")
                for i, item in enumerate(result[0]):
                    logger.debug(f"result[0][{i}]类型: {type(item)}, 值是否为None: {item is None}")
                    if item is not None:
                        logger.debug(f"result[0][{i}]값的类型: {type(item)}")
                        if isinstance(item, torch.Tensor):
                            logger.debug(f"result[0][{i}]张量形状: {item.shape}")
            
            logger.debug(f"即将返回result: {type(result)}, 长度: {len(result)}")
            if result and len(result) > 0:
                logger.debug(f"result[0]类型: {type(result[0])}, 长度: {len(result[0]) if result[0] else 'N/A'}")
            
            return result
        except Exception as e:
            logger.error(f"tokenize_with_images方法中发生错误: {str(e)}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"tokenize_with_images方法中发生错误: {str(e)}") from e

AutoProcessor.register("DeepseekVLV2Processor", DeepseekOCRProcessor)