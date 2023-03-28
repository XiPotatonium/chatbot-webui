import sys
from loguru import logger
import torch
from ...sym import sym_tbl
from ...device import empty_cache
from .. import Model
from ...history import History
from transformers import AutoModel, AutoTokenizer, BlipImageProcessor, PreTrainedTokenizer
from typing import Any, Dict, Tuple, Union
from .modeling_blip2chatglm import Blip2ChatGLM, Blip2ForChatGLM
from .modeling_chatglm import ChatGLMForConditionalGeneration
import gradio as gr
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image


class Blip2ChatGLMHistory(History):
    def append_inference(self, item: Dict[str, Any]):
        self.inference.append((item["query"]['text'], item["response"]['text']))


class Blip2ChatGLMModel(Model):
    @classmethod
    def load(cls):
        tokenizer = AutoTokenizer.from_pretrained(sym_tbl().cfg["lm_path"], trust_remote_code=True)
        lm = ChatGLMForConditionalGeneration.from_pretrained(
            sym_tbl().cfg["lm_path"], # device_map="auto"
        )

        if sym_tbl().device_info["device"] == "cpu":
            lm = lm.float()
        else:
            prec = sym_tbl().cfg["prec"]
            if prec == "fp16":
                lm = lm.half()
            elif prec == "int4":
                lm = lm.half().quantize(4)
            elif prec == "int8":
                lm = lm.half().quantize(8)

        blip2 = Blip2ForChatGLM.from_pretrained(sym_tbl().cfg["model_path"],)

        model = Blip2ChatGLM(blip2, lm)
        model.to(sym_tbl().device)
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            logger.info("Use torch.compile")
            model = torch.compile(model)

        image_size = model.blip2.config.vision_config.image_size
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
        )

        sym_tbl().model = cls(tokenizer, image_processor, model)

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            pixel_processor: BlipImageProcessor,
            model: ChatGLMForConditionalGeneration
        ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pixel_processor = pixel_processor

    def delete(self):
        del self.model
        del self.tokenizer
        empty_cache()

    def stream_generate(self, max_tokens, top_p, temperature, **kwargs):
        query = sym_tbl().history.storage[-1]["query"]
        mm_type = query["mm_type"]
        if len(mm_type) != 0 and mm_type != "Image":
            logger.warning(f"{self.__class__.__name__} is a text-image model, but got {mm_type} query. The media is ignored and only the text is used.")
        if len(query["instruction"]) != 0:
            logger.warning(f"{self.__class__.__name__} do not support instruction. It will be ignored")
        if mm_type == "Image":
            pixel_values = self.pixel_processor(
                Image.open(sym_tbl().history.folder / query["mm_path"]).convert("RGB"), return_tensors="pt"
            ).pixel_values.to(sym_tbl().device)
            mm_query = (query["text"], pixel_values)
        else:
            mm_query = query["text"]

        for i, (output, _) in enumerate(self.model.stream_chat(
            self.tokenizer, query=mm_query, history=sym_tbl().history.inference,
            max_length=max_tokens,
            top_p=top_p,
            temperature=temperature
        )):
            sym_tbl().history.storage[-1]["response"]["text"] = output
            if i == 0:
                sym_tbl().history.append_last_response_binding()
            else:
                sym_tbl().history.update_last_response_binding()
            yield       # yield to indicate that stream update has finished
        sym_tbl().history.append_last_inference()
        # logger.debug(sym_tbl().history.storage[-1])
        empty_cache()


BLIP2CHATGLM_CSS = """
"""

def blip2chatglm_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_tokens = gr.Slider(minimum=4, maximum=4096, step=4, label='max_tokens', value=2048)
        with gr.Row():
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='top_p', value=0.7)
            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='temperature', value=0.95)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_tokens, top_p, temperature]
