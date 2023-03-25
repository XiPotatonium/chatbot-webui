import sys
from loguru import logger
import torch
from ...sym import sym_tbl
from ...device import empty_cache
from .. import Model
from ...history import History
from transformers import AutoModel, AutoTokenizer
from typing import Any, Dict, Tuple, Union
import gradio as gr


class ChatGLMHistory(History):
    def append_inference(self, item: Dict[str, Any]):
        self.inference.append((item["query"]['text'], item["response"]['text']))


class ChatGLMModel(Model):
    @classmethod
    def load(cls):
        tokenizer = AutoTokenizer.from_pretrained(sym_tbl().cfg["model_path"], trust_remote_code=True)
        model = AutoModel.from_pretrained(
            sym_tbl().cfg["model_path"], trust_remote_code=True,
            # device_map="auto"
        )

        if sym_tbl().device_info["device"] == "cpu":
            model = model.float()
        else:
            prec = sym_tbl().cfg["prec"]
            if prec == "fp16":
                model = model.half()
            elif prec == "int4":
                model = model.half().quantize(4)
            elif prec == "int8":
                model = model.half().quantize(8)
        model.to(sym_tbl().device)
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            logger.info("Use torch.compile")
            model = torch.compile(model)
        sym_tbl().model = cls(tokenizer, model)

    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def delete(self):
        del self.model
        del self.tokenizer
        empty_cache()

    def forward(self, max_tokens, top_p, temperature, **kwargs):
        query = sym_tbl().history.storage[-1]["query"]
        if len(query["mm_type"]) != 0:
            logger.warning(f"{self.__class__.__name__} is a text-only model, but got mm query. The media is ignored and only the text is used.")
        if len(query["instruction"]) != 0:
            logger.warning(f"{self.__class__.__name__} do not support instruction. It will be ignored")
        tquery = query["text"]

        output, _ = self.model.chat(
            self.tokenizer, query=tquery, history=sym_tbl().history.inference,
            max_length=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        empty_cache()
        sym_tbl().history.storage[-1]["response"]["text"] = output
        sym_tbl().history.append_last_inference()
        sym_tbl().history.append_last_response_binding()
        # logger.debug(sym_tbl().history.storage[-1])


CHATGLM_CSS = """
"""

def chatglm_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_tokens = gr.Slider(minimum=4, maximum=4096, step=4, label='max_tokens', value=2048)
        with gr.Row():
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='top_p', value=0.7)
            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='temperature', value=0.95)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_tokens, top_p, temperature]
