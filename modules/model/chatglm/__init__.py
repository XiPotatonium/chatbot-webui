import sys
from loguru import logger
import torch
from ...sym import sym_tbl
from ...state import State, ROLE_BOT, ROLE_SYSTEM, ROLE_USER
from ...device import empty_cache
from ...history import append_last_message_binding, update_last_message_binding
from .. import Model
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Any, Dict, Tuple, Union, List

import gradio as gr


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
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     logger.info("Use torch.compile")
        #     model = torch.compile(model)
        sym_tbl().model = cls(tokenizer, model)

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def delete(self):
        del self.model
        empty_cache()

    def stream_generate(
            self,
            state: State,
            binding: List,
            max_tokens: int = 2048,
            top_p: float = 0.7,
            temperature: float = 0.95,
            **kwargs
    ):
        history = []
        round_roles = []
        for message in state.history:
            text = message["content"]
            role = message["role"]
            media = message.get("media", None)
            if media:
                logger.warning(
                    f"{self.__class__.__name__} is a text-only model, but got {media} input."
                    "The media is ignored and only the text is used."
                )

            if len(round_roles) == 0 or role in round_roles[-1]:
                history.append(("", ""))
                round_roles.append({role})
            else:
                round_roles[-1].add(role)
            if role == ROLE_BOT:
                history[-1] = (history[-1][0], text)
            elif role == ROLE_USER:
                history[-1] = (text, history[-1][1])
            else:
                logger.warning(f"{self.__class__.__name__} got unknown role {role}. Ignored.")
        query = history.pop()[0]

        # print(history)
        # print(query)

        for i, (output, _) in enumerate(self.model.stream_chat(
            self.tokenizer, query=query, history=history,
            max_length=max_tokens,
            top_p=top_p,
            temperature=temperature
        )):
            if i == 0:
                state.append_message_history(ROLE_BOT, output)
                yield append_last_message_binding(state, binding)
            else:
                state.history[-1]["content"] = output
                yield update_last_message_binding(state, binding)
        empty_cache()


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
