import sys
from loguru import logger
from ...sym import sym_tbl
from ...state import State
from ...history import append_response_binding, update_response_binding
from .. import Model
from typing import Any, Dict, Tuple, Union, List

import gradio as gr
import openai


class ChatGPTModel(Model):
    @classmethod
    def load(cls):
        api_key = sym_tbl().cfg["api_key"]
        model = sym_tbl().cfg["api_model"]
        # openai.organization = "xxx"
        openai.api_key = api_key
        # print(openai.Model.list())
        # TODO: check model in model list

        sym_tbl().model = cls(api_key, model)

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def delete(self):
        pass

    def stream_generate(
            self,
            state: State,
            binding: List,
            top_p: float,
            temperature: float,
            **kwargs
    ):
        history = []
        instruction = state.history[-1]["query"]["instruction"]
        if len(instruction) != 0:
            history.append({"role": "system", "content": instruction})
        for info in state.history:
            def convert(info: Dict[str, Any]):
                text = info["text"]
                mm_type = info["mm_type"]
                if len(info["mm_type"]) != 0:
                    logger.warning(
                        f"{self.__class__.__name__} is a text-only model, but got {mm_type} input."
                        "The media is ignored and only the text is used."
                    )
                return text
            history.append({"role": "user", "content": convert(info["query"])})
            history.append({"role": "assistant", "content": convert(info["response"])})
        history.pop()       # pop last assistant (it is empty)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=history,
            temperature=temperature,
            top_p=top_p,
            stream=True  # again, we set stream=True
        )

        # create variables to collect the stream of chunks
        output = ""
        # iterate through the stream of events
        for i, chunk in enumerate(response):
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            output += chunk_message.get('content', '')
            if i == 0:
                yield append_response_binding(state, binding, output)
            else:
                yield update_response_binding(state, binding, output)
        state.history[-1]["response"]["text"] = output


CHATGPT_CSS = """
"""

def chatgpt_ui():
    with gr.Column(variant="panel"):
        top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='top_p', value=1.0)
        temperature = gr.Slider(minimum=0.01, maximum=2.0, step=0.01, label='temperature', value=1.0)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [top_p, temperature]
