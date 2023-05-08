import sys
from loguru import logger
from ...sym import sym_tbl
from ...state import State, ROLE_SYSTEM, ROLE_BOT, ROLE_USER
from ...history import append_last_message_binding, update_last_message_binding
from .. import Model
from typing import Any, Dict, Tuple, Union, List

import gradio as gr
import openai


class ChatGPTModel(Model):
    @classmethod
    def load(cls):
        model = sym_tbl().cfg["api_model"]
        # openai.organization = "xxx"
        # print(openai.Model.list())
        # TODO: check model in model list

        sym_tbl().model = cls(model)

    def __init__(self, model: str) -> None:
        self.model = model

    def delete(self):
        pass

    def stream_generate(
            self,
            state: State,
            binding: List,
            api_key: str,
            top_p: float,
            temperature: float,
            **kwargs
    ):
        openai.api_key = api_key

        role_mapping = {ROLE_BOT: "assistant", ROLE_USER: "user", ROLE_SYSTEM: "system"}

        messages = []
        for message in state.history:
            media = message.get("media", None)
            if media:
                logger.warning(
                    f"{self.__class__.__name__} is a text-only model, but got {media} input."
                    "The media is ignored and only the text is used."
                )
            messages.append({"role": role_mapping[message["role"]], "content": message["content"]})

        print(messages)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
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
                state.append_message_history(ROLE_BOT, output)
                yield append_last_message_binding(state, binding)
            else:
                state.history[-1]["content"] = output
                yield update_last_message_binding(state, binding)


CHATGPT_CSS = """
"""

def chatgpt_ui():
    with gr.Column(variant="panel"):
        api_key = gr.Textbox(label="API Key", lines=1, default="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='top_p', value=1.0)
        temperature = gr.Slider(minimum=0.01, maximum=2.0, step=0.01, label='temperature', value=1.0)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [api_key, top_p, temperature]
