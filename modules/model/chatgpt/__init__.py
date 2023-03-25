import sys
from loguru import logger
from ...sym import sym_tbl
from ...device import empty_cache
from .. import Model
from ...history import History
import requests
from typing import Any, Dict, Tuple, Union
import gradio as gr


class ChatGPTHistory(History):
    def append_inference(self, item: Dict[str, Any]):
        self.append_user(item["query"]['text'])
        self.append_assistant(item["response"]['text'])

    def append_user(self, text: str):
        self.inference.append({"role": "user", "content": text})

    def append_assistant(self, text: str):
        self.inference.append({"role": "assistant", "content": text})


class ChatGPTModel(Model):
    @classmethod
    def load(cls):
        url = sym_tbl().cfg["url"]
        api_key = sym_tbl().cfg["api_key"]
        model = sym_tbl().cfg["model"]
        sym_tbl().model = cls(url, api_key, model)

    def __init__(self, url: str, api_key: str, model: str) -> None:
        self.url = url
        self.api_key = api_key
        self.model = model

    def delete(self):
        pass

    def forward(self, top_p, temperature, **kwargs):
        query = sym_tbl().history.storage[-1]["query"]
        if len(query["mm_type"]) != 0:
            logger.warning(f"{self.__class__.__name__} is a text-only model, but got mm query. The media is ignored and only the text is used.")
        if len(query["instruction"]) != 0:
            logger.warning(f"{self.__class__.__name__} do not support instruction. It will be ignored")
        tquery = query["text"]
        sym_tbl().history.append_user(tquery)

        header = {
            'Content-Type': 'application/json',
            # "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": sym_tbl().history.inference,
            # "prompt": "Explain what a named entity identification task is.",
            # "max_tokens": 1000,
            "temperature": temperature,
            "top_p": top_p,
            # "n": 1,
            # "stream": False,
            # "stop": [
            #     "\n"
            # ]
        }
        res = requests.post(self.url, headers=header, json=data)
        output = res["choices"][0]["message"]

        sym_tbl().history.storage[-1]["response"]["text"] = output
        sym_tbl().history.append_last_inference()
        sym_tbl().history.append_last_response_binding()
        # logger.debug(sym_tbl().history.storage[-1])


CHATGPT_CSS = """
"""

def chatgpt_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='top_p', value=1.0)
            temperature = gr.Slider(minimum=0., maximum=2.0, step=0.01, label='temperature', value=1.0)
    return [top_p, temperature]
