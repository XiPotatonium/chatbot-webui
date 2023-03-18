from loguru import logger
from ...sym import sym_tbl
from ...device import empty_cache
from .. import Model
from ...history import History
from transformers import AutoModel, AutoTokenizer
from typing import Any, Dict, Tuple, Union
import gradio as gr


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)


class ChatGLMHistory(History):
    def storage2inference(self, item: Dict[str, Any]):
        return (item["query"], item["response"])


class ChatGLMModel(Model):
    @classmethod
    def load(cls):
        tokenizer = AutoTokenizer.from_pretrained(sym_tbl().cfg["model_path"], trust_remote_code=True)
        model = AutoModel.from_pretrained(sym_tbl().cfg["model_path"], trust_remote_code=True)

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
        sym_tbl().model = cls(tokenizer, model)

    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def delete(self):
        del self.model
        del self.tokenizer
        empty_cache()

    def forward(self, max_length, top_p, temperature, **kwargs):
        query, _ = sym_tbl().history.binding[-1]
        if isinstance(query, str):
            iquery = query
        elif isinstance(query, tuple):
            logger.warning(f"{self.__class__.__name__} is a text-only model, but got mm query. The media is ignored and only the text is used.")
            _, iquery = query
            assert iquery is not None
        else:
            raise ValueError(f"query must be str for {self.__class__.__name__} model, but got {type(query)}")

        output, _ = self.model.chat(
            self.tokenizer, query=iquery, history=sym_tbl().history.inference,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
        )
        empty_cache()
        sym_tbl().history.binding[-1] = (query, output)
        sym_tbl().history.inference.append((iquery, output))


CHATGLM_CSS = """
"""

def chatglm_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_length = gr.Slider(minimum=4, maximum=4096, step=4, label='Max Length', value=2048)
        with gr.Row():
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.7)
        with gr.Row():
            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.95)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_length, top_p, temperature]
