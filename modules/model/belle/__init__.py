import sys
from loguru import logger
from ...sym import sym_tbl
from ...device import empty_cache
from .. import Model
from ...history import History
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Any, Dict, Tuple, Union
import torch
import gradio as gr


class BelleHistory(History):
    def append_inference(self, item: Dict[str, Any]):
        # belle has not history
        # self.inference.append((item["query"]['text'], item["response"]['text']))
        pass


class BelleModel(Model):
    @classmethod
    def load(cls):
        path = sym_tbl().cfg["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
        )

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

    def forward(
            self,
            max_length=200,
            do_sample=True,
            topp=0.85,
            topk=30,
            temperature=0.35,
            repetition_penalty=1.2,
            **kwargs
    ):
        query = sym_tbl().history.storage[-1]["query"]
        if len(query["mm_type"]) != 0:
            logger.warning(f"{self.__class__.__name__} is a text-only model, but got mm query. The media is ignored and only the text is used.")
        if len(query["instruction"]) != 0:
            logger.warning(f"{self.__class__.__name__} do not support instruction. It will be ignored")
        tquery = query["text"]

        prompt = 'Human: {}\n\nAssistant:'.format(tquery)
        # print(f"usr: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(sym_tbl().device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=topp,
            top_k=topk,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_length,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_token=True)
        # print("bot: {}".format(output.split("### Response:")[1].strip()))
        logger.debug(prompt)
        output = output[len(prompt):].strip()

        empty_cache()
        sym_tbl().history.storage[-1]["response"]["text"] = output
        sym_tbl().history.append_last_inference()
        sym_tbl().history.append_last_response_binding()


BELLE_CSS = """
"""

def belle_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_length = gr.Slider(minimum=1, maximum=2000, step=1, label='max_length', value=200)
        with gr.Row():
            topp = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='topp', value=0.85)
            topk = gr.Slider(minimum=0, maximum=100, step=1, label='topk', value=30)
        with gr.Row():
            do_sample = gr.Checkbox(label='do_sample', value=True)
            temperature = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='temperature', value=0.35)
        with gr.Row():
            repetition_penalty = gr.Slider(minimum=1, maximum=4, step=0.1, label='repetition_penalty', value=1.2)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_length, do_sample, topp, topk, temperature, repetition_penalty]