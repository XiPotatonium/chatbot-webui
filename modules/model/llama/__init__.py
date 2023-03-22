import sys
from loguru import logger
from ...sym import sym_tbl
from ...device import empty_cache
from .. import Model
from ...history import History
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from typing import Any, Dict, Tuple, Union
import torch
import gradio as gr


class LlamaHFHistory(History):
    def append_inference(self, item: Dict[str, Any]):
        # llama has not history
        # self.inference.append((item["query"]['text'], item["response"]['text']))
        pass


class LlamaHFModel(Model):
    @classmethod
    def load(cls):
        path = sym_tbl().cfg["model_path"]
        lora_path = sym_tbl().cfg["lora_path"]
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16,
        )
        model.half()
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

    def forward(self, max_length, topp, topk, temperature, beams, **kwargs):
        def generate_prompt(instruction, input=None):
            if input:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

        query = sym_tbl().history.storage[-1]["query"]
        if len(query["mm_type"]) != 0:
            logger.warning(f"{self.__class__.__name__} is a text-only model, but got mm query. The media is ignored and only the text is used.")
        tquery = query["text"]
        instruction = query["instruction"]

        prompt = generate_prompt(instruction, tquery)
        # print(f"usr: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(sym_tbl().device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=topp,
            top_k=topk,
            num_beams=beams,
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
        output = self.tokenizer.decode(s)
        # print("bot: {}".format(output.split("### Response:")[1].strip()))
        output = output.split("### Response:")[1].strip()

        empty_cache()
        sym_tbl().history.storage[-1]["response"]["text"] = output
        sym_tbl().history.append_last_inference()
        sym_tbl().history.append_last_response_binding()


LLAMA_HF_CSS = """
"""

def llama_hf_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_length = gr.Slider(minimum=1, maximum=2000, step=1, label='Max Length', value=128)
        with gr.Row():
            topp = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='Top P', value=0.75)
        with gr.Row():
            topk = gr.Slider(minimum=0, maximum=100, step=1, label='Top K', value=40)
        with gr.Row():
            temperature = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='Temperature', value=0.1)
        with gr.Row():
            beams = gr.Slider(minimum=1, maximum=4, step=1, label='Beams', value=4)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_length, topp, topk, temperature, beams]
