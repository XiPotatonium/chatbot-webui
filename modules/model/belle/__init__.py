import sys
from loguru import logger
from ...sym import sym_tbl
from ...state import State
from ...device import empty_cache
from ...history import append_response_binding, update_response_binding
from .. import Model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, PreTrainedModel
from typing import Any, Dict, Tuple, Union, List
import torch
import gradio as gr


class BelleLlamaModel(Model):
    @classmethod
    def load(cls):
        path = sym_tbl().cfg["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, low_cpu_mem_usage=True, trust_remote_code=True,
        )

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

    def generate(
            self,
            state: State,
            binding: List,
            max_tokens=200,
            do_sample=True,
            top_p=0.85,
            top_k=30,
            temperature=0.35,
            repetition_penalty=1.2,
            **kwargs
    ):
        history = []
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
            history.append((convert(info["query"]), convert(info["response"])))
        query = history.pop()
        instruction = state.history[-1]["query"]["instruction"]
        if len(instruction) != 0:
            logger.warning(f"{self.__class__.__name__} will ignore instruction {instruction}.")
        query = query[0]

        prompt = ""
        for q, r in history:
            prompt += f"Human: {q} \n\nAssistant:{r} \n\n"
        prompt += f'Human: {query} \n\nAssistant:'

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(sym_tbl().device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
            eos_token_id=2, bos_token_id=1, pad_token_id=0,
            **kwargs,
        )
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            # return_dict_in_generate=True,
            # output_scores=True,
        )
        s = generation_output[0]
        output = self.tokenizer.decode(s, skip_special_token=True, clean_up_tokenization_spaces=False)
        # logger.debug(output)
        output = output.strip()[len(prompt):].strip()
        # logger.debug(output)

        empty_cache()

        state.history[-1]["response"]["text"] = output
        return append_response_binding(state, binding, output)


BELLE_CSS = """
"""

def belle_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_tokens = gr.Slider(minimum=1, maximum=2000, step=1, label='max_tokens', value=200)
        with gr.Row():
            top_p = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='top_p', value=0.85)
            top_k = gr.Slider(minimum=0, maximum=100, step=1, label='top_k', value=30)
        with gr.Row():
            do_sample = gr.Checkbox(label='do_sample', value=True)
            temperature = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='temperature', value=0.5)
        with gr.Row():
            repetition_penalty = gr.Slider(minimum=1, maximum=4, step=0.1, label='repetition_penalty', value=1.)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_tokens, do_sample, top_p, top_k, temperature, repetition_penalty]