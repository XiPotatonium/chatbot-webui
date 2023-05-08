import sys
from loguru import logger
from ...sym import sym_tbl
from ...device import empty_cache
from ...state import State, ROLE_BOT, ROLE_USER, ROLE_SYSTEM
from ...history import append_last_message_binding
from .. import Model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, PreTrainedModel
from typing import Any, Dict, Tuple, Union, List
import torch
import gradio as gr


class LlamaHFModel(Model):
    @classmethod
    def load(cls):
        path = sym_tbl().cfg["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16,
        )
        if "lora_path" in sym_tbl().cfg:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                sym_tbl().cfg["lora_path"],
                torch_dtype=torch.float16,
            )
        model.half()
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
        del self.tokenizer
        empty_cache()

    def generate(
            self,
            state: State,
            binding: List,
            max_tokens,
            top_p,
            top_k,
            temperature,
            beams,
            **kwargs
    ):
        role_mapping = {ROLE_BOT: "### Response", ROLE_USER: "### Input", ROLE_SYSTEM: "### Instruction"}
#         def generate_prompt(instruction, input=None):
#             if input:
#                 return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# {instruction}
# ### Input:
# {input}
# ### Response:"""
#             else:
#                 return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# {instruction}
# ### Response:"""

        prompt = ""
        for message in state.history:
            role = role_mapping[message["role"]]
            text = message["content"]
            prompt += f"{role}:\n{text}\n"
        prompt += '{}:\n'.format(role_mapping[ROLE_BOT])
        # BELLE format:
        # role_mapping = {ROLE_BOT: "Assistant", ROLE_USER: "Human"}
        # for message in state.history:
        #     role = role_mapping[message["role"]]
        #     text = message["content"]
        #     prompt += f"{role}: {text} \n\n"
        # prompt += '{}: '.format(role_mapping[ROLE_BOT])

        # print(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(sym_tbl().device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=beams,
            **kwargs,
        )
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_tokens,
        )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        # print("bot: {}".format(output.split("### Response:")[1].strip()))
        output = output.strip()[len(prompt):].strip()
        # logger.debug(output)

        empty_cache()

        state.append_message_history(ROLE_BOT, output)
        return append_last_message_binding(state, binding)


LLAMA_HF_CSS = """
"""

def llama_hf_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_tokens = gr.Slider(minimum=1, maximum=2000, step=1, label='max_tokens', value=128)
        with gr.Row():
            top_p = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='top_p', value=0.75)
            top_k = gr.Slider(minimum=0, maximum=100, step=1, label='top_k', value=40)
        with gr.Row():
            temperature = gr.Slider(minimum=0., maximum=1.0, step=0.01, label='temperature', value=0.1)
        with gr.Row():
            beams = gr.Slider(minimum=1, maximum=4, step=1, label='beams', value=4)

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_tokens, top_p, top_k, temperature, beams]
