import sys
import os
from loguru import logger
import torch
from ...sym import sym_tbl
from ...state import State, ROLE_BOT, ROLE_SYSTEM, ROLE_USER
from ...device import empty_cache
from .. import Model
from ...history import append_last_message_binding, update_last_message_binding
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipImageProcessor,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from typing import Any, Dict, List, Tuple, Union
import gradio as gr
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image


class Blip2ChatGLMModel(Model):
    @classmethod
    def load(cls):
        if sym_tbl().device_info["device"] == "cpu":
            lm_dtype = "fp32"
        else:
            lm_dtype = sym_tbl().cfg["prec"]

        model = AutoModelForCausalLM.from_pretrained(
            sym_tbl().cfg["model_path"], trust_remote_code=True
        )
        model.setup_dtype(vision_encoder_dtype="fp16", lm_dtype=lm_dtype)

        if "lora_path" in sym_tbl().cfg:
            from peft import PeftModel, LoraConfig, get_peft_model
            model.language_model = PeftModel.from_pretrained(
                model.language_model,
                sym_tbl().cfg["lora_path"],
                # torch_dtype=torch.float16,
            )
            # peft_config = LoraConfig.from_pretrained(sym_tbl().cfg["lora_path"])
            # model.language_model = get_peft_model(model.language_model, peft_config)
            # model.load_state_dict(torch.load(os.path.join(sym_tbl().cfg["lora_path"], "adapter_model.bin")), strict=False)

        model.to(sym_tbl().device)
        model.eval()

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     logger.info("Use torch.compile")
        #     model = torch.compile(model)

        tokenizer = AutoTokenizer.from_pretrained(
            sym_tbl().cfg["model_path"], trust_remote_code=True
        )

        image_size = model.config.vision_config.image_size
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size},
            image_mean=OPENAI_CLIP_MEAN,
            image_std=OPENAI_CLIP_STD,
        )

        sym_tbl().model = cls(tokenizer, image_processor, model)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pixel_processor: BlipImageProcessor,
        model: PreTrainedModel,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pixel_processor = pixel_processor

    def delete(self):
        del self.model
        empty_cache()

    def prepare_input(self, state: State):
        messages = []
        # DISCUSS: should we add a field to state to store chat history for inference?
        # PROS: save conversion time
        # CONS: memory consuming, especially for mm history
        role_mapping = {ROLE_USER: "问", ROLE_BOT: "答", ROLE_SYSTEM: "指令"}
        for message in state.history:
            role = role_mapping[message["role"]]
            text = message["content"]
            medias = []
            for mm_path, mime in message.get("media", []):
                if mime == "image/jpeg" or mime == "image/png":
                    pixel_values = self.pixel_processor(
                        Image.open(state.folder / mm_path).convert("RGB"), return_tensors="pt"
                    ).pixel_values.to(sym_tbl().device)
                    medias.append((pixel_values, 0))            # insert at index 0 by default
                else:
                    logger.warning(
                        f"{self.__class__.__name__} is a text-image model, but got {mime} input."
                        "The media is ignored and only the text is used."
                    )

            messages.append((role, text, medias))
        return messages

    def stream_generate(
        self,
        state: State,
        binding: List,
        max_tokens: int = 2048,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ):
        messages = self.prepare_input(state)

        # print(list(map(lambda x: (x[0], x[1], [(m.shape, pos) for m, pos in x[2]]), messages)))

        with torch.cuda.amp.autocast(enabled=True):
            for i, output in enumerate(
                self.model.stream_chat(
                    self.tokenizer,
                    messages=messages,
                    max_length=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                )
            ):
                if i == 0:
                    state.append_message_history(ROLE_BOT, output)
                    yield append_last_message_binding(state, binding)
                else:
                    state.history[-1]["content"] = output
                    yield update_last_message_binding(state, binding)
        empty_cache()


BLIP2CHATGLM_CSS = """
"""


def blip2chatglm_ui():
    with gr.Column(variant="panel"):
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=4, maximum=4096, step=4, label="max_tokens", value=2048
            )
        with gr.Row():
            top_p = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, label="top_p", value=0.7
            )
            temperature = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, label="temperature", value=0.95
            )

        # with gr.Row():
        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
    return [max_tokens, top_p, temperature]
