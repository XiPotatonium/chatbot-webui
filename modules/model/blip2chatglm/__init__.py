import sys
from loguru import logger
import torch
from ...sym import sym_tbl
from ...state import State
from ...device import empty_cache
from .. import Model
from ...history import append_response_binding, update_response_binding
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
from peft import PeftModel
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
        history = []
        # DISCUSS: should we add a field to state to store chat history for inference?
        # PROS: save conversion time
        # CONS: memory consuming, especially for mm history
        for info in state.history:

            def convert(info: Dict[str, Any]):
                text = info["text"]
                mm_type = info["mm_type"]
                if len(info["mm_type"]) != 0:
                    mm_path = state.folder / info["mm_path"]
                    if info["mm_type"] == "Image":
                        pixel_values = self.pixel_processor(
                            Image.open(mm_path).convert("RGB"), return_tensors="pt"
                        ).pixel_values.to(sym_tbl().device)
                        return (text, pixel_values)
                    else:
                        logger.warning(
                            f"{self.__class__.__name__} is a text-image model, but got {mm_type} input."
                            "The media is ignored and only the text is used."
                        )
                return text

            history.append((convert(info["query"]), convert(info["response"])))
        query = history.pop()
        instruction = state.history[-1]["query"]["instruction"]
        if len(instruction) != 0:
            logger.warning(
                f"{self.__class__.__name__} will ignore instruction {instruction}."
            )
        query = query[0]
        return query, history

    def stream_generate(
        self,
        state: State,
        binding: List,
        max_tokens: int = 2048,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ):
        query, history = self.prepare_input(state)

        with torch.cuda.amp.autocast(enabled=True):
            for i, output in enumerate(
                self.model.stream_chat(
                    self.tokenizer,
                    query=query,
                    history=history,
                    max_length=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                )
            ):
                if i == 0:
                    yield append_response_binding(state, binding, output)
                else:
                    yield update_response_binding(state, binding, output)
        state.history[-1]["response"]["text"] = output
        empty_cache()


class Blip2ChatGLMLoraModel(Blip2ChatGLMModel):
    @classmethod
    def load(cls):
        tokenizer = AutoTokenizer.from_pretrained(
            sym_tbl().cfg["lm_path"], trust_remote_code=True
        )
        lm = ChatGLMForConditionalGeneration.from_pretrained(
            sym_tbl().cfg["lm_path"],  # device_map="auto"
        )

        if sym_tbl().device_info["device"] == "cpu":
            lm = lm.float()
        else:
            prec = sym_tbl().cfg["prec"]
            if prec == "fp16":
                lm = lm.half()
            elif prec == "int4":
                lm = lm.half().quantize(4)
            elif prec == "int8":
                lm = lm.half().quantize(8)

        blip2 = Blip2ForChatGLM.from_pretrained(
            sym_tbl().cfg["model_path"],
        )
        blip2_config = Blip2ChatGLMConfig.from_pretrained(
            sym_tbl().cfg["model_path"],
        )

        model = Blip2ChatGLM(blip2_config, blip2, lm)
        model = PeftModel.from_pretrained(
            model,
            sym_tbl().cfg["lora_path"],
            # torch_dtype=torch.float16,
        )
        model.to(sym_tbl().device)
        model.eval()

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     logger.info("Use torch.compile")
        #     model = torch.compile(model)

        image_size = model.blip2.config.vision_config.image_size
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size},
            image_mean=OPENAI_CLIP_MEAN,
            image_std=OPENAI_CLIP_STD,
        )

        sym_tbl().model = cls(tokenizer, image_processor, model)


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
