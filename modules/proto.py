from dataclasses import dataclass
from pathlib import Path
from typing import Type
from .model import Model
from .sym import sym_tbl


@dataclass
class UIProto:
    css: str
    builder: callable


@dataclass
class Proto:
    path: str
    model: Type[Model]
    ui: UIProto


MAPPING = {
    "chatglm": Proto(
        "modules.model.chatglm", "ChatGLMModel", UIProto("CHATGLM_CSS", "chatglm_ui")
    ),
    "blip2chatglm": Proto(
        "modules.model.blip2chatglm",
        "Blip2ChatGLMModel",
        UIProto("BLIP2CHATGLM_CSS", "blip2chatglm_ui"),
    ),
    "llama-hf": Proto(
        "modules.model.llama", "LlamaHFModel", UIProto("LLAMA_HF_CSS", "llama_hf_ui")
    ),
    "chatgpt": Proto(
        "modules.model.chatgpt", "ChatGPTModel", UIProto("CHATGPT_CSS", "chatgpt_ui")
    ),
}


def get_model_item(module: str, item: str):
    parts = module.split(".") + item.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def update_proto():
    model_name = sym_tbl().cfg["model"]
    _proto = MAPPING[model_name]
    if isinstance(_proto.model, str):
        # not imported
        _proto = Proto(
            path=_proto.path,
            model=get_model_item(_proto.path, _proto.model),
            ui=UIProto(
                css=get_model_item(_proto.path, _proto.ui.css),
                builder=get_model_item(_proto.path, _proto.ui.builder),
            ),
        )
        MAPPING[model_name] = _proto
    sym_tbl().proto = _proto
