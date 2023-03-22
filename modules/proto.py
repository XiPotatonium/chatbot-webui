from dataclasses import dataclass
from pathlib import Path
from .model.chatglm import *
from .model.llama import *
from .model import Model
from .history import History
from .sym import sym_tbl
import re


@dataclass
class UIProto:
    css: str
    builder: callable


@dataclass
class Proto:
    model: Model
    history: History
    ui: UIProto


def get_model_item(item: str):
    parts = ["modules", "model"] + item.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def update_proto():
    proto_cfg = sym_tbl().cfg["proto"]
    sym_tbl().proto = Proto(
        model=get_model_item(proto_cfg["model"]),
        history=get_model_item(proto_cfg["history"]),
        ui=UIProto(
            css=get_model_item(proto_cfg["ui"]["css"]),
            builder=get_model_item(proto_cfg["ui"]["builder"]),
        )
    )
