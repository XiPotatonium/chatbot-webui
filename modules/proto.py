from dataclasses import dataclass
from pathlib import Path
from .model import Model
from .sym import sym_tbl


@dataclass
class UIProto:
    css: str
    builder: callable


@dataclass
class Proto:
    model: Model
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
        ui=UIProto(
            css=get_model_item(proto_cfg["ui"]["css"]),
            builder=get_model_item(proto_cfg["ui"]["builder"]),
        )
    )
