from dataclasses import dataclass
from pathlib import Path
from .model.chatglm import *
from .model import Model
from .history import History
from .sym import sym_tbl
import re

@dataclass
class Proto:
    model: Model
    history: History
    css: str
    ui: callable

MAPPING = [
    (re.compile(r"chatglm-6b"), Proto(model=ChatGLMModel, history=ChatGLMHistory, css=CHATGLM_CSS, ui=chatglm_ui)),
]


def update_proto():
    modelname = Path(sym_tbl().cfg["model_path"]).name
    for rule, proto in MAPPING:
        if rule.match(modelname):
            sym_tbl().proto = proto
            break
    else:
        raise ValueError(f"Unknown model name: {modelname}. Available = {MAPPING}")
