# __future__.annotations will become the default in Python 3.11
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Type
import torch
from rich.console import Console


if TYPE_CHECKING:
    from .model import Model
    from .proto import Proto


# must be json serializable
DEFAULT_CFG = {
    "model_base_dir": "models",
    "history_dir": "history",
    "tmp_dir": "tmp",
}


class SymbolTbl:
    def __init__(self):
        self.console = Console()

        self.cfg: MutableMapping[str, Any] = DEFAULT_CFG
        self.device_info: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.model: Optional[Model] = None
        self.proto: Optional[Proto] = None
        self.tmp_dir: Optional[Path] = None


_sym_tbl = SymbolTbl()


def sym_tbl() -> SymbolTbl:
    return _sym_tbl
