from abc import abstractmethod
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from loguru import logger
from .sym import sym_tbl
from .state import State
from datetime import datetime


def append_query_binding(state: State, binding: List, text: str, mm_type: str = "", mm_path: str = ""):
    if len(mm_type) != 0:
        binding.append(((str(state.folder / mm_path), mm_type), None))
        # if has mm, only create a new text response if not empty
        if len(text) != 0:
            binding.append((text, None))
    else:
        # if no mm, force create a new text response
        binding.append((text, None))
    return binding


def append_response_binding(state: State, binding: List, text: str, mm_type: str = "", mm_path: str = ""):
    if len(mm_type) != 0:
        binding.append((None, (state.folder / mm_path, mm_type)))
        # if has mm, only create a new text response if not empty
        if len(text) != 0:
            binding.append((None, text))
    else:
        # if no mm, force create a new text response
        binding.append((None, text))
    return binding


def update_response_binding(state: State, binding: List, text: str, mm_type: str = "", mm_path: str = ""):
    if len(mm_type) != 0:
        binding[-2] = (None, (state.folder / mm_path, mm_type))
    if len(text) != 0:
        binding[-1] = (None, text)
    return binding


def iter_binding(binding: List) -> Iterator[Dict[str, Any]]:
    """Iterate binding and yield in dict format, can be used in store or in infernce (as history)

    Args:
        binding (List): _description_

    Yields:
        Iterator[Dict[str, Any]]: _description_
    """
    if len(binding) == 0:
        return
    info = storage_meta()
    for (q, r) in binding:
        if q is not None:
            if len(info["response"]["text"]) != 0 or len(info["response"]["mm_type"]) != 0:
                yield info
                info = storage_meta()

            if isinstance(q, str):
                # text-only query
                info["query"]["text"] = q
            else:
                # mm query
                info["query"]["mm_path"] = q["name"]
                info["query"]["mm_type"] = q["alt_txt"]
        if r is not None:
            if isinstance(r, str):
                info["response"]["text"] = r
            else:
                info["response"]["mm_path"] = r["name"]
                info["response"]["mm_type"] = r["alt_text"]
    yield info


def load_history(state: State, path: str) -> List:
    """load history from storage to binding

    Args:
        state (State): _description_
        path (str): only dirname, assumed in history_dir

    Returns:
        List: _description_
    """
    path = Path(sym_tbl().cfg["history_dir"]) / path
    state.folder = path         # update folder
    binding = []
    state.history = []
    with (path / "history.jsonl").open('r', encoding="utf8") as rf:
        for line in rf:
            info = json.loads(line)
            state.history.append(info)
            if len(info["query"]["instruction"]) != 0:
                append_query_binding(state, binding, "[INSTRUCTION] {}".format(info["query"]["instruction"]))
            append_query_binding(state, binding, info["query"]["text"], info["query"]["mm_type"], info["query"]["mm_path"])
            append_response_binding(state, binding, info["response"]["text"], info["response"]["mm_type"], info["response"]["mm_path"])
    return binding



def save_mm(src_folder: Path, save_folder: Path, fname: str) -> str:
    # do not move if already here
    if len(fname) == 0 or src_folder == save_folder:
        return fname
    shutil.move(src_folder / fname, save_folder / fname)


def save_history(state: State, path: Optional[str] = None) -> str:
    """store history from binding to storage. Always create new history dir.

    Args:
        state (State): _description_
        path (Optional[str]): save dir (only name, assumed in history_dir). Create new if not exists. If None, create with timestamp.

    Returns:
        str: only dirname, in history_dir
    """
    if path is None:
        timestamp = str(datetime.now()).replace(' ', '_').replace(':', '-')
        path: Path = Path(sym_tbl().cfg["history_dir"]) / timestamp
    else:
        path: Path = Path(sym_tbl().cfg["history_dir"]) / path
    path.mkdir(exist_ok=True, parents=True)

    with (path / "history.jsonl").open('w', encoding="utf8") as wf:
        for info in state.history:
            save_mm(state.folder, path, info["query"]["mm_path"])
            save_mm(state.folder, path, info["response"]["mm_path"])
            wf.write(json.dumps(info, ensure_ascii=False) + "\n")

    state.folder = path         # update folder
    return path.name


def sync_last_history(state: State, path: str):
    """sync last history to storage

    Args:
        state (State): _description_
        path (str): save dir (only name, assumed in history_dir).
    """
    if path is None or len(path) == 0:
        return
    path: Path = Path(sym_tbl().cfg["history_dir"]) / path
    assert state.folder == path, f"sync_last_history: state.folder ({state.folder}) != {path}"
    with (path / "history.jsonl").open('a', encoding="utf8") as wf:
        info = state.history[-1]
        # save_mm(state.folder, path, info["query"]["mm_path"])
        # save_mm(state.folder, path, info["response"]["mm_path"])
        wf.write(json.dumps(info, ensure_ascii=False) + "\n")


def storage_meta():
    return {
        "query": {
            "instruction": "",
            "text": "",
            "mm_type": "",          # Image/Video/Audio
            "mm_path": "",
        },
        "response": {
            "text": "",
            "mm_type": "",
            "mm_path": "",
        },
    }


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)
