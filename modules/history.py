from abc import abstractmethod
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from loguru import logger
from .sym import sym_tbl
from .state import State, ROLE_SYSTEM, ROLE_BOT, ROLE_USER
from datetime import datetime


def append_last_message_binding(state: State, binding: List):
    medias = []
    text = None
    message = state.history[-1]
    if message.get("media"):
        for mm_path, mime in message["media"]:
            medias.append((str(state.folder / mm_path), mime))
        # if has mm, only create a new text response if not empty
        if len(message["content"]) != 0:
            text = message["content"]
    else:
        # if no mm, force create a new text response
        text = message["content"]
    if message["role"] == ROLE_USER:
        for media in medias:
            binding.append((media, None))
        if text is not None:
            binding.append((text, None))
    elif message["role"] == ROLE_BOT:
        for media in medias:
            binding.append((None, media))
        if text is not None:
            binding.append((None, text))
    elif message["role"] == ROLE_SYSTEM:
        for media in medias:
            binding.append((media, None))
        text = message["content"]
        binding.append((f"[INSTRUCTION]{text}", None))
    else:
        raise ValueError(f"Unknown role {message['role']}")
    return binding


def update_last_message_binding(state: State, binding: List):
    medias = []
    text = None
    message = state.history[-1]
    if message.get("media"):
        for mm_path, mime in message["media"]:
            medias.append((str(state.folder / mm_path), mime))
        # if has mm, only create a new text response if not empty
        if len(message["content"]) != 0:
            text = message["content"]
    else:
        # if no mm, force create a new text response
        text = message["content"]
    if message["role"] == ROLE_USER:
        offset = 0
        if text is not None:
            binding[-1] = (text, None)
            offset = 1
        for i, media in enumerate(reversed(medias)):
            binding[-(offset + i + 1)] = (media, None)
    elif message["role"] == ROLE_BOT:
        offset = 0
        if text is not None:
            binding[-1] = (None, text)
            offset = 1
        for i, media in enumerate(reversed(medias)):
            binding[-(offset + i + 1)] = (None, media)
    elif message["role"] == ROLE_SYSTEM:
        text = message["content"]
        binding[-1] = (f"[INSTRUCTION]{text}", None)
        for i, media in enumerate(reversed(medias)):
            binding[-(i + 1)] = (media, None)
    else:
        raise ValueError(f"Unknown role {message['role']}")
    return binding


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
            append_last_message_binding(state, binding)
    return binding



def save_mm(src_folder: Path, save_folder: Path, fname: str):
    # do not move if already here
    if len(fname) == 0 or src_folder == save_folder:
        return
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
        for message in state.history:
            for mm_path, mime in message.get("media", []):
                save_mm(state.folder, path, mm_path)
            wf.write(json.dumps(message, ensure_ascii=False) + "\n")

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
        message = state.history[-1]
        for mm_path, mime in message.get("media", []):
            save_mm(state.folder, path, mm_path)
        wf.write(json.dumps(message, ensure_ascii=False) + "\n")


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
