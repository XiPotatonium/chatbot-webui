from abc import abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from .sym import sym_tbl
from datetime import datetime


class History:
    @staticmethod
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
            # You may add extra field in 2storage
        }

    @classmethod
    def new(cls):
        timestamp = str(datetime.now()).replace(' ', '_').replace(':', '-')
        history = cls(
            id=timestamp,
            folder=Path(sym_tbl().cfg["history_dir"]) / timestamp,
            meta=sym_tbl().cfg
        )

        sym_tbl().history = history

    @classmethod
    def load(cls, folder: Path):
        with (folder / "meta.json").open('r', encoding="utf8") as rf:
            meta = json.load(rf)
        history = cls(folder.name, folder, meta)
        with (folder / "history.jsonl").open('r', encoding="utf8") as rf:
            for line in rf:
                history.storage.append(json.loads(line))
                history.append_last_inference()
                history.append_last_query_binding()
                history.append_last_response_binding()

        sym_tbl().history = history

    def __init__(
            self,
            id: str,
            folder: Path,
            meta: Dict[str, Any],
        ):
        self.folder = folder
        self.id = id
        self.binding: List[Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]] = []           # Used in chatbot for rendering
        self.inference = []         # Used in model inference
        self.storage: List[Dict[str, Any]] = []
        self.meta = meta

    def save(self):
        self.folder.mkdir(parents=True)
        with (self.folder / "meta.json").open('w', encoding="utf8") as wf:
            json.dump(self.meta, wf, ensure_ascii=False)
        with (self.folder / "history.jsonl").open('w', encoding="utf8") as wf:
            for s_item in self.storage:
                wf.write(json.dumps(s_item, ensure_ascii=False))
                wf.write("\n")

    def flush_last_rounds(self, pos: int = -1):
        with (self.folder / "history.jsonl").open('a', encoding="utf8") as wf:
            for s_item in self.storage[pos:]:
                wf.write(json.dumps(s_item, ensure_ascii=False))
                wf.write("\n")

    def save_media(self, media) -> Path:
        from PIL import Image
        def next_mm_id(path: Path):
            res = -1
            for f in path.iterdir():
                if f.suffix in {".png"}:
                    try:
                        res = max(res, int(f.stem))
                    except ValueError:
                        pass
            return res + 1
        if isinstance(media, Image.Image):
            mm_path = self.folder / f"{next_mm_id(self.folder)}.png"
            media.save(mm_path, "PNG")
        else:
            raise ValueError(f"Unknown media type: {type(media)}")
        return mm_path

    @abstractmethod
    def append_inference(self, item: Dict[str, Any]):
        """Used in loading history from file

        Args:
            item (Dict[str, Any]): _description_
        """
        pass

    def append_last_inference(self):
        self.append_inference(self.storage[-1])

    def append_last_query_binding(self):
        if (
            (len(self.storage) > 1 and self.storage[-1]["query"]["instruction"] != self.storage[-2]["query"]["instruction"]) or
            (len(self.storage) == 1 and len(self.storage[-1]["query"]["instruction"]) != 0)
        ):
            self.binding.append((f"[INSTRUCTION]: {self.storage[-1]['query']['instruction']}", None))

        info = self.storage[-1]["query"]
        if len(info["mm_type"]) != 0:
            self.binding.append(((self.folder / info["mm_path"], ), None))
        if len(info["text"]) != 0:
            self.binding.append((info["text"], None))

    def append_last_response_binding(self):
        info = self.storage[-1]["response"]
        if len(info["mm_type"]) != 0:
            self.binding.append((None, (self.folder / info["mm_path"],)))
        if len(info["text"]) != 0:
            self.binding.append((None, info["text"]))


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
