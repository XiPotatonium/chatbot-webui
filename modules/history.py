from abc import abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from loguru import logger
from .sym import sym_tbl
from datetime import datetime


class History:
    @staticmethod
    def storage_meta():
        return {
            "query": {
                "text": "",
                "mm_type": "",
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
        history = cls(folder.name, folder)
        with (folder / "meta.json").open('r', encoding="utf8") as rf:
            history.meta = json.load(rf)
        with (folder / "history.jsonl").open('r', encoding="utf8") as rf:
            for line in rf:
                item = json.loads(line)
                history.binding.append(history.storage2binding(item))
                history.inference.append(history.storage2inference(item))

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
        self.meta = meta

    def save(self):
        self.folder.mkdir(parents=True)
        with (self.folder / "meta.json").open('w', encoding="utf8") as wf:
            json.dump(self.meta, wf, ensure_ascii=False)
        with (self.folder / "history.jsonl").open('w', encoding="utf8") as wf:
            for inference_item, binding_item in zip(self.inference, self.binding):
                wf.write(json.dumps(self.ib2storage(inference_item, binding_item), ensure_ascii=False))
                wf.write("\n")

    def flush_last_rounds(self, pos: int = -1):
        with (self.folder / "history.jsonl").open('a', encoding="utf8") as wf:
            for inference_item, binding_item in zip(self.inference[pos:], self.binding[pos:]):
                wf.write(json.dumps(self.ib2storage(inference_item, binding_item), ensure_ascii=False))
                wf.write('\n')

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
    def storage2inference(self, item: Dict[str, Any]):
        """Used in loading history from file

        Args:
            item (Dict[str, Any]): _description_
        """
        pass

    def storage2binding(self, item: Dict[str, Any]) -> Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]:
        """Used in loading history from file

        Args:
            item (Dict[str, Any]): _description_

        Raises:
            ValueError: _description_

        Returns:
            Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]: _description_
        """
        def _s2b(info: Dict[str, Any]):
            if info["mm_type"] == "":
                return item["text"]
            else:
                return (self.folder / item["mm_path"], item["text"])
        return _s2b(item["query"]), _s2b(item["response"])

    def ib2storage(self, inference_item, binding_item: Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]):
        """You can override this method to add extra field in storage with inference_item
        Used in storing history

        Args:
            inference_item (_type_): _description_
            binding_item (Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]): _description_

        Returns:
            _type_: _description_
        """
        def _b2s(res: Dict[str, Any], info: Union[str, None, Tuple]):
            if isinstance(info, str):
                res["text"] = info
            elif isinstance(info, tuple):
                path, text = info
                if path.suffix == ".png":
                    res["mm_type"] = "Image"
                else:
                    raise NotImplementedError()
                res["text"] = text
                res["mm_path"] = path.name
        item = self.storage_meta()
        _b2s(item["query"], binding_item[0])
        _b2s(item["response"], binding_item[1])
        return item

    @abstractmethod
    def inference2binding(self, item) -> Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]:
        """Used in updating chatbot after inference

        Args:
            item (_type_): _description_

        Returns:
            Tuple[Union[str, None, Tuple], Union[str, None, Tuple]]: _description_
        """
        pass


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
