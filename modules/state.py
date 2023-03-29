from pathlib import Path


class State:
    def __init__(self, folder: Path) -> None:
        # state in dict format
        self.history = []
        # base folder of mm in history
        self.folder = folder

    def append_history_meta(self):
        self.history.append(
            {
                "query": {
                    "instruction": "",
                    "text": "",
                    "mm_type": "",          # Image/Video/Audio
                    "mm_path": "",          # only filename, not full path
                },
                "response": {
                    "text": "",
                    "mm_type": "",
                    "mm_path": "",
                },
            }
        )
