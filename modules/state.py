from pathlib import Path


ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_BOT = "assistant"


class State:
    def __init__(self, folder: Path) -> None:
        # history in dict format
        self.history = []
        # base folder of mm in history
        self.folder = folder

    def append_message_history(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content
            # "media": [],          list of media in (path, mime) format, optional
        })
