# __future__.annotations will become the default in Python 3.11
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Type
import torch
from rich.console import Console


if TYPE_CHECKING:
    from .history import History
    from .model import Model
    from .proto import Proto


class MissingKey(RuntimeError):
    pass


class NoFrame(RuntimeError):
    pass


# must be json serializable
DEFAULT_CFG = {
    "model_base_dir": "models",
    "history_dir": "history",
}


class SymbolTbl:
    def __init__(self):
        self.reset()

    def reset(self):
        # NOTE: reset should not reset the registered modules
        # These modules might be used in the following run
        # self.registry: Dict[Type, Dict[str, Type]] = {}

        self._frames: List[Dict[str, Any]] = []
        self._global: Dict[str, Any] = {}

        self.console = Console()

        self.cfg: MutableMapping[str, Any] = DEFAULT_CFG
        self.device_info: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.model: Optional[Model] = None
        self.history: Optional[History] = None
        self.proto: Optional[Proto] = None

    def set(self, key, value):
        """设置当前局部作用域的值

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Raises:
            NoFrame: when no local frame
        """
        if len(self._frames) == 0:
            raise NoFrame()
        self._frames[-1][key] = value

    def try_set(self, key, value) -> bool:
        """设置当前局部作用域的值，如果键冲突，返回False

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Raises:
            NoFrame: when no local frame

        Returns:
            bool: _description_
        """
        if len(self._frames) == 0:
            raise NoFrame()
        if key in self._frames[-1]:
            return False
        self._frames[-1][key] = value
        return True

    def get(self, key) -> Any:
        """查找值，如果局部作用域中没有，那么去全局查找，如果依然没有，那么会raise MissingKey

        Args:
            key (_type_): _description_

        Returns:
            Any: _description_
        """
        if len(self._frames) != 0 and key in self._frames[-1]:
            return self._frames[-1][key]
        return self.get_global(key)

    def try_get(self, key, default: Optional[Any] = None) -> Any:
        """查找值，如果局部作用域中没有，那么去全局查找，如果依然没有，那么返回default

        Args:
            key (_type_): _description_
            default (Optional[Any], optional): _description_. Defaults to None.

        Returns:
            Any: _description_
        """
        if len(self._frames) != 0 and key in self._frames[-1]:
            return self._frames[-1][key]
        return self.try_get_global(key, default)

    def pop(self, key) -> Any:
        if len(self._frames) == 0:
            raise NoFrame()
        if key not in self._frames[-1]:
            raise MissingKey()
        return self._frames[-1].pop(key)

    def try_pop(self, key, default: Optional[Any] = None) -> Any:
        if len(self._frames) == 0:
            raise NoFrame()
        return self._frames[-1].pop(key, default)

    def contains(self, key) -> bool:
        if len(self._frames) == 0:
            raise NoFrame()
        return key in self._frames[-1]

    def set_global(self, key, value):
        """设置全局作用域中的值

        Args:
            key (_type_): _description_
            value (_type_): _description_
        """
        self._global[key] = value

    def try_set_global(self, key, value) -> bool:
        """设置全局作用域中的值，如果键冲突，返回False

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Returns:
            bool: _description_
        """
        if key in self._global:
            return False
        self._global[key] = value
        return True

    def get_global(self, key) -> Any:
        """读取全局作用域中的值，再找不到raise MissingKey

        Args:
            key (_type_): _description_

        Raises:
            MissingKey: _description_

        Returns:
            Any: _description_
        """
        if key in self._global:
            return self._global[key]
        raise MissingKey()

    def try_get_global(self, key, default: Optional[Any] = None) -> Any:
        """读取全局作用域中的值，再找不到则返回default

        Args:
            key (_type_): _description_
            default (Optional[Any], optional): _description_. Defaults to None.

        Returns:
            Any: _description_
        """
        if key in self._global:
            return self._global[key]
        return default

    def pop_global(self, key) -> Any:
        if key not in self._global:
            raise MissingKey()
        return self._global.pop(key)

    def try_pop_global(self, key, default: Optional[Any] = None) -> Any:
        return self._global.pop(key, default)

    def contains_global(self, key) -> bool:
        return key in self._global

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        return self.contains(key) or self.contains_global(key)


_sym_tbl = SymbolTbl()


def sym_tbl() -> SymbolTbl:
    return _sym_tbl


class new_scope:
    def __init__(self) -> None:
        pass

    def __enter__(self):
        _sym_tbl._frames.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        _sym_tbl._frames.pop()