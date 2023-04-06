from abc import abstractclassmethod, abstractmethod
from typing import Iterator, List
from ..state import State

class Model:
    @abstractclassmethod
    def load(cls):
        pass

    def generate(self, state: State, binding: List, *args, **kwargs) -> List:
        """generation configs defined in model ui will be passed in args and kwargs

        Args:
            binding (List): chatbot binding

        Raises:
            NotImplementedError: _description_

        Returns:
            List: new chatbot binding
        """
        raise NotImplementedError()

    def stream_generate(
        self, state: State, binding: List, *args, **kwargs
    ) -> Iterator[List]:
        """see Model.generate

        Args:
            binding (List): _description_

        Raises:
            NotImplementedError: _description_

        Yields:
            Iterator[List]: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self):
        pass
