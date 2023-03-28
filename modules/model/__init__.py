from abc import abstractclassmethod, abstractmethod
from ..sym import sym_tbl

class Model:
    @abstractclassmethod
    def load(cls):
        pass

    def generate(self, *args, **kwargs):
        raise NotImplementedError()

    def stream_generate(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def delete(self):
        pass
