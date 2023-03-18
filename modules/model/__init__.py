from abc import abstractclassmethod, abstractmethod
from ..sym import sym_tbl

class Model:
    @abstractclassmethod
    def load(cls):
        pass

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def delete(self):
        pass
