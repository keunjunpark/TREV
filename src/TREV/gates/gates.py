from abc import ABC, abstractmethod
from typing import List, Callable


class Gate(ABC):
    def __init__(self, matrix_fun:Callable, device:str):
        self.matrix_fun = matrix_fun
        self.device = device
    @abstractmethod
    def has_parameter(self):
        pass
