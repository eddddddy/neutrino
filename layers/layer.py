import abc
from typing import Iterable

from tensor import Tensor, Variable


class Layer(abc.ABC):
    def __init__(self):
        self.parameters = []

    def register_parameters(self, parameters: Iterable[Variable]) -> None:
        ...

    def count_parameters(self) -> int:
        return sum([np.product(parameter.data.shape) for parameter in self.parameters])

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
