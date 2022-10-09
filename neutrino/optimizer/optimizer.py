import abc
from typing import Iterable

import numpy as np

from neutrino.tensor import Variable

__all__ = ["Optimizer", "SGD", "Adam"]


class Optimizer(abc.ABC):
    def __init__(self):
        self.parameters = []

    def set_parameters(self, parameters: Iterable[Variable]) -> None:
        self.parameters = parameters

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

    def step(self) -> None:
        for param in self.parameters:
            param.data = param.data - self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1

        self.means = []
        self.variances = []

    def set_parameters(self, parameters: Iterable[Variable]) -> None:
        super().set_parameters(parameters)
        self.means = [np.zeros_like(param.data) for param in parameters]
        self.variances = [np.zeros_like(param.data) for param in parameters]

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            self.means[i] = self.beta1 * self.means[i] + (1 - self.beta1) * param.grad
            self.variances[i] = self.beta2 * self.variances[i] + (1 - self.beta2) * np.power(
                param.grad, 2
            )
            mean = self.means[i] / (1 - np.power(self.beta1, self.t))
            variance = self.variances[i] / (1 - np.power(self.beta2, self.t))
            param.data = param.data - self.lr * mean / (np.sqrt(variance) + self.eps)
        self.t += 1
