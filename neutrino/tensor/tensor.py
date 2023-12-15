import abc
from typing import Any

import numpy as np

__all__ = ["Tensor", "Matmul", "Add", "Subtract"]


class Tensor(abc.ABC):
    def __init__(self):
        self.output = None
        self.grad = None

    def __call__(self) -> np.ndarray:
        if self.output is None:
            self.output = self.forward()
        return self.output

    def __add__(self, other: Any):
        if isinstance(other, Tensor):
            return Add(self, other)
        else:
            raise NotImplemented

    def __sub__(self, other: Any):
        if isinstance(other, Tensor):
            return Subtract(self, other)
        else:
            raise NotImplemented

    def __matmul__(self, other: Any):
        if isinstance(other, Tensor):
            return Matmul(self, other)
        else:
            raise NotImplemented

    @abc.abstractmethod
    def forward(self) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def reset(self) -> None:
        self.output = None
        self.grad = None

        for attr in dir(self):
            if isinstance(getattr(self, attr), Tensor):
                getattr(self, attr).reset()


class Matmul(Tensor):
    def __init__(self, tensor1: Tensor, tensor2: Tensor):
        super().__init__()
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def forward(self) -> np.ndarray:
        return self.tensor1() @ self.tensor2()

    def backward(self, grad: np.ndarray) -> None:
        grad_dim = len(grad.shape)

        tensor1_dim = len(self.tensor1.output.shape)
        sum_axis1 = tuple(list(range(grad_dim))[:-tensor1_dim])
        tensor1_grad = np.sum(grad @ self.tensor2.output.swapaxes(-1, -2), axis=sum_axis1)

        tensor2_dim = len(self.tensor2.output.shape)
        sum_axis2 = tuple(list(range(grad_dim))[:-tensor2_dim])
        tensor2_grad = np.sum(self.tensor1.output.swapaxes(-1, -2) @ grad, axis=sum_axis2)

        self.tensor1.backward(tensor1_grad)
        self.tensor2.backward(tensor2_grad)

        super().backward(grad)


class Add(Tensor):
    def __init__(self, tensor1: Tensor, tensor2: Tensor):
        super().__init__()
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def forward(self) -> np.ndarray:
        return self.tensor1() + self.tensor2()

    def backward(self, grad: np.ndarray) -> None:
        grad_dim = len(grad.shape)

        tensor1_dim = len(self.tensor1.output.shape)
        sum_axis1 = tuple(list(range(grad_dim))[:-tensor1_dim])
        tensor1_grad = np.sum(grad, axis=sum_axis1)

        tensor2_dim = len(self.tensor2.output.shape)
        sum_axis2 = tuple(list(range(grad_dim))[:-tensor2_dim])
        tensor2_grad = np.sum(grad, axis=sum_axis2)

        self.tensor1.backward(tensor1_grad)
        self.tensor2.backward(tensor2_grad)

        super().backward(grad)


class Subtract(Tensor):
    def __init__(self, tensor1: Tensor, tensor2: Tensor):
        super().__init__()
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def forward(self) -> np.ndarray:
        return self.tensor1() - self.tensor2()

    def backward(self, grad: np.ndarray) -> None:
        grad_dim = len(grad.shape)

        tensor1_dim = len(self.tensor1.output.shape)
        sum_axis1 = tuple(list(range(grad_dim))[:-tensor1_dim])
        tensor1_grad = np.sum(grad, axis=sum_axis1)

        tensor2_dim = len(self.tensor2.output.shape)
        sum_axis2 = tuple(list(range(grad_dim))[:-tensor2_dim])
        tensor2_grad = np.sum(-grad, axis=sum_axis2)

        self.tensor1.backward(tensor1_grad)
        self.tensor2.backward(tensor2_grad)

        super().backward(grad)
