from typing import Union, Tuple

import numpy as np

from tensor import Tensor

__all__ = [
    'Reshape',
    'Pad',
    'Multiply',
    'Sum',
    'Mean',
    'Power',
    'Relu',
    'Sigmoid'
]


class Reshape(Tensor):

    def __init__(self, tensor: Tensor, new_shape: Tuple[int, ...]):
        super().__init__()
        self.tensor = tensor
        self.new_shape = new_shape
        self.input_shape = None

    def forward(self) -> np.ndarray:
        input_tensor = self.tensor()
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(self.new_shape)

    def backward(self, grad: np.ndarray) -> None:
        self.tensor.backward(grad.reshape(self.input_shape))
        super().backward(grad)


class Tile(Tensor):

    def __init__(self, tensor: Tensor, repeats: Tuple[int, ...]):
        super().__init__()
        self.tensor = tensor
        self.repeats = repeats

    def forward(self) -> np.ndarray:
        return np.tile(self.tensor(), self.repeats)

    def backward(self, grad: np.ndarray) -> None:
        ...


class Pad(Tensor):

    def __init__(self, tensor: Tensor, padding: Tuple[Tuple[int, int], ...]):
        super().__init__()
        self.tensor = tensor
        self.padding = padding

        self.input_shape = None

    def forward(self) -> np.ndarray:
        self.input_shape = self.tensor().shape
        return np.pad(self.tensor.output, self.padding)

    def backward(self, grad: np.ndarray) -> None:
        unpad = tuple([slice(self.padding[i][0], self.padding[i][0] + self.input_shape[i]) for i in range(len(self.input_shape))])
        self.tensor.backward(grad[unpad])
        super().backward(grad)


class Multiply(Tensor):

    def __init__(self, tensor: Tensor, multiplicand: float):
        super().__init__()
        self.tensor = tensor
        self.multiplicand = multiplicand

    def forward(self) -> np.ndarray:
        return self.tensor() * self.multiplicand

    def backward(self, grad: np.ndarray) -> None:
        self.tensor.backward(grad * self.multiplicand)
        super().backward(grad)


class Sum(Tensor):

    def __init__(self, tensor: Tensor, axis: Union[None, int, Tuple[int]] = None):
        super().__init__()
        self.tensor = tensor
        self.axis = axis
        self.input_shape = None

    def forward(self) -> np.ndarray:
        input_tensor = self.tensor()
        self.input_shape = input_tensor.shape
        return np.sum(input_tensor, axis=self.axis)

    def backward(self, grad: np.ndarray) -> None:
        input_dim = len(self.input_shape)

        if self.axis is None:
            axis = tuple(range(input_dim))
        elif isinstance(self.axis, int):
            axis = (self.axis,)
        else:
            axis = self.axis

        grad_reshape = grad.reshape([1 if dim in axis else self.input_shape[dim] for dim in range(input_dim)])
        self.tensor.backward(np.tile(grad_reshape, [1 if dim not in axis else self.input_shape[dim] for dim in range(input_dim)]))

        super().backward(grad)


class Mean(Tensor):

    def __init__(self, tensor: Tensor, axis: Union[None, int, Tuple[int]] = None):
        super().__init__()
        self.tensor = tensor
        self.axis = axis
        self.input_shape = None

    def forward(self) -> np.ndarray:
        input_tensor = self.tensor()
        self.input_shape = input_tensor.shape
        return np.mean(input_tensor, axis=self.axis)

    def backward(self, grad: np.ndarray) -> None:
        input_dim = len(self.input_shape)

        if self.axis is None:
            axis = tuple(range(input_dim))
        elif isinstance(self.axis, int):
            axis = (self.axis,)
        else:
            axis = self.axis

        grad_reshape = grad.reshape([1 if dim in axis else self.input_shape[dim] for dim in range(input_dim)])
        repeats = [1 if dim not in axis else self.input_shape[dim] for dim in range(input_dim)]
        self.tensor.backward(np.tile(grad_reshape, repeats) / np.product(repeats))

        super().backward(grad)


class Power(Tensor):

    def __init__(self, tensor: Tensor, exponent: float):
        super().__init__()
        self.tensor = tensor
        self.exponent = exponent

    def forward(self) -> np.ndarray:
        return np.power(self.tensor(), self.exponent)

    def backward(self, grad: np.ndarray) -> None:
        self.tensor.backward(grad * self.exponent * np.power(self.tensor.output, self.exponent - 1))
        super().backward(grad)


class Exp(Tensor):

    def __init__(self, tensor: Tensor, base: float = None):
        super().__init__()
        self.tensor = tensor
        self.factor = np.log(base) if base is not None else 1

    def forward(self) -> np.ndarray:
        return np.power(self.base, self.tensor())

    def backward(self, grad: np.ndarray) -> None:
        self.tensor.backward(grad * np.power(self.base, self.tensor.output) * self.factor)
        super().backward(grad)


class Relu(Tensor):

    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self) -> np.ndarray:
        tensor_copy = self.tensor().copy()
        tensor_copy[tensor_copy < 0] = 0
        return tensor_copy

    def backward(self, grad: np.ndarray) -> None:
        relu_grad = np.zeros_like(grad)
        relu_grad[self.tensor.output > 0] = 1
        self.tensor.backward(grad * relu_grad)
        super().backward(grad)


class Sigmoid(Tensor):

    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.tensor()))

    def backward(self, grad: np.ndarray) -> None:
        self.tensor.backward(grad * self.output * (1 - self.output))
        super().backward(grad)
