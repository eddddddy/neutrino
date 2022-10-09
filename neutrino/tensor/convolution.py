from typing import Union, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided

from neutrino.tensor import Tensor

__all__ = ["Convolve1D", "Convolve2D", "MaxPool1D", "MaxPool2D"]


def im2col_1d(A: np.ndarray, filter_size: int, stride: int = 1) -> np.ndarray:
    m = (A.shape[1] - filter_size) // stride + 1
    s0, s1, s2 = A.strides
    return as_strided(
        A, shape=(A.shape[0], m, filter_size * A.shape[2]), strides=(s0, s1 * stride, s2)
    )


def im2col_2d(
    A: np.ndarray, filter_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1)
) -> np.ndarray:
    m = (A.shape[1] - filter_size[0]) // stride[0] + 1
    n = (A.shape[2] - filter_size[1]) // stride[1] + 1
    s0, s1, s2, s3 = A.strides
    cols = as_strided(
        A,
        shape=(A.shape[0], filter_size[0], m, n, filter_size[1] * A.shape[3]),
        strides=(s0, s1, s1 * stride[0], s2 * stride[1], s3),
    )
    cols = cols.transpose((0, 2, 3, 1, 4)).reshape(cols.shape[0], m * n, -1)
    return cols, m, n


def col2im_1d(
    A_shape: Tuple[int, ...], cols: np.ndarray, filter_size: int, stride: int = 1
) -> np.ndarray:
    A = np.zeros(A_shape).reshape(-1)
    indices = np.arange(A.shape[0]).reshape(A_shape)
    indices = im2col_1d(indices, filter_size, stride=stride)
    np.add.at(A, indices, cols)
    return A.reshape(A_shape)


def col2im_2d(
    A_shape: Tuple[int, ...],
    cols: np.ndarray,
    filter_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
) -> np.ndarray:
    A = np.zeros(A_shape).reshape(-1)
    indices = np.arange(A.shape[0]).reshape(A_shape)
    indices, _, _ = im2col_2d(indices, filter_size, stride=stride)
    np.add.at(A, indices, cols)
    return A.reshape(A_shape)


class Convolve1D(Tensor):
    def __init__(self, tensor: Tensor, filter: Tensor, stride: int = 1):
        super().__init__()
        self.tensor = tensor
        self.filter = filter
        self.stride = stride

        self.input_shape = None
        self.cols = None
        self.kernel = None
        self.kernel_size = None

    def forward(self) -> np.ndarray:
        kernel = self.filter()
        self.kernel_size = kernel.shape[1]
        self.input_shape = self.tensor().shape
        cols = im2col_1d(self.tensor.output, self.kernel_size, stride=self.stride)
        kernel = kernel.reshape(kernel.shape[0], -1).transpose()
        self.cols, self.kernel = cols, kernel
        return cols @ kernel

    def backward(self, grad: np.ndarray) -> None:
        filter_grad = np.sum(self.cols.swapaxes(-1, -2) @ grad, axis=0).transpose()
        filter_grad = filter_grad.reshape((filter_grad.shape[0], self.kernel_size, -1))
        tensor_grad = col2im_1d(
            self.input_shape, grad @ self.kernel.transpose(), self.kernel_size, stride=self.stride
        )

        self.filter.backward(filter_grad)
        self.tensor.backward(tensor_grad)

        super().backward(grad)


class Convolve2D(Tensor):
    def __init__(self, tensor: Tensor, filter: Tensor, stride: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.tensor = tensor
        self.filter = filter
        self.stride = stride

        self.input_shape = None
        self.cols = None
        self.kernel = None
        self.kernel_size = None

    def forward(self) -> np.ndarray:
        kernel = self.filter()
        self.kernel_size = (kernel.shape[1], kernel.shape[2])
        self.input_shape = self.tensor().shape
        cols, m, n = im2col_2d(self.tensor.output, self.kernel_size, stride=self.stride)
        kernel = kernel.reshape(kernel.shape[0], -1).transpose()
        self.cols, self.kernel = cols, kernel
        return (cols @ kernel).reshape(self.input_shape[0], m, n, kernel.shape[1])

    def backward(self, grad: np.ndarray) -> None:
        grad_reshape = grad.reshape(grad.shape[0], -1, grad.shape[3])
        filter_grad = np.sum(self.cols.swapaxes(-1, -2) @ grad_reshape, axis=0).transpose()
        filter_grad = filter_grad.reshape(
            filter_grad.shape[0], self.kernel_size[0], self.kernel_size[1], -1
        )
        tensor_grad = col2im_2d(
            self.input_shape,
            grad_reshape @ self.kernel.transpose(),
            self.kernel_size,
            stride=self.stride,
        )

        self.filter.backward(filter_grad)
        self.tensor.backward(tensor_grad)

        super().backward(grad)


class MaxPool1D(Tensor):
    def __init__(self, tensor: Tensor, pool_size: int = 2, stride: Union[None, int] = None):
        super().__init__()
        self.tensor = tensor
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

        self.input_shape = None
        self.col_shape = None
        self.max_indices = None

    def forward(self) -> np.ndarray:
        self.input_shape = self.tensor().shape
        tensor = self.tensor.output.swapaxes(-1, -2).reshape(-1, self.input_shape[1], 1)
        tensor = im2col_1d(tensor, self.pool_size, stride=self.stride)
        self.col_shape = tensor.shape
        self.max_indices = np.argmax(tensor, axis=2)
        tensor = tensor[
            np.arange(tensor.shape[0])[:, None], np.arange(tensor.shape[1]), self.max_indices
        ]
        return tensor.reshape(self.input_shape[0], -1, tensor.shape[1]).swapaxes(-1, -2)

    def backward(self, grad: np.ndarray) -> None:
        tensor_grad = np.zeros(self.col_shape)
        grad_reshape = grad.swapaxes(-1, -2).reshape(self.max_indices.shape)
        tensor_grad[
            np.arange(self.col_shape[0])[:, None], np.arange(self.col_shape[1]), self.max_indices
        ] = grad_reshape
        tensor_grad = col2im_1d(
            (self.input_shape[0] * self.input_shape[2], self.input_shape[1], 1),
            tensor_grad,
            self.pool_size,
            self.stride,
        )
        tensor_grad = tensor_grad.reshape(
            self.input_shape[0], self.input_shape[2], self.input_shape[1]
        ).swapaxes(-1, -2)

        self.tensor.backward(tensor_grad)
        super().backward(grad)


class MaxPool2D(Tensor):
    def __init__(
        self,
        tensor: Tensor,
        pool_size: Union[int, Tuple[int, int]] = 2,
        stride: Union[None, int, Tuple[int, int]] = None,
    ):
        super().__init__()
        self.tensor = tensor

        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        self.input_shape = None
        self.col_shape = None
        self.max_indices = None

    def forward(self) -> np.ndarray:
        self.input_shape = self.tensor().shape
        tensor = self.tensor.output.transpose((0, 3, 1, 2)).reshape(
            -1, self.input_shape[1], self.input_shape[2], 1
        )
        tensor, m, n = im2col_2d(tensor, self.pool_size, stride=self.stride)
        self.col_shape = tensor.shape
        self.max_indices = np.argmax(tensor, axis=2)
        tensor = tensor[
            np.arange(tensor.shape[0])[:, None], np.arange(tensor.shape[1]), self.max_indices
        ]
        return tensor.reshape(self.input_shape[0], -1, m, n).transpose((0, 2, 3, 1))

    def backward(self, grad: np.ndarray) -> None:
        tensor_grad = np.zeros(self.col_shape)
        grad_reshape = grad.transpose((0, 3, 1, 2)).reshape(self.max_indices.shape)
        tensor_grad[
            np.arange(self.col_shape[0])[:, None], np.arange(self.col_shape[1]), self.max_indices
        ] = grad_reshape
        tensor_grad = col2im_2d(
            (
                self.input_shape[0] * self.input_shape[3],
                self.input_shape[1],
                self.input_shape[2],
                1,
            ),
            tensor_grad,
            self.pool_size,
            self.stride,
        )
        tensor_grad = tensor_grad.reshape(
            self.input_shape[0], -1, self.input_shape[1], self.input_shape[2]
        ).transpose((0, 2, 3, 1))

        self.tensor.backward(tensor_grad)
        super().backward(grad)
