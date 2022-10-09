from typing import Union, Tuple

import numpy as np

from neutrino.tensor import Tensor, Variable, Pad
from neutrino.layers import Layer
import neutrino.ops


class Conv1D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Union[None, int] = None,
        stride: int = 1,
        use_bias: bool = True,
    ):
        super().__init__()

        self.filter = Variable(
            np.random.normal(0, 0.1, size=(out_channels, kernel_size, in_channels))
        )
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Variable(np.random.normal(0, 0.1, size=(out_channels,)))
            self.register_parameters([self.filter, self.bias])
        else:
            self.register_parameters([self.filter])

    def __call__(self, input_tensor: Tensor) -> Tensor:
        if self.padding is not None:
            x = Pad(input_tensor, padding=((0, 0), (self.padding, self.padding), (0, 0)))
        else:
            x = input_tensor
        x = ops.convolve1d(x, self.filter, stride=self.stride)
        if self.use_bias:
            x = ops.add(x, self.bias)
        return x


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        padding: Union[None, int, Tuple[int, int]] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        use_bias: bool = True,
    ):
        super().__init__()

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.filter = Variable(
            np.random.normal(
                0, 0.1, size=(out_channels, kernel_size[0], kernel_size[1], in_channels)
            )
        )
        self.padding = padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Variable(np.random.normal(0, 0.1, size=(out_channels,)))
            self.register_parameters([self.filter, self.bias])
        else:
            self.register_parameters([self.filter])

    def __call__(self, input_tensor: Tensor) -> Tensor:
        if self.padding is None:
            x = input_tensor
        elif isinstance(self.padding, int):
            x = Pad(
                input_tensor,
                padding=(
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )
        else:
            x = Pad(
                input_tensor,
                padding=(
                    (0, 0),
                    (self.padding[0], self.padding[0]),
                    (self.padding[1], self.padding[1]),
                    (0, 0),
                ),
            )
        x = ops.convolve2d(x, self.filter, stride=self.stride)
        if self.use_bias:
            x = ops.add(x, self.bias)
        return x
