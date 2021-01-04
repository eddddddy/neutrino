import numpy as np

from tensor import Tensor, Variable
from layers import Layer
import ops


class Dense(Layer):

    def __init__(self, in_units: int, out_units: int, use_bias: bool = True):
        super().__init__()

        self.weight = Variable(np.random.normal(0, 0.1, size=(in_units, out_units)))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Variable(np.random.normal(0, 0.1, size=out_units))
            self.register_parameters([self.weight, self.bias])
        else:
            self.register_parameters([self.weight])

    def __call__(self, input_tensor: Tensor) -> Tensor:
        x = ops.matmul(input_tensor, self.weight)
        if self.use_bias:
            x = ops.add(x, self.bias)
        return x
