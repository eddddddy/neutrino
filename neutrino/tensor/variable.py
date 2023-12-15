import numpy as np

from neutrino.tensor import Tensor


class Variable(Tensor):
    def __init__(self, data: np.ndarray):
        super().__init__()
        self.data = data

    def forward(self) -> np.ndarray:
        return self.data
