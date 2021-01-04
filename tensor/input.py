import numpy as np

from tensor import Tensor


class Input(Tensor):

    def __init__(self):
        super().__init__()
        self.data = None

    def forward(self) -> np.ndarray:
        return self.data
