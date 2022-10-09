import abc
from typing import Union

import numpy as np

from neutrino.tensor import Tensor, Variable
import neutrino.ops

__all__ = ["Loss", "MSELoss", "CrossEntropyLoss"]


class LossTensor:
    def __init__(self, loss_tensor: Tensor, pred_tensor: Tensor, model_output: Tensor):
        self.loss_tensor = loss_tensor
        self.loss_tensor()
        self.pred_tensor = pred_tensor
        self.model_output = model_output

    def backward(self) -> None:
        self.loss_tensor.backward(np.ones_like(self.loss_tensor.output))
        self.model_output.backward(self.pred_tensor.grad)


class Loss(abc.ABC):
    def __call__(self, pred: Tensor, true: Union[np.ndarray, Tensor]):
        if not isinstance(true, Tensor):
            true = Variable(true)

        pred_copy = Variable(pred.output)
        loss = self.compute_loss(pred_copy, true)
        return LossTensor(loss, pred_copy, pred)

    @abc.abstractmethod
    def compute_loss(self, pred: Tensor, true: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    def compute_loss(self, pred: Tensor, true: Tensor) -> Tensor:
        return ops.mean(ops.sum(ops.power(ops.subtract(pred, true), 2), axis=1), axis=0)


class CrossEntropyLoss(Loss):
    class SoftmaxCrossEntropyWithLogits(Tensor):
        def __init__(self, pred: Tensor, true: Tensor):
            super().__init__()
            self.pred = pred
            self.true = true
            self.probs = None

        def forward(self) -> np.ndarray:
            logits = self.pred()
            probs = np.exp(logits)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            self.probs = probs

            return -np.einsum("ij,ij->i", self.true(), np.log(probs))

        def backward(self, grad: np.ndarray) -> None:
            self.pred.backward(grad.reshape(-1, 1) * (self.probs - self.true.output))
            super().backward(grad)

    def compute_loss(self, pred: Tensor, true: Tensor) -> Tensor:
        return ops.mean(CrossEntropyLoss.SoftmaxCrossEntropyWithLogits(pred, true), axis=0)
