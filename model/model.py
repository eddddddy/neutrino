from typing import List, Tuple

from tqdm import tqdm
import numpy as np

from tensor import Tensor, Variable, Input
from loss import Loss
from optimizer import Optimizer


class Model:

    def __init__(self, input_tensor: Input, output_tensor: Tensor):
        self.input = input_tensor
        self.output = output_tensor

        self.parameters = None
        self.optimizer = None
        self.loss_fn = None

    def __find_parameters(self) -> List[Variable]:

        def find_tensor_parameters(tensor: Tensor) -> List[Variable]:
            parameters = []
            for attr in dir(tensor):
                attr = getattr(tensor, attr)
                if isinstance(attr, Variable) and attr not in parameters:
                    parameters.append(attr)
                elif isinstance(attr, Tensor):
                    parameters.extend([param for param in find_tensor_parameters(attr) if param not in parameters])
            return parameters

        return find_tensor_parameters(self.output)

    def compile(self, optimizer: Optimizer, loss_fn: Loss) -> None:
        self.parameters = self.__find_parameters()
        self.optimizer = optimizer
        self.optimizer.set_parameters(self.parameters)
        self.loss_fn = loss_fn

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32,
              shuffle: bool = True, alpha: float = 0.95, metrics: Tuple[str] = ('loss', 'acc')):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            if shuffle:
                np.random.shuffle(indices)

            num_batches = int(np.ceil(x.shape[0] / batch_size))
            moving_loss, moving_acc = 0, 0
            train_loop = tqdm(range(num_batches))
            for batch in train_loop:
                x_batch = x[indices[batch * batch_size:(batch + 1) * batch_size]]
                y_batch = y[indices[batch * batch_size:(batch + 1) * batch_size]]

                self.input.data = x_batch
                pred = self.output()
                loss = self.loss_fn(self.output, y_batch)
                batch_loss = loss.loss_tensor.output
                batch_acc = 100 * np.count_nonzero(np.argmax(pred, axis=1) == np.argmax(y_batch, axis=1)) / x_batch.shape[0]

                loss.backward()
                self.optimizer.step()
                self.output.reset()

                if batch == 0:
                    moving_loss, moving_acc = batch_loss, batch_acc
                else:
                    moving_loss = alpha * moving_loss + (1 - alpha) * batch_loss
                    moving_acc = alpha * moving_acc + (1 - alpha) * batch_acc

                train_loop.set_description(f"Epoch {epoch + 1}/{epochs}, Loss {moving_loss:.5f}, Acc {moving_acc:.2f}")

    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        predictions = []
        num_batches = int(np.ceil(x.shape[0] / batch_size))
        predict_loop = tqdm(range(num_batches))
        for batch in predict_loop:
            x_batch = x[batch * batch_size:(batch + 1) * batch_size]
            self.input.data = x_batch
            pred = self.output()

            predictions.append(pred)
            self.output.reset()

            predict_loop.set_description("Prediction")

        return np.vstack(predictions)
