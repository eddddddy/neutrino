import numpy as np

from tensor import Tensor, Input
import ops
from optimizer import Adam
from loss import CrossEntropyLoss
from layers import Dense, Conv1D, Conv2D
from model import Model
from utils import load_mnist, load_cifar


class MnistModel:
    def __call__(self, input_tensor: Tensor) -> Tensor:
        x = ops.reshape(input_tensor, (-1, 28, 28, 1))
        x = Conv2D(1, 16, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = ops.maxpool2d(x, pool_size=2)
        x = Conv2D(16, 32, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = ops.maxpool2d(x, pool_size=2)
        x = ops.reshape(x, (-1, 1568))
        x = Dense(1568, 256)(x)
        x = ops.relu(x)
        x = Dense(256, 10)(x)
        return x


class CifarModel:
    def __call__(self, input_tensor: Tensor) -> Tensor:
        x = Conv2D(3, 24, kernel_size=3, padding=1)(input_tensor)
        x = ops.relu(x)
        x = Conv2D(24, 24, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = ops.maxpool2d(x, pool_size=2)
        x = Conv2D(24, 48, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = Conv2D(48, 48, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = ops.maxpool2d(x, pool_size=2)
        x = Conv2D(48, 72, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = Conv2D(72, 72, kernel_size=3, padding=1)(x)
        x = ops.relu(x)
        x = ops.maxpool2d(x, pool_size=2)
        x = ops.reshape(x, (-1, 1152))
        x = Dense(1152, 256)(x)
        x = ops.relu(x)
        x = Dense(256, 10)(x)
        return x


input_tensor = Input()
output_tensor = CifarModel()(input_tensor)

model = Model(input_tensor, output_tensor)
model.compile(Adam(lr=1e-3), CrossEntropyLoss())

(x_train, y_train), (x_test, y_test) = load_cifar()

model.train(x_train, y_train, batch_size=16, epochs=10)
preds = model.predict(x_test, batch_size=16)

print(
    f"Accuracy: {np.count_nonzero(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / x_test.shape[0]}"
)
