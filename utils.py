import os
import pickle
import urllib.request as request
import pathlib
import tarfile
from typing import Tuple

import numpy as np


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if not pathlib.Path("mnist.npz").is_file():
        request.urlretrieve("https://s3.amazonaws.com/img-datasets/mnist.npz", filename="mnist.npz")

    with np.load("mnist.npz") as mnist:
        x_train = mnist["x_train"]
        y_train = mnist["y_train"]
        x_test = mnist["x_test"]
        y_test = mnist["y_test"]

    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.eye(10)[y_train]
    x_test = np.expand_dims(x_test, axis=3)
    y_test = np.eye(10)[y_test]

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)


def load_cifar() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    def load_batch(path: str):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode("utf8")] = v
        return d_decoded

    if not pathlib.Path("cifar-10-batches-py").is_dir():
        request.urlretrieve(
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", filename="cifar-10.tar.gz"
        )
        with tarfile.open("cifar-10.tar.gz", mode="r:gz") as archive:
            archive.extractall()

    x_train = []
    y_train = []

    for i in range(1, 6):
        path = os.path.join("cifar-10-batches-py", f"data_batch_{i}")
        d = load_batch(path)
        x_train.append(d["data"])
        y_train.append(np.array(d["labels"]))

    x_train = np.vstack(x_train).reshape(-1, 3, 32, 32)
    y_train = np.eye(10)[np.concatenate(y_train)]

    test_path = os.path.join("cifar-10-batches-py", "test_batch")
    d = load_batch(test_path)
    x_test = d["data"].reshape(-1, 3, 32, 32)
    y_test = np.eye(10)[np.array(d["labels"])]

    x_train = x_train.transpose((0, 2, 3, 1)).astype(np.float32) / 255.0
    x_test = x_test.transpose((0, 2, 3, 1)).astype(np.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)
