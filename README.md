## Neutrino

A lightweight machine learning framework for CPU. All common deep learning ops are implemented using NumPy arrays and functions.

This is not really intended to be of any practical use since it doesn't make use of hardware accelerators. It's more of a way for me to learn about how common operations are implemented and how forward and backward passes work under the graph structure of a neural network.

If for some reason you want to try this out, you can clone this repo and run `python test.py`. This will train a small convolutional network on CIFAR-10.
