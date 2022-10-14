import pytest

import numpy as np

import neutrino


class TestMatmul:
    def test_forward(self):
        a = neutrino.Variable(np.array([[0, 1, 2], [3, 4, 5]]))
        b = neutrino.Variable(np.array([[6, 7], [8, 9], [10, 11]]))
        s = neutrino.Matmul(a, b)

        res = s()
        assert np.all(res == np.array([[28, 31], [100, 112]]))

    def test_backward(self):
        a = neutrino.Variable(np.array([[0, 1, 2], [3, 4, 5]]))
        b = neutrino.Variable(np.array([[6, 7], [8, 9], [10, 11]]))
        s = neutrino.Matmul(a, b)

        s()
        s.backward(np.ones_like(s.output))

        assert np.all(s.grad == np.ones((2, 2)))
        assert np.all(a.grad == np.array([[13, 17, 21], [13, 17, 21]]))
        assert np.all(b.grad == np.array([[3, 3], [5, 5], [7, 7]]))
