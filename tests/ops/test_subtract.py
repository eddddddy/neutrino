import pytest

import numpy as np

import neutrino


class TestSubtract:
    def test_forward(self):
        a = neutrino.Variable(np.array([[5, 3, 2], [1, 5, 4]]))
        b = neutrino.Variable(np.array([[3, 6, 8], [0, 8, 5]]))
        s = neutrino.Subtract(a, b)

        res = s()
        assert np.all(res == np.array([[2, -3, -6], [1, -3, -1]]))

    def test_backward(self):
        a = neutrino.Variable(np.array([[5, 3, 2], [1, 5, 4]]))
        b = neutrino.Variable(np.array([[3, 6, 8], [0, 8, 5]]))
        s = neutrino.Subtract(a, b)

        s()
        s.backward(np.ones_like(s.output))

        assert np.all(s.grad == np.ones((2, 3)))
        assert np.all(a.grad == np.ones((2, 3)))
        assert np.all(b.grad == -np.ones((2, 3)))
