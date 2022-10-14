import pytest

import numpy as np

import neutrino


class TestAdd:
    def test_forward(self):
        a = neutrino.Variable(np.array([[0, 1, 2], [3, 4, 5]]))
        b = neutrino.Variable(np.array([[6, 7, 8], [9, 10, 11]]))
        s = neutrino.Add(a, b)

        res = s()
        assert np.all(res == np.array([[6, 8, 10], [12, 14, 16]]))

    def test_backward(self):
        a = neutrino.Variable(np.array([[0, 1, 2], [3, 4, 5]]))
        b = neutrino.Variable(np.array([[6, 7, 8], [9, 10, 11]]))
        s = neutrino.Add(a, b)

        s()
        s.backward(np.ones_like(s.output))

        assert np.all(s.grad == np.ones((2, 3)))
        assert np.all(a.grad == np.ones((2, 3)))
        assert np.all(b.grad == np.ones((2, 3)))
