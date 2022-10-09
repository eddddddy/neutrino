import pytest
import numpy as np

import neutrino


class TestAdd:

    def test_forward(self):
        a = neutrino.Variable(np.arange(6).reshape(2, 3))
        b = neutrino.Variable(np.arange(6, 12).reshape(2, 3))
        s = neutrino.Add(a, b)

        res = s()
        assert np.all(res == np.arange(6, 18, 2).reshape(2, 3))

    def test_backward(self):
        a = neutrino.Variable(np.arange(6).reshape(2, 3))
        b = neutrino.Variable(np.arange(6, 12).reshape(2, 3))
        s = neutrino.Add(a, b)

        s()
        s.backward(np.ones_like(s.output))

        assert np.all(s.grad == np.ones((2, 3)))
        assert np.all(a.grad == np.ones((2, 3)))
        assert np.all(b.grad == np.ones((2, 3)))
