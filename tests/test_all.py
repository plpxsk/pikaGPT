import unittest

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from pika import gelu, softmax, layer_norm, linear, attention


class pikaTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tol = 1e-3
        x = [-0.8728577, 0.18746605, 1.97671091, -0.55853465, -0.58808922,
             -0.82599565, -2.20829338, 0.25707138, 0.0050708, 0.16621735]

        X = [[-1.41589, -0.260065, 0.040099], [-0.019418, -0.374672, 0.593828],
             [2.02299, 1.97057, 0.555], [0.755452, 0.28335, 0.729227]]

        # square matrix
        SQ = [[1.37818909, 2.86092843],
              [0.56502523, 0.16185584]]

        cls.randx = x
        cls.randX = X
        cls.sqX = SQ
        cls.tol = tol

    @classmethod
    def tearDownClass(cls):
        del cls.randx
        del cls.randX
        del cls.sqX
        del cls.tol

    def test_gelu(self):
        a = gelu(np.array(self.randx), numpy=True)
        b = gelu(mx.array(self.randx))
        self.assertTrue(np.allclose(a, b, atol=self.tol))

    def test_softmax(self):
        a = softmax(np.array(self.randx), numpy=True)
        b = softmax(mx.array(self.randx))
        self.assertTrue(np.allclose(a, b, atol=self.tol))

    def test_layer_norm(self):
        a = layer_norm(np.array(self.randx), g=0.5, b=-1, numpy=True)
        b = layer_norm(mx.array(self.randx), g=0.5, b=-1)
        self.assertTrue(np.allclose(a, b, atol=self.tol))

    def test_linear(self):
        w = [[1, 2, 3], [2, 2, 2], [0, 0, 0]]
        b = [0.5, 0.5, 0.5]
        A = linear(np.array(self.randX), np.array(w), np.array(b))
        B = linear(mx.array(self.randX), mx.array(w), mx.array(b))
        self.assertTrue(np.allclose(A, B, atol=self.tol))

    def test_attention(self):
        ax = np.array(self.sqX)
        bx = mx.array(self.sqX)
        A = attention(q=ax, k=ax, v=ax, mask=ax, numpy=True)
        B = attention(q=bx, k=bx, v=bx, mask=bx)
        self.assertTrue(np.allclose(A, B, atol=self.tol))

    def test_stack(self):
        # copy https://github.com/ml-explore/mlx/blob/771575d27bd560f50e12307edb90a518b272c7f0/python/tests/test_ops.py#L1766
        a = mx.ones((2,))
        np_a = np.ones((2,))
        b = mx.ones((2,))
        np_b = np.ones((2,))

        # One dimensional stack axis=0
        c = mx.stack([a, b])
        np_c = np.stack([np_a, np_b])
        self.assertTrue(np.array_equal(c, np_c))

        # One dimensional stack axis=1
        c = mx.stack([a, b], axis=1)
        np_c = np.stack([np_a, np_b], axis=1)
        self.assertTrue(np.array_equal(c, np_c))

        a = mx.ones((1, 2))
        np_a = np.ones((1, 2))
        b = mx.ones((1, 2))
        np_b = np.ones((1, 2))

        # Two dimensional stack axis=0
        c = mx.stack([a, b])
        np_c = np.stack([np_a, np_b])
        self.assertTrue(np.array_equal(c, np_c))

        # Two dimensional stack axis=1
        c = mx.stack([a, b], axis=1)
        np_c = np.stack([np_a, np_b], axis=1)
        self.assertTrue(np.array_equal(c, np_c))

    def test_hstack(self):
        # equivalent to np.hstack() seems to be mx.concatenate(..., axis=1),
        # and not mx.stack(..., axis=?)
        A = [np.array(self.randX), np.array(self.randX)]
        A = np.hstack(A)
        B = [mx.array(self.randX), mx.array(self.randX)]
        B = mx.concatenate(B, axis=1)
        assert A.shape == B.shape
