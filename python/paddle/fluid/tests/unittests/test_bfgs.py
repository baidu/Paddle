# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
from paddle.optimizer.functional import bfgs_iterates, bfgs_optimize
from paddle.optimizer.functional.bfgs import (
    SearchState,
    verify_symmetric_positive_definite_matrix,
    update_approx_inverse_hessian)
from paddle.optimizer.functional.bfgs_utils import (
    vjp
)

import tensorflow as tf

def jacfn_gen(f, create_graph=False):
    r"""Returns a helper function for computing the jacobians.
    
    Requires `f` to be single valued function. Batch input is allowed.

    The returned function, when called, returns a tensor of [Batched] gradients.
    """
    def jac(x):
        fval, grads = vjp(f, x, create_graph=create_graph)
        return grads
    
    return jac

def hesfn_gen(f):
    r"""Returns a helper function for computing the hessians.

    Requires `f` to be single valued function. Batch input is allowed.

    The returned function, when called, returns a tensor of [Batched]
    second order partial derivatives.
    """
    def hess(x):
        y = f(x)
        batch_mode = len(y.shape) > 1
        dim = x.shape[-1]
        vs = []
        for i in range(dim):
            v = paddle.zeros_like(x)
            if batch_mode:
                v[..., i] = 1
            else:
                v[i] = 1
            v = v.detach()
            vs.append(v)
        jacfn = jacfn_gen(f, create_graph=True)
        rows = [vjp(jacfn, x, v)[1] for v in vs]
        h = paddle.stack(rows, axis=-2)
        return h
    
    return hess


def update_inv_hessian_proper(H, s, y):
    batch = len(s.shape) > 1
    dtype = H.dtype
    dim = s.shape[-1]
    if not batch:
        rho = 1. / paddle.dot(s, y)
        l = paddle.eye(dim, dtype=dtype) - rho * s.unsqueeze(-1) * y.unsqueeze(0)
        r = paddle.eye(dim, dtype=dtype) - rho * y.unsqueeze(-1) * s.unsqueeze(0)
        lH = paddle.matmul(l, H)
        lHr = paddle.matmul(lH, r)
        H_next = lHr + rho * s.unsqueeze(-1) * s.unsqueeze(0)
    else:
        rho = 1. / paddle.einsum('...i, ...i', s, y)
        l = paddle.eye(dim) - paddle.einsum('...,...i,...j->...ij', rho, s, y)
        r = paddle.eye(dim) - paddle.einsum('...,...i,...j->...ij', rho, y, s)
        lH = paddle.matmul(l, H)
        lHr = paddle.matmul(lH, r)
        H_next = lHr + paddle.einsum('...,...i,...j->...ij', rho, s, s)
    
    return H_next

def _tensor_product(t1, t2):
    return tf.linalg.matmul(t1[..., tf.newaxis], t2[..., tf.newaxis, :])

class TestBFGS(unittest.TestCase):

    def setUp(self):
        pass        

    def gen_configs(self):
        dtypes = ['float32', 'float64']
        shapes = {
            '1d1v': [1],
            '1d2v': [2],
            '2d1v': [2, 1],
            '2d2v': [2, 2],
            '1d100v': [100],
            '10d10v': [10, 10]
        }
        for shape, dtype in zip(shapes.values(), dtypes):
            yield shape, dtype

    def test_update_approx_inverse_hessian(self):
        paddle.seed(12345)
        for shape, dtype in self.gen_configs():
            center = paddle.rand(shape, dtype=dtype)
            scales = paddle.exp(paddle.rand(shape, dtype=dtype))
            scales = paddle.square(paddle.rand(shape, dtype=dtype))
            def f(x):
                return paddle.sum(scales * paddle.square(x - center), axis=-1)
            x0 = paddle.ones_like(center, dtype=dtype)
            
            # The true inverse hessian value at x0
            hess = hesfn_gen(f)(x0)
            inv_hess = paddle.inverse(hess)
            
            f0, g0 = vjp(f, x0)
            state = SearchState(x0, f0, g0, inv_hess)
            
            # Verifies the estimated invese Hessian at the center equals the 
            # true value. 
            s = center - x0
            y = -g0

            H1 = update_approx_inverse_hessian(state, inv_hess, s, y)
            hess = hesfn_gen(f)(center)
            inv_hess = paddle.inverse(hess)

            self.assertTrue(paddle.allclose(H1, inv_hess))

    def _regular_quadratic(self, input_shape, dtype):
        minimum = paddle.rand(input_shape, dtype=dtype)
        scales = paddle.exp(paddle.rand(input_shape, dtype=dtype))

        def f(x):
            return paddle.sum(scales * paddle.square(x - minimum), axis=-1)

        x0 = paddle.ones_like(minimum, dtype=dtype)

        result = bfgs_optimize(f, x0, dtype=dtype)

        self.assertTrue(paddle.all(result.converged))
        self.assertTrue(paddle.allclose(result.location, minimum))

    def test_quadratic(self):
        paddle.seed(12345)
        for shape, dtype in self.gen_configs():
            self._regular_quadratic(shape, dtype)

    def _general_quadratic(self, input_shape, dtype):
        minimum = paddle.rand(input_shape, dtype=dtype)
        hessian_shape = input_shape + input_shape[-1:]
        rotation = paddle.rand(hessian_shape, dtype=dtype)
        hessian = paddle.einsum('...ik, ...jk', rotation, rotation)

        verify_symmetric_positive_definite_matrix(hessian)

        def f(x):
            y = paddle.einsum('...i, ...ij, ...j',
                              x - minimum,
                              hessian,
                              x - minimum)

            return y
        
        x0 = paddle.ones_like(minimum, dtype=dtype)

        result = bfgs_optimize(f, x0, dtype=dtype)

        self.assertTrue(paddle.all(result.converged))
        self.assertTrue(paddle.allclose(result.location, minimum))

    def test_general_quadratic(self):
        paddle.seed(12345)
        for shape, dtype in self.gen_configs():
            self._general_quadratic(shape, dtype)


shape = [2]
dtype = 'float32'

center = np.random.rand(2)
scales = np.array([0.2, 1.5])
x0 = np.ones_like(center)
s = np.array([-0.5, -0.5])

# scales = paddle.square(paddle.rand(shape, dtype=dtype))

# TF results as reference
center_tf = tf.convert_to_tensor(center)
x0_tf = tf.convert_to_tensor(x0)
s_tf = tf.convert_to_tensor(s)

def f_tf(x):
    return tf.reduce_sum(tf.square(x - center_tf), axis=-1)

with tf.GradientTape() as tape:
    tape.watch(x0_tf)
    y = f_tf(x0_tf)

g0_tf = tape.gradient(y, x0_tf)

x1_tf = x0_tf + s_tf
with tf.GradientTape() as tape:
    tape.watch(x1_tf)
    y = f_tf(x1_tf)

g1_tf = tape.gradient(y, x1_tf)

y_tf = g1_tf - g0_tf
h0_tf = tf.linalg.inv(tf.linalg.diag(g0_tf))
normalization_factor = tf.tensordot(s_tf, y_tf, 1)
h1_tf = tf_inv_hessian_update(y_tf, s_tf, normalization_factor, h0_tf)

# Applies the proper update rules.
center_pp = paddle.to_tensor(center)
scales_pp = paddle.to_tensor(scales)
x0_pp = paddle.to_tensor(x0)
s_pp = paddle.to_tensor(s)

def f(x):
    return paddle.sum(paddle.square(x - center_pp), axis=-1)

f0_pp, g0_pp = vjp(f, x0_pp)
h0_pp = paddle.inverse(paddle.diag(g0_pp))
state = SearchState(x0_pp, f0_pp, g0_pp, h0_pp)

x1_pp = x0_pp + s_pp
f1_pp, g1_pp = vjp(f, x1_pp)
y_pp = g1_pp - g0_pp

h1_pp = update_approx_inverse_hessian(state, h0_pp, s_pp, y_pp)

h1_pp_proper = update_inv_hessian_proper(h0_pp, s_pp, y_pp)



if __name__ == "__main__":
    unittest.main()
