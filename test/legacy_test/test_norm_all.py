# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import _C_ops, base
from paddle.base import core
from paddle.base.framework import in_dygraph_mode


# hack method for test p_norm final state
def p_norm_python_api(
    x, p=2.0, axis=-1, epsilon=1e-12, keepdim=False, as_vector=False
):
    if in_dygraph_mode():
        return _C_ops.p_norm(x, p, axis, epsilon, keepdim, as_vector)


def norm_public_python_api(
    x, p=2.0, axis=-1, epsilon=1e-12, keepdim=False, as_vector=False
):
    return paddle.linalg.norm(
        x,
        p,
        axis,
        keepdim,
    )


def np_linalg_vector_norm(x, axis, porder, keepdims=False):
    x_shape = list(x.shape)

    origin_axis = axis
    if origin_axis is None:
        pass
    elif isinstance(origin_axis, int):
        origin_axis = [origin_axis]
    else:
        origin_axis = list(origin_axis)

    if axis is None:
        x = x.ravel()
        axis = -1

    if not isinstance(axis, int) and len(axis) > 1:
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(x.shape)
        tmp_axis = []
        for i in range(len(axis)):
            tmp_axis.append(-1 - i)
        x = np.moveaxis(x, axis, tmp_axis)

        front_dim = x.shape[0 : len(x.shape) - len(axis)]
        back_dim = 1
        for i in range(len(x.shape) - len(axis), len(x.shape)):
            back_dim = back_dim * x.shape[i]
        front_dim = list(front_dim)
        front_dim.append(back_dim)
        x = x.reshape(front_dim)
        axis = -1
    if isinstance(axis, list):
        axis = tuple(axis)

    r = np.linalg.norm(x, ord=porder, axis=axis, keepdims=keepdims)

    r_shape = r.shape

    if keepdims:
        if origin_axis is None:
            r_shape = np.ones_like(x_shape)
        elif len(origin_axis) > 1:
            r_shape = x_shape
            for i in origin_axis:
                r_shape[i] = 1
    r = r.reshape(r_shape)
    return r


def np_linalg_matrix_norm(x, axis, porder, keepdims=False):
    axis = tuple(axis)
    r = np.linalg.norm(x, ord=porder, axis=axis, keepdims=keepdims)
    return r


def np_linalg_norm(x, axis, porder, keepdims=False):
    r = []
    if axis is None or isinstance(axis, (int, float)):
        r = np_linalg_vector_norm(x, axis, porder, keepdims)
    elif isinstance(axis, list) and len(axis) == 2:
        r = np_linalg_matrix_norm(x, axis, porder, keepdims)
    r = r.astype(x.dtype)

    return r


def numpy_frobenius_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if axis is None:
        axis = (-2, -1)
    r = np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims).astype(
        x.dtype
    )
    return r


def numpy_nuclear_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    r = np.linalg.norm(x, ord='nuc', axis=axis, keepdims=keepdims).astype(
        x.dtype
    )
    return r


def frobenius_norm(x, dim, keep_dim):
    return paddle.linalg.norm(x, p='fro', axis=dim, keepdim=keep_dim)


def nuclear_norm(x, dim, keep_dim):
    return paddle.linalg.norm(x, p='nuc', axis=dim, keepdim=keep_dim)


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.python_api = frobenius_norm
        self.op_type = "frobenius_norm"
        self.init_test_case()
        self.init_dtype()
        x = (np.random.random(self.shape) + 1.0).astype(self.dtype)
        norm = numpy_frobenius_norm(x, self.axis, self.keepdim)
        self.reduce_all = False
        self.inputs = {'X': x}
        self.attrs = {
            'dim': list(self.axis),
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all,
        }
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = (1, 2)
        self.keepdim = False

    def init_dtype(self):
        self.dtype = "float64"


class TestFrobeniusNormOp2(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5]
        self.axis = (0, 1)
        self.keepdim = True

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.python_api = p_norm_python_api
        self.public_python_api = norm_public_python_api
        self.prim_op_type = "comp"
        self.init_test_case()
        self.init_dtype()
        self.fw_comp_atol = 1e-6
        self.fw_comp_rtol = 1e-6
        self.rev_comp_atol = 1e-6
        self.rev_comp_rtol = 1e-6
        x = (np.random.random(self.shape) + 0.5).astype(self.dtype)
        norm = np_linalg_norm(x, self.axis, self.porder, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        self.outputs = {'Out': norm}
        self.gradient = self.calc_gradient()

    def test_check_output(self):
        self.check_output(check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim_pir=True)

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float64"

    def calc_gradient(self):
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        x = self.inputs["X"]
        porder = self.attrs["porder"]
        axis = self.attrs["axis"]
        asvector = self.attrs["asvector"]
        x_dtype = x.dtype
        x = x.astype(np.float32) if x.dtype == np.float16 else x
        if porder == 0:
            grad = np.zeros(x.shape).astype(x.dtype)
        elif porder in [float("inf"), float("-inf")]:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            x_abs = np.abs(x)
            grad = np.sign(x)
            grad[x_abs != norm] = 0.0
        else:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            grad = (
                np.power(norm, 1 - porder)
                * np.power(np.abs(x), porder - 1)
                * np.sign(x)
            )

        numel = 1
        for s in x.shape:
            numel *= s
        divisor = numel if asvector else x.shape[axis]
        numel /= divisor
        return [grad.astype(x_dtype) * 1 / numel]


class TestPnormOp2(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim_pir=True)


class TestPnormOp3(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = np.inf
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


class TestPnormOp4(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = -np.inf
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


class TestPnormOp5(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 0
        self.keepdim = True
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=self.gradient)


class TestPnormOp6(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = -1
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=self.gradient, check_prim_pir=True
        )


def create_test_fp16_class(parent, max_relative_error=2e-3):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestPnormFP16Op(parent):
        def init_dtype(self):
            self.dtype = "float16"

        def test_check_output(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    user_defined_grads=self.gradient,
                    max_relative_error=max_relative_error,
                )

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestPnormFP16Op.__name__ = cls_name
    globals()[cls_name] = TestPnormFP16Op


create_test_fp16_class(TestPnormOp)
create_test_fp16_class(TestPnormOp2)
create_test_fp16_class(TestPnormOp3)
create_test_fp16_class(TestPnormOp4)
create_test_fp16_class(TestPnormOp5)
create_test_fp16_class(TestPnormOp6)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestPnormBF16Op(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.prim_op_type = "comp"
        self.python_api = p_norm_python_api
        self.public_python_api = norm_public_python_api
        self.init_test_case()
        self.x = (np.random.random(self.shape) + 0.5).astype(np.float32)
        self.norm = np_linalg_norm(self.x, self.axis, self.porder, self.keepdim)
        self.gradient = self.calc_gradient()
        self.inputs = {'X': convert_float_to_uint16(self.x)}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        self.outputs = {'Out': convert_float_to_uint16(self.norm)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-3, check_prim_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            user_defined_grads=self.gradient,
            check_prim_pir=True,
        )

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = False
        self.asvector = False

    def init_dtype(self):
        self.dtype = np.uint16

    def calc_gradient(self):
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder),
            'asvector': self.asvector,
        }
        x = self.x
        porder = self.attrs["porder"]
        axis = self.attrs["axis"]
        asvector = self.attrs["asvector"]
        x_dtype = x.dtype
        x = x.astype(np.float32) if x.dtype == np.float16 else x
        if porder == 0:
            grad = np.zeros(x.shape).astype(x.dtype)
        elif porder in [float("inf"), float("-inf")]:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            x_abs = np.abs(x)
            grad = np.sign(x)
            grad[x_abs != norm] = 0.0
        else:
            norm = np_linalg_norm(x, axis=axis, porder=porder, keepdims=True)
            grad = (
                np.power(norm, 1 - porder)
                * np.power(np.abs(x), porder - 1)
                * np.sign(x)
            )

        numel = 1
        for s in x.shape:
            numel *= s
        divisor = numel if asvector else x.shape[axis]
        numel /= divisor
        return [grad.astype(x_dtype) * 1 / numel]


def check_fro_static(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = numpy_frobenius_norm(
            np_input, axis=axis, keepdims=keep_dim
        )
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_fro_dygraph(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = numpy_frobenius_norm(x_numpy, axis, keep_dim)
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.norm(x=x_paddle, p=p, axis=axis, keepdim=keep_dim)
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_nuc_static(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = numpy_nuclear_norm(
            np_input, axis=axis, keepdims=keep_dim
        )
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_nuc_dygraph(self, p, axis, shape_x, dtype, keep_dim, check_dim=False):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = numpy_nuclear_norm(x_numpy, axis, keep_dim)
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.norm(x=x_paddle, p=p, axis=axis, keepdim=keep_dim)
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_norm_static(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(x=data, p=p, axis=axis, keepdim=keep_dim)
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim
        ).astype(dtype)
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_norm_dygraph(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = np_linalg_norm(
        x_numpy, porder=p, axis=axis, keepdims=keep_dim
    )
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.linalg.norm(x=x_paddle, p=p, axis=axis, keepdim=keep_dim)
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_matrix_static(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.linalg.matrix_norm(
            x=data, p=p, axis=axis, keepdim=keep_dim
        )
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_matrix_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim
        ).astype(dtype)
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_matrix_dygraph(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    x_numpy = (np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = np_linalg_matrix_norm(
        x_numpy, porder=p, axis=axis, keepdims=keep_dim
    )
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.linalg.matrix_norm(
        x=x_paddle, p=p, axis=axis, keepdim=keep_dim
    )
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_vector_static(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    with base.program_guard(base.Program()):
        data = paddle.static.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.linalg.vector_norm(
            x=data, p=p, axis=axis, keepdim=keep_dim
        )
        place = base.CPUPlace()
        exe = base.Executor(place)
        np_input = np.array(np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = np_linalg_vector_norm(
            np_input, porder=p, axis=axis, keepdims=keep_dim
        ).astype(dtype)
        (result,) = exe.run(feed={"X": np_input}, fetch_list=[out])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


def check_linalg_vector_dygraph(
    self, p, axis, shape_x, dtype, keep_dim, check_dim=False
):
    x_numpy = np.array(np.random.random(shape_x) + 1.0).astype(dtype)
    expected_result = np_linalg_vector_norm(
        x_numpy, porder=p, axis=axis, keepdims=keep_dim
    )
    x_paddle = paddle.to_tensor(x_numpy)
    result = paddle.linalg.vector_norm(
        x=x_paddle, p=p, axis=axis, keepdim=keep_dim
    )
    result = result.numpy()
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-8)
    if keep_dim and check_dim:
        np.testing.assert_equal(result.shape, expected_result.shape)


class API_NormTest(unittest.TestCase):
    def test_basic(self):
        keep_dims = {False, True}
        for keep in keep_dims:
            check_fro_static(
                self,
                p='fro',
                axis=[-2, -1],
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_fro_static(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_nuc_static(
                self,
                p='nuc',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_norm_static(
                self,
                p=2,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_norm_static(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
            )
            check_linalg_norm_static(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_norm_static(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_static(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_vector_static(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_vector_static(
                self,
                p=4,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_vector_static(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
            )
            check_linalg_vector_static(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_vector_static(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4, 5],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=-np.inf,
                axis=[0, 1, 2],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=2,
                axis=None,
                shape_x=[],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=np.inf,
                axis=None,
                shape_x=[],
                dtype="complex64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=-np.inf,
                axis=[0, 1, 2, 3],
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=np.inf,
                axis=2,
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_static(
                self,
                p=0,
                axis=[1, 3],
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_static(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_static(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_static(
                self,
                p='nuc',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_static(
                self,
                p=-2,
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

    def test_dygraph(self):
        paddle.disable_static()
        keep_dims = {False, True}
        for keep in keep_dims:
            check_fro_dygraph(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )
            check_fro_dygraph(
                self,
                p='fro',
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )

            check_nuc_dygraph(
                self,
                p='nuc',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )
            check_nuc_dygraph(
                self,
                p='nuc',
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype='float64',
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_norm_dygraph(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
            )
            check_linalg_norm_dygraph(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_norm_dygraph(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_norm_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=None,
                shape_x=[3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float32",
                keep_dim=keep,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=0,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=None,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
            )
            check_linalg_vector_dygraph(
                self,
                p=0,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

            check_linalg_vector_dygraph(
                self,
                p=1,
                axis=1,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=0,
                axis=None,
                shape_x=[3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=-1,
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=1,
                axis=[0, 1],
                shape_x=[2, 3, 4, 5],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1, 2],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=2,
                axis=None,
                shape_x=(),
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=None,
                shape_x=[],
                dtype="complex64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1, 2, 3],
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=np.inf,
                axis=2,
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_vector_dygraph(
                self,
                p=0,
                axis=[1, 3],
                shape_x=[1, 14, 5, 14],
                dtype="complex128",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p=-np.inf,
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p='fro',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p='nuc',
                axis=[0, 1],
                shape_x=[2, 3, 4],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )
            check_linalg_matrix_dygraph(
                self,
                p=-2,
                axis=[1, 2],
                shape_x=[2, 3, 4, 5],
                dtype="float64",
                keep_dim=keep,
                check_dim=True,
            )

        paddle.enable_static()

    def test_name(self):
        if not paddle.framework.use_pir_api():
            paddle.enable_static()
            with base.program_guard(base.Program()):
                x = paddle.static.data(
                    name="x", shape=[10, 10], dtype="float32"
                )
                y_1 = paddle.norm(
                    x, p='fro', axis=[-2, -1], name='frobenius_name'
                )
                y_2 = paddle.norm(x, p=2, name='pnorm_name')
                y_3 = paddle.norm(x, p='nuc', axis=[0, 1], name='nuclear_name')
                y_4 = paddle.norm(
                    x, p=2, axis=[0, 1], name='p_matrix_norm_name'
                )
                self.assertEqual(('frobenius_name' in y_1.name), True)
                self.assertEqual(('pnorm_name' in y_2.name), True)
                self.assertEqual(('nuclear_name' in y_3.name), True)
                self.assertEqual(('p_matrix_norm_name' in y_4.name), True)

    def test_errors(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):

            def err_dtype(p, shape_x, xdtype, out=None):
                data = paddle.static.data(shape=shape_x, dtype=xdtype)
                paddle.norm(data, p=p, out=out)

            self.assertRaises(TypeError, err_dtype, "fro", [2, 2], "int64")
            self.assertRaises(ValueError, paddle.norm, "inf", [2], "int64")
            out = paddle.static.data(name="out", shape=[1], dtype="int64")
            self.assertRaises(
                TypeError, err_dtype, "fro", [2, 2], "float64", out
            )
            self.assertRaises(TypeError, err_dtype, 2, [10], "int64")
            self.assertRaises(TypeError, err_dtype, 2, [10], "float64", out)

            data = paddle.static.data(
                name="data_2d", shape=[2, 2], dtype="float64"
            )
            self.assertRaises(
                ValueError, paddle.norm, data, p="unsupported norm"
            )
            self.assertRaises(ValueError, paddle.norm, data, p=[1])
            self.assertRaises(ValueError, paddle.norm, data, p=[1], axis=-1)
            self.assertRaises(ValueError, paddle.norm, 0, [1, 0], "float64")
            data = paddle.static.data(
                name="data_3d", shape=[2, 2, 2], dtype="float64"
            )
            self.assertRaises(
                ValueError, paddle.norm, data, p='unspport', axis=[-3, -2, -1]
            )

        with base.dygraph.guard():
            # The size of input in Norm should not be 0.
            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(np.reshape(array, [0, 0]), dtype='float32')
                paddle.linalg.norm(x, axis=0)

            self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
