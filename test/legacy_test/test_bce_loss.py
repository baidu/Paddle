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

import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


def test_static_layer(
    place, input_np, label_np, reduction='mean', weight_np=None
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype='float64'
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype='float64'
        )
        if weight_np is not None:
            weight = paddle.static.data(
                name='weight', shape=weight_np.shape, dtype='float64'
            )
            bce_loss = paddle.nn.loss.BCELoss(
                weight=weight, reduction=reduction
            )
        else:
            bce_loss = paddle.nn.loss.BCELoss(reduction=reduction)
        res = bce_loss(input, label)
        exe = paddle.static.Executor(place)
        (static_result,) = exe.run(
            prog,
            feed=(
                {"input": input_np, "label": label_np}
                if weight_np is None
                else {"input": input_np, "label": label_np, "weight": weight_np}
            ),
            fetch_list=[res],
        )
    return static_result


def test_static_functional(
    place, input_np, label_np, reduction='mean', weight_np=None
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype='float64'
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype='float64'
        )
        if weight_np is not None:
            weight = paddle.static.data(
                name='weight', shape=weight_np.shape, dtype='float64'
            )
            res = paddle.nn.functional.binary_cross_entropy(
                input, label, weight=weight, reduction=reduction
            )
        else:
            res = paddle.nn.functional.binary_cross_entropy(
                input, label, reduction=reduction
            )
        exe = paddle.static.Executor(place)
        (static_result,) = exe.run(
            prog,
            feed=(
                {"input": input_np, "label": label_np}
                if weight_np is None
                else {"input": input_np, "label": label_np, "weight": weight_np}
            ),
            fetch_list=[res],
        )
    return static_result


def test_dygraph_layer(
    place, input_np, label_np, reduction='mean', weight_np=None
):
    paddle.disable_static()
    if weight_np is not None:
        weight = paddle.to_tensor(weight_np)
        bce_loss = paddle.nn.loss.BCELoss(weight=weight, reduction=reduction)
    else:
        bce_loss = paddle.nn.loss.BCELoss(reduction=reduction)
    dy_res = bce_loss(paddle.to_tensor(input_np), paddle.to_tensor(label_np))
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def test_dygraph_functional(
    place, input_np, label_np, reduction='mean', weight_np=None
):
    paddle.disable_static()
    input = paddle.to_tensor(input_np)
    label = paddle.to_tensor(label_np)

    if weight_np is not None:
        weight = paddle.to_tensor(weight_np)
        dy_res = paddle.nn.functional.binary_cross_entropy(
            input, label, weight=weight, reduction=reduction
        )
    else:
        dy_res = paddle.nn.functional.binary_cross_entropy(
            input, label, reduction=reduction
        )
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_bceloss(input_np, label_np, reduction='mean', weight_np=None):
    if weight_np is None:
        expected = -1 * (
            label_np * np.log(input_np)
            + (1.0 - label_np) * np.log(1.0 - input_np)
        )
    else:
        expected = (
            -1
            * weight_np
            * (
                label_np * np.log(input_np)
                + (1.0 - label_np) * np.log(1.0 - input_np)
            )
        )

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestBCELoss(unittest.TestCase):

    def test_BCELoss(self):
        input_np = np.random.uniform(0.1, 0.8, size=(20, 30)).astype(np.float64)
        label_np = np.random.randint(0, 2, size=(20, 30)).astype(np.float64)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                static_result = test_static_layer(
                    place, input_np, label_np, reduction
                )
                dy_result = test_dygraph_layer(
                    place, input_np, label_np, reduction
                )
                expected = calc_bceloss(input_np, label_np, reduction)
                np.testing.assert_allclose(static_result, expected, rtol=1e-05)
                np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
                np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
                static_functional = test_static_functional(
                    place, input_np, label_np, reduction
                )
                dy_functional = test_dygraph_functional(
                    place, input_np, label_np, reduction
                )
                np.testing.assert_allclose(
                    static_functional, expected, rtol=1e-05
                )
                np.testing.assert_allclose(
                    static_functional, dy_functional, rtol=1e-05
                )
                np.testing.assert_allclose(dy_functional, expected, rtol=1e-05)

    def test_BCELoss_weight(self):
        input_np = np.random.uniform(0.1, 0.8, size=(2, 3, 4, 10)).astype(
            np.float64
        )
        label_np = np.random.randint(0, 2, size=(2, 3, 4, 10)).astype(
            np.float64
        )
        weight_np = np.random.random(size=(3, 4, 10)).astype(np.float64)
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        for reduction in ['sum', 'mean', 'none']:
            static_result = test_static_layer(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            dy_result = test_dygraph_layer(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            expected = calc_bceloss(
                input_np, label_np, reduction, weight_np=weight_np
            )
            np.testing.assert_allclose(static_result, expected, rtol=1e-05)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-05)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-05)
            static_functional = test_static_functional(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            dy_functional = test_dygraph_functional(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            np.testing.assert_allclose(static_functional, expected, rtol=1e-05)
            np.testing.assert_allclose(
                static_functional, dy_functional, rtol=1e-05
            )
            np.testing.assert_allclose(dy_functional, expected, rtol=1e-05)

    def test_BCELoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.loss.BCELoss,
            reduction="unsupported reduction",
        )
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.binary_cross_entropy,
            input=input,
            label=label,
            reduction="unsupported reduction",
        )
        paddle.enable_static()


def bce_loss(input, label):
    return -1 * (label * np.log(input) + (1.0 - label) * np.log(1.0 - input))


def bce_wrapper(x, label):
    return paddle._C_ops.bce_loss(x, label)


class TestBceLossOp(OpTest):
    def setUp(self):
        self.init_test_dtype()
        self.init_test_case()
        self.op_type = "bce_loss"
        self.prim_op_type = "comp"
        self.python_api = bce_wrapper
        self.public_python_api = bce_wrapper
        input_np = np.random.uniform(0.1, 0.8, self.shape).astype(self.dtype)
        label_np = np.random.randint(0, 2, self.shape).astype(self.dtype)
        output_np = bce_loss(input_np, label_np)

        self.inputs = {'X': input_np, 'Label': label_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_case(self):
        self.shape = [10, 10]

    def init_test_dtype(self):
        self.dtype = "float64"


class TestBceLossOpCase1(OpTest):
    def init_test_cast(self):
        self.shape = [2, 3, 4, 5]


class TestBceLossOpCase2(OpTest):
    def init_test_cast(self):
        self.shape = [2, 3, 20]


class TestBceLossOpFP16(TestBceLossOp):
    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_dtype(self):
        self.dtype = np.float16


class TestBceLossOpStaticFP16(unittest.TestCase):

    def test_fp16(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.enable_static()
        shape = [2, 3, 20]
        x_data = np.random.uniform(0.1, 0.8, shape).astype("float16")
        y_data = np.random.randint(0, 2, shape).astype("float16")
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(shape=shape, name='x', dtype='float16')
            y = paddle.static.data(shape=shape, name='y', dtype='float16')
            out = paddle.nn.functional.binary_cross_entropy(
                x, y, reduction="none"
            )
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                output_pd = exe.run(
                    feed={'x': x_data, 'y': y_data}, fetch_list=[out]
                )[0]
        paddle.disable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
