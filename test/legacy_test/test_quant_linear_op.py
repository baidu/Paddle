# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, paddle_static_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

SEED = 2020


def round_array(x):
    x[x > 0] = np.ceil(x[x > 0])
    x[x <= 0] = np.floor(x[x <= 0])


def round_array_with_ties_to_even(x):
    xLower = np.floor(x)
    xUpper = np.ceil(x)
    dLower = x - xLower
    dUpper = xUpper - x
    x[(dLower == dUpper) & (xLower % 2 == 0)] = xLower[
        (dLower == dUpper) & (xLower % 2 == 0)
    ]
    x[(dLower == dUpper) & (xLower % 2 != 0)] = xUpper[
        (dLower == dUpper) & (xLower % 2 != 0)
    ]
    x[dLower < dUpper] = xLower[dLower < dUpper]
    x[dLower > dUpper] = xUpper[dLower > dUpper]


def quant_linear_refer(matrix, with_bias, with_relu=False):
    in_n, in_c, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    w_data = np.reshape(matrix.weights, [w_i, w_o])
    b_data = np.reshape(matrix.bias, [1, w_o])
    result = None

    if with_bias:
        result = np.dot(x_data, w_data) + b_data
    else:
        result = np.dot(x_data, w_data)

    if with_relu:
        return np.maximum(result, 0)
    else:
        return result


def quant_linear_quant_refer(
    matrix,
    with_bias,
    scale_in,
    scale_weights,
    quant_round_type=1,
    quant_max_bound=127,
    quant_min_bound=-127,
    with_relu=False,
):
    in_n, in_c, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
    quant_x_data = x_data.astype('float32')
    quant_x_data = quant_max_bound * scale_in * quant_x_data
    if quant_round_type == 0:
        round_array_with_ties_to_even(quant_x_data)
    else:
        round_array(quant_x_data)
    quant_x_data[quant_x_data > quant_max_bound] = quant_max_bound
    quant_x_data[quant_x_data < quant_min_bound] = quant_min_bound
    quant_x_data = quant_x_data.astype('int8')

    w_data = np.reshape(matrix.weights, [w_i, w_o])
    b_data = np.reshape(matrix.bias, [1, w_o])
    result = None
    quant_result = np.dot(quant_x_data.astype('int32'), w_data.astype('int32'))
    scale_out = scale_weights * scale_in
    result = quant_result / (quant_max_bound * quant_max_bound * scale_out)
    result = result.astype(x_data.dtype)

    if with_bias:
        result = result + b_data

    if with_relu:
        return np.maximum(result, 0)
    else:
        return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w, bias_dims=2):
        self.input = np.random.random((mb, ic, h, w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")
        if bias_dims == 2:
            self.bias = np.random.random((1, oc)).astype("float32")
        else:
            self.bias = np.random.random(oc).astype("float32")


def get_scale_in(input):
    max_v = np.max(np.abs(input))
    return 1 / max_v


def get_scale_weights(weights):
    max_v = np.max(np.abs(weights), axis=0)
    return 1 / max_v


def quant_weights(
    weights, scale_weights, quant_round_type, quant_max_bound, quant_min_bound
):
    quant_weights = weights.astype('float32')
    quant_weights = quant_max_bound * scale_weights * quant_weights
    if quant_round_type == 0:
        round_array_with_ties_to_even(quant_weights)
    else:
        round_array(quant_weights)
    quant_weights[quant_weights > quant_max_bound] = quant_max_bound
    quant_weights[quant_weights < quant_min_bound] = quant_min_bound
    quant_weights = quant_weights.astype('int8')
    return quant_weights


class TestQuantLinearOp(OpTest):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3, 2)

    def setUp(self):
        self.op_type = "quant_linear"
        self.config()

        if self.with_bias:
            self.inputs = {
                'x': self.matrix.input,
                'w': self.matrix.weights,
                'bias': self.matrix.bias,
            }
        else:
            self.inputs = {'x': self.matrix.input, 'w': self.matrix.weights}

        if self.with_relu:
            activation_type = "relu"
        else:
            activation_type = ""
        if hasattr(self, 'is_quant'):
            self.attrs = {
                'activation_type': activation_type,
                'is_quant': self.is_quant,
                'quant_round_type': self.quant_round_type,
                'quant_max_bound': self.quant_max_bound,
                'quant_min_bound': self.quant_min_bound,
                'scale_in': self.scale_in,
                'scale_weights': self.scale_weights,
            }
        else:
            self.attrs = {'activation_type': activation_type}

        if hasattr(self, 'is_quant') and self.attrs['is_quant']:
            self.outputs = {
                'out': quant_linear_quant_refer(
                    self.matrix,
                    self.with_bias,
                    self.scale_in,
                    self.scale_weights,
                    self.quant_round_type,
                    self.quant_max_bound,
                    self.quant_min_bound,
                    self.with_relu,
                )
            }
        else:
            self.outputs = {
                'out': quant_linear_refer(
                    self.matrix, self.with_bias, self.with_relu
                )
            }

    def test_check_output(self):
        if hasattr(self, 'is_quant') and self.attrs['is_quant']:
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                self.check_output_with_place(place, check_dygraph=False)
        else:
            self.check_output(check_dygraph=False)


class TestQuantLinearOpNoBias1(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(2, 8, 10, 1, 1, 2)


class TestQuantLinearOpNoBias2(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)


class TestQuantLinearOpNoBias4(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.matrix = MatrixGenerate(1, 32, 64, 3, 3, 1)


class TestQuantLinearOpWithBias1(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = False
        self.matrix = MatrixGenerate(3, 8, 10, 2, 1, 2)


class TestQuantLinearOpWithBias2(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(4, 5, 6, 2, 2, 1)


class TestQuantLinearOpWithBias3(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 64, 32, 3, 3, 1)


class TestQuantLinearOpWithPadding(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.matrix = MatrixGenerate(1, 4, 3, 128, 128, 2)


class TestQuantLinearOp_NumFlattenDims_NegOne(unittest.TestCase):
    def test_api(self):
        def run_program(num_flatten_dims):
            paddle.seed(SEED)
            np.random.seed(SEED)
            startup_program = Program()
            main_program = Program()

            with paddle_static_guard():
                with program_guard(main_program, startup_program):
                    input = np.random.random([2, 2, 25]).astype("float32")
                    x = paddle.static.data(
                        name="x",
                        shape=[2, 2, 25],
                        dtype="float32",
                    )

                    out = paddle.static.nn.quant_linear(
                        x=x, size=1, num_flatten_dims=num_flatten_dims
                    )

                place = (
                    base.CPUPlace()
                    if not core.is_compiled_with_cuda()
                    else base.CUDAPlace(0)
                )
                exe = base.Executor(place=place)
                exe.run(startup_program)
                out = exe.run(main_program, feed={"x": input}, fetch_list=[out])
                return out

        res_1 = run_program(-1)
        res_2 = run_program(2)
        np.testing.assert_array_equal(res_1, res_2)


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "QuantLinear only supports cuda kernel when is_quant is True.",
)
class TestQuantLinearOp_NumFlattenDims_NegOne_WithQuant(unittest.TestCase):
    def test_api(self):
        def run_program(num_flatten_dims):
            paddle.seed(SEED)
            np.random.seed(SEED)
            startup_program = Program()
            main_program = Program()

            with paddle_static_guard():
                with program_guard(main_program, startup_program):
                    quant_round_type = 0
                    quant_max_bound = 127.0
                    quant_min_bound = -127.0

                    input = np.random.random([2, 2, 25]).astype("float32")
                    scale_in = get_scale_in(input)
                    x = paddle.static.data(
                        name="x",
                        shape=[2, 2, 25],
                        dtype="float32",
                    )

                    weight = np.random.random([25, 1]).astype("float32")
                    scale_weight = get_scale_weights(weight)
                    weight = quant_weights(
                        weight,
                        scale_weight,
                        quant_round_type,
                        quant_max_bound,
                        quant_min_bound,
                    )
                    w = paddle.static.data(
                        name="w",
                        shape=[25, 1],
                        dtype="int8",
                    )

                    out = paddle.static.nn.quant_linear(
                        x=x,
                        size=1,
                        num_flatten_dims=num_flatten_dims,
                        w=w,
                        is_quant=True,
                        scale_in=scale_in,
                        scale_weight=scale_weight.tolist(),
                        quant_round_type=quant_round_type,
                        quant_max_bound=quant_max_bound,
                        quant_min_bound=quant_min_bound,
                    )

                place = (
                    base.CPUPlace()
                    if not core.is_compiled_with_cuda()
                    else base.CUDAPlace(0)
                )
                exe = base.Executor(place=place)
                exe.run(startup_program)
                out = exe.run(
                    main_program,
                    feed={"x": input, "w": weight},
                    fetch_list=[out],
                )
                return out

        res_1 = run_program(-1)
        res_2 = run_program(2)
        np.testing.assert_array_equal(res_1, res_2)


class TestQuantLinearOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                with paddle_static_guard():
                    # the input type must be Variable
                    paddle.static.nn.quant_linear(x=input_data, size=1)

            self.assertRaises(TypeError, test_Variable)

            def test_type():
                with paddle_static_guard():
                    # dtype must be float32 or float64
                    x2 = paddle.static.data(
                        name='x2', shape=[-1, 4], dtype='int32'
                    )
                    paddle.static.nn.quant_linear(x=x2, size=1)

            self.assertRaises(TypeError, test_type)

            with paddle_static_guard():
                # The input dtype of quant_linear can be float16 in GPU, test for warning
                x3 = paddle.static.data(
                    name='x3', shape=[-1, 4], dtype='float16'
                )
                paddle.static.nn.quant_linear(x=x3, size=1)


class TestQuantLinearOpQuantNoBias1(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(16, 10, 16, 4, 4, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantNoBias2(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.is_quant = True
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 8, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantNoBias3(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 6, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantNoBias4(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 14, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantNoBias5(TestQuantLinearOp):
    def config(self):
        self.with_bias = False
        self.with_relu = False
        self.is_quant = True
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(2, 1, 10, 1, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantWithBias1(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 64, 32, 3, 3, 1)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantWithBias2(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.is_quant = True
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(3, 8, 10, 2, 1, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantWithPadding1(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.is_quant = True
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 4, 4, 128, 128, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


class TestQuantLinearOpQuantWithPadding2(TestQuantLinearOp):
    def config(self):
        self.with_bias = True
        self.with_relu = True
        self.is_quant = True
        self.quant_round_type = 0
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.matrix = MatrixGenerate(1, 4, 3, 128, 128, 2)
        self.scale_in = get_scale_in(self.matrix.input)
        self.scale_weights = get_scale_weights(self.matrix.weights)
        self.matrix.weights = quant_weights(
            self.matrix.weights,
            self.scale_weights,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )


if __name__ == "__main__":
    unittest.main()
