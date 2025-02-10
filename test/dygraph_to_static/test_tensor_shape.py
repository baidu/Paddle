#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_pir_only,
    test_pt_only,
)

import paddle


def dyfunc_tensor_shape_1(x):
    x = paddle.to_tensor(x)
    res = paddle.reshape(x, shape=x.shape)
    return res


def dyfunc_tensor_shape_2(x):
    x = paddle.to_tensor(x)
    shape = x.shape
    shape2 = shape
    res = paddle.reshape(x, shape2)
    return res


def dyfunc_tensor_shape_3(x):
    # Transform y.shape but run y.shape actually because y is not Tensor
    x = paddle.to_tensor(x)
    y = paddle.ones([1, 5])
    res = paddle.reshape(x, shape=y.shape)
    return res


def dyfunc_tensor_shape_4(x):
    x = paddle.to_tensor(x)
    res = paddle.reshape(x, shape=(-1, x.shape[0], len(x.shape)))
    return res


def dyfunc_tensor_shape_5(x):
    # `res = base.layers.reshape(x, shape=(-1, s))` to
    # `res = base.layers.reshape(x, shape=(-1,
    #           paddle.jit.dy2static.convert_var_shape(x)[0]))`
    x = paddle.to_tensor(x)
    s = x.shape[0]
    res = paddle.reshape(x, shape=(-1, s))
    return res


def dyfunc_tensor_shape_6(x):
    # `res = base.layers.reshape(x, shape=(-1, s))` to
    # `res = base.layers.reshape(x, shape=(-1,
    #           paddle.jit.dy2static.convert_var_shape(x)[0:]))`
    x = paddle.to_tensor(x)
    s = x.shape[0:]
    res = paddle.reshape(x, shape=s)
    return res


def dyfunc_tuple_shape_1(x):
    x = paddle.to_tensor(x)
    a, b = x.shape
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_tuple_shape_2(x):
    x = paddle.to_tensor(x)
    shape = x.shape
    a, b = shape
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_tuple_shape_3(x):
    x = paddle.to_tensor(x)
    a, b = paddle.shape(x)
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_paddle_shape_api(x):
    x = paddle.to_tensor(x)
    # paddle.shape will not be converted.
    a = paddle.shape(x)[0]
    # alias api will also not be converted.
    alias_old_api = paddle.base.layers
    b = paddle.shape(x)[1]
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_with_if_1(x):
    x = paddle.to_tensor(x)
    res = paddle.reshape(x, [-1, 1])
    x_shape_0 = x.shape[0]
    if x_shape_0 < 1:
        # `res.shape[0]` is transformed into
        #   `paddle.jit.dy2static.convert_var_shape(res)[0]`
        if res.shape[0] > 1:
            res = paddle.full(shape=x.shape, fill_value=2, dtype="int32")
        else:
            res = paddle.full(shape=x.shape, fill_value=3, dtype="int32")
    return res


def dyfunc_with_if_2(x):
    x = paddle.to_tensor(x)
    # `len(x.shape)` will not be transformed because x.shape is not used by Paddle api.
    if len(x.shape) < 1:
        res = x
    else:
        res = paddle.full(shape=x.shape, fill_value=8, dtype="int32")

    return res


def dyfunc_with_for_1(x):
    x = paddle.to_tensor(x)
    res = paddle.full(shape=[1], fill_value=0, dtype="int32")
    # `x.shape[0]` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    for i in range(x.shape[0]):
        res += 1
    return res


def dyfunc_with_for_2(x):
    x = paddle.to_tensor(x)
    x_shape_0 = x.shape[0]
    res = paddle.full(shape=[1], fill_value=0, dtype="int32")

    # `x_shape_0` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    for i in range(x_shape_0):
        res += 1
    return res


def dyfunc_with_for_3(x):
    x = paddle.to_tensor(x)
    res = paddle.full(shape=[1], fill_value=0, dtype="int32")
    # `len(x.shape)` is not transformed.
    for i in range(len(x.shape)):
        res += 1

    return res


def dyfunc_with_while_1(x):
    x = paddle.to_tensor(x)
    res = paddle.full(shape=[1], fill_value=0, dtype="int32")
    # `x.shape[0]` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    i = 1
    while i < x.shape[0]:
        res += 1
        i = i + 2
    return res


def dyfunc_with_while_2(x):
    x = paddle.to_tensor(x)
    x_shape_0 = x.shape[0]
    res = paddle.full(shape=[1], fill_value=0, dtype="int32")
    i = 1
    # `x_shape_0` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    while i < x_shape_0:
        res += 1
        i = i + 2
    return res


def dyfunc_with_while_3(x):
    x = paddle.to_tensor(x)
    x_shape = x.shape
    res = paddle.full(shape=[1], fill_value=0, dtype="int32")
    i = 1

    # `len(x.shape)` is not transformed.
    while len(x_shape) > i:
        res += 1
        i += 1
    return res


def dyfunc_with_while_4(x):
    x = paddle.to_tensor(x)
    y = paddle.ones([1, 5])
    y_shape_0 = y.shape[0]
    i = 1

    # Transform y_shape_0 but run y.shape[0] actually because y is not Tensor
    while y_shape_0 > i:
        x += 1
        i += 1
    return x


def dyfunc_change_shape_after_assign(x):
    x = paddle.to_tensor(x)
    a, b = x.shape
    x = paddle.reshape(x, shape=(-1, 1))
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_len_paddle_shape():
    x = paddle.to_tensor([1, 2, 3])
    if len(paddle.shape(x)) > 0:
        print(x)


def dyfunc_dict_assign_shape():
    x = paddle.to_tensor([1, 2])
    a = {}
    a['shape'] = x.shape[0]


def walk(block, fn):
    fn(block)
    for op in block.ops:
        for sub_block in op.blocks():
            walk(sub_block, fn)


def get_op_num_in_block(block, op_name):
    num_ops = 0
    for op in block.ops:
        if op.name() == op_name:
            num_ops += 1
    return num_ops


def get_op_num_in_program(program, op_name):
    num_ops = 0

    def _calc_op_num(block):
        nonlocal num_ops
        num_ops += get_op_num_in_block(block, op_name)

    walk(program.global_block(), _calc_op_num)
    return num_ops


# 1. Basic tests without control flow
class TestTensorShapeBasic(Dy2StTestBase):
    def setUp(self):
        self.input = np.ones(5).astype("int32")
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self._set_input_spec()
        self._set_expected_op_num()
        self._set_pir_expected_op_num()
        self.init_test_func()

    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_1

    def _set_input_spec(self):
        self.input_spec = [paddle.static.InputSpec(shape=[5], dtype="int32")]

    def _run(self, to_static):
        if to_static:
            res = paddle.jit.to_static(self.dygraph_func)(self.input).numpy()
        else:
            res = self.dygraph_func(self.input).numpy()
        return res

    def get_dygraph_output(self):
        return self._run(to_static=False)

    def get_static_output(self):
        return self._run(to_static=True)

    def test_transformed_static_result(self):
        static_res = self.get_static_output()
        dygraph_res = self.get_dygraph_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)

    def _set_expected_op_num(self):
        # TODO(cleanup-legacy-ir): Remove _set_expected_op_num related code
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 3
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0

    def _compute_op_num(self, program):
        op_num = sum([len(block.ops) for block in program.blocks])
        shape_op_num = 0
        slice_op_num = 0

        for block in program.blocks:
            shape_op_num += len(
                [
                    op
                    for op in block.ops
                    if (op.type == "shape" or op.type == "shape64")
                ]
            )
            slice_op_num += len([op for op in block.ops if op.type == "slice"])
        return op_num, shape_op_num, slice_op_num

    def _compute_pir_op_num(self, program):
        op_num = program.global_block().num_ops()
        shape_op_num = get_op_num_in_program(program, "pd_op.shape")
        shape_op_num += get_op_num_in_program(program, "pd_op.shape64")
        slice_op_num = get_op_num_in_program(program, "pd_op.slice")
        return op_num, shape_op_num, slice_op_num

    @test_ast_only
    @test_pt_only
    def test_op_num(self):
        static_layer = paddle.jit.to_static(self.dygraph_func, self.input_spec)
        program = static_layer.main_program
        op_num, shape_op_num, slice_op_num = self._compute_op_num(program)
        self.assertEqual(op_num, self.expected_op_num)
        self.assertEqual(shape_op_num, self.expected_shape_op_num)
        self.assertEqual(slice_op_num, self.expected_slice_op_num)

    @test_ast_only
    @test_pir_only
    def test_pir_op_num(self):
        static_layer = paddle.jit.to_static(self.dygraph_func, self.input_spec)
        program = static_layer.main_program
        op_num, shape_op_num, slice_op_num = self._compute_pir_op_num(program)
        self.assertEqual(op_num, self.pir_expected_op_num)
        self.assertEqual(shape_op_num, self.pir_expected_shape_op_num)
        self.assertEqual(slice_op_num, self.pir_expected_slice_op_num)


class TestTensorShapeBasic2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_2

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 3
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeBasic3(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_3

    def _set_expected_op_num(self):
        self.expected_op_num = 2
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 4
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeBasic4(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_4


class TestTensorShapeBasic5(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_5

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 3
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeBasic6(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_6

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 3
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTupleShape1(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = np.ones((5, 7)).astype("int32")
        self.input_spec = [
            paddle.static.InputSpec(shape=[-1, -1], dtype="int32")
        ]
        self.dygraph_func = dyfunc_tuple_shape_1

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 2

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 11
        self.pir_expected_shape_op_num = 1
        self.pir_expected_slice_op_num = 2


class TestTupleShape2(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = np.ones((5, 7)).astype("int32")
        self.input_spec = [
            paddle.static.InputSpec(shape=[-1, 7], dtype="int32")
        ]
        self.dygraph_func = dyfunc_tuple_shape_2

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 9
        self.pir_expected_shape_op_num = 1
        self.pir_expected_slice_op_num = 1


class TestTupleShape3(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = np.ones((5, 7)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[5, 7], dtype="int32")]
        self.dygraph_func = dyfunc_tuple_shape_3

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 2

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 11
        self.pir_expected_shape_op_num = 1
        self.pir_expected_slice_op_num = 2


class TestPaddleShapeApi(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = np.ones((5, 7)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[5, 7], dtype="int32")]
        self.dygraph_func = dyfunc_paddle_shape_api

    def _set_expected_op_num(self):
        self.expected_op_num = 5
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 2

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 12
        self.pir_expected_shape_op_num = 2
        self.pir_expected_slice_op_num = 2


# 2. Tests with control flow if
class TestTensorShapeInIf1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_if_1

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 3
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeInIf2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_if_2

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 2
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


# 3. Tests with control flow for loop
class TestTensorShapeInFor1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_1

    def _set_expected_op_num(self):
        self.expected_op_num = 6
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 12
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeInFor2(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_2

    def _set_expected_op_num(self):
        self.expected_op_num = 6
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 12
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeInFor3(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_3

    def _set_expected_op_num(self):
        self.expected_op_num = 2
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 4
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


# 4. Tests with control flow while loop
class TestTensorShapeInWhile1(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_1

    def _set_expected_op_num(self):
        self.expected_op_num = 3
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 6
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeInWhile2(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_2

    def _set_expected_op_num(self):
        self.expected_op_num = 3
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 6
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeInWhile3(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_3

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 2
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


class TestTensorShapeInWhile4(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_4

    def _set_expected_op_num(self):
        self.expected_op_num = 1
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 2
        self.pir_expected_shape_op_num = 0
        self.pir_expected_slice_op_num = 0


# 5. Test op num for negative dim
class TestOpNumBasicWithTensorShape(Dy2StTestBase):
    def setUp(self):
        self._set_input_spec()
        self._set_test_func()
        self._set_expected_op_num()
        self._set_pir_expected_op_num()

    def _set_input_spec(self):
        self.input_spec = [
            paddle.static.InputSpec(shape=[-1, 5], dtype="int32")
        ]

    def _set_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_1

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 9
        self.pir_expected_shape_op_num = 1
        self.pir_expected_slice_op_num = 1

    def _compute_op_num(self, program):
        self.op_num = sum([len(block.ops) for block in program.blocks])
        self.shape_op_num = 0
        self.slice_op_num = 0

        for block in program.blocks:
            self.shape_op_num += len(
                [
                    op
                    for op in block.ops
                    if (op.type == "shape" or op.type == "shape64")
                ]
            )
            self.slice_op_num += len(
                [op for op in block.ops if op.type == "slice"]
            )

    def _compute_pir_op_num(self, program):
        op_num = program.global_block().num_ops()
        shape_op_num = get_op_num_in_program(program, "pd_op.shape")
        shape_op_num += get_op_num_in_program(program, "pd_op.shape64")
        slice_op_num = get_op_num_in_program(program, "pd_op.slice")
        return op_num, shape_op_num, slice_op_num

    @test_ast_only
    @test_pt_only
    def test_op_num(self):
        static_layer = paddle.jit.to_static(self.dygraph_func, self.input_spec)
        program = static_layer.main_program

        self._compute_op_num(program)
        self.assertEqual(self.op_num, self.expected_op_num)
        self.assertEqual(self.shape_op_num, self.expected_shape_op_num)
        self.assertEqual(self.slice_op_num, self.expected_slice_op_num)

    @test_ast_only
    @test_pir_only
    def test_pir_op_num(self):
        static_layer = paddle.jit.to_static(self.dygraph_func, self.input_spec)
        program = static_layer.main_program
        op_num, shape_op_num, slice_op_num = self._compute_pir_op_num(program)
        self.assertEqual(op_num, self.pir_expected_op_num)
        self.assertEqual(shape_op_num, self.pir_expected_shape_op_num)
        self.assertEqual(slice_op_num, self.pir_expected_slice_op_num)


class TestOpNumBasicWithTensorShape4(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_4

    def _set_expected_op_num(self):
        self.expected_op_num = 7
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 2

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 14
        self.pir_expected_shape_op_num = 2
        self.pir_expected_slice_op_num = 2


class TestOpNumWithTensorShapeTuple1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_tuple_shape_1

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 9
        self.pir_expected_shape_op_num = 1
        self.pir_expected_slice_op_num = 1


class TestOpNumWithTensorShapeInIf1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_with_if_1

    def _set_expected_op_num(self):
        self.expected_op_num = 31
        self.expected_shape_op_num = 4
        self.expected_slice_op_num = 4

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 39
        self.pir_expected_shape_op_num = 4
        self.pir_expected_slice_op_num = 4


class TestOpNumWithTensorShapeInFor1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_with_for_1

    def _set_expected_op_num(self):
        self.expected_op_num = 26
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 3

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 32
        self.pir_expected_shape_op_num = 2
        self.pir_expected_slice_op_num = 3


class TestOpNumWithTensorShapeInWhile1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_with_while_1

    def _set_expected_op_num(self):
        self.expected_op_num = 20
        self.expected_shape_op_num = 3
        self.expected_slice_op_num = 3

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 25
        self.pir_expected_shape_op_num = 3
        self.pir_expected_slice_op_num = 3


class TestChangeShapeAfterAssign(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = np.ones((2, 3)).astype("int32")
        self.input_spec = [
            paddle.static.InputSpec(shape=[-1, 3], dtype="int32")
        ]
        self.dygraph_func = dyfunc_change_shape_after_assign

    def _set_expected_op_num(self):
        self.expected_op_num = 5
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1

    def _set_pir_expected_op_num(self):
        self.pir_expected_op_num = 11
        self.pir_expected_shape_op_num = 1
        self.pir_expected_slice_op_num = 1


def dyfunc_with_static_convert_var_shape(x):
    # Note: this will create `batch_size__static_convert_var_shape_suffix_0` firstly.
    batch_size = x.shape[0]
    if len(x.shape) < 1:
        res = x
    else:
        # Test for correctly to find `batch_size__static_convert_var_shape_suffix_0` in
        # deeply nested scope.
        res = paddle.full(shape=[batch_size], fill_value=8, dtype="int32")

    return res


class TestFindStaticConvertVarShapeSuffixVar(Dy2StTestBase):
    @test_ast_only
    def test(self):
        x_spec = paddle.static.InputSpec(shape=[None, 10])
        func = paddle.jit.to_static(
            dyfunc_with_static_convert_var_shape, input_spec=[x_spec]
        )
        # Call this function to trigger program translation.
        func.concrete_program  # noqa: B018


if __name__ == '__main__':
    unittest.main()
