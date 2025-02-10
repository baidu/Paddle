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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    IrMode,
    ToStaticMode,
    disable_test_case,
    test_ast_only,
)

import paddle
from paddle import base

SEED = 2020
np.random.seed(SEED)


# Situation 1: Test list append
def test_list_append_without_control_flow(x):
    # Python list will not be transformed.
    x = paddle.assign(x)
    a = []
    # It's a plain python control flow which won't be transformed
    if 2 > 1:
        a.append(x)
    return a


def test_list_append_in_if(x):
    x = paddle.assign(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
    else:
        a.append(paddle.full(shape=[1, 2], fill_value=9, dtype="float32"))
    # TODO(Aurelius84): Currently, run_program_op doesn't support output DenseTensorArray.
    return a[0]


def test_list_append_in_for_loop(x, iter_num):
    x = paddle.assign(x)
    # Use `full` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = paddle.full(
        shape=[1], fill_value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved
    a = []
    for i in range(iter_num):
        a.append(x)
    return a[0]


def test_list_append_in_for_subscript(x):
    x = paddle.assign(x)
    iter_num = paddle.shape(x)[0]
    a = []
    for i in range(iter_num):
        x = x + 1
        a.append(x)
    out = paddle.concat(a)
    return out[0]


def test_list_append_in_while_loop_subscript(x):
    x = paddle.assign(x)
    iter_num = paddle.shape(x)[0]
    a = []
    i = 0
    while i < iter_num:
        x = x + 1
        a.append(x)
        i += 1
    out = paddle.concat(a)
    return out[0]


def test_list_append_in_for_loop_with_concat(x, iter_num):
    x = paddle.assign(x)
    a = []
    # Use `full` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = paddle.full(
        shape=[1], fill_value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved
    for i in range(iter_num):
        a.append(x)
    a = paddle.concat(a, axis=0)
    return a


def test_list_append_in_while_loop(x, iter_num):
    x = paddle.assign(x)
    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")
    a = []
    i = 0
    while i < iter_num:
        a.append(x)
        i += 1
    return a[0]


def test_list_append_in_while_loop_with_stack(x, iter_num):
    x = paddle.assign(x)
    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")
    a = []
    i = 0
    while i < iter_num.numpy()[0]:
        a.append(x)
        i += 1
    out = paddle.stack(a, axis=1)
    return out


def test_tensor_array_slice(x, iter_num):
    a = []
    for i in range(paddle.to_tensor(3)):
        a.append(paddle.to_tensor(float(i)))
    t = a[1:3]
    return a[2]


# Situation 2: Test list pop
def test_list_pop_without_control_flow_1(x):
    x = paddle.assign(x)
    a = []
    if 2 > 1:
        a.append(x)
    a.pop()
    return a


def test_list_pop_without_control_flow_2(x):
    x = paddle.assign(x)
    a = []
    if 2 > 1:
        a.append(x)
        a.append(x + 1)
    last_item = a.pop(1)
    return last_item


def test_list_pop_in_if(x):
    x = paddle.assign(x)
    a = []
    b = [x * 2 + (x + 1)]
    if x.numpy()[0] > 0:
        a.append(x)
        b.append(x + 1)
        a.append(paddle.full(shape=[1], fill_value=1, dtype="int64"))
    else:
        a.append(x + 1)
        b.append(x - 1)
        a.append(paddle.full(shape=[2], fill_value=2, dtype="int64"))
    item1 = a.pop(1)
    return item1, b[-1]


def test_list_pop_in_for_loop(x, iter_num):
    x = paddle.assign(x)
    # Use `full` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = paddle.full(
        shape=[1], fill_value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved

    a = []
    b = [x - 1, x + 1]
    for i in range(iter_num):
        a.append(x + i)
        b.append(x * 2)

    one = paddle.ones(shape=[1], dtype="int32")
    for i in range(one.numpy()[0]):
        item = a.pop()
    return a[0], item, b[1]


def test_list_pop_in_while_loop(x, iter_num):
    x = paddle.assign(x)
    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")
    a = []
    b = [x]
    b.append(x)
    b.pop()
    i = 0

    while i < iter_num:
        a.append(x + i)
        b.append(x - i)
        i += 1
        if i % 2 == 1:
            a.pop()
    return a[0], b[2]


class TestListWithoutControlFlowConfig(Dy2StTestBase):
    def setUp(self):
        self.place = base.CPUPlace()
        if base.is_compiled_with_cuda():
            self.place = base.CUDAPlace(0)
        if base.is_compiled_with_xpu():
            self.place = base.XPUPlace(0)

        self.init_data()
        self.init_dygraph_func()

    def init_data(self):
        self.input = np.random.random(3).astype('float32')

    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_list_append_without_control_flow,
            test_list_pop_without_control_flow_1,
            test_list_pop_without_control_flow_2,
        ]

    def result_to_numpy(self, res):
        if isinstance(res, (list, tuple)):
            res = paddle.utils.map_structure(lambda x: x.numpy(), res)
        else:
            res = [res.numpy()]
        return res

    def run_static_mode(self):
        return self.train(to_static=True)

    def run_dygraph_mode(self):
        return self.train(to_static=False)

    def train(self, to_static=False):
        with base.dygraph.guard():
            if to_static:
                res = paddle.jit.to_static(self.dygraph_func)(self.input)
            else:
                res = self.dygraph_func(self.input)
            return self.result_to_numpy(res)

    def compare_transformed_static_result(self):
        for dyfunc in self.all_dygraph_funcs:
            self.dygraph_func = dyfunc
            static_res_list = self.run_static_mode()
            dygraph_res_list = self.run_dygraph_mode()

            self.assertEqual(len(static_res_list), len(dygraph_res_list))
            for stat_res, dy_res in zip(static_res_list, dygraph_res_list):
                np.testing.assert_allclose(
                    stat_res,
                    dy_res,
                    rtol=1e-05,
                    err_msg=f'dygraph_res is {dy_res}\nstatic_res is {stat_res}',
                )


class TestListWithoutControlFlow(TestListWithoutControlFlowConfig):
    def test_transformed_static_result(self):
        self.compare_transformed_static_result()


class TestListInIf(TestListWithoutControlFlow):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [test_list_append_in_if]


class TestListInWhileLoop(TestListWithoutControlFlowConfig):
    def init_data(self):
        self.input = np.random.random(3).astype('float32')
        self.iter_num = 3

    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_list_append_in_while_loop,
            test_list_pop_in_while_loop,
        ]

    # TODO(zhangbo): Refine BuildOpFrom for op with sub_block
    def train(self, to_static=False):
        with base.dygraph.guard():
            if to_static:
                res = paddle.jit.to_static(self.dygraph_func)(
                    self.input, self.iter_num
                )
            else:
                res = self.dygraph_func(self.input, self.iter_num)
            return self.result_to_numpy(res)

    @disable_test_case((ToStaticMode.AST, IrMode.PT))
    def test_transformed_static_result(self):
        self.compare_transformed_static_result()


class TestListInWhileLoopWithStack(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [test_list_append_in_while_loop_with_stack]


class TestTensorArraySlice(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [test_tensor_array_slice]


class TestListInForLoop(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_list_append_in_for_loop,
            test_list_pop_in_for_loop,
        ]


class TestListInForLoopWithConcat(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_list_append_in_for_loop_with_concat,
        ]


class TestListInForLoopWithSubscript(TestListWithoutControlFlow):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_list_append_in_for_subscript,
            test_list_append_in_while_loop_subscript,
        ]

    def init_data(self):
        self.input = np.random.random((3, 4)).astype('float32')


class ListWithCondNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    # Add *args to test function.__self__ in FunctionSpec.
    # DO NOT remove *args.
    def forward(self, x, index, *args):
        y = paddle.nn.functional.relu(x)
        a = []

        for i in y:
            a.append(i)

        if index > 0:
            res = a[0] * a[0]
            y = y + 1
        else:
            res = a[-1] * a[-1]
            y = y - 1

        z = a[-1] * res * y[0]
        return z


class TestListWithCondGradInferVarType(Dy2StTestBase):
    def test_to_static(self):
        net = ListWithCondNet()
        x = paddle.to_tensor([2, 3, 4], dtype='float32')
        index = paddle.to_tensor([1])
        res = paddle.jit.to_static(net)(x, index)
        self.assertEqual(res, 48.0)


def tensor_array_dtype():
    l = []
    for i in range(paddle.to_tensor(3)):
        l.append(i)
    return l[0]


class TestTensorArrayDtype(Dy2StTestBase):
    @test_ast_only
    def test_tensor_array_dtype(self):
        fn = tensor_array_dtype
        static_fn = paddle.jit.to_static(fn)
        st_out = static_fn()
        self.assertEqual(st_out.dtype, paddle.int64)


if __name__ == '__main__':
    unittest.main()
