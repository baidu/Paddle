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

from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
import unittest
import inspect
import ast

from paddle.fluid.dygraph.jit import dygraph_to_static_output
from paddle.fluid.dygraph.dygraph_to_static.utils import is_dygraph_api

SEED = 2020
np.random.seed(SEED)
fluid.default_main_program().random_seed = SEED
fluid.default_startup_program().random_seed = SEED
"""
def dyfunc(input):
    res = fluid.layers.pool2d(input, pool_size=2, pool_type='avg', pool_stride=1, global_pooling=False)
    return res
"""


def dyfunc(input):
    pool2d = fluid.dygraph.Pool2D(
        pool_size=2, pool_type='avg', pool_stride=1, global_pooling=False)
    res = pool2d(input)
    return res


def dyfunc_to_variable(x):
    input = fluid.dygraph.to_variable(x)
    pool2d = fluid.dygraph.Pool2D(
        pool_size=2, pool_type='avg', pool_stride=1, global_pooling=False)
    res = pool2d(input)
    return res


def dyfunc_conv2d(input):
    conv2d = fluid.dygraph.Conv2D(
        num_channels=3,
        num_filters=2,
        filter_size=3,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.99)),
        bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.5)), )
    res = conv2d(input)
    return res


def dyfunc_linear(input):
    fc = fluid.dygraph.Linear(
        input_dim=10,
        output_dim=5,
        act='relu',
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.99)),
        bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.5)), )
    res = fc(input)
    return res


def dyfunc_bilinear_tensor_product(layer1, layer2):
    bilinearTensorProduct = fluid.dygraph.nn.BilinearTensorProduct(
        input1_dim=5,
        input2_dim=4,
        output_dim=1000,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.99)),
        bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.5)))
    l1 = fluid.dygraph.base.to_variable(layer1)
    l2 = fluid.dygraph.base.to_variable(layer2)

    res = bilinearTensorProduct(l1, l2)
    # res = bilinearTensorProduct(fluid.dygraph.base.to_variable(layer1),
    #                             fluid.dygraph.base.to_variable(layer2))
    return res


"""
def dyfunc_prelu(input):
    res = fluid.layers.prelu(x=input, mode='all', param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
    return res
"""


def dyfunc_prelu(input):
    prelu0 = fluid.PRelu(
        mode='all',
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
    res = prelu0(input=input)
    return res


class TestDygraphBasicAPI(unittest.TestCase):
    '''
    Compare results of dynamic graph and transformed static graph function which only
    includes basic API.
    '''

    def setUp(self):
        self.input = np.random.random((1, 4, 3, 3)).astype('float32')
        self.dygraph_func = dyfunc

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            data = fluid.dygraph.to_variable(self.input)
            res = self.dygraph_func(data).numpy()

            return res

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program, startup_program):
            data = fluid.layers.assign(self.input)
            static_out = dygraph_to_static_output(self.dygraph_func)(data)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        # print("dygraph_res\n", dygraph_res)
        # print("static_res\n", static_res)
        self.assertTrue(np.array_equal(static_res, dygraph_res))


class TestDygraphBasicAPI_Case1(TestDygraphBasicAPI):
    def setUp(self):
        self.input = np.random.random((1, 3, 3, 5)).astype('float32')
        self.dygraph_func = dyfunc_conv2d


class TestDygraphBasicAPI_Case2(TestDygraphBasicAPI):
    def setUp(self):
        # todo: add test that input dimension is greater than 2, eg: 3-D or 4-D Tensor
        # after merge fc
        self.input = np.random.random((4, 10)).astype('float32')
        self.dygraph_func = dyfunc_linear


class TestDygraphBasicAPI_Case3(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random((1, 1, 3, 20)).astype('float32')
        self.dygraph_func = dyfunc_to_variable

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input).numpy()

            return res

    def get_static_output(self):
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program):
            static_out = dygraph_to_static_output(self.dygraph_func)(self.input)

        exe = fluid.Executor(fluid.CPUPlace())
        static_res = exe.run(main_program, fetch_list=static_out)

        return static_res[0]

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        # print(static_res)
        self.assertTrue(np.array_equal(static_res, dygraph_res))


class TestDygraphBasicAPI_Case4(TestDygraphBasicAPI):
    def setUp(self):
        self.layer1 = np.random.random((5, 5)).astype('float32')
        self.layer2 = np.random.random((5, 4)).astype('float32')
        self.dygraph_func = dyfunc_bilinear_tensor_product

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            res = self.dygraph_func(self.layer1, self.layer2).numpy()
            return res

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program, startup_program):
            static_out = dygraph_to_static_output(self.dygraph_func)(
                self.layer1, self.layer2)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]


class TestDygraphBasicAPI_Case5(TestDygraphBasicAPI):
    def setUp(self):
        self.input = np.ones([5, 20, 10, 10]).astype('float32')
        self.dygraph_func = dyfunc_prelu


def _dygraph_fn():
    import paddle.fluid as fluid
    x = np.random.random((1, 3)).astype('float32')
    with fluid.dygraph.guard():
        fluid.dygraph.to_variable(x)
        np.random.random((1))


class TestDygraphAPIRecognition(unittest.TestCase):
    def setUp(self):
        self.src = inspect.getsource(_dygraph_fn)
        self.root_ast = ast.parse(self.src)

    def _get_dygraph_ast_node(self):
        return self.root_ast.body[0].body[2].body[0].value

    def _get_static_ast_node(self):
        return self.root_ast.body[0].body[2].body[1].value

    def test_dygraph_api(self):
        self.assertTrue(is_dygraph_api(self._get_dygraph_ast_node()) is True)
        self.assertTrue(is_dygraph_api(self._get_static_ast_node()) is False)


if __name__ == '__main__':
    unittest.main()
