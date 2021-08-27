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
from op_test import OpTest, skip_check_grad_ci
from numpy.linalg import multi_dot
from op_test import OpTest
import paddle
from paddle.fluid import Program, program_guard
import paddle.fluid as fluid

paddle.enable_static()


class TestMultiDotOp(OpTest):
    def setUp(self):
        self.op_type = "multi_dot"
        self.dtype = self.get_dtype()
        self.get_inputs_and_outputs()

    def get_dtype(self):
        return "float32"

    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 8)).astype(self.dtype)
        self.B = np.random.random((8, 4)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)


class TestMultiDotOpDouble(TestMultiDotOp):
    def get_dtype(self):
        return "float64"


#(A*B)*C
class TestMultiDotOp3Mat(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 10)).astype(self.dtype)
        self.B = np.random.random((10, 4)).astype(self.dtype)
        self.C = np.random.random((4, 3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)


#A*(B*C)
class TestMultiDotOp3Mat2(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((3, 4)).astype(self.dtype)
        self.B = np.random.random((4, 8)).astype(self.dtype)
        self.C = np.random.random((8, 2)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)


class TestMultiDotOp4Mat(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((8, 6)).astype(self.dtype)
        self.B = np.random.random((6, 3)).astype(self.dtype)
        self.C = np.random.random((3, 4)).astype(self.dtype)
        self.D = np.random.random((4, 5)).astype(self.dtype)
        self.inputs = {
            'X':
            [('x0', self.A), ('x1', self.B), ('x2', self.C), ('x3', self.D)]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x3'], 'Out', max_relative_error=1e-3)


class TestMultiDotOpFirst1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((4)).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}


class TestMultiDotOp3MatFirst1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((4)).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.C = np.random.random((3, 3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)


class TestMultiDotOp4MatFirst1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((4)).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.C = np.random.random((3, 4)).astype(self.dtype)
        self.D = np.random.random((4, 5)).astype(self.dtype)
        self.inputs = {
            'X':
            [('x0', self.A), ('x1', self.B), ('x2', self.C), ('x3', self.D)]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x3'], 'Out', max_relative_error=1e-3)


class TestMultiDotOpLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((3, 6)).astype(self.dtype)
        self.B = np.random.random((6)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}


class TestMultiDotOp3MatLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 4)).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.C = np.random.random((3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)


class TestMultiDotOp4MatLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 3)).astype(self.dtype)
        self.B = np.random.random((3, 2)).astype(self.dtype)
        self.C = np.random.random((2, 3)).astype(self.dtype)
        self.D = np.random.random((3)).astype(self.dtype)
        self.inputs = {
            'X':
            [('x0', self.A), ('x1', self.B), ('x2', self.C), ('x3', self.D)]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x3'], 'Out', max_relative_error=1e-3)


class TestMultiDotOpFirstAndLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((4, )).astype(self.dtype)
        self.B = np.random.random((4)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)


class TestMultiDotOp3MatFirstAndLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((6, )).astype(self.dtype)
        self.B = np.random.random((6, 4)).astype(self.dtype)
        self.C = np.random.random((4)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)


class TestMultiDotOp4MatFirstAndLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((3, )).astype(self.dtype)
        self.B = np.random.random((3, 4)).astype(self.dtype)
        self.C = np.random.random((4, 2)).astype(self.dtype)
        self.D = np.random.random((2)).astype(self.dtype)
        self.inputs = {
            'X':
            [('x0', self.A), ('x1', self.B), ('x2', self.C), ('x3', self.D)]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x1'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x2'], 'Out', max_relative_error=1e-3)
        self.check_grad(['x3'], 'Out', max_relative_error=1e-3)


#####python API test#######
class TestMultiDotOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The inputs type of multi_dot must be list matrix.
            input1 = 12
            self.assertRaises(TypeError, paddle.multi_dot, [input1, input1])

            # The inputs dtype of multi_dot must be float32, float64 or float16.
            input2 = fluid.layers.data(
                name='input2', shape=[10, 10], dtype="int32")
            self.assertRaises(TypeError, paddle.multi_dot, [input2, input2])

            # the number of tensor must be larger than 1
            x0 = fluid.data(name='x0', shape=[3, 2], dtype="float32")
            self.assertRaises(ValueError, paddle.multi_dot, [x0])

            #the first tensor must be 1D or 2D
            x1 = fluid.data(name='x1', shape=[3, 2, 3], dtype="float32")
            x2 = fluid.data(name='x2', shape=[3, 2], dtype="float32")
            self.assertRaises(ValueError, paddle.multi_dot, [x1, x2])

            #the last tensor must be 1D or 2D
            x3 = fluid.data(name='x3', shape=[3, 2], dtype="float32")
            x4 = fluid.data(name='x4', shape=[3, 2, 2], dtype="float32")
            self.assertRaises(ValueError, paddle.multi_dot, [x3, x4])

            #the tensor must be 2D, except first and last tensor
            x5 = fluid.data(name='x5', shape=[3, 2], dtype="float32")
            x6 = fluid.data(name='x6', shape=[2], dtype="float32")
            x7 = fluid.data(name='x7', shape=[2, 2], dtype="float32")
            self.assertRaises(ValueError, paddle.multi_dot, [x5, x6, x7])


class API_TestMultiDot(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            x0 = fluid.data(name='x0', shape=[3, 2], dtype="float32")
            x1 = fluid.data(name='x1', shape=[2, 3], dtype='float32')
            result = paddle.multi_dot([x0, x1])
            exe = fluid.Executor(fluid.CPUPlace())
            data1 = np.random.rand(3, 2).astype("float32")
            data2 = np.random.rand(2, 3).astype("float32")
            np_res = exe.run(feed={'x0': data1,
                                   'x1': data2},
                             fetch_list=[result])
            expected_result = np.linalg.multi_dot([data1, data2])

        self.assertTrue(
            np.allclose(
                np_res, expected_result, atol=1e-5),
            "two value is\
            {}\n{}, check diff!".format(np_res, expected_result))

    def test_dygraph_without_out(self):
        device = fluid.CPUPlace()
        with fluid.dygraph.guard(device):
            input_array1 = np.random.rand(3, 4).astype("float64")
            input_array2 = np.random.rand(4, 3).astype("float64")
            data1 = fluid.dygraph.to_variable(input_array1)
            data2 = fluid.dygraph.to_variable(input_array2)
            out = paddle.multi_dot([data1, data2])
            expected_result = np.linalg.multi_dot([input_array1, input_array2])
        self.assertTrue(np.allclose(expected_result, out.numpy()))


if __name__ == "__main__":
    unittest.main()
