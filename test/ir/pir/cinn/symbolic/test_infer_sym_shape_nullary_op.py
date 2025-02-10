# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from os.path import dirname

import numpy as np
from test_infer_sym_shape_utils import (
    TestBase,
    check_infer_results,
)

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
from utils import apply_to_static


class ArangeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, in_0, in_1, in_2):
        if in_1 is None:
            end = in_0
            out = paddle.arange(end)
        else:
            start, end, step = in_0, in_1, in_2
            out = paddle.arange(start, end, step)

        return out


class ArangeOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.start = paddle.full([1], 0, dtype='int32')
        self.end = paddle.full([1], 5, dtype='int32')
        self.step = paddle.full([1], 1, dtype='int32')
        self.expected = ['shape[Div(Add(S1, -S0), S2)], data[NULL]']

    def test_eval_symbolic(self):
        net = ArangeNet()
        input_spec = [
            InputSpec(shape=[1], dtype='int32'),
            InputSpec(shape=[1], dtype='int32'),
            InputSpec(shape=[1], dtype='int32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'builtin.shadow_output', self.expected
        )
        out = net(self.start, self.end, self.step)
        return out


class AssignNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty(shape=[3, 3])
        array = np.array([[1, 1], [3, 4], [1, 3]]).astype(np.int64)
        out = paddle.assign(array, data)
        return out


class AssignOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[3, 2], data[NULL]']

    def test_eval_symbolic(self):
        net = AssignNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'pd_op.assign_value_', self.expected
        )
        return True


class EmptyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.empty(shape=[128, 32])
        out = paddle.empty(shape=x.shape)
        return out


class EmptyOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = [
            'shape[128, 32], data[NULL]',
            'shape[S0, S1, S2], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = EmptyNet()

        x_spec = InputSpec(shape=[None, None, None], dtype='int32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.empty', self.expected)
        return True


class TriuIndicesNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.triu_indices(row=10, col=10, offset=0)
        out = paddle.triu_indices(row=10, col=10, offset=2)
        out = paddle.triu_indices(row=10, col=10, offset=-2)
        out = paddle.triu_indices(row=10, col=3, offset=0)
        out = paddle.triu_indices(row=10, col=3, offset=2)
        out = paddle.triu_indices(row=10, col=3, offset=-2)
        out = paddle.triu_indices(row=3, col=10, offset=0)
        out = paddle.triu_indices(row=3, col=10, offset=2)
        out = paddle.triu_indices(row=3, col=10, offset=-2)
        return out


class TriuIndicesOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = [
            'shape[2, 55], data[NULL]',
            'shape[2, 36], data[NULL]',
            'shape[2, 72], data[NULL]',
            'shape[2, 6], data[NULL]',
            'shape[2, 1], data[NULL]',
            'shape[2, 12], data[NULL]',
            'shape[2, 27], data[NULL]',
            'shape[2, 21], data[NULL]',
            'shape[2, 30], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = TriuIndicesNet()
        x_spec = InputSpec(shape=[None, None, None], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'pd_op.triu_indices', self.expected
        )
        return True


class TrilIndicesNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.tril_indices(row=10, col=10, offset=0)
        out = paddle.tril_indices(row=10, col=10, offset=2)
        out = paddle.tril_indices(row=10, col=10, offset=-2)
        out = paddle.tril_indices(row=10, col=3, offset=0)
        out = paddle.tril_indices(row=10, col=3, offset=2)
        out = paddle.tril_indices(row=10, col=3, offset=-2)
        out = paddle.tril_indices(row=3, col=10, offset=0)
        out = paddle.tril_indices(row=3, col=10, offset=2)
        out = paddle.tril_indices(row=3, col=10, offset=-2)
        return out


class TrilIndicesOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = [
            'shape[2, 55], data[NULL]',
            'shape[2, 72], data[NULL]',
            'shape[2, 36], data[NULL]',
            'shape[2, 27], data[NULL]',
            'shape[2, 30], data[NULL]',
            'shape[2, 21], data[NULL]',
            'shape[2, 6], data[NULL]',
            'shape[2, 12], data[NULL]',
            'shape[2, 1], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = TrilIndicesNet()
        x_spec = InputSpec(shape=[None, None, None], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'pd_op.tril_indices', self.expected
        )
        return True


class GaussianNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.tensor.random.gaussian(shape=[12, 32], mean=1.0, std=2.0)
        return out


class GaussianOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[12, 32], data[NULL]']

    def test_eval_symbolic(self):
        net = GaussianNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.gaussian', self.expected)
        return True


class RandintNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.randint(low=-5, high=5, shape=[12, 32])
        return out


class RandintOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[12, 32], data[NULL]']

    def test_eval_symbolic(self):
        net = RandintNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.randint', self.expected)
        return True


class UniformNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.tensor.random.uniform(shape=[12, 32], min=1.0, max=2.0)
        return out


class UniformOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[12, 32], data[NULL]']

    def test_eval_symbolic(self):
        net = UniformNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.uniform', self.expected)
        return True


if __name__ == '__main__':
    unittest.main()
