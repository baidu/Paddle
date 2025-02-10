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
import paddle.nn.functional as F
from paddle.framework import core
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
from utils import apply_to_static

# NOTE(SigureMo): Disable the CSE optimization to avoid op number change.
paddle.set_flags({"FLAGS_enable_cse_in_dy2st": False})


class ArgMaxMinNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        argmax_out = paddle.argmax(x)
        argmin_out = paddle.argmin(x, axis=-1)
        return argmax_out, argmin_out


class ArgMaxMinOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            ['shape[], data[NULL]'],
            ['shape[S0, S1], data[NULL]'],
        ]

    def test_eval_symbolic(self):
        net = ArgMaxMinNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )
            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'pd_op.argmax', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.argmin', self.expected[1]
            )

        return True


class AsComplexAsRealNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        complex_res = paddle.as_complex(x)
        real_res = paddle.as_real(complex_res)
        return real_res, complex_res


class AsComplexAsRealOPInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            ['shape[S0, S1], data[NULL]'],
            ['shape[S0, S1, 2], data[NULL]'],
        ]

    def test_eval_symbolic(self):
        net = AsComplexAsRealNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(shape=[None, None, 2], dtype='float32')

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'pd_op.as_complex', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.as_real', self.expected[1]
            )

        return True


class CumSumProdNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cumsum_out = paddle.cumsum(x)
        cumsum_out = paddle.cumsum(x, axis=1)
        logcumsumexp_out = paddle.logcumsumexp(x)
        logcumsumexp_out = paddle.logcumsumexp(x, axis=1)
        cumprod_out = paddle.cumprod(x, dim=1)
        return cumsum_out, logcumsumexp_out, cumprod_out


class CumSumProdOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[Mul(S0, S1, S2)], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
            ],
            [
                'shape[S0, S1, S2], data[NULL]',
            ],
        ]

    def test_eval_symbolic(self):
        net = CumSumProdNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'pd_op.cumsum', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.logcumsumexp', self.expected[0]
            )
            check_infer_results(
                net, input_spec, 'pd_op.cumprod', self.expected[1]
            )

        return True


class SumNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out_sum = paddle.sum(x)
        out_sum = paddle.sum(x, 0)
        out_sum = paddle.sum(x, 1)
        out_sum = paddle.sum(x, -1)
        out_sum = paddle.sum(x, -2)
        # keepdim=True
        out_sum = paddle.sum(x, keepdim=True)
        out_sum = paddle.sum(x, 0, keepdim=True)
        out_sum = paddle.sum(x, 1, keepdim=True)
        out_sum = paddle.sum(x, -1, keepdim=True)
        out_sum = paddle.sum(x, -2, keepdim=True)

        out_sum = paddle.sum(x, [1, 2])
        out_sum = paddle.sum(x, [1, 2], keepdim=True)

        out_logsumexp = paddle.logsumexp(x)
        out_logsumexp = paddle.logsumexp(x, 0)
        out_logsumexp = paddle.logsumexp(x, 1)
        out_logsumexp = paddle.logsumexp(x, -1)
        out_logsumexp = paddle.logsumexp(x, -2)
        # keepdim=True
        out_logsumexp = paddle.logsumexp(x, keepdim=True)
        out_logsumexp = paddle.logsumexp(x, 0, keepdim=True)
        out_logsumexp = paddle.logsumexp(x, 1, keepdim=True)
        out_logsumexp = paddle.logsumexp(x, -1, keepdim=True)
        out_logsumexp = paddle.logsumexp(x, -2, keepdim=True)

        out_logsumexp = paddle.logsumexp(x, [1, 2])
        out_logsumexp = paddle.logsumexp(x, [1, 2], keepdim=True)
        return out_sum, out_logsumexp


class SumOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            'shape[], data[NULL]',
            'shape[S1, S2], data[NULL]',
            'shape[S0, S2], data[NULL]',
            'shape[S0, S1], data[NULL]',
            'shape[S0, S2], data[NULL]',
            # keepdim=True
            'shape[1, 1, 1], data[NULL]',
            'shape[1, S1, S2], data[NULL]',
            'shape[S0, 1, S2], data[NULL]',
            'shape[S0, S1, 1], data[NULL]',
            'shape[S0, 1, S2], data[NULL]',
            'shape[S0], data[NULL]',
            'shape[S0, 1, 1], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = SumNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            check_infer_results(net, input_spec, 'pd_op.sum', self.expected)
            check_infer_results(
                net, input_spec, 'pd_op.logsumexp', self.expected
            )

        return True


class DiagEmbedNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([6])
        out = paddle.diag_embed(data)
        out = paddle.diag_embed(data, offset=-1, dim1=0, dim2=1)
        out = paddle.diag_embed(x)
        out = paddle.diag_embed(x, offset=-1, dim1=0, dim2=1)
        return out


class DiagEmbedOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[6, 6], data[NULL]',
                'shape[7, 7], data[NULL]',
                'shape[S0, S1, S2, S2], data[NULL]',
                'shape[Add(S2, 1), Add(S2, 1), S0, S1], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = DiagEmbedNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            check_infer_results(
                net, input_spec, 'pd_op.diag_embed', self.expected[i]
            )

        return True


class DiagonalNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([2, 2, 3], 'float32')
        out = paddle.diagonal(data)
        out = paddle.diagonal(data, offset=0, axis1=2, axis2=1)
        out = paddle.diagonal(x)
        out = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
        out = paddle.diagonal(x, offset=1, axis1=2, axis2=1)
        out = paddle.diagonal(x, offset=-1, axis1=2, axis2=1)
        return out


class DiagonalOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[3, 2], data[NULL]',
                'shape[2, 2], data[NULL]',
                'shape[S2, Min(S0, S1)], data[NULL]',
                'shape[S0, Min(S2, S1)], data[NULL]',
                'shape[S0, S3], data[NULL]',
                'shape[S0, S4], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = DiagonalNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            check_infer_results(
                net, input_spec, 'pd_op.diagonal', self.expected[i]
            )

        return True


class KthvalueNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty([2, 3, 3], 'float32')
        out = paddle.kthvalue(data, 2, 1)
        return out


class KthvalueOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[2, 3], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = KthvalueNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'pd_op.kthvalue', self.expected[i]
            )

        return True


class MaxMinNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out_max = paddle.max(x)
        out_max = paddle.max(x, 0)
        out_max = paddle.max(x, 1)
        out_max = paddle.max(x, -1)
        out_max = paddle.max(x, -2)
        # keepdim=True
        out_max = paddle.max(x, keepdim=True)
        out_max = paddle.max(x, 0, keepdim=True)
        out_max = paddle.max(x, 1, keepdim=True)
        out_max = paddle.max(x, -1, keepdim=True)
        out_max = paddle.max(x, -2, keepdim=True)

        out_max = paddle.max(x, [1, 2])
        out_max = paddle.max(x, [1, 2], keepdim=True)

        out_min = paddle.min(x)
        out_min = paddle.min(x, 0)
        out_min = paddle.min(x, 1)
        out_min = paddle.min(x, -1)
        out_min = paddle.min(x, -2)
        # keepdim=True
        out_min = paddle.min(x, keepdim=True)
        out_min = paddle.min(x, 0, keepdim=True)
        out_min = paddle.min(x, 1, keepdim=True)
        out_min = paddle.min(x, -1, keepdim=True)
        out_min = paddle.min(x, -2, keepdim=True)

        out_min = paddle.min(x, [1, 2])
        out_min = paddle.min(x, [1, 2], keepdim=True)
        return out_max, out_min


class MaxMinOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(2, 4, 3)]

        self.expected = [
            'shape[], data[NULL]',
            'shape[S1, S2], data[NULL]',
            'shape[S0, S2], data[NULL]',
            'shape[S0, S1], data[NULL]',
            'shape[S0, S2], data[NULL]',
            # keepdim=True
            'shape[1, 1, 1], data[NULL]',
            'shape[1, S1, S2], data[NULL]',
            'shape[S0, 1, S2], data[NULL]',
            'shape[S0, S1, 1], data[NULL]',
            'shape[S0, 1, S2], data[NULL]',
            'shape[S0], data[NULL]',
            'shape[S0, 1, 1], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = MaxMinNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )
            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(net, input_spec, 'pd_op.max', self.expected)
            check_infer_results(net, input_spec, 'pd_op.min', self.expected)

        return True


class NonzeroNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out_nonzero = paddle.nonzero(x)
        return out_nonzero


class NonzeroOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        # pdb.set_trace()

        for _ in range(np.random.randint(1, 10)):
            self.cases[0][np.random.randint(0, 3)][np.random.randint(0, 4)][
                np.random.randint(0, 5)
            ] = 0

        self.expected = [
            'shape[S3, 3], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = NonzeroNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            check_infer_results(net, input_spec, 'pd_op.nonzero', self.expected)

        return True


class PutAlongAxisNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices, value):
        out = paddle.put_along_axis(x, indices, value, axis=0)
        out = paddle.put_along_axis(x, indices, value, axis=1)
        out = paddle.put_along_axis(x, indices, value, axis=-1)

        return out


class PutAlongAxisOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [
            [
                np.random.rand(2, 3, 4),
                np.ones([2, 3, 4], dtype='int32'),
                np.ones([2, 3, 4], dtype='float32'),
            ],
        ]

        self.expected = [
            [
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, S1, S2], data[NULL]',
            ],
        ]

    def test_eval_symbolic(self):
        net = PutAlongAxisNet()

        for i in range(len(self.cases)):
            x, indices, value = self.cases[i]
            x_spec = InputSpec(
                shape=[None for _ in range(len(x.shape))], dtype='float32'
            )
            indices_spec = InputSpec(
                shape=[None for _ in range(len(indices.shape))], dtype='int32'
            )
            value_spec = InputSpec(
                shape=[None for _ in range(len(value.shape))], dtype='float32'
            )

            input_spec = [x_spec, indices_spec, value_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'pd_op.put_along_axis', self.expected[i]
            )

        return True


class RepeatInterleaveNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.repeat_interleave(x, 2, axis=0)
        out = paddle.repeat_interleave(x, 2, axis=1)
        out = paddle.repeat_interleave(x, 2, axis=-1)
        out = paddle.repeat_interleave(x, 2, axis=-2)
        return out


class RepeatInterleaveOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = [
            'shape[Mul(S0, 2), S1, S2], data[NULL]',
            'shape[S0, Mul(S1, 2), S2], data[NULL]',
            'shape[S0, S1, Mul(S2, 2)], data[NULL]',
            'shape[S0, Mul(S1, 2), S2], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = RepeatInterleaveNet()
        x_spec = InputSpec(shape=[None, None, None], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'pd_op.repeat_interleave', self.expected
        )
        return True


class TopkNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.topk(x, 2)
        out = paddle.topk(x, 2, axis=1)
        out = paddle.topk(x, 2, axis=-1)
        out = paddle.topk(x, 2, axis=-2)
        return out


class TopkOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            'shape[S0, S1, 2], data[NULL]',
            'shape[S0, 2, S2], data[NULL]',
            'shape[S0, S1, 2], data[NULL]',
            'shape[S0, 2, S2], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = TopkNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(shape=[None, None, None], dtype='float32')
            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(net, input_spec, 'pd_op.topk', self.expected)


class SplitWithNumNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        data = paddle.empty(shape=[4, 6, 5])
        out0, out1, out2 = paddle.split(data, num_or_sections=3, axis=1)
        out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
        return out0, out1, out2


class SplitWithNumOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 6, 5)]
        self.expected = [
            "shape[4, 2, 5], data[NULL], shape[4, 2, 5], data[NULL], shape[4, 2, 5], data[NULL]",
            "shape[S0, Mul(S1, 1 / (3)), S2], data[NULL], shape[S0, Mul(S1, 1 / (3)), S2], data[NULL], shape[S0, Mul(S1, 1 / (3)), S2], data[NULL]",
        ]

    def test_eval_symbolic(self):
        net = SplitWithNumNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )
            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            # check the infer result
            check_infer_results(
                net, input_spec, 'pd_op.split_with_num', self.expected
            )

        return True


class PadNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out1 = F.pad(x, [1, 2, 3, 4, 5, 6])
        out2 = F.pad(x, [0, 1, 2, 2, 0, 0])
        return out1, out2


class PadOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                'shape[Add(S0, 3), Add(S1, 7), Add(S2, 11)], data[NULL]',
                'shape[Add(S0, 1), Add(S1, 4), S2], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = PadNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                # shape=[4, None, None], dtype='float32'
                # shape=[x.shape[index] for index in range(len(x.shape))], dtype='float32'
                shape=[None for index in range(len(x.shape))],
                dtype='float32',
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            check_infer_results(net, input_spec, 'pd_op.pad', self.expected[i])

        return True


class UnbindNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out1 = paddle.unbind(x)
        out2 = paddle.unbind(x, axis=0)
        out3 = paddle.unbind(x, axis=1)
        out4 = paddle.unbind(x, axis=-2)
        out5 = paddle.unbind(x, axis=-3)
        return out1, out2, out3, out4, out5


class UnbindOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                ', '.join(['shape[5, S0], data[NULL]'] * 4),
                ', '.join(['shape[5, S0], data[NULL]'] * 4),
                ', '.join(['shape[4, S0], data[NULL]'] * 4),
                ', '.join(['shape[4, S0], data[NULL]'] * 4),
                ', '.join(['shape[5, S0], data[NULL]'] * 4),
            ]
        ]

    def test_eval_symbolic(self):
        core._set_prim_forward_blacklist("pd_op.unbind")

        net = UnbindNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[x.shape[index] for index in range(len(x.shape) - 1)]
                + [None],
                dtype='float32',
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            check_infer_results(
                net, input_spec, 'pd_op.unbind', self.expected[i]
            )

        return True


class UniqueNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out1 = paddle.unique(x)
        out2 = paddle.unique(x, axis=0)
        out3 = paddle.unique(x, axis=-1)
        out4 = paddle.unique(x, axis=2)
        return out1, out2, out3, out4


class UniqueOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                # TODO: Now only the first output is tested because only the first is exported to `sym_shape_str`.
                'shape[S3], data[NULL]',
                'shape[S4, S1, S2], data[NULL]',
                'shape[S0, S1, S5], data[NULL]',
                'shape[S0, S1, S6], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = UniqueNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            check_infer_results(
                net, input_spec, 'pd_op.unique', self.expected[i]
            )

        return True


class UniqueConsecutiveNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out1 = paddle.unique_consecutive(x)
        out2 = paddle.unique_consecutive(x, axis=0)
        out3 = paddle.unique_consecutive(x, axis=-1)
        out4 = paddle.unique_consecutive(x, axis=2)
        return out1, out2, out3, out4


class UniqueConsecutiveOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6)]
        self.expected = [
            [
                # TODO: Now only the first output is tested because only the first is exported to `sym_shape_str`.
                'shape[S3], data[NULL]',
                'shape[S4, S1, S2], data[NULL]',
                'shape[S0, S1, S5], data[NULL]',
                'shape[S0, S1, S6], data[NULL]',
            ]
        ]

    def test_eval_symbolic(self):
        net = UniqueConsecutiveNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )

            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()

            check_infer_results(
                net, input_spec, 'pd_op.unique_consecutive', self.expected[i]
            )

        return True


if __name__ == '__main__':
    unittest.main()
