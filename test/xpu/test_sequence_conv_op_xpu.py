#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import random
import sys
import unittest

import numpy as np

import paddle

sys.path.append("../")
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

paddle.enable_static()
np.set_printoptions(threshold=np.inf)


def seqconv(
    x,
    lod,
    filter,
    context_length,
    context_start,
    padding_trainable=False,
    padding_data=None,
):
    [T, M] = x.shape
    col = np.zeros((T, context_length * M)).astype('float32')
    offset = [0]
    for seq_len in lod[0]:
        offset.append(offset[-1] + seq_len)
    begin_pad = np.max([0, -context_start])
    for i in range(len(offset) - 1):
        for j in range(context_length):
            in_begin = offset[i] + context_start + j
            in_end = offset[i + 1] + context_start + j
            out_begin = offset[i]
            out_end = offset[i + 1]
            if in_begin < offset[i]:
                pad_size = np.min(
                    [offset[i] - in_begin, offset[i + 1] - offset[i]]
                )
                if padding_trainable:
                    sub_w = padding_data[j : j + pad_size, :]
                    col[
                        offset[i] : offset[i] + pad_size, j * M : (j + 1) * M
                    ] = sub_w
                out_begin = offset[i] + pad_size
                in_begin = offset[i]

            if in_end > offset[i + 1]:
                pad_size = np.min(
                    [in_end - offset[i + 1], offset[i + 1] - offset[i]]
                )
                if padding_trainable:
                    sub_w = padding_data[
                        begin_pad
                        + context_start
                        + j
                        - pad_size : begin_pad
                        + context_start
                        + j,
                        :,
                    ]
                    col[
                        offset[i + 1] - pad_size : offset[i + 1],
                        j * M : (j + 1) * M,
                    ] = sub_w
                in_end = offset[i + 1]
                out_end = offset[i + 1] - pad_size
            if in_end <= in_begin:
                continue
            in_sub = x[in_begin:in_end, :]
            col[out_begin:out_end, j * M : (j + 1) * M] += in_sub
    return np.dot(col, filter)


class XPUTestSequenceConv(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sequence_conv'

    class TestSeqProject(XPUOpTest):
        def setUp(self):
            self.init_test_case()
            self.op_type = 'sequence_conv'
            self.dtype = self.in_type
            self.use_xpu = True

            if (
                self.context_length == 1
                and self.context_start == 0
                and self.padding_trainable
            ):
                print(
                    "If context_start is 0 "
                    "and context_length is 1,"
                    " padding_trainable should be false."
                )
                return

            # one level, batch size
            x = np.random.uniform(
                -6.10907e-05,
                0.000104218,
                [self.input_size[0], self.input_size[1]],
            ).astype(self.dtype)
            w = np.random.uniform(
                -3.17068e-05,
                0.000159822,
                [
                    self.context_length * self.input_size[1],
                    self.output_representation,
                ],
            ).astype(self.dtype)

            begin_pad = np.max([0, -self.context_start])
            end_pad = np.max([0, self.context_start + self.context_length - 1])
            total_pad = begin_pad + end_pad
            padding_data = np.random.uniform(
                0, 0, [total_pad, self.input_size[1]]
            ).astype(self.dtype)
            self.pad_data = padding_data
            self.inputs = {
                'X': (x, self.lod),
                'Filter': w,
            }
            self.inputs_val = ['X', 'Filter']
            self.inputs_val_no_x = ['Filter']
            self.inputs_val_no_f = ['X']

            if total_pad != 0:
                self.inputs['PaddingData'] = padding_data
                self.inputs_val = ['X', 'PaddingData', 'Filter']
                self.inputs_val_no_x = ['PaddingData', 'Filter']
                self.inputs_val_no_f = ['PaddingData', 'X']

            self.attrs = {
                'contextStart': self.context_start,
                'contextLength': self.context_length,
                'paddingTrainable': self.padding_trainable,
                'contextStride': self.context_stride,
            }
            out = seqconv(
                x,
                self.lod,
                w,
                self.context_length,
                self.context_start,
                self.padding_trainable,
                self.pad_data,
            )
            self.outputs = {'Out': out}

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad_input(self):
            self.check_grad(['X'], 'Out', no_grad_set=set(self.inputs_val_no_x))

        def test_check_grad_padding_data(self):
            if self.padding_trainable:
                self.check_grad(
                    ['PaddingData'], 'Out', no_grad_set={'X', 'Filter'}
                )

        def test_check_grad_Filter(self):
            self.check_grad(
                ['Filter'], 'Out', no_grad_set=set(self.inputs_val_no_f)
            )

        def test_check_grad_input_filter(self):
            if self.padding_trainable:
                self.check_grad(
                    ['X', 'Filter'], 'Out', no_grad_set={'PaddingData'}
                )

        def test_check_grad_padding_input(self):
            if self.padding_trainable:
                self.check_grad(
                    self.inputs_val_no_f, 'Out', no_grad_set={'Filter'}
                )

        def test_check_grad_padding_filter(self):
            if self.padding_trainable:
                self.check_grad(self.inputs_val_no_x, 'Out', no_grad_set={'X'})

        def init_test_case(self):
            self.input_row = 7
            self.input_col = 25
            self.context_start = -2
            self.context_length = 5
            self.padding_trainable = False
            self.context_stride = 1

            self.input_size = [self.input_row, self.input_col]
            offset_lod = [[0, 1, self.input_row]]
            self.lod = [[]]
            # convert from offset-based lod to length-based lod
            for i in range(len(offset_lod[0]) - 1):
                self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
            self.output_representation = 8  # output feature size

    class TestSeqProjectCase1(TestSeqProject):
        def init_test_case(self):
            self.input_row = 11
            self.context_start = -2
            self.context_length = 5
            self.padding_trainable = False
            self.context_stride = 1

            self.input_size = [self.input_row, 50]
            offset_lod = [[0, 4, 5, 8, self.input_row]]
            self.lod = [[]]
            # convert from offset-based lod to length-based lod
            for i in range(len(offset_lod[0]) - 1):
                self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
            self.output_representation = 8  # output feature size

    class TestSeqProjectCase2Len0(TestSeqProject):
        def init_test_case(self):
            self.input_row = 11
            self.context_start = -2
            self.context_length = 5
            self.padding_trainable = False
            self.context_stride = 1

            self.input_size = [self.input_row, 50]
            offset_lod = [[0, 0, 4, 5, 5, 8, self.input_row, self.input_row]]
            self.lod = [[]]
            # convert from offset-based lod to length-based lod
            for i in range(len(offset_lod[0]) - 1):
                self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
            self.output_representation = 8  # output feature size

    class TestSeqProjectCase3(TestSeqProject):
        def init_test_case(self):
            self.input_row = 25
            self.context_start = -2
            self.context_length = 5
            self.padding_trainable = False
            self.context_stride = 1

            self.input_size = [self.input_row, 25]
            idx = list(range(self.input_size[0]))
            del idx[0]
            offset_lod = [
                [
                    0,
                    *np.sort(random.sample(idx, 8)).tolist(),
                    self.input_size[0],
                ]
            ]
            self.lod = [[]]
            # convert from offset-based lod to length-based lod
            for i in range(len(offset_lod[0]) - 1):
                self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
            self.output_representation = 8  # output feature size

    class TestSeqProjectCase4(TestSeqProject):
        def init_test_case(self):
            self.input_row = 7835
            self.input_col = 128
            self.context_start = -2
            self.context_length = 5
            self.padding_trainable = False
            self.context_stride = 1

            self.input_size = [self.input_row, self.input_col]
            offset_lod = [
                [
                    0,
                    1,
                    2,
                    3,
                    131,
                    241,
                    242,
                    263,
                    264,
                    265,
                    266,
                    267,
                    268,
                    387,
                    515,
                    516,
                    644,
                    645,
                    772,
                    794,
                    922,
                    923,
                    924,
                    944,
                    945,
                    1073,
                    1074,
                    1202,
                    1330,
                    1458,
                    1556,
                    1557,
                    1558,
                    1686,
                    1748,
                    1876,
                    1912,
                    1913,
                    1914,
                    2032,
                    2066,
                    2194,
                    2308,
                    2309,
                    2347,
                    2475,
                    2476,
                    2477,
                    2478,
                    2606,
                    2607,
                    2735,
                    2736,
                    2737,
                    2738,
                    2838,
                    2966,
                    2967,
                    2968,
                    2969,
                    3097,
                    3225,
                    3353,
                    3481,
                    3482,
                    3520,
                    3642,
                    3643,
                    3754,
                    3882,
                    3883,
                    4010,
                    4011,
                    4012,
                    4140,
                    4219,
                    4228,
                    4356,
                    4357,
                    4415,
                    4475,
                    4476,
                    4604,
                    4605,
                    4606,
                    4694,
                    4695,
                    4808,
                    4936,
                    4961,
                    4962,
                    5004,
                    5132,
                    5260,
                    5312,
                    5440,
                    5441,
                    5569,
                    5570,
                    5675,
                    5676,
                    5750,
                    5810,
                    5811,
                    5939,
                    6021,
                    6149,
                    6277,
                    6278,
                    6364,
                    6425,
                    6519,
                    6647,
                    6648,
                    6739,
                    6867,
                    6995,
                    6996,
                    7120,
                    7223,
                    7244,
                    7367,
                    7407,
                    7408,
                    7467,
                    7595,
                    7699,
                    7827,
                    7835,
                ]
            ]
            self.lod = [[]]
            # convert from offset-based lod to length-based lod
            for i in range(len(offset_lod[0]) - 1):
                self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
            self.output_representation = 8  # output feature size


support_types = get_xpu_op_support_types('sequence_conv')
for stype in support_types:
    create_test_class(globals(), XPUTestSequenceConv, stype)


class TestSeqConvApi(unittest.TestCase):
    def test_api(self):
        with paddle.pir_utils.OldIrGuard():
            from paddle import base

            x = paddle.static.data('x', shape=[-1, 32], lod_level=1)
            y = paddle.static.nn.sequence_lod.sequence_conv(
                input=x, num_filters=2, filter_size=3, padding_start=None
            )
            place = base.CPUPlace()
            x_tensor = base.create_lod_tensor(
                np.random.rand(10, 32).astype("float32"), [[2, 3, 1, 4]], place
            )
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            ret = exe.run(
                feed={'x': x_tensor}, fetch_list=[y], return_numpy=False
            )


if __name__ == '__main__':
    unittest.main()
