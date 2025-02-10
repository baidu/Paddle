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
import unittest

import numpy as np
from op_test import OpTest


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


class TestSeqProject(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = 'sequence_conv'

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
            0.1, 1, [self.input_size[0], self.input_size[1]]
        ).astype('float32')
        w = np.random.uniform(
            0.1,
            1,
            [
                self.context_length * self.input_size[1],
                self.output_representation,
            ],
        ).astype('float32')

        begin_pad = np.max([0, -self.context_start])
        end_pad = np.max([0, self.context_start + self.context_length - 1])
        total_pad = begin_pad + end_pad
        padding_data = np.random.uniform(
            0.1, 1, [total_pad, self.input_size[1]]
        ).astype('float32')
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
        # NODE(yjjiang11): This op will be deprecated.
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if self.padding_trainable:
            self.check_grad(
                set(self.inputs_val),
                'Out',
                max_relative_error=0.05,
                check_dygraph=False,
            )

    def test_check_grad_input(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(self.inputs_val_no_x),
            check_dygraph=False,
        )

    def test_check_grad_padding_data(self):
        if self.padding_trainable:
            self.check_grad(
                ['PaddingData'],
                'Out',
                no_grad_set={'X', 'Filter'},
                check_dygraph=False,
            )

    def test_check_grad_Filter(self):
        self.check_grad(
            ['Filter'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set(self.inputs_val_no_f),
            check_dygraph=False,
        )

    def test_check_grad_input_filter(self):
        if self.padding_trainable:
            self.check_grad(
                ['X', 'Filter'],
                'Out',
                max_relative_error=0.05,
                no_grad_set={'PaddingData'},
                check_dygraph=False,
            )

    def test_check_grad_padding_input(self):
        if self.padding_trainable:
            self.check_grad(
                self.inputs_val_no_f,
                'Out',
                max_relative_error=0.05,
                no_grad_set={'Filter'},
                check_dygraph=False,
            )

    def test_check_grad_padding_filter(self):
        if self.padding_trainable:
            self.check_grad(
                self.inputs_val_no_x,
                'Out',
                max_relative_error=0.05,
                no_grad_set={'X'},
                check_dygraph=False,
            )

    def init_test_case(self):
        self.input_row = 11
        self.context_start = 0
        self.context_length = 1
        self.padding_trainable = False
        self.context_stride = 1

        self.input_size = [self.input_row, 23]
        offset_lod = [[0, 4, 5, 8, self.input_row]]
        self.lod = [[]]
        # convert from offset-based lod to length-based lod
        for i in range(len(offset_lod[0]) - 1):
            self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
        self.output_representation = 8  # output feature size


class TestSeqProjectCase1(TestSeqProject):
    def init_test_case(self):
        self.input_row = 11
        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = True
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
        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = True
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
        self.context_start = 2
        self.context_length = 3
        self.padding_trainable = True
        self.context_stride = 1

        self.input_size = [self.input_row, 25]
        idx = list(range(self.input_size[0]))
        del idx[0]
        offset_lod = [
            [0, *np.sort(random.sample(idx, 8)).tolist(), self.input_size[0]]
        ]
        self.lod = [[]]
        # convert from offset-based lod to length-based lod
        for i in range(len(offset_lod[0]) - 1):
            self.lod[0].append(offset_lod[0][i + 1] - offset_lod[0][i])
        self.output_representation = 8  # output feature size


if __name__ == '__main__':
    unittest.main()
