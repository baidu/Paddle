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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestClipTensorOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.op_type = "clip_tensor"
        self.python_api = paddle.clip

        self.inputs = {}
        self.initTestCase()
        input = np.random.random(self.shape).astype(self.dtype)
        min_v = np.full(self.shape, self.min_value).astype(self.dtype)
        max_v = np.full(self.shape, self.max_value).astype(self.dtype)

        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5

        self.inputs['min'] = min_v
        self.inputs['max'] = max_v
        self.inputs['x'] = input
        self.outputs = {'out': np.clip(input, min_v, max_v)}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['x'], 'out', check_pir=True)

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 5, 6)
        self.min_value = 0.8
        self.max_value = 0.3


class TestCase1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (5, 6, 8)
        self.max_value = 0.7
        self.min_value = 0.0


class TestCase2(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 5, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestCase3(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 6)
        self.max_value = 0.7
        self.min_value = 0.2


class TestFP16Case1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (5, 6, 8)
        self.max_value = 0.7
        self.min_value = 0.0


class TestFP16Case2(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (8, 5, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestFP16Case3(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float16
        self.shape = (5, 8, 6)
        self.max_value = 0.7
        self.min_value = 0.2


class TestFP64Case1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (8, 6, 5)
        self.max_value = 0.7
        self.min_value = 0.0


class TestFP64Case2(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (8, 5, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestFP64Case3(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (4, 8, 6)
        self.max_value = 0.7
        self.min_value = 0.2


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestClipTensorBF16Op(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.op_type = "clip_tensor"
        self.python_api = paddle.clip
        self.inputs = {}
        self.initTestCase()

        self.inputs['x'] = np.random.random(self.shape).astype(np.float32)
        self.inputs['min'] = np.full(self.shape, self.min_value).astype(
            np.float32
        )
        self.inputs['max'] = np.full(self.shape, self.max_value).astype(
            np.float32
        )
        min_v = self.inputs['min']
        max_v = self.inputs['max']

        self.inputs['x'][
            np.abs(self.inputs['x'] - min_v) < self.max_relative_error
        ] = 0.5
        self.inputs['x'][
            np.abs(self.inputs['x'] - max_v) < self.max_relative_error
        ] = 0.5

        self.inputs['x'] = convert_float_to_uint16(self.inputs['x'])
        self.inputs['min'] = convert_float_to_uint16(self.inputs['min'])
        self.inputs['max'] = convert_float_to_uint16(self.inputs['max'])
        out = np.clip(self.inputs['x'], min_v, max_v)

        self.outputs = {'out': convert_float_to_uint16(out)}

    def test_check_output(self):
        place = paddle.CUDAPlace(0)
        paddle.enable_static()
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad_normal(self):
        place = paddle.CUDAPlace(0)
        paddle.enable_static()
        self.check_grad_with_place(place, ['x'], 'out', check_pir=True)

    def initTestCase(self):
        self.shape = (8, 5, 6)
        self.min_value = 0.8
        self.max_value = 0.3


class TestBF16Case1(TestClipTensorBF16Op):
    def initTestCase(self):
        self.shape = (8, 6, 5)
        self.max_value = 0.7
        self.min_value = 0.0


class TestBF16Case2(TestClipTensorBF16Op):
    def initTestCase(self):
        self.shape = (5, 8, 6)
        self.max_value = 1.0
        self.min_value = 0.0


class TestBF16Case3(TestClipTensorBF16Op):
    def initTestCase(self):
        self.shape = (4, 8, 7)
        self.max_value = 0.7
        self.min_value = 0.2


if __name__ == '__main__':
    unittest.main()
