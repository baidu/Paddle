#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
from op_mapper_test import OpMapperTest, logger

import paddle


class TestBitwiseOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32", 0, 100000),
            'y': self.random([32, 64], "int32", 0, 100000),
        }

    def set_op_type(self):
        return "bitwise_and"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        y = paddle.static.data(
            name='y',
            shape=self.feed_data['y'].shape,
            dtype=self.feed_data['y'].dtype,
        )
        return {'X': [x], 'Y': [y]}

    def set_op_attrs(self):
        return {}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestBitwiseAndOp(TestBitwiseOp):
    def set_op_type(self):
        return "bitwise_and"


class TestBitwiseOrOp(TestBitwiseOp):
    def set_op_type(self):
        return "bitwise_or"


class TestBitwiseXOrOp(TestBitwiseOp):
    def set_op_type(self):
        return "bitwise_xor"


class TestBitwiseNotOp(TestBitwiseOp):
    def set_op_type(self):
        return "bitwise_not"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
