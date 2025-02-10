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
from pass_test import PassTest

import paddle

paddle.enable_static()


class TestPlacementMatmulPass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                out = paddle.matmul(x, y)
                out = paddle.assign(out)
                self.pass_attr_list = [{'onednn_placement_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.matmul": 1,
                    "pd_op.matmul": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestPlacementReluPass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                add = paddle.matmul(x, y)
                out = paddle.nn.functional.leaky_relu(add)
                out = paddle.assign(out)
                self.pass_attr_list = [{'onednn_placement_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.matmul": 1,
                    "onednn_op.leaky_relu": 1,
                    "pd_op.leaky_relu": 0,
                    "pd_op.matmul": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestConv3dAddFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3, 32, 28, 28], dtype='float32'
                )
                bn = paddle.nn.layer.norm.BatchNorm(
                    num_channels=32,
                    data_layout='NCHW',
                    use_global_stats=True,
                    is_test=True,
                )
                bn_out = bn(x)
                out = paddle.nn.functional.relu(bn_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'onednn_placement_pass': {}}]
                self.feeds = {
                    "x": np.random.random((3, 32, 28, 28)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.batch_norm_": 1,
                    "onednn_op.relu": 1,
                    "pd_op.batch_norm": 0,
                    "pd_op.relu": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestPlacementSlicePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 5, 5, 5], dtype='float32'
                )
                out_1 = x[0, :, :, :]
                out_2 = x[0, :, :, :]
                out_1 = paddle.assign(out_1)
                out_2 = paddle.assign(out_2)
                self.pass_attr_list = [{'onednn_placement_pass': {}}]
                self.feeds = {
                    "x": np.random.random((2, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out_1, out_2]
                self.valid_op_map = {
                    "onednn_op.slice": 2,
                    "pd_op.slice": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestPlacementSlicePassCase2(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 5, 5, 5], dtype='float16'
                )
                out_1 = x[0, :, :, :]
                out_2 = x[1, :, :, :]
                out_1 = paddle.assign(out_1)
                out_2 = paddle.assign(out_2)
                self.pass_attr_list = [{'onednn_placement_pass': {}}]
                self.feeds = {
                    "x": np.random.random((2, 5, 5, 5)).astype("float16"),
                }
                self.fetch_list = [out_1, out_2]
                self.valid_op_map = {
                    "onednn_op.slice": 0,
                    "pd_op.slice": 2,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestPlacementCastFailedCase(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x1 = paddle.static.data(
                    name='x1', shape=[1, 30], dtype='float16'
                )

                out = paddle.cast(x1, 'float16')
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                ]
                self.feeds = {
                    "x1": np.random.random((1, 30)).astype("float16"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.cast": 0,
                    "pd_op.cast": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


# This case is for testing layout transformation
class TestConv2dAddPlacementPass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[1, 4, 4, 5], dtype='float32'
                )
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1, 1, 1, 3],
                    dtype='float32',
                    attr=bias_attr,
                    is_bias=False,
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=3,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NHWC',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out = paddle.add(conv2d(x), bias)

                out = paddle.assign(out)
                self.pass_attr_list = [{'onednn_placement_pass': {}}]
                self.feeds = {
                    "x": np.random.random((1, 4, 4, 5)).astype("float32"),
                    "bias": np.random.random((1, 1, 1, 3)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.conv2d": 0,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
