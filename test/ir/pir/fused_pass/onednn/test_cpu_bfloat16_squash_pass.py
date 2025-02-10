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


class TestConv2dAddBf16Pass(PassTest):
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
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out = conv2d(x)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "onednn_op.conv2d": 0,
                    "pd_op.conv2d": 0,
                    "onednn_op.dequantize": 0,
                    "onednn_op.quantize": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.skip_accuracy_verification = True
        self.check_pass_correct(atol=1e-2, rtol=1e-2)


class TestFusedConv2dBf16Pass(PassTest):
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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out_conv = conv2d(x)
                out = paddle.add(out_conv, bias)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv2d_bias_fuse_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.conv2d": 0,
                    "onednn_op.dequantize": 0,
                    "onednn_op.quantize": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.skip_accuracy_verification = True
        self.check_pass_correct(atol=1e-2, rtol=1e-2)


class TestFcGeluBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                w = paddle.static.data(name='w', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name="y", shape=[3], dtype='float32')
                out = paddle.add(paddle.matmul(x, w), y)
                out = paddle.nn.functional.gelu(out)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'matmul_add_act_fuse_pass': {}},
                    {'fc_onednn_enable_pass': {}},
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((3, 2)).astype("float32"),
                    "w": np.random.random((2, 3)).astype("float32"),
                    "y": np.random.random(3).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fc": 1,
                    "pd_op.matmul": 0,
                    "onednn_op.gelu": 1,
                    "onednn_op.dequantize": 1,
                    "onednn_op.quantize": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct(atol=5e-3, rtol=5e-3)


class TestFcGeluReluBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                w = paddle.static.data(name='w', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name="y", shape=[3], dtype='float32')
                out = paddle.add(paddle.matmul(x, w), y)
                out_1 = paddle.nn.functional.gelu(out)
                out_2 = paddle.nn.functional.relu(out)
                out_1 = paddle.assign(out_1)
                out_2 = paddle.assign(out_2)
                self.pass_attr_list = [
                    {'matmul_add_act_fuse_pass': {}},
                    {'fc_onednn_enable_pass': {}},
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((3, 2)).astype("float32"),
                    "w": np.random.random((2, 3)).astype("float32"),
                    "y": np.random.random(3).astype("float32"),
                }
                self.fetch_list = [out_1, out_2]
                self.valid_op_map = {
                    "onednn_op.fc": 1,
                    "pd_op.matmul": 0,
                    "onednn_op.gelu": 1,
                    "onednn_op.relu": 1,
                    "onednn_op.dequantize": 2,
                    "onednn_op.quantize": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct(atol=5e-3, rtol=5e-3)


class TestGeluReluBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                out_1 = paddle.nn.functional.gelu(x)
                out_2 = paddle.nn.functional.relu(out_1)
                out_2 = paddle.assign(out_2)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((3, 2)).astype("float32"),
                }
                self.fetch_list = [out_2]
                self.valid_op_map = {
                    "onednn_op.gelu": 1,
                    "onednn_op.relu": 1,
                    "onednn_op.dequantize": 1,
                    "onednn_op.quantize": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct(atol=5e-3, rtol=5e-3)


class TestFcGeluMishBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                w = paddle.static.data(name='w', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name="y", shape=[3], dtype='float32')
                out = paddle.add(paddle.matmul(x, w), y)
                out_1 = paddle.nn.functional.gelu(out)
                out_2 = paddle.nn.functional.mish(out)
                out_1 = paddle.assign(out_1)
                out_2 = paddle.assign(out_2)
                self.pass_attr_list = [
                    {'matmul_add_act_fuse_pass': {}},
                    {'fc_onednn_enable_pass': {}},
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((3, 2)).astype("float32"),
                    "w": np.random.random((2, 3)).astype("float32"),
                    "y": np.random.random(3).astype("float32"),
                }
                self.fetch_list = [out_1, out_2]
                self.valid_op_map = {
                    "onednn_op.fc": 1,
                    "pd_op.matmul": 0,
                    "onednn_op.gelu": 1,
                    "onednn_op.mish": 1,
                    "onednn_op.dequantize": 2,
                    "onednn_op.quantize": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct(atol=5e-3, rtol=5e-3)


class TestFcDqBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                w = paddle.static.data(name='w', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name="y", shape=[3], dtype='float32')
                out = paddle.add(paddle.matmul(x, w), y)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'matmul_add_act_fuse_pass': {}},
                    {'fc_onednn_enable_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((3, 2)).astype("float32"),
                    "w": np.random.random((2, 3)).astype("float32"),
                    "y": np.random.random(3).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fc": 1,
                    "pd_op.matmul": 0,
                    "onednn_op.dequantize": 0,
                    "onednn_op.quantize": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct(atol=5e-3, rtol=5e-3)


class TestCastBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x1 = paddle.static.data(
                    name='x1', shape=[1, 30], dtype='float32'
                )

                out = paddle.cast(x1, 'float32')
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_special_ops_bf16_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x1": np.random.random((1, 30)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.cast": 1,
                    "onednn_op.dequantize": 0,
                    "onednn_op.quantize": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestConvCastBf16Pass(PassTest):
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
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )
                out = conv2d(x)
                out = paddle.cast(out, 'float32')
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                    {'cpu_special_ops_bf16_pass': {}},
                    {'cpu_bf16_quantize_squash_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "onednn_op.cast": 1,
                    "onednn_op.quantize": 0,
                    "onednn_op.dequantize": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
