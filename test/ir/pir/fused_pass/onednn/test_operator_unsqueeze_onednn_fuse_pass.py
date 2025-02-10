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


class TestTransposeUnsqueezeFusePass(PassTest):
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
                transpose = paddle.transpose(
                    x, [len(x.shape) - 1, *range(0, len(x.shape) - 1)]
                )
                out = paddle.unsqueeze(transpose, [1])
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'operator_unsqueeze_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.transpose": 0,
                    "pd_op.unsqueeze": 0,
                    "onednn_op.fused_transpose": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMulUnsqueezeFusePass(PassTest):
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
                matmul = paddle.multiply(x, y)
                out = paddle.unsqueeze(matmul, [1])
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'operator_unsqueeze_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.multiply": 0,
                    "pd_op.unsqueeze": 0,
                    "onednn_op.fused_elementwise_mul": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFusedTransposeUnsqueezeFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[4, 16, 1, 32], dtype='float32'
                )

                squeeze_out = paddle.squeeze(x, axis=[2])
                transpose = paddle.transpose(squeeze_out, [0, 1, 2])
                out = paddle.unsqueeze(transpose, [1])
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'squeeze_transpose_onednn_fuse_pass': {}},
                    {'operator_unsqueeze_onednn_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((4, 16, 1, 32)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.transpose": 0,
                    "pd_op.unsqueeze": 0,
                    "onednn_op.fused_transpose": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFusedMulUnsqueezeFusePass(PassTest):
    r"""
    x     w
     \   /
     matmul
        |
     [relu]
        |
    unsqueeze
    """

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
                act_op = paddle.nn.ReLU()
                matmul = paddle.multiply(x, y)
                relu_out = act_op(matmul)
                out = paddle.unsqueeze(relu_out, [1])
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'elementwise_act_onednn_fuse_pass': {}},
                    {'operator_unsqueeze_onednn_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.multiply": 0,
                    "pd_op.unsqueeze": 0,
                    "pd_op.relu": 0,
                    "onednn_op.fused_elementwise_mul": 1,
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
