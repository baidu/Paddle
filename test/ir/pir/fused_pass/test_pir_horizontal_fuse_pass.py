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
import os

os.environ["FLAGS_enable_pir_api"] = "0"


import unittest

import numpy as np

import paddle
from paddle import nn


class MatmulHorizontalLayer(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, num_layers=32):
        super().__init__()
        self.layers = nn.LayerList(
            [
                nn.Linear(hidden_size, intermediate_size, bias_attr=False)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        results = []
        for layer in self.layers:
            result = layer(x)
            results.append(result)
        return results


class TestMatmulHorizontalFusePattern(unittest.TestCase):
    def setUp(self):
        self.bsz = 2
        self.seq_len = 16
        self.num_head = 2
        self.head_dim = 16
        self.hidden_size = self.num_head * self.head_dim
        self.intermediate_size = self.hidden_size

    def test_matmul_horizontal_fuse(self):
        if not paddle.is_compiled_with_cuda():
            return
        x = paddle.randn(shape=[self.bsz, self.seq_len, self.hidden_size])
        layer = MatmulHorizontalLayer(self.hidden_size, self.intermediate_size)
        baseline_results = layer(x)

        static_layer = paddle.incubate.jit.inference(
            layer,
            enable_new_ir=True,
            switch_ir_debug=True,
        )

        # check precision and shape
        static_results = static_layer(x)
        self.verify_results(baseline_results, static_results)

        # check fused pattern
        valid_op_map = {
            "pd_op.concat": 1,
            "pd_op.matmul": 1,
            "pd_op.split": 1,
        }
        self.verify_fuse_pattern("horizontal_fuse_pass", valid_op_map)

    def verify_fuse_pattern(self, pass_name, valid_op_map):
        str_txt = ""
        with open(pass_name, "r") as f:
            str_txt = f.read()

        for op_name, count in valid_op_map.items():
            assert str_txt.count(op_name) == count

    @staticmethod
    def verify_results(expected, actual, atol=1e-5, rtol=1e-5):
        assert len(expected) == len(
            actual
        ), f"Length mismatch: expected {len(expected)}, got {len(actual)}"
        for exp, act in zip(expected, actual):
            assert (
                exp.shape == act.shape
            ), f"Shape mismatch: expected {exp.shape}, got {act.shape}"
            np.testing.assert_allclose(
                exp.numpy(), act.numpy(), atol=atol, rtol=rtol
            )


if __name__ == "__main__":
    unittest.main()
