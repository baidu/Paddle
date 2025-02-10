# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestXpuMatmulV2WeightTransPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        # cpu
        config = self.create_inference_config(use_xpu=True)
        yield config, [
            "matmul_v2",
        ], (3e-3, 1e-3)

    def sample_program_config(self, draw):
        # 1. Generate shape and attr of matmul
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8), min_size=3, max_size=3
            )
        )
        transpose_shape = x_shape
        transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["transpose_input"]},
            outputs={"Out": ["transpose_out"]},
            axis=[0, 2, 1],
        )
        matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["matmul_x"], "Y": ["transpose_out"]},
            outputs={"Out": ["matmul_out"]},
            transpose_X=False,
            transpose_Y=False,
        )
        ops = [transpose_op, matmul_op]
        weights = {}
        inputs = {
            "matmul_x": TensorConfig(shape=x_shape),
            "transpose_input": TensorConfig(shape=transpose_shape),
        }
        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            min_success_num=5,
            passes=["matmul_weight_trans_pass"],
        )


if __name__ == "__main__":
    unittest.main()
