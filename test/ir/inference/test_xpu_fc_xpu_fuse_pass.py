# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestFcXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fc_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        # 1. matmul_v2
        # Generate shape of input:X of matmul_v2
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        # Generate attr trans_x, trans_y
        trans_x = False
        trans_y = draw(st.booleans())
        # Generate legal shape of input:Y of mul
        y_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8), min_size=2, max_size=2
            )
        )
        if trans_y:
            y_shape[1] = x_shape[-1]
        else:
            y_shape[0] = x_shape[-1]
        # 2. elementwise_add
        # Generate legal attr:axis of elementwise_add
        axis = -1
        # Generate legal shape of input:Y of elementwise_add
        bias_shape = [y_shape[0]] if trans_y else [y_shape[1]]
        # 3. activation
        # Random choose if add a relu operator
        has_relu = draw(st.booleans())

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        matmul_v2_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["matmul_v2_x"], "Y": ["matmul_v2_y"]},
            outputs={"Out": ["matmul_v2_out"]},
            trans_x=trans_x,
            trans_y=trans_y,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["matmul_v2_out"], "Y": ["bias"]},
            outputs={"Out": ["add_out"]},
            axis=axis,
        )
        ops = [matmul_v2_op, add_op]
        if has_relu:
            relu_op = OpConfig(
                "relu", inputs={"X": ["add_out"]}, outputs={"Out": ["relu_out"]}
            )
            ops.append(relu_op)
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "matmul_v2_y": TensorConfig(shape=y_shape),
                "bias": TensorConfig(shape=bias_shape),
            },
            inputs={
                "matmul_v2_x": TensorConfig(shape=x_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False, max_examples=25, passes=["fc_xpu_fuse_pass"]
        )


if __name__ == "__main__":
    unittest.main()
