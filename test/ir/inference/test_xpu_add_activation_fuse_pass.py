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
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestAddActXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["add_act_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=50))

        # Generate shape of input:X Y of ele_add
        def generate_input():
            return np.random.random([batch_size, 3, 100, 100]).astype(
                np.float32
            )

        axis = -1

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        elementwise_op = OpConfig(
            type='elementwise_add',
            inputs={'X': ['eltwise_X'], 'Y': ['eltwise_Y']},
            outputs={'Out': ['eltwise_output']},
            axis=axis,
        )
        relu_op = OpConfig(
            "relu",
            inputs={"X": ["eltwise_output"]},
            outputs={"Out": ["relu_out"]},
        )
        mini_graph = [elementwise_op, relu_op]

        program_config = ProgramConfig(
            ops=mini_graph,
            weights={},
            inputs={
                "eltwise_X": TensorConfig(data_gen=partial(generate_input)),
                "eltwise_Y": TensorConfig(data_gen=partial(generate_input)),
            },
            outputs=mini_graph[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["add_activation_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
