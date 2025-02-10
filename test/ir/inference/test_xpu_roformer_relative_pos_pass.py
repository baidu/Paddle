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

from paddle.base import core


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Unsupported on XPU3",
)
class TestRoformerRelativePosXPUPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        # config.switch_ir_optim(True)
        # config.switch_ir_debug(True)
        yield config, ["roformer_relative_embedding_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=10), min_size=4, max_size=4
            )
        )
        x_shape[1] = draw(st.integers(min_value=12, max_value=12))
        x_shape[2] = draw(st.integers(min_value=512, max_value=512))
        x_shape[3] = draw(st.integers(min_value=32, max_value=32))
        sin_emb_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=1),
                min_size=4,
                max_size=4,
            )
        )
        sin_emb_shape[1] = draw(st.integers(min_value=1, max_value=1))
        sin_emb_shape[2] = draw(st.integers(min_value=512, max_value=512))
        sin_emb_shape[3] = draw(st.integers(min_value=32, max_value=32))
        cos_emb_shape = sin_emb_shape

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while running
        split_op = OpConfig(
            "split",
            inputs={"X": ["x"]},
            outputs={"Out": ["split_out1", "split_out2"]},
            axis=3,
            num=2,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["split_out2"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
        )
        concat_op = OpConfig(
            "concat",
            inputs={"X": ["scale_out", "split_out1"]},
            outputs={"Out": ["concat_out"]},
            axis=-1,
        )
        shape_op = OpConfig(
            "shape",
            inputs={"Input": ["x"]},
            outputs={"Out": ["shape_out"]},
        )
        slice1_op = OpConfig(
            "slice",
            inputs={"Input": ["shape_out"]},
            outputs={"Out": ["slice1_out"]},
            axes=[0],
            starts=[-2],
            ends=[-1],
            infer_flags=[1],
            decrease_axis=[0],
        )
        slice_sin_op = OpConfig(
            "slice",
            inputs={"Input": ["sin_emb"], "EndsTensorList": ["slice1_out"]},
            outputs={"Out": ["slice_sin_out"]},
            axes=[2],
            starts=[0],
            ends=[-1],
            infer_flags=[-1],
            decrease_axis=[],
        )
        slice_cos_op = OpConfig(
            "slice",
            inputs={"Input": ["cos_emb"], "EndsTensorList": ["slice1_out"]},
            outputs={"Out": ["slice_cos_out"]},
            axes=[2],
            starts=[0],
            ends=[-1],
            infer_flags=[-1],
            decrease_axis=[],
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["concat_out"], "Y": ["slice_sin_out"]},
            outputs={"Out": ["mul1_out"]},
        )
        mul2_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["x"], "Y": ["slice_cos_out"]},
            outputs={"Out": ["mul2_out"]},
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul2_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add_out"]},
        )

        ops = [
            split_op,
            scale_op,
            concat_op,
            shape_op,
            slice1_op,
            slice_sin_op,
            slice_cos_op,
            mul1_op,
            mul2_op,
            add_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "x": TensorConfig(data_gen=partial(generate_data, x_shape)),
                "sin_emb": TensorConfig(
                    data_gen=partial(generate_data, sin_emb_shape)
                ),
                "cos_emb": TensorConfig(
                    data_gen=partial(generate_data, cos_emb_shape)
                ),
            },
            weights={},
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["roformer_relative_pos_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
