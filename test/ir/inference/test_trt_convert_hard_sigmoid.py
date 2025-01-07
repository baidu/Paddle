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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertHardSigmoidTest_dim_2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for batch in [1, 4]:
            for shape in [[batch, 32], [batch, 16, 32], [batch, 32, 16, 128]]:
                self.input_dim = len(shape)
                for slope in [0.1, 0.5]:
                    for offset in [0.2, 0.7]:
                        dics = [{"slope": slope, "offset": offset}]
                        ops_config = [
                            {
                                "op_type": "hard_sigmoid",
                                "op_inputs": {
                                    "X": ["input_data"],
                                },
                                "op_outputs": {"Out": ["output_data"]},
                                "op_attrs": dics[0],
                            }
                        ]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input, shape)
                                )
                            },
                            outputs=["output_data"],
                        )

                        yield program_config

    def generate_dynamic_shape(self):
        if self.input_dim == 2:
            self.dynamic_shape.min_input_shape = {"input_data": [1, 8]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 32]}
            self.dynamic_shape.opt_input_shape = {"input_data": [2, 16]}
        elif self.input_dim == 3:
            self.dynamic_shape.min_input_shape = {"input_data": [1, 8, 8]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 16, 32]}
            self.dynamic_shape.opt_input_shape = {"input_data": [4, 16, 32]}
        elif self.input_dim == 4:
            self.dynamic_shape.min_input_shape = {"input_data": [1, 8, 8, 4]}
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 32, 16, 128]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [4, 32, 16, 128]
            }
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if not run_pir:
            # for static_shape
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), (1, 2), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), (1, 2), 1e-3

        # for dynamic_shape
        clear_dynamic_shape()
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 2), 1e-3

    def test(self):
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
