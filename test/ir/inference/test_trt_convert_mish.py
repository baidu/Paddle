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
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertMishTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(batch, dim1, dim2, dim3):
            shape = [batch]
            if dim1 != 0:
                shape.append(dim1)
            if dim2 != 0:
                shape.append(dim2)
            if dim3 != 0:
                shape.append(dim3)
            return np.random.random(shape).astype(np.float32)

        for batch in [1, 4]:
            for dim1 in [0, 3]:
                for dim2 in [0, 16]:
                    for dim3 in [0, 32]:
                        for threshold in [5.0, 20.0]:
                            self.dim1 = dim1
                            self.dim2 = dim2
                            self.dim3 = dim3

                            if dim1 == 0 and dim2 != 0:
                                continue
                            if dim1 == 0 and dim2 == 0 and dim3 != 0:
                                continue

                            ops_config = [
                                {
                                    "op_type": "mish",
                                    "op_inputs": {"X": ["input_data"]},
                                    "op_outputs": {"Out": ["mish_output_data"]},
                                    "op_attrs": {"threshold": threshold},
                                }
                            ]

                            ops = self.generate_op_config(ops_config)
                            program_config = ProgramConfig(
                                ops=ops,
                                weights={},
                                inputs={
                                    "input_data": TensorConfig(
                                        data_gen=partial(
                                            generate_input,
                                            batch,
                                            dim1,
                                            dim2,
                                            dim3,
                                        )
                                    )
                                },
                                outputs=["mish_output_data"],
                            )

                            yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            if self.dim1 == 0:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2],
                }
            else:
                if self.dim2 == 0 and self.dim3 == 0:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 1],
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64],
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [2, 3],
                    }
                elif self.dim2 != 0 and self.dim3 != 0:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 1, 1, 1],
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64, 128, 128],
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [2, 3, 16, 32],
                    }
                elif self.dim3 == 0:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 1, 1],
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64, 256],
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [2, 3, 128],
                    }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if self.dim1 == 0 and self.dim2 == 0 and self.dim3 == 0:
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "Trt does not support 1-dimensional input.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
