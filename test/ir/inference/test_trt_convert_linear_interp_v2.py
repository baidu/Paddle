# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertLinearInterpV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        ver = paddle_infer.get_trt_compile_version()
        # here is consistent with op_teller.cc
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7100:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            return np.random.uniform(low=0.0, high=1.0, size=[1, 3, 64]).astype(
                np.float32
            )

        def generate_input2(attrs: list[dict[str, Any]]):
            return np.random.uniform(low=0.5, high=6.0, size=(1)).astype(
                "float32"
            )

        for data_layout in ["NCHW", "NHWC"]:
            for align_corners in [False, True]:
                dics = [
                    {
                        "data_layout": data_layout,
                        "interp_method": "linear",
                        "align_corners": align_corners,
                        "align_mode": 0,
                        "scale": [],
                        "out_h": -1,
                        "out_w": -1,
                    }
                ]

                ops_config = [
                    {
                        "op_type": "linear_interp_v2",
                        "op_inputs": {
                            "X": ["input_data"],
                            "Scale": ["input_scale"],
                        },
                        "op_outputs": {"Out": ["linear_interp_v2_output_data"]},
                        "op_attrs": dics[0],
                    }
                ]
                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "input_scale": TensorConfig(
                            data_gen=partial(generate_input2, dics)
                        )
                    },
                    inputs={
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["linear_interp_v2_output_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64]}

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
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2

    def test(self):
        self.run_test()


class TrtConvertLinearInterpV2Test1(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        ver = paddle_infer.get_trt_compile_version()
        # here is consistent with op_teller.cc
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7100:
            return False
        return True

    def sample_program_configs(self):
        self.workspace_size = 1 << 32

        def generate_input1(attrs: list[dict[str, Any]]):
            return np.random.uniform(
                low=0.0, high=1.0, size=[1, 18, 144]
            ).astype(np.float32)

        for data_layout in ["NCHW", "NHWC"]:
            for align_corners in [False, True]:
                for out_w in [288]:
                    dics = [
                        {
                            "data_layout": data_layout,
                            "interp_method": "linear",
                            "align_corners": align_corners,
                            "align_mode": 0,
                            "scale": [],
                            "out_h": -1,
                            "out_w": out_w,
                        }
                    ]

                    ops_config = [
                        {
                            "op_type": "linear_interp_v2",
                            "op_inputs": {
                                "X": ["input_data"],
                            },
                            "op_outputs": {
                                "Out": ["linear_interp_v2_output_data"]
                            },
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input1, dics)
                            )
                        },
                        outputs=["linear_interp_v2_output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 18, 144]}
            self.dynamic_shape.max_input_shape = {"input_data": [8, 18, 144]}
            self.dynamic_shape.opt_input_shape = {"input_data": [4, 18, 144]}

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
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
