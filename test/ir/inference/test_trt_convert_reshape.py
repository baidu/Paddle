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
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertReshapeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if self.dims == 1:
            if len(attrs[0]['shape']) != 1:
                return False

        # To test if the shape contains 0
        if len(attrs[0]['shape']) == 3:
            if attrs[0]['shape'][1] == 0:
                if self.dims != 3:
                    return False

        if len(attrs[0]['shape']) == 4:
            if attrs[0]['shape'][2] == 0:
                if self.dims != 4:
                    return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            if self.dims == 4:
                self.input_shape = [1, 2, 4, 6]
                return np.ones([1, 2, 4, 6]).astype(np.float32)
            elif self.dims == 3:
                self.input_shape = [1, 8, 6]
                return np.ones([1, 8, 6]).astype(np.float32)
            elif self.dims == 2:
                self.input_shape = [1, 48]
                return np.ones([1, 48]).astype(np.float32)
            elif self.dims == 1:
                self.input_shape = [48]
                return np.ones([48]).astype(np.float32)

        def generate_weight1(attrs: list[dict[str, Any]]):
            return np.array([1, 48]).astype(np.int32)

        def generate_shapeT1_data(attrs: list[dict[str, Any]]):
            return np.array([2]).astype(np.int32)

        def generate_shapeT2_data(attrs: list[dict[str, Any]]):
            return np.array([24]).astype(np.int32)

        for dims in [4, 3, 2, 1]:
            for shape in [
                [1, 6, 8],
                [1, 2, 4, 6],
                [1, 1, 0, 12],
                [1, 0, 6],
                [1, -1, 12],
                [2, -1],
                [3, 16],
                [3, 4, 4],
                [48],
                [-1, 48],
            ]:
                dics = [
                    {
                        "shape": shape,
                    },
                ]
                self.dims = dims
                dics_input = [{"X": ["reshape_input"]}]

                ops_config = [
                    {
                        "op_type": "reshape",
                        "op_inputs": dics_input[0],
                        "op_outputs": {"Out": ["reshape_out"]},
                        "op_attrs": dics[0],
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "reshape_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["reshape_out"],
                )

                yield program_config

    def generate_dynamic_shape(self):
        if self.dims == 4:
            self.dynamic_shape.min_input_shape = {"reshape_input": [1, 2, 4, 6]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [4, 2, 4, 6]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 2, 4, 6]}
        elif self.dims == 3:
            self.dynamic_shape.min_input_shape = {"reshape_input": [1, 8, 6]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [4, 8, 6]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 8, 6]}
        elif self.dims == 2:
            self.dynamic_shape.min_input_shape = {"reshape_input": [1, 48]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [4, 48]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 48]}
        elif self.dims == 1:
            self.dynamic_shape.min_input_shape = {"reshape_input": [48]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [48]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [48]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            # in static shape mode, here is consistent with op_teller.cc
            if not dynamic_shape:
                if attrs[0]['shape'][0] == 0:
                    return 1, 2
                elif len(attrs[0]['shape']) == 1:
                    return 0, 3
                elif np.prod(attrs[0]['shape'][1:]) == np.prod(
                    self.input_shape[1:]
                ):
                    return 1, 2
                else:
                    return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if not run_pir:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), 1e-3

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


# reshape having three inputs.
class TrtConvertReshapeTest2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            if self.dims == 4:
                return np.random.random([1, 2, 4, 6]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([1, 8, 6]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([1, 48]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([48]).astype(np.float32)

        for dims in [4, 3, 2, 1]:
            for shape in [[-1, 48]]:
                dics = [
                    {
                        "shape": shape,
                    },
                    {},
                ]
                self.dims = dims
                dics_input = [
                    {
                        "X": ["reshape_input"],
                        "ShapeTensor": ["shapeT1_data", "shapeT2_data"],
                    },
                ]
                ops_config = [
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT1_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "2",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shapeT2_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "24",
                            "shape": [1],
                        },
                    },
                    {
                        "op_type": "reshape",
                        "op_inputs": dics_input[0],
                        "op_outputs": {"Out": ["reshape_out"]},
                        "op_attrs": dics[0],
                    },
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "reshape_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["reshape_out"],
                )

                yield program_config

    def generate_dynamic_shape(self):
        if self.dims == 4:
            self.dynamic_shape.min_input_shape = {"reshape_input": [1, 2, 4, 6]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [4, 2, 4, 6]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 2, 4, 6]}
        elif self.dims == 3:
            self.dynamic_shape.min_input_shape = {"reshape_input": [1, 8, 6]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [4, 8, 6]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 8, 6]}
        elif self.dims == 2:
            self.dynamic_shape.min_input_shape = {"reshape_input": [1, 48]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [4, 48]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 48]}
        elif self.dims == 1:
            self.dynamic_shape.min_input_shape = {"reshape_input": [48]}
            self.dynamic_shape.max_input_shape = {"reshape_input": [48]}
            self.dynamic_shape.opt_input_shape = {"reshape_input": [48]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 2), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


# reshape having 2 inputs.
class TrtConvertReshapeTest3(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            if self.dims == 4:
                return np.random.random([1, 2, 12, 6]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([1, 8, 18]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([1, 144]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([144]).astype(np.float32)

        for dims in [4, 3, 2, 1]:
            for shape in [[-1, 144]]:
                dics = [
                    {
                        "shape": shape,
                    },
                    {},
                ]
                self.dims = dims
                dics_input = [
                    {
                        "X": ["reshape_input"],
                        "shape_data": ["shape_data"],
                    },
                ]
                ops_config = [
                    {
                        "op_type": "fill_constant",
                        "op_inputs": {},
                        "op_outputs": {"Out": ["shape_data"]},
                        "op_attrs": {
                            "dtype": 2,
                            "str_value": "12",
                            "shape": [2],
                        },
                    },
                    {
                        "op_type": "reshape",
                        "op_inputs": dics_input[0],
                        "op_outputs": {"Out": ["reshape_out"]},
                        "op_attrs": dics[0],
                    },
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "reshape_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["reshape_out"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(self):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "reshape_input": [1, 2, 12, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "reshape_input": [4, 2, 12, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "reshape_input": [1, 2, 12, 6]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "reshape_input": [1, 8, 18]
                }
                self.dynamic_shape.max_input_shape = {
                    "reshape_input": [4, 8, 18]
                }
                self.dynamic_shape.opt_input_shape = {
                    "reshape_input": [1, 8, 18]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"reshape_input": [1, 144]}
                self.dynamic_shape.max_input_shape = {"reshape_input": [4, 144]}
                self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 144]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"reshape_input": [144]}
                self.dynamic_shape.max_input_shape = {"reshape_input": [144]}
                self.dynamic_shape.opt_input_shape = {"reshape_input": [144]}

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 2), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


class TrtConvertReshapeZeroDimsTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            if self.dims > 0:
                self.input_shape = [1] * self.dims
                return np.random.random(self.input_shape).astype(np.float32)
            elif self.dims == 0:
                self.input_shape = []
                return np.random.random([]).astype(np.float32)

        for dims in [0, 1, 2, 3]:
            for shape in [
                [],
                [1, 1],
            ]:
                dics = [
                    {
                        "shape": shape,
                    },
                ]
                self.dims = dims
                dics_input = [{"X": ["reshape_input"]}]

                ops_config = [
                    {
                        "op_type": "reshape",
                        "op_inputs": dics_input[0],
                        "op_outputs": {"Out": ["reshape_out"]},
                        "op_attrs": dics[0],
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "reshape_input": TensorConfig(
                            data_gen=partial(generate_input1, dics)
                        )
                    },
                    outputs=["reshape_out"],
                )

                yield program_config

    def generate_dynamic_shape(self):
        self.dynamic_shape.min_input_shape = {"reshape_input": self.input_shape}
        self.dynamic_shape.max_input_shape = {"reshape_input": self.input_shape}
        self.dynamic_shape.opt_input_shape = {"reshape_input": self.input_shape}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            # only test dynamic shape mode
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
