# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TrtConvertLogicalTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for shape in [[2, 16], [2, 16, 32], [1, 32, 16, 32]]:
            for op_type in ["logical_and", "logical_or", "logical_xor"]:
                for axis in [-1]:
                    self.dims = len(shape)
                    dics = [
                        {"axis": axis},
                        {"in_dtype": 5, "out_dtype": 0},
                        {"in_dtype": 0, "out_dtype": 5},
                    ]
                    ops_config = [
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data1"]},
                            "op_outputs": {"Out": ["cast_output_data1"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data1": np.bool_},
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data2"]},
                            "op_outputs": {"Out": ["cast_output_data3"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data3": np.bool_},
                        },
                        {
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["cast_output_data1"],
                                "Y": ["cast_output_data3"],
                            },
                            "op_outputs": {"Out": ["cast_output_data0"]},
                            "op_attrs": dics[0],
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["cast_output_data0"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[2],
                        },
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
            if self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400:
                    return 0, 7
                return 1, 3
            return 0, 7

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
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertCompareTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for shape in [[2, 16], [2, 16, 32], [1, 32, 16, 32]]:
            for op_type in ["less_than", "greater_than"]:
                for axis in [-1]:
                    self.dims = len(shape)
                    dics = [
                        {"axis": axis},
                        {"in_dtype": 0, "out_dtype": 5},
                    ]
                    ops_config = [
                        {
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["input_data1"],
                                "Y": ["input_data2"],
                            },
                            "op_outputs": {"Out": ["cast_output_data0"]},
                            "op_attrs": dics[0],
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["cast_output_data0"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[1],
                        },
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, list[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
            if self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400:
                return 0, 5
            if not dynamic_shape:
                return 0, 5
            return 1, 3

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
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertLessEqualTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for shape in [[2, 16], [2, 16, 32], [1, 32, 16, 32]]:
            for op_type in ["less_equal"]:
                for axis in [-1]:
                    self.dims = len(shape)
                    dics = [
                        {"axis": axis},
                        {"in_dtype": 5, "out_dtype": 2},
                        {"in_dtype": 0, "out_dtype": 5},
                    ]
                    ops_config = [
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data1"]},
                            "op_outputs": {"Out": ["cast_output_data1"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data1": np.int32},
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data2"]},
                            "op_outputs": {"Out": ["cast_output_data2"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data2": np.int32},
                        },
                        {
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["cast_output_data1"],
                                "Y": ["cast_output_data2"],
                            },
                            "op_outputs": {"Out": ["cast_output_data0"]},
                            "op_attrs": dics[0],
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["cast_output_data0"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[2],
                        },
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, list[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
            if self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if (
                ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400
                or not dynamic_shape
            ):
                return 2, 5
            else:
                return 1, 3

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
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertGreaterEqualTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for shape in [[2, 16], [2, 16, 32], [1, 32, 16, 32]]:
            for op_type in ["greater_equal"]:
                for axis in [-1]:
                    self.dims = len(shape)
                    dics = [
                        {"axis": axis},
                        {"in_dtype": 5, "out_dtype": 2},
                        {"in_dtype": 0, "out_dtype": 5},
                    ]
                    ops_config = [
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data1"]},
                            "op_outputs": {"Out": ["cast_output_data1"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data1": np.int32},
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["input_data2"]},
                            "op_outputs": {"Out": ["cast_output_data2"]},
                            "op_attrs": dics[1],
                            "outputs_dtype": {"cast_output_data2": np.int32},
                        },
                        {
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["cast_output_data1"],
                                "Y": ["cast_output_data2"],
                            },
                            "op_outputs": {"Out": ["cast_output_data0"]},
                            "op_attrs": dics[0],
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["cast_output_data0"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[2],
                        },
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, list[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [2, 16],
                }
            if self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16, 32],
                    "input_data2": [2, 16, 32],
                }
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [1, 32, 16, 32],
                    "input_data2": [1, 32, 16, 32],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if (
                ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400
                or not dynamic_shape
            ):
                return 2, 5
            else:
                return 1, 3

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
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertCompareSkipTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.int32)

        for shape in [[2, 16], [2, 16, 32], [1, 32, 16, 32]]:
            for op_type in ["less_than", "greater_than"]:
                self.dims = len(shape)
                ops_config = [
                    {
                        "op_type": op_type,
                        "op_inputs": {
                            "X": ["input_data1"],
                            "Y": ["input_data2"],
                        },
                        "op_outputs": {"Out": ["output_data"]},
                        "op_attrs": {},  #
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data1": TensorConfig(
                            data_gen=partial(generate_input, shape)
                        ),
                        "input_data2": TensorConfig(
                            data_gen=partial(generate_input, shape)
                        ),
                    },
                    outputs=["output_data"],
                )

                yield program_config

    def generate_dynamic_shape(self, attrs):
        if self.dims == 2:
            shape_data = [2, 16]
        if self.dims == 3:
            shape_data = [2, 16, 32]
        if self.dims == 4:
            shape_data = [1, 32, 16, 32]

        shape_info = {
            "input_data1": shape_data,
            "input_data2": shape_data,
        }
        self.dynamic_shape.min_input_shape = shape_info
        self.dynamic_shape.max_input_shape = shape_info
        self.dynamic_shape.opt_input_shape = shape_info

        return self.dynamic_shape

    def sample_predictor_configs(
        self,
        program_config,
        run_pir=False,
    ) -> (paddle_infer.Config, list[int], float):

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400:
                return 0, 7
            if not dynamic_shape:
                return 0, 7
            return 3, 4

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
        ), (1e-3, 1e-3)

        # for dynamic_shape
        self.generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
