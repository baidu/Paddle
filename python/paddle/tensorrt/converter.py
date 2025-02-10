# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import ctypes
import hashlib
import logging

import paddle

paddle.base.core.register_paddle_plugin()
import tensorrt as trt

import paddle
from paddle import pir
from paddle.base.core import clear_shape_info, get_value_shape_range_info
from paddle.base.log_helper import get_logger

from .impls.activation import *  # noqa: F403
from .impls.attribute import *  # noqa: F403
from .impls.common import *  # noqa: F403
from .impls.conv import *  # noqa: F403
from .impls.creation import *  # noqa: F403
from .impls.einsum import *  # noqa: F403
from .impls.input import *  # noqa: F403
from .impls.linalg import *  # noqa: F403
from .impls.logic import *  # noqa: F403
from .impls.manipulation import *  # noqa: F403
from .impls.math import *  # noqa: F403
from .impls.norm import *  # noqa: F403
from .impls.ops import *  # noqa: F403
from .impls.others import *  # noqa: F403
from .impls.pooling import *  # noqa: F403
from .impls.search import *  # noqa: F403
from .impls.stat import *  # noqa: F403
from .impls.vision import *  # noqa: F403
from .register import converter_registry
from .util import (
    TensorRTConfigManager,
    TensorRTConstantManager,
    get_cache_path,
    get_trt_version,
    get_trt_version_list,
    is_shape_tensor,
    map_dtype,
    remove_duplicate_value,
    weight_to_tensor,
    zero_dims_to_one_dims,
)

version_list = get_trt_version_list()

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class PaddleToTensorRTConverter:
    def __init__(self, paddle_program, scope, trt_config=None):
        self.scope = scope
        self.program = paddle_program
        self.trt_config = trt_config
        constant_manager = TensorRTConstantManager()
        params = paddle_program.global_block().all_parameters()
        param_dict = {}
        # save parameters
        for v in params:
            name = v.get_defining_op().attrs()["parameter_name"]
            weight_tensor = self.scope.var(name).get_tensor()
            constant_manager.set_constant_value(name, weight_tensor, v)

        self.input_info = {}
        self.trt_output_value_map = {}
        self.engine_num = 0
        # init tensorrt plugin
        trt_plugin_lib = ctypes.CDLL('libnvinfer_plugin.so')
        trt_plugin_lib.initLibNvInferPlugins(None, "")

    def find_graph_inputs_outputs(self, group_op):
        operations = next(iter(group_op.blocks())).ops
        all_values = {}
        output_values = {}

        graph_output_values = []

        def __is_output_value(value):
            for op in value.all_used_ops():
                if op.name() == "cf.yield":
                    return True
            return False

        # Collect all output values from all operations
        for op in operations:
            for result in op.results():
                output_values[result.id] = result
                all_values[result.id] = result
                if __is_output_value(result):
                    graph_output_values.append(result)
            for operand in op.operands():
                source = operand.source()
                if not source.initialized():
                    _logger.warning(f"Skipping uninitialized source: {source}")
                    continue
                else:
                    all_values[source.id] = source

        # Input values are those that are in all_values but not in output_values
        input_values = [
            value
            for value_id, value in all_values.items()
            if value_id not in output_values
        ]

        return input_values, graph_output_values

    def convert_subgraph_to_trt(self, program, group_op):
        from .export import PrecisionMode

        trt_manager = TensorRTConfigManager(self.trt_config)
        if self.trt_config is not None and self.trt_config.ops_run_float:
            _logger.info(f"force_fp32_ops: {trt_manager.get_force_fp32_ops()}")

        _logger.info(f"start process {group_op}")

        operations = next(iter(group_op.blocks())).ops
        input_values, output_values = self.find_graph_inputs_outputs(group_op)
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        profile = builder.create_optimization_profile()

        # Mapping from Value id to TensorRT ITensor
        value_to_trt_tensor = {}
        min_shape_map = {}
        opt_shape_map = {}
        max_shape_map = {}
        min_value_map = {}
        opt_value_map = {}
        max_value_map = {}
        input_names = []
        new_input_values = []
        constant_manager = TensorRTConstantManager()

        # Because one of the inputs to pd_op.concat is builtin.combine,
        # during the conversion process using the converter,
        # it is necessary to obtain the input of builtin.combine.
        origin_input_value = []
        for value in input_values:
            defining_op = value.get_defining_op()
            if defining_op.name() == "builtin.combine":
                for operand in defining_op.operands():
                    source = operand.source()
                    origin_input_value.append(source)
            else:
                origin_input_value.append(value)

        origin_input_value = remove_duplicate_value(origin_input_value)
        # create TRT Weight and TRT Input
        for value in origin_input_value:
            defining_op = value.get_defining_op()
            if defining_op.name() == "builtin.parameter":
                param_name = defining_op.attrs()["parameter_name"]
                weight = trt.Weights(
                    constant_manager.get_constant_value(param_name)
                )
                value_to_trt_tensor[value.id] = weight
            elif defining_op.name() == "builtin.constant":
                constant_value_name = defining_op.attrs()["value"]
                constant_tensor = self.scope.var(
                    constant_value_name
                ).get_tensor()
                constant_manager.set_constant_value(
                    constant_value_name, constant_tensor, value
                )
                constant_tensor = trt.Weights(
                    constant_manager.get_constant_value(constant_value_name)
                )
                value_to_trt_tensor[value.id] = constant_tensor
            else:
                shape = value.shape
                dtype = map_dtype(value.dtype.name)
                input_name = f"input_{value.id}"
                # 0-dims -> 1-dims
                if len(shape) == 0:
                    shape = [1]
                input_tensor = network.add_input(
                    name=input_name, dtype=dtype, shape=shape
                )
                input_names.append(input_name)
                new_input_values.append(value)
                value_to_trt_tensor[value.id] = input_tensor

        for op in operations:
            # Adding marker labels to builtin ops facilitates convert processing, but they ultimately do not enter the TensorRT subgraph.
            if op.name() == "builtin.split" or op.name() == "builtin.combine":
                continue
            operands = []
            for operand in op.operands():
                source = operand.source()
                if not source.initialized():
                    operands.append(None)
                    continue
                vec_type = source.type().as_vec_type()
                if vec_type is not None and len(vec_type.as_list()) == 0:
                    continue
                define_op_name = source.get_defining_op().name()
                if define_op_name == "builtin.combine":
                    operand_list = []
                    for combined_operand in source.get_defining_op().operands():
                        combined_source = combined_operand.source()
                        combined_source_id = combined_source.id
                        if combined_source_id in value_to_trt_tensor:
                            trt_input_tensor = weight_to_tensor(
                                network,
                                combined_source,
                                value_to_trt_tensor[combined_source_id],
                                op.name(),
                            )
                            trt_input_tensor = zero_dims_to_one_dims(
                                network, trt_input_tensor
                            )
                            operand_list.append(trt_input_tensor)
                        else:
                            raise RuntimeError(
                                f'{combined_source_id} not found in value_to_trt_tensor'
                            )
                    operands.append(operand_list)
                else:
                    source_id = source.id
                    if source_id in value_to_trt_tensor:
                        trt_input_tensor = weight_to_tensor(
                            network,
                            source,
                            value_to_trt_tensor[source_id],
                            op.name(),
                        )
                        trt_input_tensor = zero_dims_to_one_dims(
                            network, trt_input_tensor
                        )
                        operands.append(trt_input_tensor)
                    else:
                        raise RuntimeError(
                            f'{source_id} not found in value_to_trt_tensor'
                        )

            trt_outs = self.convert(network, op, operands)

            results = []

            for idx, result in enumerate(op.results()):
                if result.is_combine():
                    # empty vec value condition
                    if len(result.type().as_vec_type().as_list()) == 0:
                        results.append(result)
                        continue
                    used_ops = result.all_used_ops()
                    for use_op in used_ops:
                        if use_op.name() == "builtin.split":
                            split_outputs = use_op.results()
                            results.extend(split_outputs)
                else:
                    results.append(result)

            for idx, result in enumerate(results):
                if idx < len(trt_outs):
                    value_to_trt_tensor[result.id] = trt_outs[idx]
                else:
                    value_to_trt_tensor[result.id] = None

        # Set TRT min/opt/max input shape and the value of shape tensor
        for i, value in enumerate(origin_input_value):
            trt_input = value_to_trt_tensor[value.id]
            defining_op_name = value.get_defining_op().name()
            if (
                defining_op_name == "builtin.parameter"
                or defining_op_name == "builtin.constant"
            ):
                # constant/parameter condition, needn't get min/opt/max shape
                continue
            input_name = trt_input.name
            _logger.info(
                f"set shape of {value}, op is: {value.get_defining_op()}"
            )
            min_shape = []
            opt_shape = []
            max_shape = []
            min_value = []
            opt_value = []
            max_value = []

            value_define_op = value.get_defining_op()
            # if the input value is generated by the other trt_engine_op, so the shape is searched by origin value
            if (
                value_define_op.name() == "builtin.split"
                and value_define_op.operand_source(0).get_defining_op().name()
                == "pd_op.tensorrt_engine"
            ):
                min_shape = self.input_info[value.id]["min_shape"]
                opt_shape = self.input_info[value.id]["opt_shape"]
                max_shape = self.input_info[value.id]["max_shape"]
                if trt_input.is_shape_tensor:
                    min_value = self.input_info[value.id]["min_value"]
                    opt_value = self.input_info[value.id]["opt_value"]
                    max_value = self.input_info[value.id]["max_value"]
            else:
                min_shape = get_value_shape_range_info(
                    value, False, paddle.base.core.ShapeMode.kMIN
                )
                opt_shape = get_value_shape_range_info(
                    value, False, paddle.base.core.ShapeMode.kOPT
                )
                max_shape = get_value_shape_range_info(
                    value, False, paddle.base.core.ShapeMode.kMAX
                )

                if trt_input.is_shape_tensor:
                    min_value = get_value_shape_range_info(
                        value, True, paddle.base.core.ShapeMode.kMIN
                    )
                    opt_value = get_value_shape_range_info(
                        value, True, paddle.base.core.ShapeMode.kOPT
                    )
                    max_value = get_value_shape_range_info(
                        value, True, paddle.base.core.ShapeMode.kMAX
                    )
            if not trt_input.is_shape_tensor:
                _logger.info(f"set min_shape of {value} as {min_shape}")
                _logger.info(f"set opt_shape of {value} as {opt_shape}")
                _logger.info(f"set max_shape of {value} as {max_shape}")
                profile.set_shape(
                    input_name, min=min_shape, opt=opt_shape, max=max_shape
                )
            else:
                _logger.info(
                    f"set min_value of shape input: {value} as {min_value}"
                )
                _logger.info(
                    f"set max_value of shape input: {value} as {opt_value}"
                )
                _logger.info(
                    f"set opt_value of shape input: {value} as {max_value}"
                )
                profile.set_shape_input(
                    input_name, min=min_value, opt=opt_value, max=max_value
                )

            min_shape_map[input_name] = min_shape
            opt_shape_map[input_name] = opt_shape
            max_shape_map[input_name] = max_shape
            min_value_map[input_name] = min_value
            opt_value_map[input_name] = opt_value
            max_value_map[input_name] = max_value

        out_shapes = []
        out_names = []
        out_types = []
        for out_index in range(len(output_values)):
            result_value = output_values[out_index]
            output_tensor = value_to_trt_tensor[result_value.id]
            if output_tensor is None:
                out_names.append("")
                out_shapes.append([])
                out_types.append(None)
                continue
            network.mark_output(output_tensor)
            out_names.append(output_tensor.name)
            out_shapes.append(result_value.shape)
            out_types.append(result_value.dtype)
            if group_op.result(out_index).use_empty():
                # if result value is not used, it doesn't need get shape, continue
                continue
            min_shape = get_value_shape_range_info(
                result_value, False, paddle.base.core.ShapeMode.kMIN
            )
            opt_shape = get_value_shape_range_info(
                result_value, False, paddle.base.core.ShapeMode.kOPT
            )
            max_shape = get_value_shape_range_info(
                result_value, False, paddle.base.core.ShapeMode.kMAX
            )
            min_value = []
            opt_value = []
            max_value = []
            if is_shape_tensor(result_value):
                min_value = get_value_shape_range_info(
                    result_value, True, paddle.base.core.ShapeMode.kMIN
                )
                opt_value = get_value_shape_range_info(
                    result_value, True, paddle.base.core.ShapeMode.kOPT
                )
                max_value = get_value_shape_range_info(
                    result_value, True, paddle.base.core.ShapeMode.kMAX
                )

            self.input_info[result_value.id] = {
                "min_shape": min_shape,
                "opt_shape": opt_shape,
                "max_shape": max_shape,
                "min_value": min_value,
                "opt_value": opt_value,
                "max_value": max_value,
            }

        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        if version_list[0] > 8 or (
            version_list[0] == 8 and version_list[1] >= 6
        ):  # trt version >= 8.6
            config.builder_optimization_level = (
                self.trt_config.optimization_level
            )
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, self.trt_config.workspace_size
        )

        if self.trt_config is not None:
            precision_mode = self.trt_config.precision_mode
        if self.trt_config is not None and precision_mode == PrecisionMode.FP16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                _logger.info("Run Paddle-TRT FP16 mode")
            else:
                _logger.warning(
                    "Hardware does not support FP16. Continuing in FP32 mode."
                )
        elif (
            self.trt_config is not None and precision_mode == PrecisionMode.BF16
        ):
            if version_list[0] >= 9:
                if builder.platform_has_fast_bfp16 and hasattr(
                    builder, 'plateform_has_fast_bf16'
                ):
                    config.set_flag(trt.BuilderFlag.BF16)
                    _logger.info("Run Paddle-TRT BF16 mode")
                else:
                    _logger.warning(
                        "Hardware does not support BF16. Continuing in FP32 mode."
                    )
            else:
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    _logger.warning(
                        "Because the version of TensorRT is less than 9.0, run  Paddle-TRT FP16 mode"
                    )
                else:
                    _logger.warning(
                        "Hardware does not support FP16. Continuing in FP32 mode."
                    )
        elif self.trt_config is not None:
            _logger.info(
                f"Default precision mode {self.trt_config.precision_mode}"
            )

        if (
            version_list[0] > 8
            or version_list[0] == 8
            and version_list[1] >= 2
            and version_list[2] >= 1
        ):
            if self.trt_config is not None and self.trt_config.ops_run_float:
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        trt_engine = builder.build_serialized_network(network, config)
        assert (
            trt_engine is not None
        ), 'Failed to build engine. please see ERROR log from trt.Logger'
        trt_params = paddle.base.libpaddle.TRTEngineParams()
        trt_params.min_input_shape = min_shape_map
        trt_params.max_input_shape = max_shape_map
        trt_params.optim_input_shape = opt_shape_map
        trt_params.min_shape_tensor = min_value_map
        trt_params.max_shape_tensor = max_value_map
        trt_params.optim_shape_tensor = opt_value_map
        group_str = str(group_op)
        engine_name = (
            int(hashlib.sha256(group_str.encode('utf-8')).hexdigest(), 16)
            % 10**8
        )
        CACHE_ROOT = get_cache_path()
        CACHE_FILE = f"{CACHE_ROOT}/engine_{engine_name}_{self.engine_num}.trt"
        with open(CACHE_FILE, "wb") as f:
            f.write(trt_engine)
        PIR_DUMP_FILE = (
            f"{CACHE_ROOT}/engine_{engine_name}_{self.engine_num}.pir"
        )
        with open(PIR_DUMP_FILE, "w") as f:
            f.write(group_str)
        trt_params.engine_serialized_data = CACHE_FILE

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(program):
            pir.set_insertion_point(group_op)
            out = paddle._C_ops.tensorrt_engine(
                new_input_values,
                trt_params,
                input_names,
                out_names,
                out_shapes,
                out_types,
                "",
            )

            for out_index in range(len(out)):
                if group_op.result(out_index).use_empty():
                    # if result value is not been used, it doesn't need get shape, continue
                    continue
                ori_value = output_values[out_index]
                current_value = out[out_index]
                orin_min_shape = self.input_info[ori_value.id]["min_shape"]
                orin_opt_shape = self.input_info[ori_value.id]["opt_shape"]
                orin_max_shape = self.input_info[ori_value.id]["max_shape"]
                orin_min_value = self.input_info[ori_value.id]["min_value"]
                orin_opt_value = self.input_info[ori_value.id]["opt_value"]
                orin_max_value = self.input_info[ori_value.id]["max_value"]
                self.input_info[current_value.id] = {
                    "min_shape": orin_min_shape,
                    "opt_shape": orin_opt_shape,
                    "max_shape": orin_max_shape,
                    "min_value": orin_min_value,
                    "opt_value": orin_opt_value,
                    "max_value": orin_max_value,
                }

        return out

    def convert(self, network, paddle_op, inputs):
        trt_version = get_trt_version()
        op_name = paddle_op.name()
        if op_name in ["cf.yield"]:
            return
        else:
            converter_func = converter_registry.get(op_name, trt_version)
            if converter_func is None:
                raise NotImplementedError(
                    f"Converter for {op_name} not implemented."
                )
            outs = converter_func(network, paddle_op, inputs)
        if isinstance(outs, trt.ITensor):
            return (outs,)
        else:
            return outs

    def convert_program_to_trt(self):
        for op in self.program.global_block().ops:
            if op.name() == "cinn_op.group" or op.name() == "builtin.group":
                _logger.info(f"start process {op.name()}")
                self.engine_num += 1
                new_out = self.convert_subgraph_to_trt(self.program, op)
                orin_out_values = op.results()
                for o_i in range(len(orin_out_values)):
                    orin_out_values[o_i].replace_all_uses_with(new_out[o_i])

                self.program.global_block().remove_op(op)

        save_one_parameter = (
            False  # We need to keep at least one parameter for save
        )
        for op in self.program.global_block().ops:
            if op.name() == "builtin.parameter":
                parameter_name = op.attrs()["parameter_name"]
                if (
                    not save_one_parameter
                    and "constant_folding" not in parameter_name
                ):
                    save_one_parameter = True
                    continue
                if op.results()[0].use_empty():
                    self.program.global_block().remove_op(op)
            if op.name() == "builtin.constant":
                if op.results()[0].use_empty():
                    self.program.global_block().remove_op(op)

        # Call clear_shape_info to clear the previous shape information
        clear_shape_info()
