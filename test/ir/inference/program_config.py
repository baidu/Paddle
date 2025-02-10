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

import copy
import enum
import os
from typing import Any, Callable

import numpy as np

import paddle
from paddle.base import core, framework
from paddle.base.executor import global_scope
from paddle.base.framework import (
    IrGraph,
    IrNode,
    Operator,
    OpProtoHolder,
    convert_np_dtype_to_proto_type,
)
from paddle.static.log_helper import get_logger
from paddle.static.quantization import (
    QuantizationFreezePass,
    QuantizationTransformPass,
)

LOGLEVEL = os.environ.get("PADDLE_TEST_LOGLEVEL", "INFO").upper()
logging = get_logger(
    __name__, LOGLEVEL, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class TensorConfig:
    '''
    A config builder for a input or a weight.
    '''

    def __init__(
        self,
        lod: list[list[int]] | None = None,
        data_gen: Callable[..., np.array] | None = None,
        shape: list[list[int]] | None = None,
    ):
        '''
        shape: The shape of the tensor.
        dtype: The data type of the tensor.
        data: The value of WeightVar. for input, it should be None
        '''
        self.lod = lod
        if data_gen is not None:
            self.data_gen = data_gen
            self.data = data_gen()
            self.dtype = self.data.dtype
            self.shape = self.data.shape
        else:
            assert (
                shape is not None
            ), "While data_gen is not defined, shape must not be None"
            self.data = np.random.normal(0.0, 1.0, shape).astype(np.float32)
            self.shape = shape
            self.dtype = self.data.dtype

    def __repr__(self):
        return str({'shape': self.shape, 'lod': self.lod, 'dtype': self.dtype})

    def convert_type_inplace(self, type: np.dtype):
        self.data = self.data.astype(type)
        self.dtype = self.data.dtype
        return self


class VarType(enum.Enum):
    DENSE_TENSOR = 1
    DENSE_TENSOR_ARRAY = 2
    STEP_SCOPES = 3


class OpConfig:
    '''A config builder for generating a Op.'''

    def __init__(
        self,
        type: str,
        inputs: dict[str, list[str]],
        outputs: dict[str, list[str]],
        attrs: dict[str, Any] | None = None,
        outputs_var_type: dict[str, VarType] | None = None,
        outputs_dtype: dict[str, np.dtype] | None = None,
        **kwargs,
    ):
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.outputs_dtype = outputs_dtype
        self.outputs_var_type = outputs_var_type
        self.attrs = attrs
        if self.attrs is None:
            self.attrs = {}
        self.attrs.update(kwargs)

    def __repr__(self):
        log_str = self.type
        log_str += str(self.attrs)
        return log_str


_OP_WITHOUT_KERNEL_SET = {
    'feed',
    'fetch',
    'go',
    'conditional_block',
    'static_pylayer',
    'while',
    'send',
    'recv',
    'listen_and_serv',
    'fl_listen_and_serv',
    'select',
    'checkpoint_notify',
    'gen_bkcl_id',
    'c_gen_bkcl_id',
    'gen_nccl_id',
    'c_gen_nccl_id',
    'c_comm_init',
    'c_sync_calc_stream',
    'c_sync_comm_stream',
    'heter_listen_and_serv',
    'c_wait_comm',
    'c_wait_compute',
}


class BlockConfig:
    '''A config builder for generating a Block.'''

    def __init__(
        self,
        ops: list[OpConfig],
        vars: list[str],
        vars_dtype: dict[str, np.dtype] | None = None,
        vars_var_type: dict[str, VarType] | None = None,
        vars_lod_level: dict[str, int] | None = None,
    ):
        self.ops = ops
        self.vars = vars
        self.vars_dtype = vars_dtype
        self.vars_var_type = vars_var_type
        self.vars_lod_level = vars_lod_level

    def fill_block_desc(self, block_desc):
        for name in self.vars:
            var_desc = block_desc.var(name.encode())
            var_desc.set_type(core.VarDesc.VarType.DENSE_TENSOR)
            if (
                self.vars_lod_level is not None
                and name in self.vars_lod_level.keys()
            ):
                var_desc.set_lod_level(self.vars_lod_level[name])
            if (
                self.vars_var_type is not None
                and name in self.vars_var_type.keys()
            ):
                if self.vars_var_type[name] == VarType.DENSE_TENSOR_ARRAY:
                    var_desc.set_type(core.VarDesc.VarType.DENSE_TENSOR_ARRAY)
                elif self.vars_var_type[name] == VarType.STEP_SCOPES:
                    var_desc.set_type(core.VarDesc.VarType.STEP_SCOPES)
                    continue

            var_desc.set_dtype(convert_np_dtype_to_proto_type(np.float32))
            if self.vars_dtype is not None and name in self.vars_dtype.keys():
                var_desc.set_dtype(
                    convert_np_dtype_to_proto_type(self.vars_dtype[name])
                )

        for op_config in self.ops:
            op_desc = block_desc.append_op()
            op_desc.set_type(op_config.type)
            for name, values in op_config.inputs.items():
                op_desc.set_input(name, values)
            # canonicalize scalar attrs
            if OpProtoHolder.instance().has_op_proto(op_config.type):
                proto = OpProtoHolder.instance().get_op_proto(op_config.type)
                canonicalized_attrs = framework.canonicalize_attrs(
                    op_config.attrs, proto
                )
            else:
                canonicalized_attrs = op_config.attrs
            for name, values in canonicalized_attrs.items():
                op_desc._set_attr(name, values)
            for name, values in op_config.outputs.items():
                op_desc.set_output(name, values)
                for v in values:
                    if block_desc.has_var_recursive(v.encode()):
                        continue
                    var_desc = block_desc.var(v.encode())
                    var_desc.set_type(core.VarDesc.VarType.DENSE_TENSOR)
                    if (
                        op_config.outputs_var_type is not None
                        and v in op_config.outputs_var_type.keys()
                    ):
                        if (
                            op_config.outputs_var_type[v]
                            == VarType.DENSE_TENSOR_ARRAY
                        ):
                            var_desc.set_type(
                                core.VarDesc.VarType.DENSE_TENSOR_ARRAY
                            )
                        elif (
                            op_config.outputs_var_type[v] == VarType.STEP_SCOPES
                        ):
                            var_desc.set_type(core.VarDesc.VarType.STEP_SCOPES)
                            continue
                    var_desc.set_dtype(
                        convert_np_dtype_to_proto_type(np.float32)
                    )
                    if (
                        op_config.outputs_dtype is not None
                        and v in op_config.outputs_dtype.keys()
                    ):
                        var_desc.set_dtype(
                            convert_np_dtype_to_proto_type(
                                op_config.outputs_dtype[v]
                            )
                        )
            if op_config.type not in _OP_WITHOUT_KERNEL_SET:
                op_desc.infer_var_type(block_desc)
                op_desc.infer_shape(block_desc)
            op_desc.check_attrs()


class ProgramConfig:
    '''A config builder for generating a Program.
    input_type : (np.dtype, default=None), the inputs will be casted to input_type before
                fed into TRT engine. If set to None, no casting will be performed.
    no_cast_list : (list[str], default=None), specify the tensors that will skip the casting
    '''

    def __init__(
        self,
        ops: list[OpConfig],
        weights: dict[str, TensorConfig],
        inputs: dict[str, TensorConfig],
        outputs: list[str],
        input_type: np.dtype | None = None,
        no_cast_list: list[str] | None = None,
    ):
        self.ops = ops
        # if no weight need to save, we create a place_holder to help serialize params.
        if not weights:

            def generate_weight():
                return np.array([1]).astype(np.float32)

            self.weights = {
                "place_holder_weight": TensorConfig(data_gen=generate_weight)
            }
        else:
            self.weights = weights
        self.inputs = inputs
        self.outputs = outputs
        self.input_type = input_type
        self.no_cast_list = [] if no_cast_list is None else no_cast_list
        self.supported_cast_type = [np.float32, np.float16]

    def __repr__(self):
        log_str = ''
        for i in range(len(self.ops)):
            if i != len(self.ops) - 1:
                log_str += repr(self.ops[i]) + ' + '
            else:
                log_str += repr(self.ops[i])
        log_str += ' -- '
        for t, v in self.inputs.items():
            log_str += '[' + t + ': ' + str(v) + ']'
        for t, v in self.weights.items():
            log_str += '[' + t + ': ' + str(v) + ']'
        log_str += f"['input_type': {self.input_type}]"
        return log_str

    def set_input_type(self, _type: np.dtype) -> None:
        assert (
            _type in self.supported_cast_type or _type is None
        ), "PaddleTRT only supports FP32 / FP16 IO"

        ver = paddle.inference.get_trt_compile_version()
        trt_version = ver[0] * 1000 + ver[1] * 100 + ver[2] * 10
        if trt_version < 8600:
            logging.info("set_input_type is ignored for TRT version < 8600")
            return

        self.input_type = _type

    def get_feed_data(self) -> dict[str, dict[str, Any]]:
        feed_data = {}
        for name, tensor_config in self.inputs.items():
            data = tensor_config.data
            # Cast to target input_type
            if (
                self.input_type is not None
                and name not in self.no_cast_list
                and data.dtype in self.supported_cast_type
            ):
                data = data.astype(self.input_type)
            # Truncate FP32 tensors to FP16 precision for FP16 test stability
            if data.dtype == np.float32 and name not in self.no_cast_list:
                data = data.astype(np.float16).astype(np.float32)

            feed_data[name] = {
                'data': data,
                'lod': tensor_config.lod,
            }
        return feed_data

    def _cast(self) -> None:
        if self.input_type is None:
            return
        for name, inp in self.inputs.items():
            if name in self.no_cast_list:
                continue
            if inp.dtype not in self.supported_cast_type:
                continue
            inp.convert_type_inplace(self.input_type)
        for name, weight in self.weights.items():
            if name in self.no_cast_list:
                continue
            if weight.dtype not in self.supported_cast_type:
                continue
            weight.convert_type_inplace(self.input_type)
        return self


def convert_to_dynamic_shape(dynamic_shape, name):
    if dynamic_shape.min_input_shape == {}:
        return tuple(dynamic_shape.min_input_shape)
    min_shape = tuple(dynamic_shape.min_input_shape[name])
    opt_shape = tuple(dynamic_shape.opt_input_shape[name])
    max_shape = tuple(dynamic_shape.max_input_shape[name])
    result_shape = []
    for i in range(len(min_shape)):
        if min_shape[i] == opt_shape[i] == max_shape[i]:
            result_shape.append(min_shape[i])
        else:
            result_shape.append(-1)
    return tuple(result_shape)


def create_fake_model(program_config, run_pir=False, dynamic_shape=None):
    '''Create a Paddle model(in memory) according to the given config.'''
    program_config = copy.deepcopy(program_config)
    program_config._cast()
    paddle.enable_static()
    with paddle.pir_utils.OldIrGuard():
        main_program_desc = core.ProgramDesc()
        # util_program = base.Program()
        util_program = paddle.static.Program()
        main_block_desc = main_program_desc.block(0)

        var_desc = main_block_desc.var(b"feed")
        var_desc.set_type(core.VarDesc.VarType.FEED_MINIBATCH)
        var_desc.set_persistable(True)

        index = 0
        for name, tensor_config in program_config.inputs.items():
            var_desc = main_block_desc.var(name.encode())
            var_desc.set_type(core.VarDesc.VarType.DENSE_TENSOR)
            var_desc.set_dtype(
                convert_np_dtype_to_proto_type(tensor_config.dtype)
            )
            if dynamic_shape is not None:
                dynamic_shape_copy = convert_to_dynamic_shape(
                    dynamic_shape, name
                )
                var_desc.set_shape(dynamic_shape_copy)
            else:
                var_desc.set_shape(tensor_config.shape)
            var_desc.set_need_check_feed(True)
            if tensor_config.lod is not None:
                var_desc.set_lod_level(len(tensor_config.lod))
            op_desc = main_block_desc._prepend_op()
            op_desc.set_type("feed")
            op_desc.set_input('X', ["feed"])
            op_desc.set_output('Out', [name])
            op_desc._set_attr("col", index)
            index = index + 1

        save_var_map = {}
        for name, tensor_config in program_config.weights.items():
            var_desc = main_block_desc.var(name.encode())
            var_desc.set_type(core.VarDesc.VarType.DENSE_TENSOR)
            var_desc.set_dtype(
                convert_np_dtype_to_proto_type(tensor_config.dtype)
            )
            var_desc.set_shape(tensor_config.shape)
            var_desc.set_persistable(True)

            save_var_map[name] = util_program.global_block().create_parameter(
                dtype=tensor_config.dtype,
                shape=tensor_config.shape,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                name=name,
                initializer=paddle.nn.initializer.Assign(tensor_config.data),
            )
        in_vars = []
        for name in sorted(save_var_map.keys()):
            in_vars.append(save_var_map[name])

        out_var = util_program.global_block().create_var(
            type=core.VarDesc.VarType.RAW, name="out_var_0"
        )
        out_var.desc.set_persistable(True)
        if not run_pir:
            util_program.global_block().append_op(
                type='save_combine',
                inputs={'X': in_vars},
                outputs={'Y': out_var},
                attrs={'file_path': '', 'save_to_memory': True},
            )
        for op_config in program_config.ops:
            op_desc = main_block_desc.append_op()
            op_desc.set_type(op_config.type)
            # canonicalize scalar attrs
            if OpProtoHolder.instance().has_op_proto(op_config.type):
                proto = OpProtoHolder.instance().get_op_proto(op_config.type)
                canonicalized_attrs = framework.canonicalize_attrs(
                    op_config.attrs, proto
                )
            else:
                canonicalized_attrs = op_config.attrs

            for name, values in op_config.inputs.items():
                op_desc.set_input(name, values)
            for name, values in canonicalized_attrs.items():
                if name == 'sub_block':
                    sub_block_desc = main_program_desc.append_block(
                        main_block_desc
                    )
                    values.fill_block_desc(sub_block_desc)
                    op_desc._set_attr(name, sub_block_desc)
                else:
                    op_desc._set_attr(name, values)
            for name, values in op_config.outputs.items():
                op_desc.set_output(name, values)
                for v in values:
                    if main_block_desc.has_var_recursive(v.encode()):
                        continue
                    var_desc = main_block_desc.var(v.encode())
                    var_desc.set_type(core.VarDesc.VarType.DENSE_TENSOR)
                    if (
                        op_config.outputs_var_type is not None
                        and v in op_config.outputs_var_type.keys()
                    ):
                        if (
                            op_config.outputs_var_type[v]
                            == VarType.DENSE_TENSOR_ARRAY
                        ):
                            var_desc.set_type(
                                core.VarDesc.VarType.DENSE_TENSOR_ARRAY
                            )
                        elif (
                            op_config.outputs_var_type[v] == VarType.STEP_SCOPES
                        ):
                            var_desc.set_type(core.VarDesc.VarType.STEP_SCOPES)
                            continue
                    if run_pir:
                        var_desc.set_dtype(
                            convert_np_dtype_to_proto_type(tensor_config.dtype)
                        )
                    else:
                        var_desc.set_dtype(
                            convert_np_dtype_to_proto_type(np.float32)
                        )
                    if (
                        op_config.outputs_dtype is not None
                        and v in op_config.outputs_dtype.keys()
                    ):
                        var_desc.set_dtype(
                            convert_np_dtype_to_proto_type(
                                op_config.outputs_dtype[v]
                            )
                        )
            if op_config.type not in _OP_WITHOUT_KERNEL_SET:
                op_desc.infer_var_type(main_block_desc)
                op_desc.infer_shape(main_block_desc)
            op_desc.check_attrs()

        for index, name in enumerate(program_config.outputs):
            var_desc = main_block_desc.var(b"fetch")
            var_desc.set_type(core.VarDesc.VarType.FETCH_LIST)
            var_desc.set_need_check_feed(True)
            op_desc = main_block_desc.append_op()
            op_desc.set_type("fetch")
            op_desc.set_input('X', [name])
            op_desc.set_output('Out', ["fetch"])
            op_desc._set_attr("col", index)
        util_program._sync_with_cpp()

    return main_program_desc, util_program


def create_quant_model(
    model,
    params,
    activation_quantize_type='moving_average_abs_max',
    weight_quantize_type='channel_wise_abs_max',
    save=False,
):
    place = paddle.CUDAPlace(0)
    scope = global_scope()
    exe = paddle.static.Executor(place)
    [
        inference_program,
        feed_target_names,
        fetch_targets,
    ] = paddle.static.io.load_inference_model(
        path_prefix=None,
        executor=exe,
        model_filename=model,
        params_filename=params,
    )
    graph = IrGraph(core.Graph(inference_program.desc), for_test=True)

    out_scale_op_list = [
        "conv2d",
        "depthwise_conv2d",
        "mul",
        "matmul",
        "relu",
        "leaky_relu",
        "relu6",
        "sigmoid",
        "tanh",
        "prelu",
        "swish",
        "softmax",
        "batch_norm",
        "layer_norm",
        "elementwise_add",
        "pool2d",
        "reshape2",
        "transpose2",
        "concat",
        "elementwise_mul",
        "scale",
        "slice",
        "hard_swish",
        "hard_sigmoid",
        "conv2d_transpose",
        "gru",
        "bilinear_interp",
        "nearest_interp",
        "trilinear_interp",
        "flatten",
        "flatten2",
        "transpose",
        "pad2d",
        "reshape",
        "layer_norm",
        "fusion_gru",
        "multi_gru",
        "quantize",
        "dequantize",
    ]
    op_real_in_out_name = {
        "conv2d": [["Input", "Filter"], ["Output"]],
        "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
        "conv2d_transpose": [["Input", "Filter"], ["Output"]],
        "mul": [["X", "Y"], ["Out"]],
        "matmul": [["X", "Y"], ["Out"]],
        "pool2d": [["X"], ["Out"]],
        "elementwise_add": [["X", "Y"], ["Out"]],
        "concat": [["X"], ["Out"]],
        "softmax": [["X"], ["Out"]],
        "argmax": [["X"], ["Out"]],
        "transpose": [["X"], ["Out"]],
        "equal": [["X", "Y"], ["Out"]],
        "gather": [["X"], ["Out"]],
        "greater_equal": [["X", "Y"], ["Out"]],
        "greater_than": [["X", "Y"], ["Out"]],
        "less_equal": [["X", "Y"], ["Out"]],
        "less_than": [["X", "Y"], ["Out"]],
        "mean": [["X"], ["Out"]],
        "not_equal": [["X", "Y"], ["Out"]],
        "reshape": [["X"], ["Out"]],
        "reshape2": [["X"], ["Out"]],
        "transpose2": [["X"], ["Out"]],
        "bilinear_interp": [["X"], ["Out"]],
        "nearest_interp": [["X"], ["Out"]],
        "trilinear_interp": [["X"], ["Out"]],
        "slice": [["Input"], ["Out"]],
        "squeeze": [["X"], ["Out"]],
        "elementwise_sub": [["X", "Y"], ["Out"]],
        "relu": [["X"], ["Out"]],
        "relu6": [["X"], ["Out"]],
        "leaky_relu": [["X"], ["Out"]],
        "prelu": [["X"], ["Out"]],
        "tanh": [["X"], ["Out"]],
        "swish": [["X"], ["Out"]],
        "dropout": [["X"], ["Out"]],
        "batch_norm": [["X"], ["Y"]],
        "layer_norm": [["X"], ["Y"]],
        "sigmoid": [["X"], ["Out"]],
        "elementwise_mul": [["X", "Y"], ["Out"]],
        "scale": [["X"], ["Out"]],
        "hard_swish": [["X"], ["Out"]],
        "hard_sigmoid": [["X"], ["Out"]],
        "gru": [["Input", "Weight"], ["Hidden"]],
        "lstm": [["Input", "Weight"], ["Hidden"]],
        "pad2d": [["X"], ["Out"]],
        "flatten": [["X"], ["Out"]],
        "flatten2": [["X"], ["Out"]],
        "fusion_gru": [["X", "WeightX", "WeightH"], ["Hidden", "XX"]],
        "multi_gru": [["X", "WeightX", "WeightH"], ["Hidden"]],
        "quantize": [["Input"], ["Output"]],
        "dequantize": [["Input"], ["Output"]],
    }

    def _get_op_output_var_names(op):
        """ """
        assert isinstance(
            op, (IrNode, Operator)
        ), "The input op should be IrNode or Operator."
        var_names = []
        op_name = op.name() if isinstance(op, IrNode) else op.type
        if op_name not in op_real_in_out_name:
            return []

        name_list = op_real_in_out_name[op_name][1]
        for name in name_list:
            var_name = op.output(name)
            if isinstance(var_name, list):
                var_names.extend(var_name)
            else:
                var_names.append(var_name)
        return var_names

    transform_pass = QuantizationTransformPass(
        scope=scope,
        place=place,
        activation_quantize_type=activation_quantize_type,
        weight_quantize_type=weight_quantize_type,
    )
    transform_pass.apply(graph)

    op_nodes = graph.all_op_nodes()
    for op_node in op_nodes:
        if op_node.name() in out_scale_op_list:
            var_names = _get_op_output_var_names(op_node)
            for var_name in var_names:
                in_node = graph._find_node_by_name(op_node.outputs, var_name)
                if in_node.dtype() not in [
                    core.VarDesc.VarType.FP64,
                    core.VarDesc.VarType.FP32,
                ]:
                    continue

                op_node.op()._set_attr("out_threshold", 3.0)

    # Freeze graph for inference, but the weight of fc/conv is still float type.
    freeze_pass = QuantizationFreezePass(
        scope=scope, place=place, weight_quantize_type=weight_quantize_type
    )
    freeze_pass.apply(graph)

    main_program = graph.to_program()

    # modify fake_quantize_moving_average_abs_max(InScale) and fake_channel_wise_dequantize_max_abs(Scales)
    op_nodes = graph.all_op_nodes()
    for op_node in op_nodes:
        if op_node.name() == 'fake_quantize_moving_average_abs_max':
            var_name = op_node.input("InScale")[0]
            tensor = scope.var(var_name).get_tensor()
            tensor.set(np.array([1], dtype=np.float32), place)
        elif op_node.name() == 'fake_channel_wise_dequantize_max_abs':
            var_name = op_node.input("Scales")[0]
            tensor = scope.var(var_name).get_tensor()
            tensor.set(np.ones(tensor.shape(), dtype=np.float32), place)

    feed_vars = [
        main_program.global_block().var(name) for name in feed_target_names
    ]

    if save:
        paddle.static.io.save_inference_model(
            'test_inference_model',
            feed_vars,
            fetch_targets,
            exe,
            program=main_program,
        )

    serialized_program = paddle.static.serialize_program(
        feed_vars, fetch_targets, program=main_program
    )
    serialized_params = paddle.static.serialize_persistables(
        feed_vars, fetch_targets, executor=exe, program=main_program
    )
    return serialized_program, serialized_params
