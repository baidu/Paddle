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

import logging
import os

import numpy as np

import paddle

try:
    import tensorrt as trt
except Exception as e:
    pass
from paddle import pir
from paddle.base.log_helper import get_logger
from paddle.pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def map_dtype(pd_dtype):
    version_list = get_trt_version_list()
    if pd_dtype == "FLOAT32":
        return trt.float32
    elif pd_dtype == "FLOAT16":
        return trt.float16
    elif pd_dtype == "INT32":
        return trt.int32
    elif pd_dtype == "INT8":
        return trt.int8
    elif pd_dtype == "BOOL":
        return trt.bool
    # trt version<10.0 not support int64,so convert int64 to int32
    elif pd_dtype == "INT64":
        return trt.int64 if version_list[0] >= 10 else trt.int32
    # Add other dtype mappings as needed
    else:
        raise TypeError(f"Unsupported dtype: {pd_dtype}")


def all_ops_into_trt(program):
    for op in program.global_block().ops:
        if (
            op.name() == "pd_op.fetch"
            or op.name() == "pd_op.data"
            or op.name().split('.')[0] == "builtin"
        ):
            continue
        if op.has_attr("__l_trt__") is False:
            return False
        if op.attrs()["__l_trt__"] is False:
            return False
    _logger.info("All ops convert to trt.")
    return True


def run_pir_pass(program, disable_passes=[], scope=None):
    def _add_pass_(pm, passes, disable_passes):
        for pass_item in passes:
            for pass_name, pass_attr in pass_item.items():
                if pass_name in disable_passes:
                    continue
                pm.add_pass(pass_name, pass_attr)

    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    if scope is None:
        scope = paddle.static.global_scope()
    place = paddle.CUDAPlace(0)

    # run marker pass
    passes = [
        {'trt_op_marker_pass': {}},
    ]
    _add_pass_(pm, passes, disable_passes)
    pm.run(program)

    # run other passes
    pm.clear()
    passes = []
    if all_ops_into_trt(program):
        # only run constant_folding_pass when all ops into trt
        passes.append(
            {
                'constant_folding_pass': {
                    "__place__": place,
                    "__param_scope__": scope,
                }
            }
        )

        passes.append({'conv2d_add_fuse_pass': {}})
    passes.append({'trt_op_marker_pass': {}})  # for op that created by pass
    _add_pass_(pm, passes, disable_passes)
    pm.run(program)

    # delete unused op
    for op in program.global_block().ops:
        if op.name() == "builtin.constant" or op.name() == "builtin.parameter":
            if op.results()[0].use_empty():
                program.global_block().remove_op(op)

    return program


def run_trt_partition(program):
    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    pm.add_pass("trt_sub_graph_extract_pass", {})
    pm.run(program)
    return program


def forbid_op_lower_trt(program, disabled_ops):
    if isinstance(disabled_ops, str):
        disabled_ops = [disabled_ops]
    for op in program.global_block().ops:
        if op.name() in disabled_ops:
            op.set_bool_attr("__l_trt__", False)


def enforce_op_lower_trt(program, op_name):
    for op in program.global_block().ops:
        if op.name() == op_name:
            op.set_bool_attr("__l_trt__", True)


def predict_program(program, feed_data, fetch_var_list, scope=None):
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            place = paddle.CUDAPlace(0)
            executor = paddle.static.Executor(place)
            output = executor.run(
                program,
                feed=feed_data,
                fetch_list=fetch_var_list,
                scope=scope,
            )
            return output


def warmup_shape_infer(
    program, min_shape_feed, opt_shape_feed, max_shape_feed, scope=None
):
    paddle.framework.set_flags({"FLAGS_enable_collect_shape": True})
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            executor = paddle.static.Executor()
            # Run the program with input_data
            for _ in range(1):
                executor.run(program, feed=min_shape_feed, scope=scope)

            for _ in range(1):
                executor.run(program, feed=opt_shape_feed, scope=scope)

            # Run the program with input_data_max_shape (fake max_shape input)
            for _ in range(1):
                executor.run(program, feed=max_shape_feed, scope=scope)

            exe_program, _, _ = (
                executor._executor_cache.get_pir_program_and_executor(
                    program,
                    feed=max_shape_feed,
                    fetch_list=None,
                    feed_var_name='feed',
                    fetch_var_name='fetch',
                    place=paddle.framework._current_expected_place_(),
                    scope=scope,
                    plan=None,
                )
            )
    paddle.framework.set_flags({"FLAGS_enable_collect_shape": False})

    return exe_program


def get_trt_version_list():
    version = trt.__version__
    return list(map(int, version.split('.')))


# Adding marker labels to builtin ops facilitates convert processing, but they ultimately do not enter the TensorRT subgraph.
def mark_builtin_op(program):
    for op in program.global_block().ops:
        if op.name() == "builtin.split":
            defining_op = op.operands()[0].source().get_defining_op()
            if defining_op is not None:
                if (
                    defining_op.has_attr("__l_trt__")
                    and defining_op.attrs()["__l_trt__"]
                ):
                    op.set_bool_attr("__l_trt__", True)
        if op.name() == "builtin.combine":
            defining_op = op.results()[0].all_used_ops()[0]
            if defining_op is not None:
                if (
                    defining_op.has_attr("__l_trt__")
                    and defining_op.attrs()["__l_trt__"]
                ):
                    op.set_bool_attr("__l_trt__", True)


class TensorRTConfigManager:
    _instance = None

    def __new__(cls, trt_config=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.trt_config = trt_config
        return cls._instance

    def _init(self, trt_config=None):
        self.trt_config = trt_config

    def get_precision_mode(self):
        if self.trt_config and self.trt_config.precision_mode:
            return self.trt_config.precision_mode
        return None

    def get_force_fp32_ops(self):
        if self.trt_config and self.trt_config.ops_run_float:
            return self.trt_config.ops_run_float
        return []


class TensorRTConstantManager:
    _instance = None

    def __new__(cls, trt_config=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.constant_dict = {}
        return cls._instance

    def set_constant_value(self, name, tensor_data, value):
        out_dtype = np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[value.dtype])
        if out_dtype == np.dtype("float64"):
            out_dtype = np.dtype("float32")
        if out_dtype == np.dtype("int64"):
            out_dtype = np.dtype("int32")
        constant_array = np.array(tensor_data, dtype=out_dtype)
        self.constant_dict.update({name: constant_array})

    def get_constant_value(self, name):
        return self.constant_dict[name]


# In TensorRT FP16 inference, this function sets the precision of specific
# operators to FP32, ensuring numerical accuracy for these operations.
def support_fp32_mix_precision(op_type, layer, trt_config=None):
    trt_manager = TensorRTConfigManager()
    force_fp32_ops = trt_manager.get_force_fp32_ops()
    if op_type in force_fp32_ops:
        layer.reset_precision()
        layer.precision = trt.DataType.FLOAT


def weight_to_tensor(network, paddle_value, trt_tensor, use_op_name):
    # the following op needn't cast trt.Weight to ITensor, because the layer need weight as input
    forbid_cast_op = [
        "pd_op.depthwise_conv2d",
        "pd_op.conv2d",
        "pd_op.conv2d_transpose",
        "pd_op.batch_norm",
        "pd_op.batch_norm_",
        "pd_op.layer_norm",
        "pd_op.depthwise_conv2d_transpose",
        "pd_op.fused_conv2d_add_act",
        "pd_op.affine_channel",
        "pd_op.fused_bias_dropout_residual_layer_norm",
        "pd_op.deformable_conv",
    ]
    if use_op_name in forbid_cast_op:
        return trt_tensor
    input_shape = paddle_value.shape
    if type(trt_tensor) == trt.Weights:
        return network.add_constant(input_shape, trt_tensor).get_output(0)
    return trt_tensor


def zero_dims_to_one_dims(network, trt_tensor):
    if trt_tensor is None:
        return None
    if type(trt_tensor) == trt.Weights:
        return trt_tensor
    if len(trt_tensor.shape) != 0:
        return trt_tensor
    shuffle_layer = network.add_shuffle(trt_tensor)
    shuffle_layer.reshape_dims = (1,)
    return shuffle_layer.get_output(0)


# We use a special rule to judge whether a paddle value is a shape tensor.
# The rule is consistent with the rule in C++ source code(collect_shape_manager.cc).
# We use the rule for getting min/max/opt value shape from collect_shape_manager.
# We don't use trt_tensor.is_shape_tensor, because sometimes, the trt_tensor that corresponding to paddle value is not a shape tensor
# when it is a output in this trt graph, but it is a shape tensor when it is a input in next trt graph.
def is_shape_tensor(value):
    dims = value.shape
    total_elements = 1
    if (
        dims.count(-1) > 1
    ):  # we can only deal with the situation that is has one dynamic dims
        return False
    for dim in dims:
        total_elements *= abs(dim)  # add abs for dynamic shape -1
    is_int_dtype = value.dtype == paddle.int32 or value.dtype == paddle.int64
    return total_elements <= 8 and total_elements >= 1 and is_int_dtype


def get_cache_path():
    home_path = os.path.expanduser("~")
    cache_path = os.path.join(home_path, ".pp_trt_cache")

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    return cache_path


def remove_duplicate_value(value_list):
    ret_list = []
    ret_list_id = []
    for value in value_list:
        if value.id not in ret_list_id:
            ret_list.append(value)
            ret_list_id.append(value.id)
    return ret_list


def get_trt_version():
    return trt.__version__
