#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import logging

from paddle.fluid import core
from paddle.fluid.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

# lookup_table fp16 is slower than fp32, though fp16 is supported.
_extra_unsupported_list = {
    'lookup_table',
    'lookup_table_v2',
    'scatter',
    'scatter_grad',
}


def check_amp_dtype(dtype):
    """
    Check amp_dtype: float16 or bfloat16
    """
    if isinstance(dtype, str):
        dtype = dtype.lower()
    if dtype not in ['float16', 'bfloat16']:
        raise ValueError(
            "If enable AMP, dtype should be 'float16' or 'bfloat16'."
        )
    return dtype


def get_low_precision_vartype(dtype):
    if isinstance(dtype, core.VarDesc.VarType):
        return dtype
    elif isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "float16":
            var_type = core.VarDesc.VarType.FP16
        elif dtype == "bfloat16":
            var_type = core.VarDesc.VarType.BF16
        else:
            raise ValueError(
                "If enable AMP, dtype should be 'float16' or 'bfloat16'."
            )
        return var_type
    else:
        raise TypeError(
            "The type of dtype is expected to be string or core.VarDesc.VarType, but recieved {}.".format(
                type(dtype)
            )
        )


def get_low_precision_dtypestr(dtype):
    if isinstance(dtype, str):
        return check_amp_dtype(dtype)
    elif isinstance(dtype, core.VarDesc.VarType):
        if dtype == core.VarDesc.VarType.FP16:
            return "float16"
        elif dtype == core.VarDesc.VarType.BF16:
            return "bfloat16"
        else:
            raise ValueError(
                "If enable AMP, dtype should be core.VarDesc.VarType.FP16 or core.VarDesc.VarType.BF16."
            )
    else:
        raise TypeError(
            "The type of dtype is expected to be string or core.VarDesc.VarType, but recieved {}.".format(
                type(dtype)
            )
        )


def _get_sys_unsupported_list(dtype):
    var_type = get_low_precision_vartype(dtype)

    # The set of ops that don't support fp16 calculation
    device = None
    if core.is_compiled_with_xpu():
        device = 'XPU'
    elif core.is_compiled_with_custom_device('npu'):
        device = 'NPU'
    else:
        device = 'GPU'
    _, _, sys_unsupported_list = core.op_supported_infos(device, var_type)
    return device, sys_unsupported_list


def _get_unsupported_list(dtype):
    # The set of ops that don't support fp16 calculation
    _, _sys_unsupported_list = _get_sys_unsupported_list(dtype)
    unsupported_list = _extra_unsupported_list | _sys_unsupported_list
    return unsupported_list


class AutoMixedPrecisionLists:
    """
    AutoMixedPrecisionLists is a class for black/white list. It can update
    pre-defined black list and white list according to users' custom black
    white lists. The lists are used for an algorithm which determines op's
    execution mode (fp32, fp16 or bf16).

    Args:
        custom_white_list (set): Users' custom white list.
        custom_black_list (set): Users' custom black list.
        custom_black_varnames (set): Users' custom black varibles' names.
        dtype (str): the low precision dtype, which can be set to 'float16' or 'bfloat16'.
    """

    def __init__(
        self,
        custom_white_list=None,
        custom_black_list=None,
        custom_black_varnames=None,
        dtype="float16",
    ):
        self.amp_dtype = check_amp_dtype(dtype)
        self._custom_white_list = custom_white_list
        self._custom_black_list = custom_black_list
        self.white_list = copy.copy(white_list)
        self.black_list = copy.copy(black_list)
        self.gray_list = copy.copy(gray_list)
        self.unsupported_list = copy.copy(_get_unsupported_list(self.amp_dtype))
        self.black_varnames = copy.copy(custom_black_varnames)
        self._update_list()

    def _update_list(self):
        """
        Update black and white list according to users' custom list.
        """
        if self._custom_white_list and self._custom_black_list:
            for op_name in self._custom_white_list:
                if op_name in self._custom_black_list:
                    raise ValueError(
                        f"The given custom_white_list overlaps custom_black_list with < {op_name} >!"
                    )
        if self._custom_white_list:
            for op_name in self._custom_white_list:
                if op_name in self.black_list:
                    self.black_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.white_list.add(op_name)
                if op_name in _extra_unsupported_list:
                    self.unsupported_list.remove(op_name)
        if self._custom_black_list:
            for op_name in self._custom_black_list:
                if op_name in self.white_list:
                    self.white_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.black_list.add(op_name)
                self.unsupported_list.add(op_name)
        device, sys_unsupported_list = _get_sys_unsupported_list(self.amp_dtype)
        actual_unsupported_list = []
        for op_name in sys_unsupported_list:
            if op_name in self.white_list:
                actual_unsupported_list.append(op_name)
        if len(actual_unsupported_list) > 0:
            _logger.warning(
                f"On current {device}, {self.amp_dtype} is not supported for operators < {actual_unsupported_list} > in white_list!"
            )


# The three sets listed below are changed dynamiclly. They don't contain all
# paddle ops currently.

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.
white_list = {
    'conv2d',
    'matmul',
    'matmul_v2',
    'mul',
}

# The set of ops that support fp16 calculation and are considered numerically-
# dangerous and whose effects may also be observed in downstream ops.
black_list = {
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'c_softmax_with_cross_entropy',
    'cross_entropy',
    'cross_entropy2',
    # fp16 is slower than fp32, though fp16 is supported.
    'lookup_table',
    'lookup_table_v2',
    'linear_interp_v2',
    'nearest_interp_v2',
    'bilinear_interp_v2',
    'bicubic_interp_v2',
    'trilinear_interp_v2',
    # default fp32 can avoid return inf when the sum value large than 65504
    'reduce_sum',
}

# This set contains two types of ops. All ops supported fp16 calculation. One
# of two types is considered numerically-safe, but may be made unsafe by an
# upstream blacklist op. Another type do not have numerically-significant
# effects, like stack, flatten2.
gray_list = {
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'batch_norm',
    'layer_norm',
    'tanh',
    'sigmoid',
    'top_k',
    'pool2d',
    'pool3d',
    'dropout',
    'relu',
    'relu6',
    'leaky_relu',
    'soft_relu',
    'flatten2',
    'stack',
    'unstack',
    'uniform_random',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'gaussian_random_batch_size_like',
    'slice',
    'rank',
    'scale',
    'transpose2',
    'reshape2',
    'gather',
    'fill_constant',
    'get_tensor_from_selected_rows',
    'sign',
    'cast',
    'fused_bn_add_activation',
    'c_identity',
    'c_concat',
    'c_allreduce_sum',
    'concat',
    'split',
    'fused_feedforward',
    'fused_attention',
    'fused_multi_transformer',
}

<<<<<<< HEAD
=======
# The set of ops that don't support fp16 calculation
# lookup_table fp16 is slower than fp32, though fp16 is supported.
_sys_unsupported_fp16_list = []
if core.is_compiled_with_xpu():
    _, _, _sys_unsupported_fp16_list = core.op_supported_infos(
        'XPU', core.VarDesc.VarType.FP16
    )
elif core.is_compiled_with_custom_device('npu'):
    _, _, _sys_unsupported_fp16_list = core.op_supported_infos(
        'NPU', core.VarDesc.VarType.FP16
    )
else:
    _, _, _sys_unsupported_fp16_list = core.op_supported_infos(
        'GPU', core.VarDesc.VarType.FP16
    )

supported_fp16_list = {
    "conditional_block_grad",
    "conditional_block",
    "conditional_block_infer",
    "select_input",
    "while",
    "while_grad",
    "cast",
    "tensor_array_to_tensor",
    "lod_array_length",
    "write_to_array",
}

unsupported_fp16_list = (
    _extra_unsupported_fp16_list | _sys_unsupported_fp16_list
) - supported_fp16_list

>>>>>>> unify o1 and o2
CustomOpLists = AutoMixedPrecisionLists
