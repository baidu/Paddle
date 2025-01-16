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


import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    cast_tensor,
    replenish_layer_and_output,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register(
    "pd_op.one_hot", trt_version="trt_version_ge=8.5.1"
)
def one_hot_converter(network, paddle_op, inputs):
    input_tensor, num_classes_tensor = inputs

    input_type = input_tensor.dtype

    trt_dtype_map = {
        trt.DataType.INT32: trt.int32,
    }
    trt_dtype = trt_dtype_map.get(input_type, None)

    trt_dtype = trt_dtype_map[input_type]

    if trt_dtype == trt.int32:
        values_data = [0, 1]
        np_dtype = np.int32
    # trt version>10  support int64
    elif trt_dtype == trt.int64:
        values_data = [0, 1]
        np_dtype = np.int64
    else:
        raise ValueError(f"Unsupported trt_dtype for one_hot: {trt_dtype}")

    values_tensor = add_1D_constant_layer(network, values_data, dtype=np_dtype)

    if isinstance(num_classes_tensor, trt.Weights):
        num_classes_tensor = network.add_constant(
            paddle_op.operands()[1].source().shape, num_classes_tensor
        ).get_output(0)

    reshape_layer = network.add_shuffle(num_classes_tensor)
    reshape_layer.reshape_dims = ()
    depth_tensor = reshape_layer.get_output(0)

    depth_tensor = cast_tensor(network, depth_tensor, trt.int32)

    one_hot_layer = network.add_one_hot(
        input_tensor, values_tensor, depth_tensor, axis=-1
    )
    one_hot_layer.set_output_type(0, trt_dtype)
    output_tensor = one_hot_layer.get_output(0)

    replenish_layer_and_output(
        one_hot_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return [output_tensor]
