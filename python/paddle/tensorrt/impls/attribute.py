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

import tensorrt as trt

from paddle.tensorrt.converter_utils import replenish_layer_and_output
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.shape", trt_version="trt_version_ge=8.0")
def shape_converter(network, paddle_op, inputs):
    version = trt.__version__
    version_list = list(map(int, version.split('.')))
    shape_layer = network.add_shape(inputs[0])
    if version_list[0] >= 10:
        identity_layer = network.add_identity(shape_layer.get_output(0))
        identity_layer.set_output_type(0, trt.int32)
        identity_layer.get_output(0).dtype = trt.int32
        replenish_layer_and_output(
            identity_layer, paddle_op.name(), paddle_op.get_output_names()
        )
        return identity_layer.get_output(0)
    replenish_layer_and_output(
        shape_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return shape_layer.get_output(0)


@converter_registry.register("pd_op.shape64", trt_version="trt_version_ge=8.0")
def shape64_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    shape_layer = network.add_shape(input_tensor)
    replenish_layer_and_output(
        shape_layer, paddle_op.name(), paddle_op.get_output_names()
    )
    return shape_layer.get_output(0)
