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


from paddle.tensorrt.converter_utils import (
    convert_conv2d,
    convert_conv3d,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.depthwise_conv2d", trt_version="8.x")
@converter_registry.register("pd_op.conv2d", trt_version="trt_version_ge=8.0")
@converter_registry.register(
    "pd_op.fused_conv2d_add_act", trt_version="trt_version_ge=8.0"
)
@converter_registry.register("pd_op.conv2d_transpose", trt_version="8.x")
@converter_registry.register(
    "pd_op.depthwise_conv2d_transpose", trt_version="8.x"
)
def conv2d_converter(network, paddle_op, inputs):
    return convert_conv2d(network, paddle_op, inputs)


@converter_registry.register("pd_op.conv3d_transpose", trt_version="8.x")
@converter_registry.register("pd_op.conv3d", trt_version="8.x")
def conv3d_converter(network, paddle_op, inputs):
    return convert_conv3d(network, paddle_op, inputs)
