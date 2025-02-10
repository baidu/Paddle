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

# import tensorrt as trt

# from paddle.tensorrt.converter_utils import get_axes_for_reduce_op
# from paddle.tensorrt.register import converter_registry


# @converter_registry.register("pd_op.mean", trt_version="trt_version_ge=8.0")
# def mean_converter(network, paddle_op, inputs):
#     input_tensor = inputs[0]
#     keep_dim = paddle_op.attrs().get("keepdim")
#     dim = paddle_op.attrs().get("axis")

#     mean_layer = network.add_reduce(
#         input_tensor,
#         trt.ReduceOperation.AVG,
#         axes=get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
#         keep_dims=keep_dim,
#     )
#     return mean_layer.get_output(0)
