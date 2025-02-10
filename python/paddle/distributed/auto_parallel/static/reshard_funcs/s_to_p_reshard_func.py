# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from .base_reshard_func import ReshardFunction, is_partial, is_shard
from .r_to_p_reshard_func import RToPReshardFunction
from .s_to_r_reshard_func import SToRReshardFunction
from .same_status_reshard_func import SameStatusReshardFunction


class SToPReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_partial(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh.ndim != 1:
            return False
        if out_mesh.ndim != 1:
            return False
        if in_mesh != out_mesh:
            return False
        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        # step 1, create tmp dist attr and tmp dist tensor
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            dst_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        tmp_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            dst_type, tmp_dist_attr
        )
        # step 2, do s to r reshard on `in` to `tmp`
        s_to_r = SToRReshardFunction()
        tmp_tensor = s_to_r.reshard(
            src_dist_attr, tmp_dist_attr, src_value, tmp_type
        )
        print(f"tmp_tensor is {tmp_tensor}")
        # step 3, do r to p reshard on `tmp` to `dst_value`
        r_to_p = RToPReshardFunction()
        dst_value = r_to_p.reshard(
            tmp_dist_attr, dst_dist_attr, tmp_tensor, dst_type
        )
        print(f"dst_value is {dst_value}")
        return dst_value


class SToPReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_partial(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if (
            in_mesh.ndim != 1
            or out_mesh.ndim != 1
            or in_mesh.shape != out_mesh.shape
        ):
            return False

        if in_mesh == out_mesh:
            return False

        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        same_status_func = SameStatusReshardFunction()
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            src_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        tmp_dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            src_value.type(), tmp_dist_attr
        )
        out_value = same_status_func.reshard(
            src_dist_attr, tmp_dist_attr, src_value, tmp_dst_type
        )

        s_to_p_func = SToPReshardFunction()
        assert s_to_p_func.is_suitable(
            tmp_dist_attr, dst_dist_attr
        ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
        return s_to_p_func.reshard(
            tmp_dist_attr, dst_dist_attr, out_value, dst_type
        )
