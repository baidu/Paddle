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

from .base_reshard_func import register_reshard_func
from .global_to_sub_mesh_func import GlobaleToSubMeshFunction
from .nd_mesh_reshard_func import (
    NdMeshReshardFunction,
    NdMeshReshardFunctionCrossMesh,
)
from .p_to_r_reshard_func import (
    PToRReshardFunction,
    PToRReshardFunctionCrossMesh,
)
from .p_to_s_reshard_func import (
    PToSReshardFunction,
)
from .r_to_p_reshard_func import RToPReshardFunction
from .r_to_s_reshard_func import (
    RToSReshardFunction,
    RToSReshardFunctionCrossMesh,
)
from .s_to_r_reshard_func import (
    SToRReshardFunction,
    SToRReshardFunctionCrossMesh,
)
from .s_to_s_reshard_func import SToSReshardFunction
from .same_status_reshard_func import SameStatusReshardFunction
from .sub_to_global_mesh_func import SubToGlobalMeshFunction


def register_reshard_funcs():
    register_reshard_func(PToRReshardFunction())
    register_reshard_func(PToRReshardFunctionCrossMesh())
    register_reshard_func(PToSReshardFunction())
    register_reshard_func(RToSReshardFunction())
    register_reshard_func(RToSReshardFunctionCrossMesh())
    register_reshard_func(RToPReshardFunction())
    register_reshard_func(SameStatusReshardFunction())
    register_reshard_func(SToRReshardFunction())
    register_reshard_func(SToRReshardFunctionCrossMesh())
    register_reshard_func(NdMeshReshardFunction())
    register_reshard_func(NdMeshReshardFunctionCrossMesh())
    register_reshard_func(GlobaleToSubMeshFunction())
    register_reshard_func(SubToGlobalMeshFunction())
    register_reshard_func(SToSReshardFunction())


register_reshard_funcs()
