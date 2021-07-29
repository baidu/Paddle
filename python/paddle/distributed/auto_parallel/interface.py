#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.distributed_attribute import TensorDistributedAttribute
from paddle.fluid.distributed_attribute import get_default_distributed_config


def validate_check():
    pass


def shard_tensor(tensor, mesh, dims_mapping):
    """
    Add distributed attributes for tensors.
    Inputs:
        tensor (Variable): tensor to process， it's an instance of Variable (framework.py)
        mesh (ProcessMesh): an instance of ProcessMesh
        dims_mapping (list): a list to describe the mapping between tensor shape and mesh topology
    Returns:
        The tensor itself.
    """
    validate_check()
    assert len(tensor.shape) == len(dims_mapping)
    dist_config = get_default_distributed_config()
    tensor_dist_attr = dist_config.get_tensor_distributed_attr_program(
        tensor.desc)
    if tensor_dist_attr is None:
        tensor_dist_attr = TensorDistributedAttribute(tensor.desc)
        dist_config.set_tensor_distributed_attr_program(tensor.desc,
                                                        tensor_dist_attr)
    tensor_dist_attr.set_process_mesh(mesh, is_annotated=True)
    tensor_dist_attr.set_dims_mapping(dims_mapping, is_annotated=True)
    return tensor


def set_shard_mask(tensor, mask):
    """
    Set the mask for a tensor which mask out the tensor from some processes in its mesh.
    Inputs:
        tensor (Variable): tensor to process， it's an instance of Variable (framework.py)
        mask (Variable): mask out tensor from some processes in mesh.
    Returns:
        The tensor itself.
    """
    validate_check()
    dist_config = get_default_distributed_config()
    tensor_dist_attr = dist_config.get_tensor_distributed_attr_program(
        tensor.desc)
    if tensor_dist_attr is None:
        tensor_dist_attr = TensorDistributedAttribute(tensor.desc)
        dist_config.set_tensor_distributed_attr_program(tensor.desc,
                                                        tensor_dist_attr)
    tensor_dist_attr.set_shard_mask(mask, is_annotated=True)
    return tensor


def shard_op(op_name, mesh, inputs_dims_mapping, outputs_dims_mapping):
    """
    Add distributed attributes for ops.
    Inputs:
        op_name (string): the name of the  op to process
        mesh (ProcessMesh): an instance of ProcessMesh
        inputs_dims_mapping (dict): a mapping from input name to the input's dims_mapping
        outputs_dims_mapping(dict): a mapping from output name to the output's dims_mapping
    Returns:
        Output variables of the op named op_name(tuple).
    """
    validate_check()
    pass


def set_offload_device(tensor, offload_device):
    """
    Set the device that the tensor on.
    Inputs:
        op (tensor): tensor to process, it's an instance of Variable (framework.py)
        offload_device: the device that the tensor on, e.g., 'gpu', 'cpu'.
    Returns:
        None.
    """
    validate_check()
    dist_config = get_default_distributed_config()
    tensor_dist_attr = dist_config.get_tensor_distributed_attr_program(
        tensor.desc)
    if tensor_dist_attr is None:
        tensor_dist_attr = TensorDistributedAttribute(tensor.desc)
        dist_config.set_tensor_distributed_attr_program(tensor.desc,
                                                        tensor_dist_attr)
    tensor_dist_attr.set_offload_device(offload_device, is_annotated=True)
    return tensor


def set_pipeline_stage(stage):
    """
    Set the pipeline stage of the following ops.
    Inputs:
        stage: the pipeline stage the following ops belonging to
    Returns:
        None.
    """
    global _CURRENT_PIPELINE_STAGE_
    _CURRENT_PIPELINE_STAGE_ = stage
