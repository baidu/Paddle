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
# limitations under the License

import copy
import logging
import os
import threading
import warnings
from functools import reduce

import numpy as np

import paddle
from paddle.base.framework import use_pir_api
from paddle.base.libpaddle import pir
from paddle.base.wrapped_decorator import (
    wrap_decorator,
)
from paddle.framework import core
from paddle.framework.io_utils import is_belong_to_optimizer, is_parameter
from paddle.static import Variable

from ..process_mesh import ProcessMesh, merge_process_meshes
from .dist_attribute import DistTensorSpec, OperatorDistAttr, TensorDistAttr

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()

__no_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.DENSE_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]

__not_naive_data_parallel_op__ = ["expand_v2"]
_g_gradient_clip_ops = [
    "sum",
    "sqrt",
    "fill_constant",
    "elementwise_max",
    "elementwise_div",
    "stack",
    "reduce_sum",
]

partition_skip_op_list = [
    "builtin.combine",
    "builtin.split",
    "pd_op.pylayer",
    "cf.yield",
    "cf.tuple_push",
    "cf.tuple_pop",
    "cf.stack_create",
    "cf.has_elements",
]


def get_logger(log_level, name="auto_parallel"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(log_level)
        log_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
        )
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
    else:
        logger.setLevel(log_level)
    return logger


def is_valid_list_index(list, index):
    if index >= -len(list) and index < len(list):
        return True
    else:
        return False


def is_dim_shard(mapping):
    if mapping != -1:
        return True
    else:
        return False


def is_dim_replicate(mapping):
    if mapping == -1:
        return True
    else:
        return False


def verify_dims_mapping(dims_mapping, process_mesh):
    if dims_mapping is None:
        return False
    if not all(isinstance(d, int) for d in dims_mapping):
        return False
    for i in range(len(dims_mapping)):
        if dims_mapping[i] < -1 or dims_mapping[i] >= len(process_mesh.shape):
            return False
    for i in range(len(process_mesh.shape)):
        if dims_mapping.count(i) > 1:
            return False
    return True


def convert_to_dims_mapping(shard_spec, process_mesh):
    dims_mapping = []
    for shard in shard_spec:
        if shard is None:
            dims_mapping.append(-1)
        elif process_mesh.shape[process_mesh.dim_names.index(shard)] == 1:
            dims_mapping.append(-1)
        else:
            dims_mapping.append(process_mesh.dim_names.index(shard))
    return dims_mapping


def convert_to_shard_spec(dims_mapping, process_mesh):
    shard_spec = []
    for dim_mapping in dims_mapping:
        if dim_mapping == -1:
            shard_spec.append(None)
        else:
            shard_spec.append(process_mesh.dim_names[dim_mapping])
    return shard_spec


def verify_shard_spec(shard_spec, tensor_shape, process_mesh):
    if len(shard_spec) != len(tensor_shape):
        return False
    for shard in shard_spec:
        if shard is not None and not isinstance(shard, str):
            return False
        if shard is not None and shard not in process_mesh.dim_names:
            return False
    dims_mapping = convert_to_dims_mapping(shard_spec, process_mesh)
    if not verify_dims_mapping(dims_mapping, process_mesh):
        return False
    for i in range(len(tensor_shape)):
        if (
            dims_mapping[i] != -1
            and tensor_shape[i] > 0
            and tensor_shape[i] % process_mesh.shape[dims_mapping[i]] != 0
        ):
            return False
    return True


def compute_compatible_dim_mapping(dim_mappings):
    if not dim_mappings:
        return None
    compatible_mapping = dim_mappings[0]
    for mapping in dim_mappings:
        if compatible_mapping == -1:
            compatible_mapping = mapping
        elif mapping == -1:
            continue
        elif compatible_mapping == mapping:
            continue
        else:
            return None
    return compatible_mapping


def compute_compatible_dims_mapping(dims_mapping_list):
    if not dims_mapping_list:
        return None
    length = len(dims_mapping_list[0])
    for dims_mapping in dims_mapping_list:
        assert (
            dims_mapping is not None
        ), "Dims mapping must not be None for compatible computation"
        assert (
            len(dims_mapping) == length
        ), "The length of dims_mapping in list must be same for compatible computation."
    compatible_result = []
    for dim_mappings in zip(*dims_mapping_list):
        compatible_dim_mapping = compute_compatible_dim_mapping(
            list(dim_mappings)
        )
        if compatible_dim_mapping is None:
            return None
        compatible_result.append(compatible_dim_mapping)
    return compatible_result


def compute_compatible_process_mesh(process_mesh_list):
    compatible_process_mesh = None
    if not process_mesh_list:
        return compatible_process_mesh
    for process_mesh in process_mesh_list:
        if process_mesh is not None:
            if (
                compatible_process_mesh is None
                or compatible_process_mesh == process_mesh
            ):
                compatible_process_mesh = process_mesh
            else:
                return None
    return compatible_process_mesh


def compute_compatible_and_update_dim_mapping(dims_mapping_list, index_list):
    assert len(dims_mapping_list) == len(index_list)
    changed = False
    dim_mappings = []
    for i in range(len(dims_mapping_list)):
        assert is_valid_list_index(dims_mapping_list[i], index_list[i])
        dim_mappings.append(dims_mapping_list[i][index_list[i]])
    compatible_dim_mapping = compute_compatible_dim_mapping(dim_mappings)
    if compatible_dim_mapping is None:
        return False
    for i in range(len(dims_mapping_list)):
        if compatible_dim_mapping != dims_mapping_list[i][index_list[i]]:
            dims_mapping_list[i][index_list[i]] = compatible_dim_mapping
            changed = True
    return changed


def append_distributed_attr_suffix(name):
    """
    Append auto parallel suffix for distributed attribute name.
    """
    return name + core.kAutoParallelSuffix()


def remove_distributed_attr_suffix(name):
    """
    Remove auto parallel suffix from distributed attribute name.
    """
    return name.strip(core.kAutoParallelSuffix())


def check_distributed_attr_for_program(program, dist_context=None):
    from .dist_context import get_default_distributed_context

    if dist_context is None:
        dist_context = get_default_distributed_context()
    assert (
        dist_context.is_initialized_for_program()
    ), "Distributed attributes must be initialized before check."
    for block in program.blocks:
        for tensor in block.vars.values():
            dist_tensor = dist_context.get_dist_tensor_for_graph(tensor)
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                tensor
            )
            if (tensor_dist_attr is not None) and (not dist_tensor.is_valid()):
                return False
        for op in block.ops:
            dist_op = dist_context.get_dist_op_for_graph(tensor)
            op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
            if (op_dist_attr is not None) and (not dist_op.is_valid()):
                return False
    return True


def print_program_with_dist_attr(program, dist_context=None):
    """
    This function reuses the original program output ability with a distributed context.
    Using lock can avoid multiple threads change the default distributed context simultaneously.
    """
    lock = threading.Lock()
    lock.acquire()
    from .dist_context import (
        get_default_distributed_context,
        set_default_distributed_context,
    )

    if dist_context is None:
        dist_context = get_default_distributed_context()
        print(program, flush=True)
    else:
        original_default_context = get_default_distributed_context()
        set_default_distributed_context(dist_context)
        print(program, flush=True)
        set_default_distributed_context(original_default_context)
    lock.release()


def _get_comm_group(processes, shape, axis, rank):
    """
    Given a rank and the processes mesh the rank belongs to,
    compute the communication peers of the rank based on the give axis in the mesh.

    Example: 16 processes managed in a 4-Dimensional mesh with shape of [2, 2, 2, 2].
    the rank communication peers of rank 0 (included) are following:
    in axis 0: [0, 1]
    in axis 1: [0, 2]
    in axis 2: [0, 4]
    in axis 3: [0, 8]
    """

    # NOTE _linear_idx2coordinate assume processes mesh start with 0 and continuous
    # tricks to support processes mesh when it is not start with 0 or continuous
    assert (
        rank in processes
    ), f"rank [{rank}] is NOT in processes group {processes}"
    rank_relative = processes.index(rank)
    coordinate = _linear_idx2coordinate(shape, rank_relative)
    coordinates_in_group = [coordinate[:] for i in range(shape[axis])]

    # select comm group
    for i in range(shape[axis]):
        coordinates_in_group[i][axis] = i

    ranks_in_group_relative = [
        _coordinate2linear_idx(shape, coordinate)
        for coordinate in coordinates_in_group
    ]
    ranks_in_group = [processes[idx] for idx in ranks_in_group_relative]

    return sorted(ranks_in_group)


def _get_idx_in_axis(processes, shape, axis, rank):
    """
    Given a rank and the processes mesh the rank belongs to,
    compute the index of the rank in given axis.

    Example: 27 processes managed in a 3-Dimensional mesh with shape of [3, 3, 3].
    the index of rank 22 are:
    in axis 0: 1
    in axis 1: 1
    in axis 2: 2
    """

    # NOTE _linear_idx2coordinate assume processes mesh start with 0 and continuous
    #  tricks to support processes mesh when it is not start with 0 or continuous
    rank_relative = processes.index(rank)
    coordinate = _linear_idx2coordinate(shape, rank_relative)
    return coordinate[axis]


def _coordinate2linear_idx(mesh_shape, coordinate):
    """
    convert a coordinate in multidimensional mesh space into a scala idx in linear space.

    it use Row-major order for dimension conversion.
    so it has:  [most_significant_dim, ..., least_significant_dim]
    assume:

        the size of i-th dimension to be:  S[i]
        the index of j-th dimension is: I[j]

    linear_idx of a n dimensional coordinate is:

        I[n-1] * (S[n-2] * S[n-3] * S[n-4] *     ....    S[0]) +
        I[n-2] * (         S[n-3] * S[n-4] *     ....    S[0]) +
        I[n-3] * (                  S[n-4] *     ....    S[0]) +
        ...
        I[1]   * (                                       S[0]) +
        I[0]

    """
    # NOTE the following function work based on a strong an assumption
    # that the processes in mesh are
    #    1. starts from 0
    #    2. continuous
    # it will be wrong if ths above condition does not meet,
    # e.g. process_mesh = { process_groups = [7, 8, 9,10, 12, 13, 14, 15], mesh = [2, 4]}
    # if you want a more general mapping, you should use cartesian product

    assert len(mesh_shape) == len(
        coordinate
    ), f"coordinate should have the same size as mesh shape, but got shape: {mesh_shape}, coordinate: {coordinate}"
    for i in range(len(mesh_shape)):
        assert (
            coordinate[i] >= 0
        ), f"index in dimension [{i}] is least than zero. coordinate: {coordinate}"
        assert (
            coordinate[i] < mesh_shape[i]
        ), f"index beyond extent in dimension [{i}]. shape: {mesh_shape}, coordinate: {coordinate}"

    base = mesh_shape[-1]
    linear_idx = coordinate[-1]

    # row major order
    for i in range(len(mesh_shape) - 2, -1, -1):
        linear_idx += base * coordinate[i]
        base *= mesh_shape[i]

    return linear_idx


def _linear_idx2coordinate(mesh_shape, linear_idx):
    """
    mapping a linear scala into multidimensional mesh space, return it coordinate in that space.

    it is the inverse function of _coordinate2linear_idx.
    assume:

        the size of i-th dimension to be:  S[i]
        the index of j-th dimension is: I[j]

    the coordinate given linear_idx is:

        I[0] = linear_idx                                  % S[0]
        I[0] = (linear_idx / S[0])                         % S[1]
        I[0] = (linear_idx / (S[0] * S[1]))                % S[2]
        ....

    """

    assert linear_idx >= 0, f"linear index [{linear_idx}] is least than zero"
    assert linear_idx < np.prod(
        mesh_shape
    ), f"linear index beyond the extent of mesh shape. shape: {mesh_shape}, linear index: {linear_idx}"

    base = 1
    coordinate = [-1] * len(mesh_shape)

    for i in reversed(range(len(mesh_shape))):
        offset = linear_idx / base
        coordinate[i] = int(offset % mesh_shape[i])
        base *= mesh_shape[i]

    # row major order
    return coordinate


def _get_corresponding_rank(dist_context, target_mesh, rank):
    # TODO(JZ-LIANG) a hack method to support varying mesh in Pipeline parallelism case.
    # we assume that all mesh are evenly divide from a parent mesh and should have same size.
    # to revise this in future.

    coordinate = None
    for mesh in dist_context.process_meshes:
        if rank in mesh.process_ids and mesh.shape == target_mesh.shape:
            coordinate = _linear_idx2coordinate(
                mesh.shape, mesh.process_ids.index(rank)
            )
            break

    # assert coordinate is not None, "could NOT found rank [{}] in any registered mesh".format(
    #     rank)
    if coordinate is not None:
        return target_mesh.process_ids[
            _coordinate2linear_idx(mesh.shape, coordinate)
        ]
    else:
        return target_mesh.process_ids[0]


def _get_unshard_dist_shape(var, dist_attr):
    var_shape = var.shape
    mapping = dist_attr.dims_mapping
    mesh = dist_attr.process_mesh.shape
    assert len(var_shape) == len(
        mapping
    ), f"variable shape [{var_shape}] and dim_mapping [{mapping}] is NOT match !"
    new_shape = []
    for idx in range(len(var_shape)):
        if var_shape[idx] == -1 or mapping[idx] == -1:
            new_shape.append(var_shape[idx])
        else:
            new_shape.append(var_shape[idx] * mesh[mapping[idx]])

    return new_shape


def make_data_unshard(dist_main_prog, dist_startup_prog, dist_context=None):
    from .dist_context import get_default_distributed_context

    if dist_context is None:
        dist_context = get_default_distributed_context()

    for var in dist_main_prog.list_vars():
        if var.is_data:
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                var
            )
            inverse_shape = _get_unshard_dist_shape(var, tensor_dist_attr)
            var.desc.set_shape(inverse_shape)
            dim_mapping = tensor_dist_attr.dims_mapping
            dim_mapping = [-1] * len(dim_mapping)
            tensor_dist_attr.dims_mapping = dim_mapping
            dist_context.set_tensor_dist_attr_for_program(var, tensor_dist_attr)


def _update_addition_info(addition_info):
    """Update default addition_info with inputs"""
    add_info = {"epoch": 0, "batch": 0, "batch_size": 0}
    if not addition_info:
        return add_info
    elif not isinstance(addition_info, dict):
        raise TypeError(
            "The type of 'addition_info' should be 'dict', "
            f"but got '{type(addition_info)}'."
        )
    else:
        for item, value in addition_info.items():
            if item not in ["epoch", "batch", "batch_size"]:
                raise ValueError(
                    "The key of 'addition_info' should be one of the "
                    f"['epoch', 'batch', 'batch_size'], but got '{item}'."
                )
            if not isinstance(value, int):
                raise ValueError(
                    "The value of 'addition_info' should be 'int', "
                    f"but got '{type(value)}'."
                )
            add_info[item] = value
        return add_info


def _check_valid_path(file_path):
    """Validity check of input file path"""
    if not file_path:
        return file_path
    elif isinstance(file_path, list):
        for file in file_path:
            if not isinstance(file, str):
                raise TypeError(
                    "The type of file path should be 'str', "
                    f"but got '{type(file)}'."
                )
            if not os.path.exists(file):
                raise ValueError(f"The file path '{file}' does not exist.")
        return file_path
    else:
        raise TypeError(
            "The type of file path should be 'list', "
            f"but got '{type(file_path)}'."
        )


def _check_param_dict(param_dict):
    if not param_dict:
        raise ValueError("'param_dict' cannot be None.")
    elif not isinstance(param_dict, dict):
        raise TypeError(
            "The type of 'param_dict' should be 'dict', "
            f"but got '{type(param_dict)}'."
        )
    else:
        for name, value in param_dict.items():
            if not isinstance(name, str):
                raise TypeError(
                    "The type of key of 'param_dict' should be 'str', "
                    f"but got '{type(name)}'."
                )
            if not isinstance(value, paddle.base.DenseTensor):
                raise TypeError(
                    "The type of value of 'param_dict' should be 'DenseTensor', "
                    f"but got '{type(value)}'."
                )
        return param_dict


def _check_dist_attr(dist_attr):
    if not dist_attr:
        return dist_attr
    elif not isinstance(dist_attr, dict):
        raise TypeError(
            "The type of 'dist_attr' should be 'dict', "
            f"but got '{type(dist_attr)}'."
        )
    else:
        for name, value in dist_attr.items():
            if not isinstance(name, str):
                raise TypeError(
                    "The type of param name of 'dist_attr' should be 'str', "
                    f"but got '{type(name)}'."
                )
            if not isinstance(value, dict):
                raise TypeError(
                    "The type of distributed attribute should be 'dict', "
                    f"but got '{type(value)}'"
                )
            attr = [
                'process_shape',
                'process_group',
                'dims_mapping',
                'dim_names',
            ]
            if list(value.keys()) != attr:
                raise ValueError(
                    "The key of distributed attribute should be "
                    "'['process_shape', 'process_group', 'dims_mapping']', "
                    f"but got {value.keys()}."
                )
        return dist_attr


def save_distributed_checkpoint(
    program,
    checkpoint_path,
    dist_attr_path,
    addition_info=None,
    is_integrated=False,
    dist_context=None,
):
    """
    Save model parameter state, optimizer state, distributed attribute and
    additional information of each rank.

    Args:
        program(Program): The program to be saved.
        checkpoint_path(str): The path of the checkpoint file to be saved.
        dist_attr_path(str): The path of distributed attribute file to be saved.
        addition_info(dict, optional): Additional information, key should be selected in ['epoch', 'batch', 'batch_size'].
            Default values are 0, when 'addition_info' is None. Default: None.
        is_integrated(bool, optional): Whether to integrate param before save. Default: False.
        dist_context(DistributedContext ,optional): collect related distributed information for program

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import os
            >>> from paddle.distributed.auto_parallel.static.utils import save_distributed_checkpoint

            >>> step = 16000
            >>> global_batch_size = 32
            >>> path = os.path.join("./output", "step_%d" % step)
            >>> os.makedirs(path, exist_ok=True)
            >>> program = paddle.static.Program()

            >>> add_info = {'batch': step, "batch_size": global_batch_size}
            >>> save_distributed_checkpoint(program, path, path, add_info)

    """
    from .dist_context import get_default_distributed_context

    assert isinstance(program, paddle.static.Program)
    assert isinstance(is_integrated, bool)
    if dist_context is None:
        dist_context = get_default_distributed_context()
    addition_info = _update_addition_info(addition_info)

    if not is_integrated:
        _save_distributed_state_dict(program, addition_info, checkpoint_path)
        _save_distributed_attribute(program, dist_attr_path, dist_context)
    else:
        # TODO: integrate param before save
        raise NotImplementedError(
            "Integrating parameter has not been implemented."
        )


def load_distributed_checkpoint(checkpoint_path, dist_attr_path):
    """
    Load parameter, optimizer, distributed attribute and addition_info.

    Args:
        checkpoint_path(list[str]): model parameter file path, must be in order of rank id.
        dist_attr_path(list[str]): distributed attribute file path, must be in order of rank id.

    Returns:
        param_dict(dict): parameters' value of all ranks.
        dist_attr(dict): parameters' distributed attribute.
        addition_info(dict): additional information user saved in last training.

    Notes:
        The return, 'addition_info', is belonging to the first file of checkpoint_path by default.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('Depends on external files.')
            >>> from paddle.distributed.auto_parallel.static.utils import load_distributed_checkpoint

            >>> ckpt_path = [
            ...     './model_state_rank0.pdmodel',
            ...     './model_state_rank1.pdmodel',
            ... ]
            >>> dist_attr_path = [
            ...     './dist_attr_rank0.pdattr',
            ...     './dist_attr_rank1.pdattr',
            ... ]
            >>> param_dict, dist_attr, add_info = load_distributed_checkpoint(ckpt_path, dist_attr_path)
    """
    assert _check_valid_path(
        checkpoint_path
    ), "'checkpoint_path' cannot be None."
    assert _check_valid_path(dist_attr_path), "'dist_attr_path' cannot be None."

    state_dict_info = _load_distributed_state_dict(checkpoint_path)
    dist_attr = _load_distributed_attribute(dist_attr_path)
    param_dict = state_dict_info["model"]
    addition_info = state_dict_info["addition_info"]
    return param_dict, dist_attr, addition_info


def load_checkpoint_into_program(
    checkpoint_path, dist_attr_path, program, dist_context=None
):
    """
    Load parameter, optimizer, distributed attribute and addition_info into model.

    Args:
        checkpoint_path(list[str]): model parameter file path, must be in order of rank id.
        dist_attr_path(list[str]): distributed attribute file path, must be in order of rank id.
        program(Program): the program to be updated with checkpoint_path.
        dist_context(DistributedContext ,optional): collect related distributed information for program

    Returns:
        addition_info(dict): user saved in last train.

    Notes:
        The return, 'addition_info', is belonging to the first file of checkpoint_path by default.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('Depends on external files.')
            >>> from paddle.distributed.auto_parallel.static.utils import load_checkpoint_into_program

            >>> exe.run(startup_program)
            >>> ckpt_path = [
            ...     './model_state_rank0.pdmodel',
            ...     './model_state_rank1.pdmodel',
            ... ]
            >>> dist_attr_path = [
            ...     './dist_attr_rank0.pdattr',
            ...     './dist_attr_rank1.pdattr',
            ... ]
            >>> load_checkpoint_into_program(ckpt_path, dist_attr_path, main_program)
    """
    from .dist_context import get_default_distributed_context

    assert isinstance(program, paddle.static.Program)
    assert _check_valid_path(
        checkpoint_path
    ), "'checkpoint_path' cannot be None."
    assert _check_valid_path(dist_attr_path), "'dist_attr_path' cannot be None."
    if dist_context is None:
        dist_context = get_default_distributed_context()
    all_state_dict_info = _load_distributed_state_dict(checkpoint_path)
    all_pre_dist_attr = _load_distributed_attribute(dist_attr_path)
    all_cur_dist_attr = get_dist_attr(program, dist_context)
    all_param_dict = all_state_dict_info["model"]
    addition_info = all_state_dict_info["addition_info"]
    sliced_param_dict = merge_and_slice_parameter(
        all_param_dict, all_pre_dist_attr, all_cur_dist_attr
    )
    load_parameter_into_program(sliced_param_dict, program)

    return addition_info


def load_parameter_into_program(param_dict, program):
    """
    Load parameters into program.

    Args:
        param_dict(dict): parameters' name and value.
        program(Program): the program to be updated
    """
    assert isinstance(param_dict, dict)
    assert program and isinstance(program, paddle.static.Program)
    if not param_dict:
        return
    program.set_state_dict(param_dict)


def _save_distributed_attribute(program, dist_attr_path, dist_context):
    """Save distributed attribute of all parameters"""
    # TODO: just save a complete distributed attribute file
    rank_id = paddle.distributed.get_rank()
    dist_attr_name = os.path.join(
        dist_attr_path, f"dist_attr_rank{rank_id}.pdattr"
    )
    dist_attr_dict = {
        "model": get_dist_attr(program, dist_context),
        "world_size": paddle.distributed.get_world_size(),
    }
    paddle.save(dist_attr_dict, dist_attr_name)
    logging.info(f"Already saved distributed attribute to '{dist_attr_path}'.")


def _load_distributed_attribute(dist_attr_path):
    """Load parameters' distributed attribute from dist_attr_path"""
    total_dist_attr = {}
    for dist_attr_file in dist_attr_path:
        dist_attr = paddle.load(dist_attr_file)
        pre_world_size = dist_attr["world_size"]
        assert pre_world_size == len(
            dist_attr_path
        ), "The number of 'dist_attr_path' must be equal to the last training world size."
        for name, attr in dist_attr["model"].items():
            if name not in total_dist_attr:
                total_dist_attr[name] = attr

    return total_dist_attr


def _save_distributed_state_dict(program, addition_info, checkpoint_path):
    """Save parameters' state_dict"""
    rank = paddle.distributed.get_rank()
    ckpt_file_name = os.path.join(
        checkpoint_path, f"model_state_rank{rank}.pdmodel"
    )
    state_dict = {
        "model": program.state_dict(),
        "world_size": paddle.distributed.get_world_size(),
        "addition_info": addition_info,
    }
    paddle.save(state_dict, ckpt_file_name)
    logging.info(f"Already saved model to '{checkpoint_path}'.")


def _load_distributed_state_dict(checkpoint_path):
    """Load parameters' state_dict from checkpoint_path"""
    all_state_dict = {}
    for idx, ckpt_file in enumerate(checkpoint_path):
        state_dict_info = paddle.load(ckpt_file, return_numpy=True)
        pre_world_size = state_dict_info["world_size"]
        assert pre_world_size == len(
            checkpoint_path
        ), "The number of 'checkpoint_path' must be equal to the last training world size."
        if idx == 0:
            addition_info = state_dict_info["addition_info"]
        for name, value in state_dict_info["model"].items():
            if name in all_state_dict:
                all_state_dict[name].append(np.array(value))
            else:
                all_state_dict[name] = [np.array(value)]

    all_state_dict_info = {
        "model": all_state_dict,
        "addition_info": addition_info,
    }
    return all_state_dict_info


def get_dist_attr(program, dist_context=None):
    """
    Get distributed attribute of current rank.

    Args:
        program(Program): main program for training
    """
    dist_attr = {}
    if use_pir_api():
        ops = program.global_block().ops
        for op in ops:
            if op.name() == "builtin.parameter" or (
                op.name() == "pd_op.data"
                and op.has_attr("persistable")
                and op.attrs()["persistable"]
            ):
                op_dist_attr = op.dist_attr
                var_dist_attr = op_dist_attr.result(0).as_tensor_dist_attr()
                var_name = (
                    op.str_attr("parameter_name")
                    if op.name() == "builtin.parameter"
                    else op.str_attr("name")
                )
                process_mesh = var_dist_attr.process_mesh
                dist_attr[var_name] = {
                    "process_shape": process_mesh.shape,
                    "process_group": process_mesh.process_ids,
                    "dims_mapping": var_dist_attr.dims_mapping,
                    "dim_names": process_mesh.dim_names,
                }
    else:
        from .dist_context import get_default_distributed_context

        assert isinstance(program, paddle.static.Program)
        if dist_context is None:
            dist_context = get_default_distributed_context()
        for var in program.list_vars():
            if is_parameter(var) or is_belong_to_optimizer(var):
                tensor_dist_attr = (
                    dist_context.get_tensor_dist_attr_for_program(var)
                )
                process_mesh = tensor_dist_attr.process_mesh
                dims_mapping = tensor_dist_attr.dims_mapping
                dim_names = tensor_dist_attr.process_mesh.dim_names
                dist_attr[var.name] = {
                    "process_shape": process_mesh.shape,
                    "process_group": process_mesh.process_ids,
                    "dims_mapping": dims_mapping,
                    "dim_names": dim_names,
                }
    return dist_attr


def merge_and_slice_parameter(dist_param_dict, pre_dist_attr, cur_dist_attr):
    """
    Merge parameters with previous dist_attr and slice parameters with current dist_attr

    Args:
        dist_param_dict(dict): parameters' value of all ranks.
        pre_dist_attr(dict): parameters' dist_attr of last training process.
        cur_dist_attr(dict): parameters' dist_attr of current training process.

    Returns:
        dist_param_dict(dict): parameters' value of current rank.
    """
    assert _check_dist_attr(pre_dist_attr), "'pre_dist_attr' cannot be None."
    assert isinstance(
        dist_param_dict, dict
    ), f"The type of 'dist_param_dict' should be 'dict', but got {type(dist_param_dict)}."
    for name, value in dist_param_dict.items():
        if not isinstance(name, str):
            raise TypeError(
                "The key of 'dist_param_dict' is parameter's name, "
                f"and its type should be 'str', but got {type(name)}."
            )
        if not isinstance(value, list) or not all(
            isinstance(v, np.ndarray) for v in value
        ):
            raise TypeError(
                "The value of 'dist_param_dict' is parameter's value of all ranks, "
                "and its type should be 'list(numpy.ndarray)'."
            )

    if cur_dist_attr is None:
        return {}

    param_not_in_pre = []
    param_not_in_cur = []
    logging.info("Start to merge and slice parameters.")
    for var_name in cur_dist_attr.keys():
        if var_name not in pre_dist_attr:
            param_not_in_pre.append(var_name)
            continue

        pre_attr = pre_dist_attr[var_name]
        cur_attr = cur_dist_attr[var_name]
        if pre_attr == cur_attr:
            # skip merge and slice
            rank_id = paddle.distributed.get_rank()
            index = cur_attr["process_group"].index(rank_id)
            param = dist_param_dict[var_name][index]
            dist_param_dict[var_name] = param
            continue

        pre_param = dist_param_dict[var_name]
        pre_dims_mapping = pre_attr["dims_mapping"]
        cur_dims_mapping = cur_attr["dims_mapping"]
        if len(set(pre_dims_mapping)) > 1 or -1 not in pre_dims_mapping:
            complete_param = _merge_parameter_with_dist_attr(
                pre_param, pre_attr
            )
            dist_param_dict[var_name] = complete_param
        else:
            complete_param = pre_param[0]
            dist_param_dict[var_name] = complete_param

        if len(set(cur_dims_mapping)) > 1 or -1 not in cur_dims_mapping:
            sliced_param = _slice_parameter_with_dist_attr(
                complete_param, cur_attr
            )
            dist_param_dict[var_name] = sliced_param

    for var_name in pre_dist_attr:
        if var_name not in cur_dist_attr:
            param_not_in_cur.append(var_name)
            dist_param_dict.pop(var_name)

    if param_not_in_pre:
        warnings.warn(
            f"Parameters '{param_not_in_pre}' are not found in last training process."
        )
    if param_not_in_cur:
        warnings.warn(
            f"Parameters '{param_not_in_cur}' are not found in current training process."
        )

    return dist_param_dict


def _merge_parameter_with_dist_attr(param_list, dist_attr):
    """Merge parameter with distributed attribute"""
    from .reshard import Resharder

    dims_mapping = dist_attr["dims_mapping"]
    process_shape = dist_attr["process_shape"]
    process_group = dist_attr["process_group"]
    # get the complete shape of the parameter
    complete_shape = Resharder.compute_complete_shape(
        param_list[0].shape, process_shape, dims_mapping
    )
    # merge the parameter with dist_attr
    partition_param_list = []
    merged_partition = []
    for process in process_group:
        partition_index = Resharder.compute_partition_index(
            process, complete_shape, dims_mapping, process_shape, process_group
        )
        index = process_group.index(process)
        if partition_index not in merged_partition:
            merged_partition.append(partition_index)
            _merge_parameter(
                partition_param_list,
                param_list[index],
                partition_index,
                complete_shape,
            )

    assert (
        len(partition_param_list) == 1 or not partition_param_list
    ), "Fail to merge parameter"
    complete_param = partition_param_list[0][0]
    return complete_param


def _slice_parameter_with_dist_attr(param, dist_attr):
    """Slice parameter with distributed attribute"""
    param = (
        np.array(param) if isinstance(param, paddle.base.DenseTensor) else param
    )
    dims_mapping = dist_attr["dims_mapping"]
    process_shape = dist_attr["process_shape"]
    process_group = dist_attr["process_group"]
    # slice the parameter with dist_attr
    partition_index_list = _get_split_indices(
        param.shape, dims_mapping, process_shape, process_group
    )
    sliced_param_list = _slice_parameter(
        param, partition_index_list, len(partition_index_list)
    )
    # get the current parameter's index in sliced_param_list
    rank_id = paddle.distributed.get_rank()
    sliced_param_index = _get_sliced_param_index(
        rank_id, param.shape, dims_mapping, process_shape, process_group
    )
    sliced_param = sliced_param_list[sliced_param_index]
    return sliced_param


def _merge_parameter(
    partition_param_list, param, partition_index, complete_shape
):
    """
    Merge partial parameters to a complete one.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from paddle.distributed.auto_parallel.static.utils import _merge_parameter

            >>> partition_param_list = [(np.array([[[1.11, 1.12]]]), [[0, 1],[0, 1],[0, 2]])]
            >>> param = np.array([[[1.13, 1.14]]])
            >>> partition_index = [[0, 1],[0, 1],[2, 4]]
            >>> complete_shape = [2, 2, 4]

            >>> _merge_parameter(partition_param_list, param, partition_index, complete_shape)
            >>> print(partition_param_list)
            [(array([[[1.11, 1.12, 1.13, 1.14]]]), [[0, 1],[0, 1],[0, 4]])]

    """
    from .reshard import Resharder

    if len(partition_param_list) == 1:
        is_complete_data = True
        for idx, item in enumerate(partition_param_list[0][1]):
            if item[0] != 0 or item[1] != complete_shape[idx]:
                is_complete_data = False
                break
        if is_complete_data:
            return

    if not partition_param_list:
        partition_param_list.append((param, partition_index))
    else:
        i = 0
        while i < len(partition_param_list):
            (
                concat_axis,
                first_order,
                new_partition,
            ) = Resharder.compute_concat_info(
                partition_param_list[i][1], partition_index
            )
            if concat_axis != -1:
                if first_order == 0:
                    new_param = np.concatenate(
                        (partition_param_list[i][0], param), axis=concat_axis
                    )
                else:
                    new_param = np.concatenate(
                        (param, partition_param_list[i][0]), axis=concat_axis
                    )

                partition_param_list.pop(i)
                _merge_parameter(
                    partition_param_list,
                    new_param,
                    new_partition,
                    complete_shape,
                )
                break
            i += 1


def _complete_op_dist_attr(program, block=None):
    if block is None:
        block = program.global_block()
    for op in block.ops:
        for sub_block in op.blocks():
            _complete_op_dist_attr(program, block=sub_block)
        if op.name() in partition_skip_op_list:
            continue

        if op.dist_attr is None:
            meshes = []
            operand_attrs = []
            result_attrs = []
            for operand in op.operands_source():
                tmp_attr = operand.dist_attr()
                if tmp_attr is None:
                    operand_attrs.append(pir.Attribute())
                    value_mesh = None
                    tmp_op_dist_attr = operand.get_defining_op().dist_attr
                    if tmp_op_dist_attr is not None:
                        value_mesh = tmp_op_dist_attr.process_mesh
                else:
                    operand_attrs.append(tmp_attr)
                    value_mesh = tmp_attr.process_mesh
                if value_mesh is not None and value_mesh not in meshes:
                    meshes.append(value_mesh)

            for result in op.results():
                tmp_attr = result.dist_attr()
                if tmp_attr is None:
                    result_attrs.append(pir.Attribute())
                else:
                    result_attrs.append(tmp_attr)
                    if tmp_attr.process_mesh not in meshes:
                        meshes.append(tmp_attr.process_mesh)
            if len(meshes) > 0:
                if len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    mesh = merge_process_meshes(meshes)
                op.dist_attr = pir.create_op_dist_attribute(
                    mesh,
                    operand_attrs,
                    result_attrs,
                )


def _slice_parameter(complete_param, partition_index_list, length):
    """
    Slice a complete parameter.

    Returns:
        sliced_param_list(list): sliced parameters with 'partition_index_list'

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from paddle.distributed.auto_parallel.static.utils import _slice_parameter

            >>> complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
            >>> rank = 2
            >>> complete_shape = [1, 1, 6]
            >>> dims_mapping = [-1, -1, 0]
            >>> process_shape = [3]
            >>> process_group = [0, 1, 2]

            >>> sliced_param_list = _slice_parameter(complete_param, [[], [], [2, 4]], 3)
            >>> print(sliced_param_list)
            [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]

    """
    sliced_param_list = []
    axis = len(complete_param.shape) - length
    sliced_param = np.split(
        complete_param, partition_index_list[axis], axis=axis
    )
    if length == 1:
        return sliced_param
    for param in sliced_param:
        sliced_param_list.extend(
            _slice_parameter(param, partition_index_list, length - 1)
        )
    return sliced_param_list


def _get_sliced_param_index(
    rank, complete_shape, dims_mapping, process_shape, process_group
):
    """
    Get sliced_param's index of current rank in all sliced parameters list.

    Returns:
        sliced_param_index(int): the index of sliced param in sliced_param_list

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from paddle.distributed.auto_parallel.static.utils import _get_sliced_param_index

            >>> complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
            >>> rank = 2
            >>> complete_shape = [1, 1, 6]
            >>> dims_mapping = [-1, -1, 0]
            >>> process_shape = [3]
            >>> process_group = [0, 1, 2]

            >>> slice_param = _slice_parameter(complete_param, [[], [], [2, 4]], 3)
            >>> print(slice_param)
            [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]

            >>> index = _get_sliced_param_index(rank, complete_shape, dims_mapping,
            ...                                 process_shape, process_group)
            >>> print(index)
            2
    """
    from .reshard import Resharder

    partition_index = Resharder.compute_partition_index(
        rank, complete_shape, dims_mapping, process_shape, process_group
    )
    sliced_param_index = 0
    for i, shape in enumerate(complete_shape):
        if dims_mapping[i] == -1:
            slice_shape = shape
        else:
            slice_shape = shape // process_shape[dims_mapping[i]]
        if slice_shape == 1:
            index = partition_index[i][0]
        else:
            index = (partition_index[i][0] + 1) // slice_shape
        sliced_param_index = sliced_param_index * (shape // slice_shape) + index
    return sliced_param_index


def _get_split_indices(
    complete_shape, dims_mapping, process_shape, process_group
):
    """
    Get split indices of every dimension.

    Returns:
        split_indices_list(list): the split indices of every dimension of the parameter

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from paddle.distributed.auto_parallel.static.utils import _get_split_indices

            >>> complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
            >>> complete_shape = [1, 1, 6]
            >>> dims_mapping = [-1, -1, 0]
            >>> process_shape = [3]
            >>> process_group = [0, 1, 2]

            >>> index = _get_split_indices(complete_shape, dims_mapping, process_shape, process_group)
            >>> print(index)
            [[], [], [2, 4]]
    """
    from .reshard import Resharder

    split_indices_list = []
    for process in process_group:
        partition_index = Resharder.compute_partition_index(
            process, complete_shape, dims_mapping, process_shape, process_group
        )
        if split_indices_list:
            for dim in range(len(partition_index)):
                split_indices_list[dim].extend(partition_index[dim])
        else:
            split_indices_list = partition_index
    split_indices_list = list(
        map(
            lambda x, y: list(set(x) - {y} - {0}),
            split_indices_list,
            complete_shape,
        )
    )
    split_indices_list = [sorted(x) for x in split_indices_list]
    return split_indices_list


def is_forward_op(op):
    op_role = int(op.attr('op_role'))
    return OP_ROLE_KEY in op.attr_names and (
        op_role == int(OpRole.Forward) or op_role == int(OpRole.Loss)
    )


def is_backward_op(op):
    return OP_ROLE_KEY in op.attr_names and int(
        op.all_attrs()[OP_ROLE_KEY]
    ) & int(OpRole.Backward)


def is_optimize_op(op):
    return OP_ROLE_KEY in op.attr_names and int(
        op.all_attrs()[OP_ROLE_KEY]
    ) & int(OpRole.Optimize)


def is_lr_sched_op(op):
    return OP_ROLE_KEY in op.attr_names and int(
        op.all_attrs()[OP_ROLE_KEY]
    ) & int(OpRole.Optimize.LRSched)


def is_loss_op(op):
    return OP_ROLE_KEY in op.attr_names and int(
        op.all_attrs()[OP_ROLE_KEY]
    ) == (int(OpRole.Forward) | int(OpRole.Loss))


def is_loss_grad_op(op):
    if OP_ROLE_KEY not in op.attr_names:
        return False
    op_role = int(op.all_attrs()[OP_ROLE_KEY])
    return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)


def is_gradient_clip_op(op):
    return op.desc.has_attr("op_namescope") and op.desc.attr(
        "op_namescope"
    ).startswith("/gradient_clip")


def is_reshard_op(op):
    return op.desc.has_attr(
        "op_namescope"
    ) and "/auto_parallel/reshard" in op.desc.attr('op_namescope')


def is_prim_op(op):
    return op.type.endswith("_p")


def is_comm_op(op):
    return op.has_attr("ring_id")


def get_loss_op(block):
    loss_ops = []
    for op in block.ops:
        if is_loss_op(op):
            assert (
                len(op.desc.output_arg_names()) == 1
            ), "loss op should only output loss var"
            loss_ops.append(op)

    assert len(loss_ops) == 1, "num of loss op is not equal to one"
    return loss_ops[0]


def set_var_dist_attr(dist_context, var, dims_mapping, process_mesh, **kwargs):
    tensor_dist_attr = TensorDistAttr()
    tensor_dist_attr.dims_mapping = dims_mapping
    # TODO get global mesh group
    if isinstance(process_mesh, (list, np.ndarray)):
        tensor_dist_attr.process_mesh = ProcessMesh(process_mesh)
    elif isinstance(process_mesh, core.ProcessMesh):
        tensor_dist_attr.process_mesh = process_mesh
    else:
        raise ValueError(
            f"{process_mesh} must be a instance of ProcessMesh or list, but receive {type(process_mesh)}"
        )
    if kwargs.get("mark_annotated"):
        tensor_dist_attr.mark_annotated("dims_mapping")
        tensor_dist_attr.mark_annotated("process_mesh")
    if kwargs.get("chunk_id"):
        tensor_dist_attr.chunk_id = kwargs["chunk_id"]
    dist_context.set_tensor_dist_attr_for_program(var, tensor_dist_attr)
    return tensor_dist_attr


def naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
    new_op, process_mesh, ref_mapping, ctx, **kwargs
):
    assert process_mesh is not None
    assert ref_mapping is not None

    new_op_dist_attr = OperatorDistAttr()

    for input_varname in new_op.desc.input_arg_names():
        new_op_dist_attr.set_input_dims_mapping(input_varname, ref_mapping)
    for output_varname in new_op.desc.output_arg_names():
        new_op_dist_attr.set_output_dims_mapping(output_varname, ref_mapping)

    new_op_dist_attr.process_mesh = process_mesh
    if kwargs.get("chunk_id"):
        new_op_dist_attr.chunk_id = kwargs["chunk_id"]
    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def naive_set_dist_op_attr_for_program_by_mesh(
    new_op, process_mesh, ctx, **kwargs
):
    assert process_mesh is not None

    new_op_dist_attr = OperatorDistAttr()

    for input_varname in new_op.desc.input_arg_names():
        var = new_op.block.var(input_varname)
        mapping = ctx.get_tensor_dist_attr_for_program(var).dims_mapping
        new_op_dist_attr.set_input_dims_mapping(input_varname, mapping)
    for output_varname in new_op.desc.output_arg_names():
        var = new_op.block.var(output_varname)
        mapping = ctx.get_tensor_dist_attr_for_program(var).dims_mapping
        new_op_dist_attr.set_output_dims_mapping(output_varname, mapping)

    new_op_dist_attr.process_mesh = process_mesh
    if "is_recompute" in kwargs:
        new_op_dist_attr.is_recompute = kwargs["is_recompute"]
    if "chunk_id" in kwargs:
        new_op_dist_attr.chunk_id = kwargs["chunk_id"]
    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def update_op_dims_mapping_by_default_dist_impl(dist_op):
    changed = False
    op_dist_attr = dist_op.dist_attr
    op_desc = dist_op.serial_op.desc
    # The following statement will be replaced by a more elegant way
    if op_desc.type() == "shape" or op_desc.type() == "slice":
        return False
    output_names = op_desc.output_names()
    xshape_arg_names = []
    if "XShape" in output_names:
        xshape_arg_names = op_desc.output("XShape")
    batch_dim_mappings = []
    for arg_name in op_desc.input_arg_names():
        serial_tensor = dist_op.get_serial_input(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if len(dims_mapping) > 1:
            for idx, mapping in enumerate(dims_mapping[1:]):
                assert (
                    mapping == -1
                ), f"{op_desc.type()} only the batch dimension (0-dim) can be sharded, but the dimension {idx} is sharded by {mapping} part."
        if len(dims_mapping) >= 1:
            batch_dim_mappings.append(dims_mapping[0])
    for arg_name in op_desc.output_arg_names():
        serial_tensor = dist_op.get_serial_output(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if arg_name not in xshape_arg_names:
            if len(dims_mapping) > 1:
                for idx, mapping in enumerate(dims_mapping[1:]):
                    assert (
                        mapping == -1
                    ), f"{op_desc.type()} only the batch dimension (0-dim) can be sharded, but the dimension {idx} is sharded by {mapping} part."
            if len(dims_mapping) >= 1:
                batch_dim_mappings.append(dims_mapping[0])
        else:
            assert (
                dims_mapping[0] == -1
            ), f"{op_desc.type()} only the batch dimension (1-dim) of XShape can be sharded, but the dimension 0 is sharded by {mapping} part."
            if len(dims_mapping) > 2:
                for idx, mapping in enumerate(dims_mapping[2:]):
                    assert (
                        mapping == -1
                    ), f"{op_desc.type()} only the batch dimension (1-dim) of XShape can be sharded, but the dimension {idx} is sharded by {mapping} part."
            batch_dim_mappings.append(dims_mapping[1])

    compatible_dim_mapping = compute_compatible_dim_mapping(batch_dim_mappings)
    assert (
        compatible_dim_mapping is not None
    ), "There is no compatible dim mapping."
    for arg_name in op_desc.input_arg_names():
        serial_tensor = dist_op.get_serial_input(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if len(dims_mapping) >= 1 and compatible_dim_mapping != dims_mapping[0]:
            dims_mapping[0] = compatible_dim_mapping
            changed = True
    for arg_name in op_desc.output_arg_names():
        serial_tensor = dist_op.get_serial_output(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if arg_name not in xshape_arg_names:
            if (
                len(dims_mapping) >= 1
                and compatible_dim_mapping != dims_mapping[0]
            ):
                dims_mapping[0] = compatible_dim_mapping
                changed = True
        else:
            if compatible_dim_mapping != dims_mapping[1]:
                dims_mapping[1] = compatible_dim_mapping
                changed = True

    return changed


def update_op_dims_mapping_by_elementwise_like_dist_impl(dist_op):
    changed = False
    op_dist_attr = dist_op.dist_attr
    op_desc = dist_op.serial_op.desc
    input_arg_names = op_desc.input_arg_names()
    input_dims_mapping_dict = {}
    input_dims_mapping_lens = {}
    max_dims_mapping_len = -1
    for arg_name in input_arg_names:
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if max_dims_mapping_len < len(dims_mapping):
            max_dims_mapping_len = len(dims_mapping)
        input_dims_mapping_dict[arg_name] = dims_mapping
        input_dims_mapping_lens[arg_name] = len(dims_mapping)

    dims_mapping_list = []
    for arg_name in input_arg_names:
        if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
            new_dims_mapping = [-1 for _ in range(max_dims_mapping_len)]
            for i in range(input_dims_mapping_lens[arg_name]):
                new_idx = (
                    max_dims_mapping_len - input_dims_mapping_lens[arg_name]
                ) + i
                new_dims_mapping[new_idx] = input_dims_mapping_dict[arg_name][i]
            dims_mapping_list.append(new_dims_mapping)
        else:
            dims_mapping_list.append(input_dims_mapping_dict[arg_name])
    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        assert len(dims_mapping) == max_dims_mapping_len
        dims_mapping_list.append(dims_mapping)

    compatible_dims_mapping = compute_compatible_dims_mapping(dims_mapping_list)
    assert (
        compatible_dims_mapping is not None
    ), "There is no compatible dim mapping."

    for arg_name in input_arg_names:
        if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
            new_dims_mapping = [
                -1 for _ in range(input_dims_mapping_lens[arg_name])
            ]
            for i in range(input_dims_mapping_lens[arg_name]):
                new_idx = (
                    max_dims_mapping_len - input_dims_mapping_lens[arg_name]
                ) + i
                new_dims_mapping[i] = compatible_dims_mapping[new_idx]
            if new_dims_mapping != input_dims_mapping_dict[arg_name]:
                op_dist_attr.set_input_dims_mapping(arg_name, new_dims_mapping)
                changed = True
        else:
            if compatible_dims_mapping != input_dims_mapping_dict[arg_name]:
                op_dist_attr.set_input_dims_mapping(
                    arg_name, compatible_dims_mapping
                )
                changed = True

    for arg_name in output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if compatible_dims_mapping != dims_mapping:
            op_dist_attr.set_output_dims_mapping(
                arg_name, compatible_dims_mapping
            )
            changed = True

    return changed


def get_all_distributed_main_program(
    serial_program_info, dist_context, parallelizer
):
    "Get all distributed main programs by dist_context."
    from .dist_context import DistributedOperatorContext

    cluster = serial_program_info.cluster
    copied_parallelizer = copy.deepcopy(parallelizer)
    all_dist_main_program = []
    ranks = (
        paddle.distributed.get_world_size()
        if cluster is None
        else len(cluster.get_all_devices("GPU"))
    )
    for rank_id in range(ranks):
        used_dist_context = copy.deepcopy(dist_context)
        used_dist_context._dist_op_context = DistributedOperatorContext()
        (
            _,
            _,
            dist_startup_program,
            dist_main_program,
            _,
        ) = copied_parallelizer._get_dist_program(rank_id, used_dist_context)
        all_dist_main_program.append(dist_main_program)

    return all_dist_main_program


class SerialProgramInfo:
    def __init__(
        self, train_program, startup_program, loss, optimizer, cluster=None
    ):
        self._train_program = train_program
        self._startup_program = startup_program
        self._loss = loss
        self._optimizer = optimizer
        self._cluster = cluster

    @property
    def train_program(self):
        return self._train_program

    @property
    def startup_program(self):
        return self._startup_program

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def cluster(self):
        return self._cluster


def get_standalone_cost_data(distributed_programs):
    def _compute_runtime(op_cost, op, vars):
        runtime = 0
        try:
            runtime = float(op_cost["op_time"])
        except:
            return runtime
        op_config = op_cost["config"]
        total_static_input_size = 0
        total_actual_input_size = 0
        parsed_info = op_config.split("\n")
        variable = "(Variable)"
        for info in parsed_info:
            variable = (
                "(Variable)" if "(Variable)" in info else "(list<Variable>"
            )
            if variable in info:
                arg_name_lower = info[: info.find(variable) - 1]
                shape_left_boundary = info.find("[")
                shape_right_boundary = info.find("]")
                assert (
                    shape_left_boundary > 0
                    and shape_right_boundary > 0
                    and shape_right_boundary > shape_left_boundary
                ), "Get shape failed."
                shape = info[
                    shape_left_boundary + 1 : shape_right_boundary
                ].split(",")
                shape = [int(x.strip()) for x in shape]
                dtype_factor = 1
                total_static_input_size += reduce(lambda x, y: x * y, shape, 1)
                if op.type == "c_embedding":
                    arg_name_lower = (
                        "w" if arg_name_lower == "weight" else "ids"
                    )
                for arg_name in op.input_names:
                    if arg_name.lower() == arg_name_lower:
                        for var_name in op.input(arg_name):
                            var = vars[var_name]
                            total_actual_input_size += reduce(
                                lambda x, y: x * y, var.shape
                            )
                        break
        assert (
            total_static_input_size > 0 and total_actual_input_size > 0
        ), "Get input size failed."

        actual_runtime = (
            total_actual_input_size / total_static_input_size * runtime
        )
        return actual_runtime

    import paddle.cost_model as cm

    cost_model = cm.CostModel()
    cost_model.static_cost_data()
    DEFAULT_MULTIPLE = 2
    OP_NAME_MAPPING = {
        "c_embedding": "embedding",
        "matmul_v2": "matmul",
        "transpose2": "transpose",
        "reshape2": "reshape",
        "unsqueeze2": "unsqueeze",
        "reduce_sum": "sum",
        "elementwise_div": "divide",
    }

    standalone_cost_data = []
    # skip ops
    not_enum_ops = [
        "create_py_reader",
        "create_double_buffer_reader",
        "read",
        "assign",
    ]
    for distributed_program in distributed_programs:
        cost_data = {}
        vars = distributed_program.global_block().vars
        for op in distributed_program.global_block().ops:
            runtime = 0
            if op.type in not_enum_ops:
                cost_data[op.desc.id()] = runtime
                continue
            dtype = (
                str(vars[op.input_arg_names[0]].dtype)
                if op.input_arg_names
                else "float32"
            )
            if int(op.attr('op_role')) == int(OpRole.Backward):
                if "_grad" in op.type:
                    forward_op_name = op.type[:-5]
                    if forward_op_name in OP_NAME_MAPPING.keys():
                        forward_op_name = OP_NAME_MAPPING[forward_op_name]
                    op_cost = cost_model.get_static_op_time(
                        forward_op_name, forward=False, dtype=dtype
                    )
                    if op_cost:
                        runtime = _compute_runtime(op_cost, op, vars)
                    else:
                        op_cost = cost_model.get_static_op_time(
                            forward_op_name, dtype=dtype
                        )
                        if op_cost:
                            runtime = 2 * _compute_runtime(op_cost, op, vars)
            elif int(op.attr('op_role')) == int(OpRole.Forward):
                op_name = (
                    OP_NAME_MAPPING[op.type]
                    if op.type in OP_NAME_MAPPING.keys()
                    else op.type
                )
                op_cost = cost_model.get_static_op_time(op_name)
                if op_cost:
                    runtime = _compute_runtime(op_cost, op, vars)

            cost_data[op.desc.id()] = runtime

        standalone_cost_data.append(cost_data)

    return standalone_cost_data


def set_dist_op_desc_original_id(dist_op_desc, op_desc, dist_context):
    op_id = op_desc.id()
    op_original_id = op_desc.original_id()
    # First, try to set the original id to the id of the op_desc
    if op_id in dist_context._dist_ops_for_program:
        dist_op_desc.set_original_id(op_id)
        return
    # Second, try to set the original id to the original_id of the op_desc
    elif op_original_id in dist_context._dist_ops_for_program:
        dist_op_desc.set_original_id(op_original_id)
        return
    # Third, print error information if we cannot find the original id
    else:
        raise AssertionError(
            "Cannot find the original id in the distributed context"
        )


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def debug_program(program, path, name):
    filename = os.path.join(
        path, f"{name}_program.{paddle.distributed.get_rank()}"
    )
    with open(filename, 'w') as f:
        f.write(str(program))


def ring_id_to_process_group(ring_id):
    from .process_group import get_all_process_groups

    for g in get_all_process_groups():
        if g.id == ring_id:
            return g
    return None


def find_higher_order_backward_op(program):
    higher_order_op_suffix = ['_grad_grad', 'triple_grad']
    for block in program.blocks:
        for op in block.ops:
            for suffix in higher_order_op_suffix:
                if suffix in op.type:
                    return True

    return False


def get_var_numel(var):
    """
    input:
        - var: variable
    return:
        number of element in var
    """
    assert isinstance(var, Variable)
    assert -1 not in var.shape
    return reduce(lambda x, y: x * y, var.shape, 1)


def get_lr(optimizer):
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer.get_lr()
    elif isinstance(optimizer, paddle.static.Optimizer):
        if isinstance(optimizer._learning_rate, float):
            return optimizer._learning_rate
        else:
            return optimizer._learning_rate()
    else:
        raise TypeError(
            "'optimizer' must be object of class `paddle.optimizer.Optimizer`"
            f" or `paddle.static.Optimizer`, but got {type(optimizer)}."
        )


def initialize_pg_in_full_mode(all_process_groups, cur_rank):
    import socket

    from ...collective import _get_global_env

    has_recv_by_socket = []
    # This is a magic number
    magic_num = 500
    genv = _get_global_env()
    cur_rank_ip, cur_rank_port = genv.current_endpoint.split(":")
    cur_rank_recv_port = int(cur_rank_port) + magic_num
    server_socket = None
    # Large enough for recv rank
    buff_size = 1024
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((cur_rank_ip, cur_rank_recv_port))
    # The 10 is an empirical value
    server_socket.listen(10)
    client_sockets = {}
    for process_group in all_process_groups:
        if cur_rank not in process_group.ranks:
            continue
        if len(process_group.ranks) == 2:
            index = process_group.ranks.index(cur_rank)
            is_send = True if index == 0 else False
            if is_send:
                recv_rank = process_group.ranks[1]
                recv_rank_ip, recv_rank_port = genv.trainer_endpoints[
                    recv_rank
                ].split(":")
                connect_port = int(recv_rank_port) + magic_num
                client_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                )
                client_socket.connect((recv_rank_ip, connect_port))
                client_socket.send(str(cur_rank).encode('utf-8'))
                rank = client_socket.recv(buff_size).decode('utf-8')
                rank = int(rank)
                if rank != recv_rank:
                    raise ValueError(
                        f"Please check comm pair, the recv rank should be {recv_rank} but got {rank}."
                    )
                else:
                    print(
                        f"It is able to instantiate {process_group.ranks} as sender now."
                    )
                client_socket.close()
            else:
                send_rank = process_group.ranks[0]
                while True:
                    if send_rank not in has_recv_by_socket:
                        client_socket, recv_addr = server_socket.accept()
                        rank = int(client_socket.recv(buff_size).decode())
                        client_sockets[rank] = client_socket
                        has_recv_by_socket.append(rank)
                    else:
                        client_sockets[send_rank].send(
                            str(cur_rank).encode("utf-8")
                        )
                        client_sockets[send_rank].close()
                        print(
                            f"It is able to instantiate {process_group.ranks} as receiver now."
                        )
                        break
        process_group.instantiate()
    server_socket.close()


def is_recompute_op(op):
    return (
        op.has_attr('op_namescope')
        and "/auto_parallel/rc" in op.attr('op_namescope')
        and 'exclude_rc' not in op.attr('op_namescope')
    )


def is_recompute_exclude_op(op):
    return op.has_attr('op_namescope') and 'exclude_rc' in op.attr(
        'op_namescope'
    )


def set_recompute_segments(model, losses, strategy, program):
    from ...passes.auto_parallel_recompute import RecomputeState

    if not losses:
        return

    recompute = strategy.recompute
    if not recompute.enable:
        return

    # NOTE: hack to enable recompute in engine api for GPT-3
    # TODO support more PaddleNLP/CV models here
    # extract ckpts by specific model
    ckpts = []
    if isinstance(model, paddle.nn.Layer):
        if (
            hasattr(model, "gpt")
            and model.__class__.__name__
            in [
                'GPTForPretraining',
                'GPTForPretrainingAuto',
            ]
            and hasattr(model.gpt, "checkpoints")
        ):
            ckpts = model.gpt.checkpoints
            # last recompute segment is not need to recompute
            if len(ckpts) > 2:
                ckpts.pop()
        else:
            ckpts = recompute.checkpoints
    else:
        ckpts = recompute.checkpoints

    if not ckpts:
        return

    block = program.global_block()
    rc_state = RecomputeState(block, block.ops)
    rc_state.build_stats()
    checkpoints = rc_state.sort_checkpoints(ckpts)

    segments = []
    start_idx = -1
    pre_segment_end_idx = -1
    while start_idx + 1 < len(checkpoints):
        if start_idx == -1:
            ckpt_name = checkpoints[start_idx + 1]
            if ckpt_name not in rc_state.var_op_deps:
                start_idx += 1
                continue
            op_idx_list = rc_state.var_op_deps[ckpt_name]["var_as_output_ops"]
            if op_idx_list and max(op_idx_list) > 0:
                segments.append([0, max(op_idx_list) + 1])
        else:
            flag, min_idx, max_idx = rc_state.is_subgraph(
                [checkpoints[start_idx]], [checkpoints[start_idx + 1]]
            )
            if flag:
                min_idx = rc_state._update_segment_start(
                    min_idx, pre_segment_end_idx
                )
                segments.append([min_idx, max_idx + 1])
            else:
                logging.debug(
                    f"Could not recompute op range [{min_idx}] - [{max_idx + 1}] "
                )
        start_idx += 1

    for i, segment in enumerate(segments):
        for j in range(segment[0], segment[1]):
            block.ops[j]._set_attr(
                'op_namescope', "/auto_parallel/rc_" + str(i)
            )


def get_input_split_info(cur_rank, var, dist_context):
    # deduce how the input data is split among the cluster
    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
    process_mesh = tensor_dist_attr.process_mesh
    dims_mapping = tensor_dist_attr.dims_mapping

    if cur_rank not in process_mesh.process_ids:
        rank_id = _get_corresponding_rank(dist_context, process_mesh, cur_rank)
    else:
        rank_id = cur_rank

    batch_size_axis = dims_mapping[0]
    if batch_size_axis > -1 and process_mesh.shape[batch_size_axis] > 1:
        group_ranks = _get_comm_group(
            process_mesh.process_ids,
            process_mesh.shape,
            batch_size_axis,
            rank_id,
        )
        return len(group_ranks), group_ranks.index(rank_id)

    return 1, 0


def validate_opt(optimizer):
    if optimizer is not None:
        optimizer._parameter_list = None
        optimizer._param_groups = None
        if optimizer._grad_clip and isinstance(
            optimizer._grad_clip, paddle.nn.ClipGradByGlobalNorm
        ):
            optimizer._grad_clip._async_add_n = True
    return optimizer


def set_data_parallel(x):
    from ..interface import ProcessMesh, shard_tensor
    from .process_group import get_world_process_group

    world_ranks = get_world_process_group().ranks
    process_mesh = ProcessMesh(world_ranks, ['dp'])
    shard_spec = ['dp' if len(world_ranks) > 1 else None] + [
        None for _ in range(len(x.shape) - 1)
    ]

    return shard_tensor(x, process_mesh, shard_spec)


def is_naive_data_parallel(dist_context):
    # Naive data parallel only completes dist_attr once from the front to back.
    if not dist_context.data_parallel:
        return False

    ops_type = [
        op.type
        for op in dist_context._original_serial_main_program.global_block().ops
    ]
    if (
        not set(ops_type) & set(__not_naive_data_parallel_op__)
    ) and dist_context.data_parallel:
        return True
    return False


def _copy_tensor_dist_attr_to_cpp(cpp_dist_attr, py_dist_attr):
    py_process_mesh = py_dist_attr.process_mesh
    if py_process_mesh is not None:
        cpp_dist_attr.process_mesh = core.ProcessMesh(
            py_process_mesh.shape,
            py_process_mesh.process_ids,
            ["d" + str(i) for i in range(len(py_process_mesh.shape))],
        )
    cpp_dist_attr.dims_mapping = py_dist_attr.dims_mapping
    cpp_dist_attr.annotated = py_dist_attr.annotated


def _copy_tensor_dist_attr_from_cpp(cpp_dist_attr, py_dist_attr):
    from ..process_mesh import ProcessMesh

    cpp_process_mesh = cpp_dist_attr.process_mesh
    if cpp_process_mesh is not None:
        py_dist_attr.process_mesh = ProcessMesh(
            shape=cpp_process_mesh.shape,
            process_ids=cpp_process_mesh.process_ids,
        )
    py_dist_attr.dims_mapping = cpp_dist_attr.dims_mapping
    py_dist_attr.annotated = cpp_dist_attr.annotated


def _copy_op_dist_attr_to_cpp(cpp_dist_attr, py_dist_attr):
    py_process_mesh = py_dist_attr.process_mesh
    if py_process_mesh is not None:
        cpp_dist_attr.process_mesh = core.ProcessMesh(
            py_process_mesh.shape,
            py_process_mesh.process_ids,
            ["d" + str(i) for i in range(len(py_process_mesh.shape))],
        )
    cpp_dist_attr.impl_type = py_dist_attr.impl_type
    cpp_dist_attr.impl_idx = py_dist_attr.impl_idx
    cpp_dist_attr.is_recompute = py_dist_attr.is_recompute
    cpp_dist_attr.annotated = py_dist_attr.annotated
    for name, py_tensor_dist_attr in py_dist_attr.inputs_dist_attrs.items():
        cpp_tensor_dist_attr = cpp_dist_attr.get_input_dist_attr(name)
        _copy_tensor_dist_attr_to_cpp(cpp_tensor_dist_attr, py_tensor_dist_attr)
    for name, py_tensor_dist_attr in py_dist_attr.outputs_dist_attrs.items():
        cpp_tensor_dist_attr = cpp_dist_attr.get_output_dist_attr(name)
        _copy_tensor_dist_attr_to_cpp(cpp_tensor_dist_attr, py_tensor_dist_attr)


def _copy_op_dist_attr_from_cpp(cpp_dist_attr, py_dist_attr):
    from ..process_mesh import ProcessMesh

    cpp_process_mesh = cpp_dist_attr.process_mesh
    if cpp_process_mesh is not None:
        py_dist_attr.process_mesh = ProcessMesh(
            shape=cpp_process_mesh.shape,
            process_ids=cpp_process_mesh.process_ids,
        )
    py_dist_attr.impl_type = cpp_dist_attr.impl_type
    py_dist_attr.impl_idx = cpp_dist_attr.impl_idx
    py_dist_attr.is_recompute = cpp_dist_attr.is_recompute
    py_dist_attr.annotated = cpp_dist_attr.annotated
    for name, cpp_tensor_dist_attr in cpp_dist_attr.inputs_dist_attrs.items():
        py_tensor_dist_attr = py_dist_attr.get_input_dist_attr(name)
        _copy_tensor_dist_attr_from_cpp(
            cpp_tensor_dist_attr, py_tensor_dist_attr
        )
    for name, cpp_tensor_dist_attr in cpp_dist_attr.outputs_dist_attrs.items():
        py_tensor_dist_attr = py_dist_attr.get_output_dist_attr(name)
        _copy_tensor_dist_attr_from_cpp(
            cpp_tensor_dist_attr, py_tensor_dist_attr
        )


def _copy_dist_attr_to_cpp(dist_context):
    for dist_tensor in dist_context._dist_tensors_for_program.values():
        _copy_tensor_dist_attr_to_cpp(
            dist_tensor.serial_tensor.dist_attr, dist_tensor.dist_attr
        )

    for dist_op in dist_context._dist_ops_for_program.values():
        _copy_op_dist_attr_to_cpp(
            dist_op.serial_op.dist_attr, dist_op.dist_attr
        )


def _copy_dist_attr_from_cpp(dist_context):
    for dist_tensor in dist_context._dist_tensors_for_program.values():
        _copy_tensor_dist_attr_from_cpp(
            dist_tensor.serial_tensor.dist_attr, dist_tensor.dist_attr
        )

    for dist_op in dist_context._dist_ops_for_program.values():
        _copy_op_dist_attr_from_cpp(
            dist_op.serial_op.dist_attr, dist_op.dist_attr
        )


def _copy_dist_attr_to_cpp_for_graph(dist_context):
    for node in dist_context.serial_ordered_nodes:
        if node.is_var() and node.var() is not None:
            py_dist_attr = dist_context.get_tensor_dist_attr_for_graph(node)
            cpp_dist_attr = node.var().dist_attr
            _copy_tensor_dist_attr_to_cpp(cpp_dist_attr, py_dist_attr)
        if node.is_op() and node.op() is not None:
            py_dist_attr = dist_context.get_op_dist_attr_for_graph(node)
            cpp_dist_attr = node.op().dist_attr
            _copy_op_dist_attr_to_cpp(cpp_dist_attr, py_dist_attr)


def _copy_dist_attr_from_cpp_for_graph(dist_context):
    for node in dist_context.serial_ordered_nodes:
        if node.is_var() and node.var() is not None:
            py_dist_attr = dist_context.get_tensor_dist_attr_for_graph(node)
            cpp_dist_attr = node.var().dist_attr
            _copy_tensor_dist_attr_from_cpp(cpp_dist_attr, py_dist_attr)
        if node.is_op() and node.op() is not None:
            py_dist_attr = dist_context.get_op_dist_attr_for_graph(node)
            cpp_dist_attr = node.op().dist_attr
            _copy_op_dist_attr_from_cpp(cpp_dist_attr, py_dist_attr)


def insert_dependencies_for_two_ops(
    block,
    idx,
    prior_op,
    posterior_op,
    dist_context,
    is_recompute=False,
    sync=False,
    op_namescope=None,
):
    """
    dependency: prior_op should be run before posterior_op
    """

    if is_sequential_run():
        return

    assert (
        len(prior_op.output_arg_names) >= 1
    ), f"first op of dependency should at least have one output. [{prior_op}]"
    assert (
        len(posterior_op.input_arg_names) >= 1
    ), f"second op of dependency should at least have one input. [{posterior_op}]"
    prior_op_mesh = dist_context.get_op_dist_attr_for_program(
        prior_op
    ).process_mesh
    posterior_mesh = dist_context.get_op_dist_attr_for_program(
        posterior_op
    ).process_mesh
    assert (
        prior_op_mesh == posterior_mesh
    ), f"two ops of dependency should have same mesh but got [{prior_op_mesh}] and [{posterior_mesh}]"

    def _select_best_depend_var(vars):
        # parameter should not be dep var since it maybe partition in sharding pass
        vars = [var for var in vars if not var.is_parameter]
        assert len(vars) > 0
        vars_with_numels = [(var, get_var_numel(var)) for var in vars]
        vars_with_numels.sort(key=lambda x: x[1])

        return vars_with_numels[-1][0]

    first_var = _select_best_depend_var(
        [block.var(name) for name in prior_op.output_arg_names]
    )
    second_var = _select_best_depend_var(
        [block.var(name) for name in posterior_op.input_arg_names]
    )

    return insert_dependencies_for_vars(
        block,
        idx,
        first_var,
        second_var,
        dist_context,
        OpRole.Backward,
        process_mesh=prior_op_mesh,
        is_recompute=is_recompute,
        sync=sync,
        op_namescope=op_namescope,
        use_nop=False,
    )


def insert_dependencies_for_vars(
    block,
    idx,
    prior_vars,
    post_vars,
    dist_context,
    oprole,
    process_mesh=None,
    is_recompute=False,
    sync=False,
    op_namescope=None,
    use_nop=False,
    skip_insert_when_sequential_run=True,
):
    """
    dependency: op that generates prior_vars should be run before op that generates post_vars
    """

    if skip_insert_when_sequential_run and is_sequential_run():
        return

    if isinstance(prior_vars, Variable):
        prior_vars = [prior_vars]
    if isinstance(post_vars, Variable):
        post_vars = [post_vars]
    for prior_var in prior_vars:
        assert block.has_var(prior_var.name)
    for post_var in post_vars:
        assert block.has_var(post_var.name)

    post_dist_attr = dist_context.get_tensor_dist_attr_for_program(post_vars[0])
    if process_mesh is None:
        process_mesh = post_dist_attr.process_mesh
    assert process_mesh is not None

    use_nop = True
    if use_nop:
        depend_op = block._insert_op_without_sync(
            idx,
            type='nop',
            inputs={
                "X": prior_vars,
            },
            outputs={"Out": post_vars},
        )
    else:
        depend_op = block._insert_op_without_sync(
            idx,
            type='depend',
            inputs={
                "X": post_vars,
                "Dep": prior_vars,
            },
            outputs={"Out": post_vars},
        )
    depend_op._set_attr(OP_ROLE_KEY, oprole)

    # TODO: condition can be removed when add correct dist_attr for coalesce vars and ops in sharding_pass
    if is_recompute or process_mesh != [-1]:
        depend_op_dist_attr = OperatorDistAttr()
        depend_op_dist_attr.impl_idx = 0
        depend_op_dist_attr.impl_type = "default"
        depend_op_dist_attr.process_mesh = process_mesh
        depend_op_dist_attr.is_recompute = is_recompute
        depend_op_dist_attr.chunk_id = post_dist_attr.chunk_id
        for input_varname in depend_op.desc.input_arg_names():
            var = block.var(input_varname)
            mapping = dist_context.get_tensor_dist_attr_for_program(
                var
            ).dims_mapping
            depend_op_dist_attr.set_input_dims_mapping(input_varname, mapping)
        for output_varname in depend_op.desc.output_arg_names():
            var = block.var(output_varname)
            mapping = dist_context.get_tensor_dist_attr_for_program(
                var
            ).dims_mapping
            depend_op_dist_attr.set_output_dims_mapping(output_varname, mapping)
        dist_context.set_op_dist_attr_for_program(
            depend_op, depend_op_dist_attr
        )

    if op_namescope is not None:
        depend_op._set_attr('op_namescope', f"/{op_namescope}")

    if sync:
        block._sync_with_cpp()

    return depend_op


def is_dep_skip_op(op):
    if "c_" in op.type:
        return True

    return False


def _dygraph_guard_(func):
    def __impl__(*args, **kwargs):
        if paddle.framework.in_dynamic_mode():
            return func(*args, **kwargs)
        else:
            with paddle.base.dygraph.guard():
                return func(*args, **kwargs)

    return __impl__


dygraph_guard = wrap_decorator(_dygraph_guard_)


def is_sequential_run():
    return bool(
        paddle.get_flags("FLAGS_new_executor_sequential_run")[
            "FLAGS_new_executor_sequential_run"
        ]
    )


def get_pp_degree(dist_context):
    if len(dist_context.process_meshes) < 2:
        return 0, []

    sub_process_meshes = get_sub_process_mesh(dist_context.process_meshes)
    return len(sub_process_meshes), sub_process_meshes


def get_sub_process_mesh_by_program(dist_program):
    all_ops = dist_program.global_block().ops
    process_meshes = []

    for idx, op in enumerate(all_ops):
        if "pd_op" in op.name() and op.dist_attr:
            process_mesh = op.dist_attr.process_mesh
            if process_mesh not in process_meshes:
                process_meshes.append(process_mesh)

    sub_process_meshes = get_sub_process_mesh(process_meshes)
    sub_process_meshes = sorted(
        sub_process_meshes, key=lambda x: x.process_ids[0]
    )

    return sub_process_meshes


def get_sub_process_mesh(process_meshes):
    process_ids = set()
    sub_process_meshes = copy.deepcopy(process_meshes)

    for pm in sub_process_meshes:
        process_ids |= set(pm.process_ids)

    global_pm_idx = []
    has_sub_pm = False
    for idx, pm in enumerate(sub_process_meshes):
        if len(set(pm.process_ids)) == len(process_ids):
            global_pm_idx.append(idx)
        elif set(pm.process_ids) < process_ids:
            has_sub_pm = True

    if has_sub_pm:
        for idx in reversed(global_pm_idx):
            sub_process_meshes.pop(idx)

    return sub_process_meshes


def get_pp_stage(dist_context, rank):
    pp_idx = None
    for idx, process_mesh in enumerate(dist_context.process_meshes):
        if rank in process_mesh.process_ids:
            pp_idx = idx
            break
    return pp_idx


def get_pp_stage_by_pp_degree(pp_degree):
    cur_rank = paddle.distributed.get_rank()
    return get_pp_stage_by_rank(cur_rank, pp_degree)


def get_pp_stage_by_process_mesh(process_mesh, pp_degree):
    pp_stage_for_process_mesh = None
    for rank in process_mesh.process_ids:
        pp_stage = get_pp_stage_by_rank(rank, pp_degree)
        if pp_stage_for_process_mesh is not None:
            if pp_stage != pp_stage_for_process_mesh:
                return None
            assert (
                pp_stage == pp_stage_for_process_mesh
            ), f"Can't get pp_stage by process_mesh with different pp_stage {pp_stage} and {pp_stage_for_process_mesh}"
        pp_stage_for_process_mesh = pp_stage

    return pp_stage_for_process_mesh


def get_pp_stage_by_rank(rank, pp_degree):
    word_size = paddle.distributed.get_world_size()
    pp_group_size = word_size // pp_degree
    pp_stage = rank // pp_group_size
    return pp_stage


def wrap_data_for_completion(
    dist_op, input_names: list, output_names: list, attr_names: list
):
    """
    Get data used in inferring distributed attributes, including:
      1. DistTensorSpec for each input and output tensor of this dist_op.
      2. Operator attributes of this dist_op, e.g. transpose_x in matmul op.

    Args:
      dist_op: the DistributedOperator
      input_names: list, name of the dist_op's input tensors
      output_names: list, name of the dist_op's output tensors
      attr_names: list, attribute name of the dist_op's corresponding serial op

    Returns:
      input_specs: list, DistTensorSpec for each input tensor of the dist_op
      output_specs: list, DistTensorSpec for each output tensor of the dist_op
      attrs: dict, attribute map of the dist op

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('Depends on other ops.')
            >>> from paddle.distributed.auto_parallel.static.utils import wrap_data_for_completion

            >>> op_desc = dist_op.serial_op.desc
            >>> input_name_list = []
            >>> output_name_list = []

            >>> input_name_list.append(op_desc.input('X')[0]) # 'X' is the arg name for op
            >>> input_name_list.append(op_desc.input('Y')[0])
            >>> output_name_list.append(op_desc.output('Out')[0])

            >>> attr_name_list = ['trans_x', 'trans_y']
            >>> input_specs, output_specs, attrs = wrap_data_for_completion(
            ...        dist_op,
            ...        input_name_list,
            ...        output_name_list,
            ...        attr_name_list)

    """

    input_specs = []
    output_specs = []
    attrs = {}

    serial_op = dist_op.serial_op

    # Construct each input tensor's DistTensorSpec with shape and dist_attr
    for name in input_names:
        tensor_dist_attr = dist_op.dist_attr.get_input_dist_attr(name)
        var = serial_op.block._var_recursive(name)
        tensor_shape = var.shape
        dist_spec = DistTensorSpec(tensor_shape, tensor_dist_attr)
        input_specs.append(dist_spec)

    # Construct each output tensor's DistTensorSpec with shape and dist_attr
    for name in output_names:
        tensor_dist_attr = dist_op.dist_attr.get_output_dist_attr(name)
        var = serial_op.block._var_recursive(name)
        tensor_shape = var.shape
        dist_spec = DistTensorSpec(tensor_shape, tensor_dist_attr)
        output_specs.append(dist_spec)

    for attr_name in attr_names:
        attrs[attr_name] = serial_op.desc.attr(attr_name)

    return input_specs, output_specs, attrs


def get_dist_tensor_spec(dist_op, name, is_input=True):
    tensor_shape = dist_op.serial_op.block._var_recursive(name).shape
    if is_input:
        tensor_dist_attr = dist_op.dist_attr.get_input_dist_attr(name)
    else:
        tensor_dist_attr = dist_op.dist_attr.get_output_dist_attr(name)
    return DistTensorSpec(tensor_shape, tensor_dist_attr)


# get grad_var_to_var from distributed context, recording the mapping from backward grad variable to forward variable
# which is used for decomposing backward ops when enabling prim after distributed
def get_grad_var_to_var(dist_context):
    # get grad_var_to_var in distributed context
    grad_var_to_var_map = dist_context._dist_op_context.grad_var_to_var
    assert len(grad_var_to_var_map.keys()) == 1, "invalid grad_var_to_var"
    grad_var_to_var = grad_var_to_var_map[1]
    return grad_var_to_var


# update grad_var_to_var manually according to different distributed pass or strategy, thus recording complete and correct mapping between backward to forward
def update_grad_var_to_var(program, strategy, grad_var_to_var):
    from paddle.distributed.fleet.meta_optimizers.common import (
        OP_ROLE_KEY,
        OpRole,
    )

    # update grad_var_to_var according to different distributed pass
    first_backward_op_idx = -1
    for idx, op in enumerate(program.global_block().ops):
        # process @RESHARD variable in distributed training
        if (
            op.has_attr("op_namescope")
            and op.attr("op_namescope") == "/auto_parallel/reshard"
        ):
            reshard_op_types = [
                "split",
                "assign",
                "cast",
                "c_concat",
                "concat",
                "slice",
                "all_gather",
            ]
            if op.desc.type() in reshard_op_types:
                input_names = op.desc.input_names()
                if (
                    "X" in input_names
                    or "Input" in input_names
                    or "x" in input_names
                ):
                    inputs = (
                        op.desc.input("X")
                        if "X" in input_names
                        else (
                            op.desc.input("Input")
                            if "Input" in input_names
                            else op.desc.input("x")
                        )
                    )
                output_names = op.desc.output_names()
                if "Out" in output_names or "out" in output_names:
                    outputs = (
                        op.desc.output("Out")
                        if "Out" in output_names
                        else op.desc.output("out")
                    )
                if inputs[0] in grad_var_to_var.keys():
                    for output in outputs:
                        grad_var_to_var[output] = grad_var_to_var[inputs[0]]

        # process amp pass in distributed training
        if (
            strategy.amp.enable
            and op.has_attr(OP_ROLE_KEY)
            and (op.attr(OP_ROLE_KEY) & int(OpRole.Backward))
            and (op.attr(OP_ROLE_KEY) & int(OpRole.Loss))
        ):
            first_backward_op_idx = idx

    # process amp pass in distributed training
    if first_backward_op_idx != -1:
        scale_loss_op = program.global_block().ops[first_backward_op_idx - 1]
        scale_loss_var_name = scale_loss_op.desc.output("Out")[0]
        first_backward_op = program.global_block().ops[first_backward_op_idx]
        scale_loss_grad_var_name = first_backward_op.desc.output("Out")[0]
        if scale_loss_grad_var_name not in grad_var_to_var.keys():
            grad_var_to_var[scale_loss_grad_var_name] = scale_loss_var_name


def set_all_ops_op_role(block, op_role):
    all_ops = block.ops
    for op in all_ops:
        if op.op_role == -1:
            op.op_role = op_role
        for sub_block in op.blocks():
            set_all_ops_op_role(sub_block, op_role)


def fuse_param_func(
    fuse_params, is_qkv=False, num_heads=None, num_key_value_heads=None
):
    """fuse function for fusing weights

    (1) fuse_attention_qkv
        q => [q1,q2,q3,q4]
        k => [k1,k2,k3,k4] or [k1,k2] for GQA
        v => [v1,v2,v3,v4] or [v1,v2] for GQA
        fused weight => [q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4]
                or for GQA [q1,q2,k1,v1,q3,q4,k2,v2]
    (2) fuse_attention_ffn
        directly fuse weights to 1 parts
        [gate_weight], [up_weight] => [gate_weight, up_weight]

    Args:
        fuse_params (_type_): to be fused weights
        is_qkv (bool, optional): for attention qkv weights. Defaults to False.
        num_heads (_type_, optional): query heads. Defaults to None.
        num_key_value_heads (_type_, optional): key and value heads. Defaults to None.

    Returns:
        _type_: fused weights
    """
    concat_fn = paddle.concat
    split_fn = paddle.split

    if is_qkv:
        # fuse_attention_qkv
        assert (
            num_heads
        ), f"num_heads should be number of heads for Q, but got {num_heads}"
        assert (
            num_key_value_heads
        ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
        assert (
            len(fuse_params) == 3
        ), f"fuse_params length is not equal 3, it should be Q K V list. but got length {len(fuse_params)}"
        num_query_groups = num_heads // num_key_value_heads
        q_list = split_fn(fuse_params[0], num_heads, axis=-1)
        k_list = split_fn(fuse_params[1], num_key_value_heads, axis=-1)
        v_list = split_fn(fuse_params[2], num_key_value_heads, axis=-1)

        qkv_pairs = []
        for i in range(num_key_value_heads):
            qkv_pairs += q_list[
                i * num_query_groups : (i + 1) * num_query_groups
            ]
            qkv_pairs.append(k_list[i])
            qkv_pairs.append(v_list[i])
        return concat_fn(qkv_pairs, axis=-1)
    else:
        # fuse_attention_ffn
        return concat_fn(fuse_params, axis=-1)


def split_param_func(
    fused_param,
    split_nums=2,
    is_qkv=False,
    num_heads=None,
    num_key_value_heads=None,
):
    """split function for splitting weights

    (1) fuse_attention_qkv
        fused weight => [q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4]
                or for GQA [q1,q2,k1,v1,q3,q4,k2,v2]
        after split
        q => [q1,q2,q3,q4]
        k => [k1,k2,k3,k4] or [k1,k2] for GQA
        v => [v1,v2,v3,v4] or [v1,v2] for GQA
    (2) fuse_attention_ffn
        directly split weight to 2 parts
        [gate_weight, up_weight] => [gate_weight], [up_weight]

    Args:
        fused_param (_type_): len(fused_param)=1, only one weight to be splitted
        split_nums (int, optional): split_nums. Defaults to 2.
        is_qkv (bool, optional): for attention qkv weights. Defaults to False.
        num_heads (_type_, optional): query heads. Defaults to None.
        num_key_value_heads (_type_, optional): key and value heads. Defaults to None.

    Returns:
        _type_: splitted weights
    """
    concat_fn = paddle.concat
    split_fn = paddle.split

    if is_qkv:
        # fuse_attention_qkv
        assert (
            num_heads
        ), f"num_heads should be number of heads for Q, but got {num_heads}"
        assert (
            num_key_value_heads
        ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
        num_query_groups = num_heads // num_key_value_heads
        q_list, k_list, v_list = [], [], []
        split_heads = split_fn(
            fused_param, num_heads + 2 * num_key_value_heads, axis=-1
        )
        for i in range(num_key_value_heads):
            q_list += split_heads[
                i * (num_query_groups + 2) : (i + 1) * (num_query_groups + 2)
                - 2
            ]
            k_list.append(split_heads[(i + 1) * (num_query_groups + 2) - 2])
            v_list.append(split_heads[(i + 1) * (num_query_groups + 2) - 1])
        return (
            concat_fn(q_list, axis=-1),
            concat_fn(k_list, axis=-1),
            concat_fn(v_list, axis=-1),
        )
    else:
        # fuse_attention_ffn
        return split_fn(fused_param, split_nums, axis=-1)


def split_mesh(global_mesh: ProcessMesh, sub_mesh_dim: int):
    mesh_shape = global_mesh.shape
    mesh_ndim = len(mesh_shape)
    if sub_mesh_dim >= mesh_ndim or (
        sub_mesh_dim < 0 and -sub_mesh_dim > mesh_ndim
    ):
        raise ValueError(
            f"The sub_mesh_dim should between (-{mesh_ndim}, {mesh_ndim}]"
        )
    if sub_mesh_dim < 0:
        sub_mesh_dim += mesh_ndim

    process_ids = np.array(global_mesh.process_ids).reshape(mesh_shape)
    splitted_process_ids = np.split(
        process_ids, mesh_shape[sub_mesh_dim], axis=sub_mesh_dim
    )
    sub_mesh_list = []
    for sub_process_ids in splitted_process_ids:
        sub_mesh_list.append(
            ProcessMesh(sub_process_ids, global_mesh.dim_names)
        )

    return sub_mesh_list


# Note: This function is intended for internal use within the PaddlePaddle framework for optimizing computational graphs.
def update_pylayer_output(trivial_value):
    """
    Update the subblock within a pylayer operation by modifying its output argument.

    This function optimizes a pylayer operation by removing unnecessary outputs from the 'cf.yield' step.

    Args:
        trivale_value (pir::Value): The output argument of the pylayer operation to be modified.

    Example:
        (1) Original pylayer operation:
            (%1, %2) = "pd_op.pylayer" (%0) {
                () = "cf.tuple_pop" [id:1]
                (%3, %4) = "dist_op.xxx" [id:2]
                () = "cf.yield" [id:3] (%3, %4)
            }
        (2) After calling `update_pylayer_output(%4)`, the updated pylayer operation removes the unused output:
            (%1) = "pd_op.pylayer" (%0) {
                () = "cf.tuple_pop" [id:1]
                (%3) = "dist_op.xxx" [id:2]
                () = "cf.yield" [id:3] (%3)
            }

    Args:
        trivale_value(pir::Value): The output argument of the pylayer op to be updated.
    """
    define_op = trivial_value.get_defining_op()
    if define_op.get_parent_block().parent_op.name() != "pd_op.pylayer":
        return
    paddle.pir.set_insertion_point(define_op)
    fake_value = paddle.static.data(
        name="_fake_pylayer_out",
        shape=trivial_value.shape,
        dtype=trivial_value.dtype,
    )
    fake_value.set_type(trivial_value.type())
    trivial_value.replace_all_uses_with(fake_value)
