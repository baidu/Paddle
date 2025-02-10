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

import abc
import logging
import warnings

import paddle
import paddle.distributed as dist
from paddle.base.log_helper import get_logger
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..dist_attribute import OperatorDistAttr
from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    _get_corresponding_rank,
    compute_compatible_dims_mapping,
    is_optimize_op,
    set_dist_op_desc_original_id,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

_g_distributed_operator_impl_containers = {}

_g_elementwise_ops = [
    "assign",
    "elementwise",
    "gelu",
    # "dropout",
    "scale",
    "relu",
    "cast",
    # "gather",
    # "concat",
    "silu",
    "fused_softmax_mask_upper_triangle",
]
BACKWARD_ONLY_DIST_OPS = {'check_finite_and_unscale', 'update_loss_scaling'}

_gradient_sync_by_partial_ops = [
    "matmul_v2_grad",
    "elementwise_add_grad",
    "layer_norm_grad",
    "lookup_table_v2_grad",
    # "conv",
]


class ParallelMode:
    """
    the parallel mode for communication or auxiliary operator
    """

    DataParallel = "auto_parallel/data_parallel"
    TensorParallel = "auto_parallel/tensor_parallel"
    PipelineParallel = "auto_parallel/pipeline_parallel"
    MoEParallel = "auto_parallel/moe_parallel"


class SyncMode:
    """
    the synchronization mode for communication or auxiliary operator
    """

    AmpFlagSync = "auto_parallel/amp_flag_synchronization"
    GlobalNormSync = "auto_parallel/global_norm_synchronization"


def is_elementwise_op(op_type):
    if op_type in _g_elementwise_ops:
        return True
    if "elementwise" in op_type:
        return True
    return False


class DistributedOperatorImplContainer(abc.ABC):
    def __init__(self, op_type):
        self._type = op_type
        self._impls = []

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, op_type):
        self._type = op_type

    @property
    def impls(self):
        return self._impls

    def register_impl(self, dist_impl):
        assert (
            self.type == dist_impl.type
        ), "Op type of container must be same as that of the implementation."
        impl_idx = len(self.impls)
        dist_impl.idx = impl_idx
        self._impls.append(dist_impl)

    def get_impl(self, impl_idx):
        return self._impls[impl_idx]

    def get_input_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_input_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    def get_output_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_output_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    def get_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_auto_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    # (NOTE) Currently, both DistributedOperatorImplContainer and DistributedOperatorImpl have update_dims_mapping method.
    # But this method is supposed to be maintained by DistributedOperatorImplContainer, and we are ongoing adding method
    # to DistributedOperatorImplContainer and removing those in DistributedOperatorImpl.
    # @abc.abstractmethod
    def update_dims_mapping(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    # (NOTE) Currently we has limited DistributedOperatorImpls for an op to deal with different parallel patterns of this op.
    # This function help to choose the correct DistributedOperatorImpl based on the result from InferSPMD.
    # @abc.abstractmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        raise NotImplementedError("Please Implement this method in Subclass.")


class DistributedOperatorImpl(abc.ABC):
    def __init__(self, name):
        self._name = name
        self._type = None
        self._idx = None
        self._forward_implemented = False
        self._backward_implemented = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, op_type):
        self._type = op_type

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, impl_idx):
        self._idx = impl_idx

    # to be deprecated
    @abc.abstractmethod
    def is_input_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    # to be deprecated
    @abc.abstractmethod
    def is_output_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    # to be deprecated
    @abc.abstractmethod
    def is_auto_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    @abc.abstractmethod
    def forward(dist_ctx, *args, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    @abc.abstractmethod
    def backward(dist_ctx, *grad_outputs, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    # to be deprecated
    def update_dims_mapping(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")


def register_distributed_operator_impl_container(container):
    global _g_distributed_operator_impl_containers
    _g_distributed_operator_impl_containers[container.type] = container


def get_distributed_operator_impl_container(op_type):
    global _g_distributed_operator_impl_containers
    return _g_distributed_operator_impl_containers.get(op_type, None)


def register_distributed_operator_impl(op_type, dist_impl):
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    if dist_op_impl_container is not None:
        dist_impl.type = op_type
        dist_op_impl_container.register_impl(dist_impl)
    else:
        raise AssertionError(
            "Must register distributed operator registry first."
        )


def find_compatible_distributed_operator_impls(dist_op, fwd=True, partial=True):
    """
    Here just return the first compatible implementation.
    This will be improved by cost model in the future.
    """
    op_type = dist_op.serial_op.type
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    dist_op_eltwise_impl_container = get_distributed_operator_impl_container(
        "elementwise"
    )
    dist_op_default_impl_container = get_distributed_operator_impl_container(
        "default"
    )
    compatible_impls = []
    if partial:
        if fwd:
            # First, find impls in the corresponding container
            if dist_op_impl_container:
                compatible_impls.extend(
                    dist_op_impl_container.get_input_compatible_impls(dist_op)
                )
            # Second, find impls in the elementwise container
            if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
                compatible_impls.extend(
                    dist_op_eltwise_impl_container.get_input_compatible_impls(
                        dist_op
                    )
                )
            # Third, find impls in the default container
            if dist_op_default_impl_container:
                compatible_impls.extend(
                    dist_op_default_impl_container.get_input_compatible_impls(
                        dist_op
                    )
                )
        else:
            # First, find impls in the corresponding container
            if dist_op_impl_container:
                compatible_impls.extend(
                    dist_op_impl_container.get_output_compatible_impls(dist_op)
                )
            # Second, find impls in the elementwise container
            if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
                compatible_impls.extend(
                    dist_op_eltwise_impl_container.get_output_compatible_impls(
                        dist_op
                    )
                )
            # Third, find impls in the default container
            if dist_op_default_impl_container:
                compatible_impls.extend(
                    dist_op_default_impl_container.get_output_compatible_impls(
                        dist_op
                    )
                )
    else:
        # First, find impls in the corresponding container
        if dist_op_impl_container:
            compatible_impls.extend(
                dist_op_impl_container.get_compatible_impls(dist_op)
            )
        # Second, find impls in the elementwise container
        if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
            compatible_impls.extend(
                dist_op_eltwise_impl_container.get_compatible_impls(dist_op)
            )
        # Third, find impls in the default container
        if dist_op_default_impl_container:
            compatible_impls.extend(
                dist_op_default_impl_container.get_compatible_impls(dist_op)
            )

    if compatible_impls:
        # For now, just return the first compatible impl
        # best_compatible_impl = compatible_impls[0]
        best_compatible_impl = compatible_impls
    else:
        best_compatible_impl = None
    return best_compatible_impl


def find_distributed_operator_impl_container(dist_op):
    """
    Return a unique container for dist op.
    If not specific container found, default container will be return.
    """
    op_type = dist_op.serial_op.type

    # Op has a  match container
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    if dist_op_impl_container is None:
        # if op is register to elemwise spmd rule and has NO specific container implemented
        if is_elementwise_op(op_type):
            dist_op_impl_container = get_distributed_operator_impl_container(
                "elementwise"
            )
        # default container for all bottom line cases
        else:
            dist_op_impl_container = get_distributed_operator_impl_container(
                "default"
            )

    _logger.debug(
        f"Op [{op_type}] Complete DistAttr using {type(dist_op_impl_container).__name__}"
    )
    return dist_op_impl_container


def is_parameter_related(varname, block, dist_context=None):
    # TODO(zhaoyingli): maintain a dict in dist_context to record all variables which are be renamed
    if ".subprog_" in varname:
        varname = varname[: varname.index(".subprog_")]
    if ".cast_fp" in varname:
        varname = varname[: varname.index(".cast_fp")]
    if ".cast_bf" in varname:
        varname = varname[: varname.index(".cast_bf")]
    if ".quantized" in varname:
        varname = varname[: varname.index(".quantized")]
    assert block._find_var_recursive(
        varname
    ), f"cannot find var {varname} in cur block"
    var = block._var_recursive(varname)
    # NOTE(hack method): to find the param which is resharded
    if dist_context and "@RESHARD" in varname:
        varname = varname[: varname.index("@RESHARD")]
        serial_program = dist_context.serial_main_program
        var = serial_program.global_block()._find_var_recursive(varname)
        if var is None:
            return False
    # NOTE(liym27): when Y_var is not a parameter, but Y_var is resharded by a parameter.
    elif "reshard_api" in varname:
        for op in block.ops:
            if op.type == "assign" and varname in op.output("Out"):
                in_varname = op.input("X")[0]
                var = block._find_var_recursive(in_varname)
                if var is not None and var.is_parameter:
                    return True
    return var.is_parameter


def infer_shape(block, src_var, src_var_dist_attr, op_input_dist_attr):
    var_shape = block._var_recursive(src_var.name).shape
    var_topology = src_var_dist_attr.process_mesh.shape
    var_dims_mapping = src_var_dist_attr.dims_mapping

    complete_shape = []
    for idx, shape in enumerate(var_shape):
        if var_dims_mapping[idx] == -1:
            complete_shape.append(shape)
        else:
            new_shape = shape * var_topology[var_dims_mapping[idx]]
            complete_shape.append(new_shape)

    exact_shape = []
    input_topology = op_input_dist_attr.process_mesh.shape
    input_dims_mapping = op_input_dist_attr.dims_mapping
    for idx, shape in enumerate(complete_shape):
        if input_dims_mapping[idx] == -1:
            exact_shape.append(shape)
        else:
            new_shape = shape // input_topology[input_dims_mapping[idx]]
            exact_shape.append(new_shape)

    return exact_shape


def set_comm_op_dist_attr_for_program(
    new_op, process_mesh, tensor_dist_attr, ctx, **kwargs
):
    assert process_mesh is not None
    assert tensor_dist_attr is not None

    new_op_dist_attr = OperatorDistAttr()
    new_op_dist_attr.process_mesh = process_mesh
    if "chunk_id" in kwargs:
        new_op_dist_attr.chunk_id = kwargs["chunk_id"]
    for input_varname in new_op.desc.input_arg_names():
        new_op_dist_attr.set_input_dist_attr(input_varname, tensor_dist_attr)
    for output_varname in new_op.desc.output_arg_names():
        new_op_dist_attr.set_output_dist_attr(output_varname, tensor_dist_attr)
    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def naive_copy_op_dist_attr_for_program(new_op, ref_op, ctx):
    ref_dist_attr = ctx.get_op_dist_attr_for_program(ref_op)
    new_op_dist_attr = OperatorDistAttr()
    new_op_dist_attr.process_mesh = ref_dist_attr.process_mesh
    new_op_dist_attr.impl_type = ref_dist_attr.impl_type
    new_op_dist_attr.impl_idx = ref_dist_attr.impl_idx
    new_op_dist_attr.chunk_id = ref_dist_attr.chunk_id

    for input_name in ref_op.input_names:
        assert input_name in new_op.input_names
        assert len(ref_op.input(input_name)) == 1
        assert len(new_op.input(input_name)) == 1

        ref_tensor_dist_attr = ref_dist_attr.get_input_dist_attr(
            ref_op.input(input_name)[0]
        )
        new_op_dist_attr.set_input_dist_attr(
            new_op.input(input_name)[0], ref_tensor_dist_attr
        )

    for output_name in ref_op.output_names:
        assert output_name in new_op.output_names
        assert len(ref_op.output(output_name)) == 1
        assert len(new_op.output(output_name)) == 1

        ref_tensor_dist_attr = ref_dist_attr.get_output_dist_attr(
            ref_op.output(output_name)[0]
        )
        new_op_dist_attr.set_output_dist_attr(
            new_op.output(output_name)[0], ref_tensor_dist_attr
        )

    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def get_data_parallel_group(dist_ctx, op, act_grad_names, rank):
    """
    deduce the data parallel communication group for current operator.

    Args:
        dist_ctx (DistributedContext): dist context.
        op (Operator): the current (backward) operator which might need.
        act_grad_names (list): list of input activation grads variable name to the current operator.
        rank (int): global ranks index for current process.
    """
    dp_group = None

    op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
    process_mesh = op_dist_attr.process_mesh
    mesh_shape = process_mesh.shape
    # FIXME Hack for Pipeline Parallelism where the current operator
    # not belong to the mesh the current rank belong to.
    if rank not in process_mesh.process_ids:
        rank = _get_corresponding_rank(dist_ctx, process_mesh, rank)

    for var_name in act_grad_names:
        var_dim_mapping = op_dist_attr.get_input_dims_mapping(var_name)
        # consider that the variable's shape is [], which is 0-D
        # TODO utilize the batch_dim attr instead of "0" in future
        batch_size_axis = var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1

        if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
            group_ranks = _get_comm_group(
                process_mesh.process_ids,
                process_mesh.shape,
                batch_size_axis,
                rank,
            )
            dp_group = new_process_group(group_ranks)
            break
    if dp_group is not None:
        return [dp_group]
    else:
        return []


def sync_and_scale_gradients(dist_ctx, op, groups, allreduce_var_names):
    """
    insert the allreduce and scale ops for gradients of model
    parameters for operator in data parallelism.

    Args:
        dist_ctx (DistributedContext): dist context.
        op (Operator): the current (backward) operator which might need.
        allreduce_var_names (list): list of the parameter's grads variable name in the current operator output.
    """

    op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
    process_mesh = op_dist_attr.process_mesh
    chunk_id = op_dist_attr.chunk_id
    dist_op_context = dist_ctx.dist_op_context
    main_block = dist_op_context.work_block

    allreduce_type = "c_allreduce_sum"
    need_scale = dist_ctx.gradient_scale
    scale_using_allreduce_avg = dist_ctx.gradient_scale_using_allreduce_avg

    # With nccl_version > 2.10.00, we can use c_allreduce_avg to replace c_allreduce_sum and eliminate the scale op.
    if (
        need_scale
        and scale_using_allreduce_avg
        and int(paddle.version.nccl()) > 21000
    ):
        allreduce_type = "c_allreduce_avg"
        need_scale = False

    for group in groups:
        group_size = len(group.ranks)

        for var_name in allreduce_var_names:
            added_ops = []
            grad_var = main_block.var(var_name)
            allreduce_op = main_block.append_op(
                type=allreduce_type,
                inputs={'X': [grad_var]},
                outputs={'Out': [grad_var]},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    OP_ROLE_KEY: OpRole.Backward,
                },
            )
            allreduce_op._set_attr(
                'op_namescope', '/' + ParallelMode.DataParallel
            )
            added_ops.append(allreduce_op)

            if need_scale:
                scale_op = main_block.append_op(
                    type='scale',
                    inputs={'X': grad_var},
                    outputs={'Out': grad_var},
                    attrs={
                        'scale': 1.0 / group_size,
                        OP_ROLE_KEY: OpRole.Backward,
                    },
                )
                scale_op._set_attr(
                    'op_namescope', '/' + ParallelMode.DataParallel
                )
                added_ops.append(scale_op)

            dims_mapping = op_dist_attr.get_output_dims_mapping(grad_var.name)
            assert (
                dims_mapping is not None
            ), f"Unexpected: dims_mapping of output [{grad_var.name}] of op [{op_dist_attr.op_type}] is None"
            # NOTE auxiliary op's dist attr should follow dist_op not dist_tensor
            for new_op in added_ops:
                new_op_attr = OperatorDistAttr()
                new_op_attr.process_mesh = process_mesh
                new_op_attr.chunk_id = chunk_id
                new_op_attr.set_output_dims_mapping(grad_var.name, dims_mapping)
                new_op_attr.set_input_dims_mapping(grad_var.name, dims_mapping)
                dist_ctx.set_op_dist_attr_for_program(new_op, new_op_attr)


def get_partial_groups(dist_ctx, op, out_grad_names, rank):
    """
    deduce the partial communication group for current operator output vars.

    Args:
        dist_ctx (DistributedContext): dist context.
        op (Operator): the current (backward) operator which might need.
        out_grad_names (list): list of the output parameter's grads variable name of the current operator.
        rank (int): global ranks index for current process.
    """
    op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
    process_mesh = op_dist_attr.process_mesh
    mesh_shape = process_mesh.shape

    groups = []

    partial_dims = None
    for var_name in out_grad_names:
        var_dist_attr = op_dist_attr.get_output_dist_attr(var_name)
        if partial_dims is None:
            partial_dims = var_dist_attr._partial_dims()
        else:
            assert (
                partial_dims == var_dist_attr._partial_dims()
            ), f"Partial dims of outputs {out_grad_names} of op [{op.type}] is not consistent"

    partial_dims = list(partial_dims)
    partial_dims.sort()

    # FIXME Hack for Pipeline Parallelism where the current operator
    # not belong to the mesh the current rank belong to.
    if rank not in process_mesh.process_ids:
        rank = _get_corresponding_rank(dist_ctx, process_mesh, rank)

    for dim in partial_dims:
        if mesh_shape[dim] > 1:
            group_ranks = _get_comm_group(
                process_mesh.process_ids,
                process_mesh.shape,
                dim,
                rank,
            )
            groups.append(new_process_group(group_ranks))

    return groups


def gradient_synchronization(
    dist_ctx, op, act_grad_names, out_grad_names, rank
):
    """
    conduct the allreduce and scaling for gradients of model
    parameters for operator in parallelism train.

    Args:
        dist_ctx (DistributedContext): dist context.
        op (Operator): the current (backward) operator which might need.
        act_grad_names (list): list of input activation grads variable name to the current operator.
        out_grad_names (list): list of the output parameter's grads variable name of the current operator.
        rank (int): global ranks index for current process.
    """

    if not is_in_backward_phase(dist_ctx):
        return

    if (
        is_optimize_op(op)
        or len(act_grad_names) == 0
        or len(out_grad_names) == 0
    ):
        return

    if op.type in _gradient_sync_by_partial_ops:
        sync_groups = get_partial_groups(dist_ctx, op, out_grad_names, rank)
    # NOTE we reverse the following old branch to support operators (e.g. fuse operators) that haven't been adopted for partial inferspmd,
    # and remove this branch after all operators are adopted for partial inferspmd.
    else:
        sync_groups = get_data_parallel_group(
            dist_ctx, op, act_grad_names, rank
        )

    if len(sync_groups) < 1:
        return

    sync_and_scale_gradients(dist_ctx, op, sync_groups, out_grad_names)


def is_data_parallel_scale_op(op):
    return (
        op.type == "scale"
        and op.desc.has_attr("op_namescope")
        and ParallelMode.DataParallel in op.desc.attr("op_namescope")
    )


def is_data_parallel_reduce_op(op):
    is_allreduce_op = op.type in [
        "c_allreduce_sum",
        "c_allreduce_avg",
    ]
    is_reduce_op = op.type == "reduce" and op.desc.attr("reduce_type") in [
        dist.ReduceOp.SUM,
        dist.ReduceOp.AVG,
    ]
    return (
        (is_allreduce_op or is_reduce_op)
        and op.desc.has_attr("op_namescope")
        and ParallelMode.DataParallel in op.desc.attr("op_namescope")
    )


def is_amp_flag_sync_op(op):
    return (
        op.type == "c_allreduce_max"
        and op.desc.has_attr("op_namescope")
        and SyncMode.AmpFlagSync in op.desc.attr("op_namescope")
    )


def is_global_norm_sync_op(op):
    return (
        op.type == "c_allreduce_sum"
        and op.desc.has_attr("op_namescope")
        and SyncMode.GlobalNormSync in op.desc.attr("op_namescope")
    )


def is_in_backward_phase(dist_ctx):
    # NOTE currently high-order differential in Paddle dose NOT distinguish gradient computation operators
    # in Forward phase and operators in Backward phase (both with op_role=1), which will mislead
    # auto parallel to add gradient synchronization for gradient computation operators in Forward phase.
    # we use this FLAG to distinguish these two phases temporarily.

    return dist_ctx.dist_op_context.in_backward_phase()


def merge_forward_backward_dims_mapping(fw_results, bw_results):
    flatten_fw_inputs = paddle.utils.flatten(fw_results[0])
    flatten_fw_outputs = paddle.utils.flatten(fw_results[1])
    flatten_bw_inputs = paddle.utils.flatten(bw_results[0])
    flatten_bw_outputs = paddle.utils.flatten(bw_results[1])
    ninputs = len(flatten_fw_inputs)
    noutputs = len(flatten_fw_outputs)
    inferred_input_dims_mappings = []
    inferred_output_dims_mappings = []

    for i in range(ninputs):
        compatible_dims_mapping = compute_compatible_dims_mapping(
            [
                flatten_fw_inputs[i].dims_mapping,
                flatten_bw_inputs[i].dims_mapping,
            ]
        )
        inferred_input_dims_mappings.append(compatible_dims_mapping)

    for i in range(noutputs):
        compatible_dims_mapping = compute_compatible_dims_mapping(
            [
                flatten_fw_outputs[i].dims_mapping,
                flatten_bw_outputs[i].dims_mapping,
            ]
        )
        inferred_output_dims_mappings.append(compatible_dims_mapping)
    return inferred_input_dims_mappings, inferred_output_dims_mappings


def update_op_dims_mapping(
    dist_op, input_arg_names, output_arg_names, fw_results, bw_results
):
    (
        inferred_input_dims_mappings,
        inferred_output_dims_mappings,
    ) = merge_forward_backward_dims_mapping(fw_results, bw_results)

    op_dist_attr = dist_op.dist_attr
    changed = False
    if len(input_arg_names) != len(inferred_input_dims_mappings):
        warnings.warn(
            f"dims mapping is NOT Match, inferred [{len(inferred_input_dims_mappings)}], original: [{len(input_arg_names)}]; dist op: [{dist_op}]"
        )
    if len(output_arg_names) != len(inferred_output_dims_mappings):
        warnings.warn(
            f"dims mapping is NOT Match, inferred [{len(inferred_output_dims_mappings)}], original: [{len(output_arg_names)}]; dist op: [{dist_op}]"
        )

    for i in range(len(input_arg_names)):
        original_dims_mapping = op_dist_attr.get_input_dims_mapping(
            input_arg_names[i]
        )
        inferred_dims_mapping = inferred_input_dims_mappings[i]
        if (inferred_dims_mapping is not None) and (
            original_dims_mapping != inferred_dims_mapping
        ):
            _logger.debug(
                f"Changed: Op [{dist_op.serial_op.type}], name [{input_arg_names[i]}], Original [{original_dims_mapping}], Inferred [{inferred_dims_mapping}]"
            )
            changed = True
            op_dist_attr.set_input_dims_mapping(
                input_arg_names[i], inferred_dims_mapping
            )
        # TODO support partial for inputs

    for i in range(len(output_arg_names)):
        original_dims_mapping = op_dist_attr.get_output_dims_mapping(
            output_arg_names[i]
        )
        inferred_dims_mapping = inferred_output_dims_mappings[i]
        if (inferred_dims_mapping is not None) and (
            original_dims_mapping != inferred_dims_mapping
        ):
            _logger.debug(
                f"Changed: Op [{dist_op.serial_op.type}], name [{output_arg_names[i]}], Original [{original_dims_mapping}], Inferred [{inferred_dims_mapping}]"
            )
            changed = True
            op_dist_attr.set_output_dims_mapping(
                output_arg_names[i], inferred_dims_mapping
            )

        # NOTE in partial stage-I, we infer partial for output in infer_forward only
        output_dist_attr = op_dist_attr.get_output_dist_attr(
            output_arg_names[i]
        )
        output_idx = output_arg_names.index(output_arg_names[i])
        if (
            fw_results[1][output_idx]._partial_dims()
            != output_dist_attr._partial_dims()
        ):
            # _logger.info(
            #     "Changed: Op [{}], tensor name [{}], Original partial on [{}], Inferred partial on [{}]".format(
            #         dist_op.serial_op.type,
            #         output_arg_names[i],
            #         output_dist_attr._partial_dims(),
            #         fw_results[1][output_idx]._partial_dims(),
            #     )
            # )
            output_dist_attr._clean_partial_status()
            output_dist_attr._set_partial_dims(
                list(fw_results[1][0]._partial_dims())
            )
            changed = True

    return changed


def get_default_distributed_operator_impl():
    dist_op_default_impl_container = get_distributed_operator_impl_container(
        "default"
    )
    num_impls = len(dist_op_default_impl_container.impls)
    assert num_impls == 1, f"Default dist op has [{num_impls}] impls"
    return dist_op_default_impl_container.get_impl(0)


def copy_op_without_infer_shape(src_op, block, ctx, varname_kwargs):
    new_op = block.append_op(type='nop')
    new_op_desc = new_op.desc
    new_op_desc.copy_from(src_op.desc)
    set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
    for input_name in src_op.desc.input_names():
        new_op_desc.set_input(input_name, varname_kwargs[input_name])
    for output_name in src_op.desc.output_names():
        new_op_desc.set_output(output_name, varname_kwargs[output_name])
    # TODO: should we add a new dist attr for the new op here?
    return new_op
