# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

from paddle.base import core
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY
from paddle.static import Parameter

_supported_optimizer_type = [
    "adam",
    "adamax",
    "adamw",
    "decayed_adagrad",
    "momentum",
    "dgc_momentum",
    "lars_momentum",
    "merged_momentum",
    "lamb",
    "sgd",
]


def tensor_parallel_sync_filter_fn(
    param, pos_emb=True, layer_norm=True, bias=True
):
    """
    Layer filter function for tensor parallelism transformer.

    In tensor parallelism of transformer like model, there is 4 kind of param
    that are supposed to be the same in all tensor parallel peers:
        * position embedding
        * scale of layer norm
        * bias of layer norm
        * bias of row parallel linear

    set corresponding input args to select specific layers.
    NOTE  adopting the param name pattern for different transformer blocks.
    """
    p_name = param.name
    if pos_emb and p_name.startswith("pos_embedding"):
        return True

    elif layer_norm and p_name.endswith("_layer_norm_bias"):
        return True

    elif layer_norm and p_name.endswith("_layer_norm_scale"):
        return True

    elif bias and ".b_" in p_name and (param.is_distributed is False):
        return True

    else:
        return False


def resolute_tensor_parallel_ring_id(program):
    ops = program.global_block().ops
    ring_id = None

    for op in ops:
        if op.type == "c_identity":
            if ring_id is None:
                ring_id = int(op.attr("ring_id"))
            else:
                assert ring_id == int(
                    op.attr("ring_id")
                ), "Found two different ring_id for Tensor Parallel: ring_id={} and ring_id={}.".format(
                    ring_id, int(op.attr("ring_id"))
                )
    assert ring_id is not None, "Could NOT found ring_id for Tensor Parallel."

    return ring_id


def copy_parameters(block_, params):
    for param in params:
        new_p = Parameter(
            block=block_,
            shape=param.shape,
            dtype=param.dtype,
            type=param.type,
            lod_level=(
                param.lod_level
                if param.type == core.VarDesc.VarType.DENSE_TENSOR
                else None
            ),
            stop_gradient=param.stop_gradient,
            trainable=param.trainable,
            optimize_attr=param.optimize_attr,
            regularizer=param.regularizer,
            error_clip=param.error_clip,
            name=param.name,
        )
        assert (
            param.is_distributed is False
        ), f"Try to sync Distributed Parameter: {param}"
        new_p.is_distributed = False

    block_.vars[new_p.name] = new_p


def insert_sync_op(
    block, idx, tp_degree, sync_mode, sync_ring_id, src_rank, varname, op_role
):
    if sync_mode == "broadcast":
        block._insert_op_without_sync(
            idx,
            type='broadcast',
            inputs={'x': varname},
            outputs={'out': varname},
            attrs={
                'ring_id': sync_ring_id,
                'root': src_rank,
                OP_ROLE_KEY: op_role,
            },
        )

    elif sync_mode == "average":
        block._insert_op_without_sync(
            idx,
            type='scale',
            inputs={'X': varname},
            outputs={'Out': varname},
            attrs={'scale': 1.0 / tp_degree, OP_ROLE_KEY: op_role},
        )
        block._insert_op_without_sync(
            idx,
            type='c_allreduce_sum',
            inputs={'X': varname},
            outputs={'Out': varname},
            attrs={
                'ring_id': sync_ring_id,
                'use_calc_stream': True,
                OP_ROLE_KEY: op_role,
            },
        )
    else:
        raise NotImplementedError(
            f'Sync mode of [{sync_mode}] is NOT supported.'
        )


def insert_synchronization(
    block,
    params_to_sync,
    tp_degree,
    sync_ring_id,
    sync_param,
    sync_grad,
    sync_moment,
    sync_mode,
    src_rank,
):
    unsync_param_names = [p.name for p in params_to_sync]

    for idx, op in reversed(list(enumerate(block.ops))):
        if op.type in _supported_optimizer_type:
            assert "Param" in op.input_names
            assert len(op.input("Param")) == 1
            param_name = op.input("Param")[0]
            op_role = op.attr(OP_ROLE_KEY)

            if param_name in unsync_param_names:
                unsync_param_names.remove(param_name)

                # Param sync after opt
                if sync_param:
                    assert (
                        "ParamOut" in op.output_names
                        and op.output("ParamOut")[0] == param_name
                    )
                    insert_sync_op(
                        block,
                        idx + 1,
                        tp_degree,
                        sync_mode,
                        sync_ring_id,
                        src_rank,
                        param_name,
                        op_role,
                    )

                    if (
                        "MasterParamOut" in op.output_names
                        and len(op.output("MasterParamOut")) == 1
                    ):
                        sync_var = op.output("MasterParamOut")[0]
                        insert_sync_op(
                            block,
                            idx + 1,
                            tp_degree,
                            sync_mode,
                            sync_ring_id,
                            src_rank,
                            sync_var,
                            op_role,
                        )

                # Moment sync after opt
                if sync_moment:
                    if (
                        "Moment1Out" in op.output_names
                        and len(op.output("Moment1Out")) == 1
                    ):
                        sync_var = op.output("Moment1Out")[0]
                        insert_sync_op(
                            block,
                            idx + 1,
                            tp_degree,
                            sync_mode,
                            sync_ring_id,
                            src_rank,
                            sync_var,
                            op_role,
                        )

                    if (
                        "Moment2Out" in op.output_names
                        and len(op.output("Moment2Out")) == 1
                    ):
                        sync_var = op.output("Moment2Out")[0]
                        insert_sync_op(
                            block,
                            idx + 1,
                            tp_degree,
                            sync_mode,
                            sync_ring_id,
                            src_rank,
                            sync_var,
                            op_role,
                        )

                # Grad sync before opt
                if sync_grad:
                    assert (
                        "Grad" in op.input_names and len(op.input("Grad")) == 1
                    )
                    sync_var = op.input("Grad")[0]
                    insert_sync_op(
                        block,
                        idx,
                        tp_degree,
                        sync_mode,
                        sync_ring_id,
                        src_rank,
                        sync_var,
                        op_role,
                    )

    assert (
        len(unsync_param_names) == 0
    ), f"The following param is unsync by some error: {unsync_param_names}"


def add_extra_synchronization(
    program,
    params_filter_fn=tensor_parallel_sync_filter_fn,
    tp_degree=8,
    sync_mode="broadcast",
    sync_param=True,
    sync_grad=False,
    sync_moment=False,
    src_rank=0,
    sync_ring_id=None,
):
    """
    Inplace add extra synchronization for input program.

    program(Paddle.Program): distributed train program.

    params_filter_fn(callable): function to filter out parameter for synchronization.

    sync_mode(string): select from
        "broadcast": parameter is sync by broadcasted from 'src_rank' to all other ranks.
        "average": parameter is sync by average among all ranks

    src_rank(int): the src used in broadcast sync_mode.

    sync_param(bool): extra synchronize parameters.

    sync_grad(bool): extra synchronize gradients.

    sync_grad(bool): extra synchronize optimizer momentum.

    sync_ring_id(int): communicator id use for synchronization, if it is None, use the ring_id of tensor parallel.
    """

    logger.info("Constructing Extra Parameter Synchronization.")
    logger.info(
        f"Tensor Parallel Degree: {tp_degree}, Synchronization mode: {sync_mode}"
    )

    # adopt for pipeline opt
    if program._pipeline_opt is not None:
        assert (
            program._pipeline_opt['section_program'] is not None
        ), "Pipeline is enable but section_program is None"
        program = program._pipeline_opt['section_program']

    # step1: collect the param that need to be sync
    params_to_sync = []
    # TODO support multiple blocks with different parameter.
    all_params = program.global_block().all_parameters()
    for param in all_params:
        if params_filter_fn(param):
            params_to_sync.append(param)
    logger.info(
        "The following param are going to be synchronization everytime the optimizer update phase of the program is run: "
    )
    logger.info([p.name for p in params_to_sync])

    # step2: resolute synchronization communicator group (ring_id)
    if sync_ring_id is None:
        sync_ring_id = resolute_tensor_parallel_ring_id(program)

    # step3: insert synchronization
    # TODO support gradient merge with different update block
    block = program.global_block()
    insert_synchronization(
        block,
        params_to_sync,
        tp_degree,
        sync_ring_id,
        sync_param,
        sync_grad,
        sync_moment,
        sync_mode,
        src_rank,
    )
