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
import logging

import paddle
from paddle.base.log_helper import get_logger
from paddle.framework import core
from paddle.utils import unique_name

from ...random import determinate_rng, is_enable_auto_rand_ctrl
from ..completion import get_phi_spmd_rule
from ..utils import (
    get_dist_tensor_spec,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    set_var_dist_attr,
)
from .common import (
    DistributedOperatorImplContainer,
    merge_forward_backward_dims_mapping,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)
from .dist_eltwise import DistributedDefaultImpl0, DistributedElementwiseImpl0

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class DistributedDropout(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc

        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        mask_name = op_desc.output('Mask')[0]
        # seed_name = op_desc.input('Seed')[0]  // seed is a scalar and leave it to be unsharded

        x_spec = get_dist_tensor_spec(dist_op, x_name)
        output_spec = get_dist_tensor_spec(dist_op, out_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("dropout")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(x_spec)
        bw_results = rule.infer_backward(x_spec, output_spec)

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op, [x_name], [out_name], fw_results, bw_results
        )

        # step5: update mask and seed dropout special
        if changed:
            (
                _,
                inferred_output_dims_mappings,
            ) = merge_forward_backward_dims_mapping(fw_results, bw_results)
            dist_op.dist_attr.set_output_dims_mapping(
                mask_name, inferred_output_dims_mappings[0]
            )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        # all dropout op use Dropout with Random Control dist operator impl.
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.impl_type = "dropout"
        op_dist_attr.impl_idx = 0

        return False


register_distributed_operator_impl_container(DistributedDropout("dropout"))


# Dist Dropout with Random Control
# Dropout re-use the compatible and cost function of elementwise
class DistributedDropoutImpl0(DistributedElementwiseImpl0):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    @staticmethod
    def forward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), f"forward op [{src_op}] don't have dist attribute !"

        if is_enable_auto_rand_ctrl() and not op_dist_attr.is_recompute:
            # check validation of inputs / outputs
            assert 'X' in kwargs, "input [{}] is not given".format('X')
            assert (
                len(kwargs['X']) == 1
            ), "input X should be only one tensor but got {}".format(
                kwargs['X']
            )
            assert 'Seed' in kwargs, "input [{}] is not given".format('Seed')

            if (
                src_op.has_attr("fix_seed")
                and src_op.attr("fix_seed")
                and src_op.has_attr("seed")
                and src_op.attr("seed")
            ):
                _logger.info(
                    f"Auto Parallel Random Control Skipped Since manual seed is set by user: {src_op}"
                )
            elif rank_id not in op_dist_attr.process_mesh.process_ids:
                pass
            # NOTE Adopt for recompute
            # If user already set seed, We should not modify it. But if the seed is added by recompute pass, it should be under control.
            # TODO  in future recompute pass should happen after parallel partition. and remove this at that time.
            elif len(kwargs['Seed']) > 0 or len(src_op.input("Seed")) > 0:
                seed_var_name = kwargs['Seed'][0]
                if seed_var_name.startswith('rc_seed'):
                    pre_op = main_block.ops[-1]
                    assert (
                        pre_op.type == "seed"
                        and len(pre_op.attr("rng_name")) == 0
                    ), f"found exception op {pre_op}"

                    # determinate rng
                    X_var = main_block._var_recursive(kwargs['X'][0])
                    X_dims_mapping = op_dist_attr.get_input_dims_mapping(
                        X_var.name
                    )
                    process_mesh = op_dist_attr.process_mesh
                    rng_name = determinate_rng(
                        rank_id, X_dims_mapping, process_mesh
                    )
                    # make recompute seed under control
                    pre_op._set_attr("rng_name", rng_name)
                    pre_op._set_attr("deterministic", True)
                    pre_op._set_attr("force_cpu", True)
                else:
                    _logger.info(
                        f"Auto Parallel Random Control Skipped Since manual seed is set by user: {src_op}"
                    )
            else:
                # determinate rng
                X_var = main_block._var_recursive(kwargs['X'][0])
                X_dims_mapping = op_dist_attr.get_input_dims_mapping(X_var.name)
                process_mesh = op_dist_attr.process_mesh

                rng_name = determinate_rng(
                    rank_id, X_dims_mapping, process_mesh
                )
                assert rng_name is not None and rng_name != ""

                # insert seed op
                seed_var = main_block.create_var(
                    name=unique_name.generate_with_ignorable_key(
                        ".".join(["tensor_parallel_seed", 'tmp'])
                    ),
                    dtype=paddle.int32,
                    type=core.VarDesc.VarType.DENSE_TENSOR,
                    persistable=False,
                    stop_gradient=False,
                )

                # set new seed_var's dist_attr
                seed_var_dims_mapping = [-1]
                seed_var_dist_attr = set_var_dist_attr(
                    ctx,
                    seed_var,
                    seed_var_dims_mapping,
                    process_mesh,
                    chunk_id=op_dist_attr.chunk_id,
                )

                # adopt for recompute
                # force_cpu to reduce sync copy from CPU->GPU->CPU, and reduce pipeline hang
                seed_op = main_block.append_op(
                    type='seed',
                    outputs={'Out': seed_var},
                    attrs={
                        'deterministic': True,
                        'rng_name': rng_name,
                        'force_cpu': True,
                    },
                )
                seed_op._set_attr('op_namescope', 'auto_tensor_parallel_seed')
                # set new seed op's dist_attr
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    seed_op,
                    process_mesh,
                    seed_var_dims_mapping,
                    ctx,
                    chunk_id=op_dist_attr.chunk_id,
                )

                # modify dropout op
                src_op.desc.set_input("Seed", [seed_var.name])
                src_op.desc._set_attr("fix_seed", False)
                src_op.desc._set_attr("seed", 0)
                op_dist_attr.set_input_dist_attr(
                    seed_var.name, seed_var_dist_attr
                )
                kwargs['Seed'] = [seed_var.name]

        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        # dropout backward is deterministic by mask, and not need for random state control
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "dropout", DistributedDropoutImpl0("random_control")
)
