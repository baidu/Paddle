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

import re
import warnings

import paddle
import paddle.distributed as dist
from paddle.base.core import TensorDistAttr
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.framework import core

from .pass_base import PassBase, register_pass


@register_pass("auto_parallel_c_embedding_pass")
class AutoParallelCEmbeddingPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        hcg = fleet.get_hybrid_communicate_group()
        mp_size = hcg.get_model_parallel_world_size()
        if mp_size > 1:
            return True
        warnings.warn("c_embedding pass is only applicable to tnesor parallel.")
        return False

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        concrete_program = self.get_attr("concrete_program")
        ops = main_program.global_block().ops
        for i, op in enumerate(ops):
            if op.name() == 'pd_op.embedding':
                # update weight dims mapping
                mp_axis = self._update_weight(op, concrete_program)
                # update startup_program
                self._update_startup_program(startup_program, mp_axis)
                # replace embedding with c_embedding
                c_emb_op = self._replace_embedding_with_c_embedding(op)
                # insert allreduce reshard
                comm_op = self._insert_allreduce_reshard(c_emb_op)
                # update dims_mapping before c_embedding
                self._update_before_dims_mapping(c_emb_op)
                # update dims_mapping after c_embedding
                self._update_after_dims_mapping(comm_op)

    def _update_weight(self, op, concrete_program):
        # update weight dims_mapping concrete_program
        placements = op.operand(1).source().placements
        dim_map, partial_status = dist.auto_parallel.placement_type.to_dim_map(
            placements, op.operand(1).source().ndim
        )
        # mp_axis is used to  specify the axis for row parallel
        mp_axis = -1
        dim_map = [-1, -1]
        hcg = fleet.get_hybrid_communicate_group()
        mp_size = hcg.get_model_parallel_world_size()
        if mp_size > 1:
            strategy = fleet.DistributedStrategy()
            # get mp_axis from DistributedStrategy
            mp_axis = strategy.hybrid_configs['mp_degree']
            dim_map = [mp_axis, -1]
        dist_attr_w = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            op.operand(1).source().process_mesh,
            dim_map,
            partial_status,
        )
        dist_type_input0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
            op.operand(1).source().type(), dist_attr_w
        )
        op.operand(1).source().set_type(dist_type_input0)
        # update c_emebedding weight dynamic parameters
        dy_params = concrete_program.parameters[0]
        pattern = re.compile(r'embedding_.*\.w_0\.dist')
        for index, param in enumerate(dy_params):
            if pattern.match(param.name):
                var_dist_attr = TensorDistAttr()
                var_dist_attr.process_mesh = dist_attr_w.process_mesh
                var_dist_attr.dims_mapping = dist_attr_w.dims_mapping
                tmp = paddle.base.core.reshard(param, var_dist_attr)
                param.get_tensor()._share_data_with(tmp.get_tensor())
        return mp_axis

    def _replace_embedding_with_c_embedding(self, op):
        paddle.pir.set_insertion_point(op)
        num_embeddings = op.operand(1).source().type().shape[0]
        hcg = fleet.get_hybrid_communicate_group()
        # compute the start_index using the MP's world size and rank
        mp_size = hcg.get_model_parallel_world_size()
        mp_rank = hcg.get_model_parallel_rank()
        per_part_size = num_embeddings // mp_size
        vocab_start_index = mp_rank * per_part_size
        t_op = paddle._C_ops.c_embedding(
            op.operand(1).source(),
            op.operand(0).source(),
            vocab_start_index,
            num_embeddings,
        )
        t_op.get_defining_op().op_role = int(OpRole.Forward)
        new_op = t_op.get_defining_op()
        op.result(0).replace_all_uses_with(t_op)
        op.erase()
        return new_op

    def _insert_allreduce_reshard(self, c_emb_op):
        result = c_emb_op.result(0)
        paddle.pir.set_insertion_point_after(c_emb_op)
        placements = result.dist_attr().placements
        dim_map, partial_status = dist.auto_parallel.placement_type.to_dim_map(
            placements, result.ndim
        )
        partial_status = {}
        dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            result.process_mesh,
            dim_map,
            partial_status,
        )
        # insert allreduce by inserting reshard with an empty partial.
        comm_op_t = paddle._C_ops.reshard_v2(result, dist_attr_new)
        comm_op_t.get_defining_op().op_role = int(OpRole.Forward)
        result.replace_all_uses_with(comm_op_t)
        comm_op = comm_op_t.get_defining_op()
        comm_op.operand(0).set_source(result)
        return comm_op

    def _update_before_dims_mapping(self, new_op):
        placements = new_op.operand(0).source().placements
        stack = [new_op.operand(0).source().get_defining_op()]
        # adjust all ops before c_embedding until parameters input
        while stack:
            op = stack.pop()
            operands, results = [], []
            if op.num_results() > 0:
                for result, result_dist in zip(
                    op.results(), op.dist_attr.results()
                ):
                    placements_dist = (
                        result_dist.as_tensor_dist_attr().placements
                    )
                    if placements != placements_dist:
                        dim_map, partial_status = (
                            dist.auto_parallel.placement_type.to_dim_map(
                                placements, result.ndim
                            )
                        )
                        dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            result.process_mesh,
                            dim_map,
                            partial_status,
                        )
                        dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                            result.type(), dist_attr_new
                        )
                        result.set_type(dist_type)
                        results.append(dist_attr_new)
                        sub_name = op.name().split('.')[1]
                        if op.num_operands() > 0:
                            assert (
                                sub_name != "cast"
                            ), "Need to add support for {sub_name}."
                            operands.append(dist_attr_new)
                            next_op = op.operand(0).source().get_defining_op()
                            stack.append(next_op)
                        process_mesh = (
                            op.results()[0].process_mesh
                            if op.num_results() > 0
                            else op.operand(0).source().process_mesh
                        )
                        op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                process_mesh,
                                operands,
                                results,
                            )
                        )

    def _update_after_dims_mapping(self, new_op):
        placements = new_op.result(0).placements
        pre_id = new_op.id()
        stack = list(new_op.result(0).all_used_ops())
        # adjust all ops after c_embedding until the placements are consistent
        while stack:
            op = stack.pop()
            operands, results = [], []
            if op.num_operands() > 0:
                for operand, operand_dist in zip(
                    op.operands_source(), op.dist_attr.operands()
                ):
                    if operand.get_defining_op().id() != pre_id:
                        continue
                    placements_dist = (
                        operand_dist.as_tensor_dist_attr().placements
                    )
                    if placements != placements_dist:
                        dim_map, partial_status = (
                            dist.auto_parallel.placement_type.to_dim_map(
                                placements, operand.ndim
                            )
                        )
                        dist_attr_new = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            operand.process_mesh,
                            dim_map,
                            partial_status,
                        )
                        dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                            operand.type(), dist_attr_new
                        )
                        operand.set_type(dist_type)
                        operands.append(dist_attr_new)
                        sub_name = op.name().split('.')[1]
                        if sub_name == 'reshard':
                            # only change reshard‘s inputs
                            placements_out0 = op.results()[0].placements
                            dim_map_out0, partial_status_out0 = (
                                dist.auto_parallel.placement_type.to_dim_map(
                                    placements_out0,
                                    op.results()[0].ndim,
                                )
                            )
                            dist_attr_out0 = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                op.results()[0].process_mesh,
                                dim_map_out0,
                                partial_status_out0,
                            )
                            results.append(dist_attr_out0)
                        elif core.contains_spmd_rule(sub_name):
                            # redo the infer spmd_rule
                            rule = core.get_phi_spmd_rule(sub_name)
                            tensor_dist_attr = TensorDistAttr()
                            tensor_dist_attr.dims_mapping = dim_map
                            partial_dims = []
                            for i, p in enumerate(placements):
                                if isinstance(p, dist.Partial):
                                    partial_dims.append(i)
                            if len(partial_dims) > 0:
                                tensor_dist_attr._set_partial_dims(partial_dims)
                            tensor_dist_attr.process_mesh = operand.process_mesh
                            inputs = DistTensorSpec(
                                operand.shape, tensor_dist_attr
                            )
                            attr_names = op.get_attr_names()
                            input_specs = []
                            input_specs.append(inputs)
                            for attr_name in attr_names:
                                input_specs.append(op.attrs()[attr_name])
                            inferred_dist_attrs = rule.infer_forward(
                                *input_specs
                            )
                            dims_mapping_new_out = inferred_dist_attrs[1][
                                0
                            ].dims_mapping
                            partial_status = {}
                            if inferred_dist_attrs[1][0]._is_partial():
                                partial_dims = inferred_dist_attrs[1][
                                    0
                                ]._partial_dims()
                                for i in partial_dims:
                                    partial_status[i] = (
                                        paddle.base.core.ReduceType.kRedSum
                                    )
                            dist_attr_new_out = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                operand.process_mesh,
                                dims_mapping_new_out,
                                partial_status,
                            )
                            dist_type = (
                                paddle.base.libpaddle.pir.cvt_to_dist_type(
                                    op.result(0).type(), dist_attr_new_out
                                )
                            )
                            op.result(0).set_type(dist_type)
                            results.append(dist_attr_new_out)
                            next_op = op.results()[0].all_used_ops()[0]
                            stack.append(next_op)
                            pre_id = op.id()
                            placements = dist_attr_new_out.placements
                        else:
                            results.append(dist_attr_new)
                            next_op = op.results()[0].all_used_ops()[0]
                            stack.append(next_op)
                            pre_id = op.id()

                        process_mesh = (
                            op.results()[0].process_mesh
                            if op.num_results() > 0
                            else op.operand(0).source().process_mesh
                        )
                        op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                process_mesh,
                                operands,
                                results,
                            )
                        )

    def _update_startup_program(self, startup_program, mp_axis):
        # modify the startup_program because the optimizer needs to use
        startup_block = startup_program.global_block()
        for op in startup_block.ops:
            if op.name() == 'pd_op.full':
                next_op = op.result(0).all_used_ops()[0]
                parameter_name = next_op.str_attr("parameter_name")
                pattern = re.compile(r'embedding_.*\.w_0\.dist')
                if pattern.match(parameter_name):
                    placements = op.results()[0].placements
                    dim_map, partial_status = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements, len(placements)
                        )
                    )
                    dim_map = [mp_axis, -1]
                    dist_attr = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            op.results()[0].process_mesh,
                            dim_map,
                            partial_status,
                        )
                    )
                    dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                        op.results()[0].type(), dist_attr
                    )
                    op.results()[0].set_type(dist_type)
                    op.dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            op.results()[0].process_mesh, [], [dist_attr]
                        )
                    )
