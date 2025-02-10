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

from paddle.distributed.fleet.meta_optimizers.common import OpRole

from ..completion import get_phi_spmd_rule
from ..cost import (
    Transpose2GradOpCost,
    Transpose2OpCost,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..utils import (
    compute_compatible_and_update_dim_mapping,
    get_dist_tensor_spec,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    get_default_distributed_operator_impl,
    is_parameter_related,
    merge_forward_backward_dims_mapping,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)
from .dist_default import DistributedDefaultImpl0


class DistributedTranspose2(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc
        assert (
            dist_op.serial_op.type == "transpose2"
        ), f"{dist_op.serial_op.type} is not supported by dist transpose yet."

        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        xshape_name = op_desc.output('XShape')[0]
        axes = op_desc.attr('axis')

        x_spec = get_dist_tensor_spec(dist_op, x_name)
        output_spec = get_dist_tensor_spec(dist_op, out_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("transpose")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(x_spec, axes)
        bw_results = rule.infer_backward(x_spec, output_spec, axes)

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op, [x_name], [out_name], fw_results, bw_results
        )

        # step4: update xshape
        inferred_input_dims_mappings, _ = merge_forward_backward_dims_mapping(
            fw_results, bw_results
        )
        dist_op.dist_attr.set_output_dims_mapping(
            xshape_name, [-1] + inferred_input_dims_mappings[0]
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        # all elementwise op use default dist operator impl.
        op_dist_attr = dist_op.dist_attr
        default_impl = get_default_distributed_operator_impl()
        op_dist_attr.impl_type = default_impl.type
        op_dist_attr.impl_idx = default_impl.idx

        return False


register_distributed_operator_impl_container(
    DistributedTranspose2("transpose2")
)


class DistributedTranspose2Impl(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = False
        self._backward_implemented = False

    def is_input_compatible(self, dist_op):
        return True

    def is_output_compatible(self, dist_op):
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        perm = op_desc.attr('axis')
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        new_dims_mapping = [-1 for i in range(len(x_dims_mapping))]
        for i in range(len(x_dims_mapping)):
            new_dims_mapping[i] = x_dims_mapping[perm[i]]

        if len(x_dims_mapping) != len(out_dims_mapping):
            return False

        if new_dims_mapping != out_dims_mapping:
            return False

        if x_shape_dims_mapping[0] != -1:
            return False

        if x_shape_dims_mapping[1:] != x_dims_mapping[:]:
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )
        perm = op_desc.attr('axis')

        assert len(x_dims_mapping) == len(perm)

        new_dims_mapping = [-1 for i in range(len(x_dims_mapping))]
        for i in range(len(x_dims_mapping)):
            new_dims_mapping[i] = x_dims_mapping[perm[i]]

        for i in range(len(out_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [new_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        for i in range(len(x_dims_mapping)):
            if x_dims_mapping[perm[i]] != new_dims_mapping[i]:
                x_dims_mapping[perm[i]] = new_dims_mapping[i]
                changed = True

        for i in range(len(x_dims_mapping)):
            x_shape_dims_mapping[i + 1] = x_dims_mapping[i]

        if changed:
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
            op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)
            op_dist_attr.set_output_dims_mapping(
                x_shape_name, x_shape_dims_mapping
            )

        return changed

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        else:
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        op_type = dist_op.serial_op.type
        cost_mapping = build_comp_costs_from_descs(
            Transpose2OpCost, ctx, processes, desc_mapping, cluster
        )

        res_cost = [cost_mapping]
        return res_cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        res = []
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        op_type = dist_op.serial_op.type
        cost_mapping = build_comp_costs_from_descs(
            Transpose2GradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and is_parameter_related(
                    varname, main_block
                ):
                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)

                    mesh_shape = process_mesh.shape
                    batch_size_axis = (
                        var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1
                    )
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                        parallel_axis = batch_size_axis
                        attrs = {"use_calc_stream": True}
                        var_names = [varname + "@GRAD"]
                        build_dp_costs(
                            res,
                            dist_op,
                            ctx,
                            var_names,
                            attrs,
                            parallel_axis,
                            cluster,
                        )
        return res

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "transpose2", DistributedTranspose2Impl("same_mapping_transpose")
)
