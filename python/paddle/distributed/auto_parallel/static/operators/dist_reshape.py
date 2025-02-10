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
    Reshape2GradOpCost,
    Reshape2OpCost,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..utils import (
    compute_compatible_and_update_dim_mapping,
    get_dist_tensor_spec,
    is_dim_shard,
    set_dist_op_desc_original_id,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    is_parameter_related,
    merge_forward_backward_dims_mapping,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)
from .dist_default import DistributedDefaultImpl0


class DistributedReshape2(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc
        assert (
            dist_op.serial_op.type == "reshape2"
        ), f"{dist_op.serial_op.type} is not supported by dist reshape yet."

        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        xshape_name = op_desc.output('XShape')[0]
        shape = op_desc.attr('shape')

        x_spec = get_dist_tensor_spec(dist_op, x_name)
        output_spec = get_dist_tensor_spec(dist_op, out_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("reshape")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(x_spec, shape)
        bw_results = rule.infer_backward(x_spec, output_spec, shape)

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
        reverted = False
        op_dist_attr = dist_op.dist_attr

        # all reshape mapping to impl0
        op_dist_attr.impl_type = "reshape2"
        op_dist_attr.impl_idx = 0

        return reverted


register_distributed_operator_impl_container(DistributedReshape2("reshape2"))


class DistributedReshapeImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = False

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        else:
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        res = []
        op = dist_op.serial_op
        dist_attr = dist_op.dist_attr

        shape_list = op.desc.attr("shape")
        # got dist attribute info
        dim_mapping = dist_attr.get_output_dims_mapping(op.output("Out")[0])
        process_mesh_shape = dist_attr.process_mesh.shape

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_attr.process_mesh.process_ids
        for key in desc_mapping:
            desc_mapping[key]["shape"] = shape_list

        cost_mapping = build_comp_costs_from_descs(
            Reshape2OpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        return res

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
            Reshape2GradOpCost, ctx, processes, desc_mapping, cluster
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

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) - 1:
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) - 1:
            return False

        if is_dim_shard(out_dims_mapping[-1]):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for idx, dim_mapping in enumerate(out_dims_mapping[:-1]):
            if x_dims_mapping[idx] != dim_mapping:
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

        for i in range(len(x_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
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

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), f"backward op [{src_op}] don't have dist attribute !"

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, f"input [{input_name}] is not given"
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), f"number of tensor for input [{input_name}] is not match"
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, f"input [{output_name}] is not given"
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), f"number of tensor for input [{output_name}] is not match"

        X_var = main_block._var_recursive(kwargs['X'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
        XShape_var = main_block._var_recursive(kwargs['XShape'][0])
        shape_list = src_op.desc.attr("shape")
        ShapeTensor_var_list = []
        for name in kwargs['ShapeTensor']:
            ShapeTensor_var_list.append(name)
        Shape_var_list = []
        for name in kwargs['Shape']:
            Shape_var_list.append(name)

        # got dist attribute info
        dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
        process_mesh_shape = op_dist_attr.process_mesh.shape

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # create op
        new_op = main_block.append_op(type='nop')
        new_op_desc = new_op.desc
        new_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
        new_op_desc.set_input('ShapeTensor', ShapeTensor_var_list)
        new_op_desc.set_input('Shape', Shape_var_list)
        new_op_desc.set_input('X', [X_var.name])
        new_op_desc.set_output('XShape', [XShape_var.name])
        new_op_desc.set_output('Out', [Out_var.name])
        new_op_desc._set_attr('shape', shape_list)
        # TODO: should we add a new dist attr for the new op here?

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


class DistributedReshapeImpl1(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = False

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        else:
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        res = []
        op = dist_op.serial_op
        dist_attr = dist_op.dist_attr

        shape_list = op.desc.attr("shape")
        # got dist attribute info
        dim_mapping = dist_attr.get_output_dims_mapping(op.output("Out")[0])
        process_mesh_shape = dist_attr.process_mesh.shape

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_attr.process_mesh.process_ids
        for key in desc_mapping:
            desc_mapping[key]["shape"] = shape_list

        cost_mapping = build_comp_costs_from_descs(
            Reshape2OpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        return res

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
            Reshape2GradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
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

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) + 1:
            return False

        if is_dim_shard(x_dims_mapping[-1]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) + 1:
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

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

        if is_dim_shard(x_dims_mapping[-1]):
            return False

        for idx, item in enumerate(x_dims_mapping[:-1]):
            if out_dims_mapping[idx] != item:
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

        for i in range(len(out_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
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

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), f"backward op [{src_op}] don't have dist attribute !"

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, f"input [{input_name}] is not given"
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), f"number of tensor for input [{input_name}] is not match"
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, f"input [{output_name}] is not given"
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), f"number of tensor for input [{output_name}] is not match"

        X_var = main_block._var_recursive(kwargs['X'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
        XShape_var = main_block._var_recursive(kwargs['XShape'][0])
        shape_list = src_op.desc.attr("shape")
        ShapeTensor_var_list = []
        for name in kwargs['ShapeTensor']:
            ShapeTensor_var_list.append(name)
        Shape_var_list = []
        for name in kwargs['Shape']:
            Shape_var_list.append(name)

        # got dist attribute info
        dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
        process_mesh_shape = op_dist_attr.process_mesh.shape

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # create op
        new_op = main_block.append_op(type='nop')
        new_op_desc = new_op.desc
        new_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
        new_op_desc.set_input('ShapeTensor', ShapeTensor_var_list)
        new_op_desc.set_input('Shape', Shape_var_list)
        new_op_desc.set_input('X', [X_var.name])
        new_op_desc.set_output('XShape', [XShape_var.name])
        new_op_desc.set_output('Out', [Out_var.name])
        new_op_desc._set_attr('shape', shape_list)
        # TODO: should we add a new dist attr for the new op here?

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


class DistributedReshapeImpl2(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = False

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        else:
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        res = []
        op = dist_op.serial_op
        dist_attr = dist_op.dist_attr

        shape_list = op.desc.attr("shape")
        # got dist attribute info
        dim_mapping = dist_attr.get_output_dims_mapping(op.output("Out")[0])
        process_mesh_shape = dist_attr.process_mesh.shape

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_attr.process_mesh.process_ids
        for key in desc_mapping:
            desc_mapping[key]["shape"] = shape_list

        cost_mapping = build_comp_costs_from_descs(
            Reshape2OpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        return res

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
            Reshape2GradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
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

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        x_name = op_desc.input('X')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

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

        for idx, item in enumerate(x_dims_mapping[:-1]):
            if out_dims_mapping[idx] != item:
                return False

        if x_shape_dims_mapping[0] != -1:
            return False

        if x_shape_dims_mapping[1:] != out_dims_mapping[:]:
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

        for i in range(len(out_dims_mapping) - 1):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        for i in range(len(out_dims_mapping)):
            x_shape_dims_mapping[i + 1] = out_dims_mapping[i]

        if changed:
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
            op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)
            op_dist_attr.set_output_dims_mapping(
                x_shape_name, x_shape_dims_mapping
            )

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), f"backward op [{src_op}] don't have dist attribute !"

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, f"input [{input_name}] is not given"
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), f"number of tensor for input [{input_name}] is not match"
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, f"input [{output_name}] is not given"
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), f"number of tensor for input [{output_name}] is not match"

        X_var = main_block._var_recursive(kwargs['X'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
        XShape_var = main_block._var_recursive(kwargs['XShape'][0])
        shape_list = src_op.desc.attr("shape")
        ShapeTensor_var_list = []
        for name in kwargs['ShapeTensor']:
            ShapeTensor_var_list.append(name)
        Shape_var_list = []
        for name in kwargs['Shape']:
            Shape_var_list.append(name)

        # got dist attribute info
        out_dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
        process_mesh_shape = op_dist_attr.process_mesh.shape

        # modify target shape
        for idx, axis in enumerate(out_dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # create op
        new_op = main_block.append_op(type='nop')
        new_op_desc = new_op.desc
        new_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
        new_op_desc.set_input('ShapeTensor', ShapeTensor_var_list)
        new_op_desc.set_input('Shape', Shape_var_list)
        new_op_desc.set_input('X', [X_var.name])
        new_op_desc.set_output('XShape', [XShape_var.name])
        new_op_desc.set_output('Out', [Out_var.name])
        new_op_desc._set_attr('shape', shape_list)
        # TODO: should we add a new dist attr for the new op here?

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "reshape2", DistributedReshapeImpl0("add_one_dim_back")
)
register_distributed_operator_impl(
    "reshape2", DistributedReshapeImpl1("remove_one_dim_back")
)
register_distributed_operator_impl(
    "reshape2", DistributedReshapeImpl2("same_dim_shape")
)
