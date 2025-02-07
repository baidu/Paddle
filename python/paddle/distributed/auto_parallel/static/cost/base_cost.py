# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

import numpy as np

import paddle
from paddle.utils.flops import flops

from ..cluster import DeviceType, LinkType, get_default_cluster
from ..dist_tensor import DistributedTensor
from ..process_group import get_process_group
from ..utils import _get_comm_group, _get_idx_in_axis

COMM_OP_TYPE = [
    "p_send",
    "p_recv",
    "broadcast",
    "all_gather",
    "c_allreduce_sum",
    "c_identity",
]
NON_COMP_TYPE = ["while", *COMM_OP_TYPE]
_g_op_cost_factory = {}


def build_comp_desc_from_op(op):
    """Build the description of computation op."""
    # NOTE: The desc is for serial op.
    from ..reshard import get_var_with_recursion

    desc = {}
    # The desc of concat op is {"op": "concat", "inputs": {"X": [(paddle.float32, [20, 20]), (paddle.float32, [20, 20])]}, "outputs": {"Out": [(paddle.float32, [20, 40])], "attrs": {"axis": -1}}}
    vars = op.block.vars
    desc["op"] = op.type
    input_desc = OrderedDict()
    for input_name in op.input_names:
        var_name_list = op.input(input_name)
        var_desc = []
        for var_name in var_name_list:
            var = get_var_with_recursion(var_name, op.block, op.block.program)
            shape = var.shape
            var_desc.append((var.dtype, shape))
        input_desc[input_name] = var_desc
    desc["inputs"] = input_desc

    output_desc = OrderedDict()
    for out_name in op.output_names:
        var_name_list = op.output(out_name)
        var_desc = []
        for var_name in var_name_list:
            var = get_var_with_recursion(var_name, op.block, op.block.program)
            shape = var.shape
            var_desc.append((var.dtype, shape))
            desc["dtype"] = var.dtype
        output_desc[out_name] = var_desc
    desc["outputs"] = output_desc

    attr_desc = op.all_attrs
    desc["attrs"] = attr_desc

    return desc


def build_comp_desc_from_dist_op(dist_op, dist_context):
    """Build descriptions of computation op distributed on the processes."""
    from ..reshard import get_var_with_recursion

    op_descs = {}
    op = dist_op.serial_op
    dist_attr = dist_op.dist_attr
    process_mesh = dist_attr.process_mesh
    assert process_mesh, "Process mesh must not be None."
    processes = process_mesh.process_ids
    for process in processes:
        desc = {}
        desc["op"] = op.type
        attr_desc = op.all_attrs()
        # NOTE: The attrs of desc is replica of serial op, there may be a bug if shape need to be partitioned involved in attrs.
        desc["attrs"] = attr_desc
        input_desc = OrderedDict()
        output_desc = OrderedDict()

        # Get partitioned shape of input
        input_var_desc = {}
        for input_name in op.input_names:
            var_name_list = op.input(input_name)
            input_var_desc[input_name] = []
            for var_name in var_name_list:
                var = get_var_with_recursion(
                    var_name, op.block, op.block.program
                )
                # Use op input_dims_mapping
                dims_mapping = dist_attr.get_input_dims_mapping(var_name)
                global_sizes = var.shape
                # NOTE: When support uneven partition, the shard_sizes will be got from dist_attr.
                shard_sizes = None
                topology = process_mesh.shape
                shape = DistributedTensor.get_local_sizes(
                    global_sizes,
                    dims_mapping,
                    topology,
                    processes,
                    process,
                    shard_sizes,
                )
                input_var_desc[input_name].append(shape)

                # For special op such as embedding and its grad op
                if (
                    op.type == "c_embedding"
                    or op.type == "lookup_table_v2"
                    or op.type == "c_embedding_grad"
                    or op.type == "lookup_table_v2_grad"
                ):
                    if input_name == "W":
                        embedding_row_dim_mapping = (
                            dist_attr.get_input_dims_mapping(
                                op.input(input_name)[0]
                            )[0]
                        )
                        relative_idx = _get_idx_in_axis(
                            processes,
                            dist_attr.process_mesh.shape,
                            embedding_row_dim_mapping,
                            process,
                        )
                        per_part_size = shape[0]
                        relative_idx = relative_idx * per_part_size
                        desc["attrs"]["start_index"] = relative_idx

        desc["inputs"] = input_var_desc

        for out_name in op.output_names:
            var_name_list = op.output(out_name)
            var_desc = []
            for var_name in var_name_list:
                # Use op output_dims_mapping
                var = get_var_with_recursion(
                    var_name, op.block, op.block.program
                )
                dist_attr = dist_op.dist_attr
                dims_mapping = dist_attr.get_output_dims_mapping(var_name)
                process_mesh = dist_attr.process_mesh
                global_sizes = var.shape
                shard_sizes = None
                processes = process_mesh.process_ids
                topology = process_mesh.shape
                shape = DistributedTensor.get_local_sizes(
                    global_sizes,
                    dims_mapping,
                    topology,
                    processes,
                    process,
                    shard_sizes,
                )
                var_desc.append((var.dtype, shape))
                desc["dtype"] = var.dtype

                # For special op such as fill_constant_batch_size_like
                if op.type == "fill_constant_batch_size_like":
                    # Modify shape attr according to how output are partitioned
                    out_name = var_name_list[0]
                    dims_mapping = dist_attr.get_output_dims_mapping(out_name)
                    process_mesh_shape = dist_attr.process_mesh.shape
                    shape_list = op.attr("shape")
                    # Modify target shape
                    for idx, axis in enumerate(dims_mapping):
                        if axis >= 0:
                            shape_list[idx] = (
                                shape_list[idx] // process_mesh_shape[axis]
                            )
                    desc["attrs"]["shape"] = shape_list
            output_desc[out_name] = var_desc

        desc["outputs"] = output_desc

        op_descs[process] = desc

    return op_descs


def build_comp_desc_str_for_predict(desc):
    # NOTE: The description format may change in the future.
    def _parse_dtype(dtype):
        dtype_str = ""
        if dtype == paddle.float32:
            dtype_str = "float32"
        elif dtype == paddle.float16:
            dtype_str = "float16"
        elif dtype == paddle.int32:
            dtype_str = "int32"
        elif dtype == paddle.int64:
            dtype_str = "int64"
        elif dtype == paddle.unit8:
            dtype_str = "unit8"
        else:
            raise TypeError(f"Unsupported dtype {dtype}")
        return dtype_str

    assert isinstance(desc, dict)
    desc_str_list = []
    desc_str = None
    dtype_str_list = []
    dims_list = []
    shape_list = []

    desc_str_list.append(desc["op"])
    inputs = desc["inputs"]
    for key, item in inputs.items():
        for dtype, shape in item:
            dtype_str_list.append(_parse_dtype(dtype))
            shape_list += list(shape)
            dims = len(shape)
            dims_list.append(dims)

    dtype_str = "*".join(dtype_str_list)
    dims_list = [str(item) for item in dims_list]
    dims_str = "*".join(dims_list)

    shape_list = [str(item) for item in shape_list]
    shape_str = "[" + ",".join(shape_list) + "]"
    desc_str_list += [dtype_str, dims_str, shape_str]
    desc_str = "_".join(desc_str_list)
    attrs = desc["attrs"]
    parse_result = (desc_str, attrs)
    return parse_result


def build_comm_desc_from_dist_op(
    op_type,
    dist_op,
    ctx,
    var_names,
    attrs=None,
    parallel_axis=None,
    group_ranks=None,
):
    """Build descriptions of communication op distributed on the processes."""
    from ..reshard import get_var_with_recursion

    specific_op_type = []
    dist_attr = dist_op.dist_attr
    assert dist_attr, "Dist attr must not be None."
    process_mesh = dist_attr.process_mesh
    assert process_mesh, "Process mesh must not be None."

    processes = process_mesh.process_ids
    op_descs = {}
    for process in processes:
        rank_id = process
        desc = {}
        desc["op"] = op_type
        op_attrs = None
        comm_group_ranks = None

        if op_type not in specific_op_type:
            serial_op = dist_op.serial_op
            input_list = []
            # The var_names usually contain just one item.
            for var_name in var_names:
                dist_attr = dist_op.dist_attr
                has_found = False
                # Find var_name in serial op input or output
                for name in dist_op.serial_op.input_arg_names:
                    # If a tensor is the input of multi ops, sum the grad of all ops, so the name will be varname@RENAME@block@0 and so on.
                    if var_name in name:
                        var_name = name
                        has_found = True
                        break

                if not has_found:
                    for name in dist_op.serial_op.output_arg_names:
                        if var_name in name:
                            var_name = name
                            has_found = True
                            break
                assert has_found
                var = get_var_with_recursion(
                    var_name, serial_op.block, serial_op.block.program
                )

                dims_mapping = (
                    dist_attr.get_input_dims_mapping(var_name)
                    if var_name in dist_op.serial_op.input_arg_names
                    else dist_attr.get_output_dims_mapping(var_name)
                )
                global_sizes = var.shape
                shard_sizes = None
                topology = process_mesh.shape
                shape = DistributedTensor.get_local_sizes(
                    global_sizes,
                    dims_mapping,
                    topology,
                    processes,
                    process,
                    shard_sizes,
                )
                input_list.append((var.dtype, shape))

            # NOTE: The input_name of comm ops used usually is X.
            desc["inputs"] = {"X": input_list}

            # Get comm group by parallel_axis or the given group_ranks.
            if parallel_axis is not None:
                process_mesh_shape = process_mesh.shape
                process_mesh_group = process_mesh.process_ids
                comm_group_ranks = _get_comm_group(
                    process_mesh_group,
                    process_mesh_shape,
                    parallel_axis,
                    rank_id,
                )
            elif group_ranks is not None:
                comm_group_ranks = group_ranks
            else:
                raise ValueError(
                    "The parallel_axis and group_ranks can not be None in the same."
                )

            if attrs is not None:
                assert isinstance(attrs, dict)
                op_attrs = attrs
            else:
                op_attrs = {}

            desc["attrs"] = op_attrs
            desc["group_ranks"] = comm_group_ranks

            op_descs[rank_id] = desc

    return op_descs


def build_comm_desc(op_type, group_ranks, dtype, shape, attrs=None):
    """Build a comm desc directly."""
    desc = {}
    desc["op"] = op_type
    desc["group_ranks"] = group_ranks
    desc["inputs"] = {"X": [(dtype, shape)]}
    desc["attrs"] = attrs
    return desc


def build_comm_costs_from_descs(
    op_cost_class, ctx, processes, descs, cluster, is_dp=False
):
    """Build comm costs by descriptions"""
    comm_context = CommContext(cluster)
    group_ranks_list = []
    comm_op_cost_list = []
    for process in processes:
        desc = descs[process]
        group_ranks = desc["group_ranks"]
        if group_ranks not in group_ranks_list:
            group_ranks_list.append(group_ranks)
            comm_op_cost = op_cost_class(
                op_desc=desc, comm_context=comm_context
            )
            if is_dp:
                comm_op_cost.cost.time *= 0.9
            comm_op_cost_list.append(comm_op_cost)
    return comm_op_cost_list


def build_comp_costs_from_descs(op_cost_class, ctx, processes, descs, cluster):
    """Build comp costs by descriptions."""
    costs = {}
    for process in processes:
        costs[process] = op_cost_class(
            op_desc=descs[process], cluster=cluster, rank=process
        )
    return costs


def build_dp_costs(
    result, dist_op, ctx, var_names, attrs, parallel_axis, cluster
):
    """DP cost contains a allreduce_sum op cost and a scale op cost"""
    # The costs will be appended in the given result.
    from ..reshard import get_var_with_recursion

    dist_attr = dist_op.dist_attr
    process_mesh = dist_attr.process_mesh
    processes = process_mesh.process_ids
    assert len(var_names) == 1
    vars = dist_op.serial_op.block.vars
    var_name = var_names[0]
    has_found = False
    is_input = True
    for name in dist_op.serial_op.input_arg_names:
        if var_name in name:
            var_name = name
            has_found = True
            break

    if not has_found:
        for name in dist_op.serial_op.output_arg_names:
            if var_name in name:
                var_name = name
                has_found = True
                is_input = False
                break
    if not has_found:
        return

    c_allreduce_sum_descs = build_comm_desc_from_dist_op(
        "c_allreduce_sum",
        dist_op,
        ctx,
        var_names,
        attrs=attrs,
        parallel_axis=parallel_axis,
    )
    comm_cost_list = build_comm_costs_from_descs(
        _g_op_cost_factory["c_allreduce_sum"],
        ctx,
        processes,
        c_allreduce_sum_descs,
        cluster,
        is_dp=True,
    )
    result.append(comm_cost_list)

    # The scale op just on the group_ranks
    for comm_cost in comm_cost_list:
        group_ranks = comm_cost.group_ranks
        dp_degree = len(group_ranks)
        scale_costs = {}
        op_type = "scale"
        for rank in group_ranks:
            desc = {}
            desc["op"] = op_type
            desc["inputs"] = {}
            dims_mapping = (
                dist_attr.get_input_dims_mapping(var_name)
                if is_input
                else dist_attr.get_output_dims_mapping(var_name)
            )
            var = get_var_with_recursion(
                var_name,
                dist_op.serial_op.block,
                dist_op.serial_op.block.program,
            )
            global_sizes = var.shape
            shard_sizes = None
            topology = process_mesh.shape
            shape = DistributedTensor.get_local_sizes(
                global_sizes,
                dims_mapping,
                topology,
                processes,
                rank,
                shard_sizes,
            )
            desc["inputs"]["X"] = [(var.dtype, shape)]
            attrs = {"scale": 1.0 / dp_degree}
            desc["attrs"] = attrs
            desc["dtype"] = var.dtype
            scale_op_cost = _g_op_cost_factory["scale"](
                op_desc=desc, cluster=cluster, rank=rank
            )
            scale_costs[rank] = scale_op_cost
        result.append(scale_costs)


class CommContext:
    _instance = None
    _has_instance = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            _has_instance = True
        return cls._instance

    def __init__(self, cluster):
        if CommContext._has_instance:
            return
        self.beta = {}
        self.hops = {}
        assert cluster is not None
        self.cluster = cluster
        # if cluster has no info about those vars, it will be set by default
        self.base_ring = None
        self.base_tree = None
        self.intra_ring = None
        self.intra_tree = None
        self.inter_ring = None
        self.inter_tree = None
        self.switch = None
        self._post_init()

    def _post_init(self):
        alpha_latency = self.cluster.alpha_latency
        if alpha_latency is None:
            # set default
            self.base_ring = 8.4
            self.base_tree = 0.0
            # NVL in default
            self.intra_ring = 3.4
            self.intra_tree = 28
            # NET in default
            self.inter_ring = 9.6
            self.inter_tree = 28
            self.switch = 10.0
        else:
            base_ring = alpha_latency.base_ring
            self.base_ring = base_ring if base_ring is not None else 8.4

            base_tree = alpha_latency.base_tree
            self.base_tree = base_tree if base_tree is not None else 0.0

            intra_ring = alpha_latency.intra_ring
            if intra_ring == LinkType.NVL:
                self.intra_ring = 3.4
            elif intra_ring == LinkType.PHB:
                self.intra_ring = 5.7
            elif intra_ring is not None:
                self.intra_ring = intra_ring
            else:
                # NVL Default
                self.intra_ring = 3.4

            intra_tree = alpha_latency.intra_tree
            if intra_tree == LinkType.NVL:
                self.intra_tree = 28
            elif intra_tree == LinkType.PHB:
                self.intra_tree = 28
            elif intra_tree is not None:
                self.intra_tree = intra_tree
            else:
                # NVL Default
                self.intra_tree = 28

            inter_ring = alpha_latency.inter_ring
            if inter_ring == LinkType.NET:
                self.inter_ring = 9.6
            elif inter_ring is not None:
                self.inter_ring = inter_ring
            else:
                # NET Default
                self.inter_ring = 9.6

            inter_tree = alpha_latency.inter_tree
            if inter_tree == LinkType.NET:
                self.inter_tree = 28
            elif inter_tree is not None:
                self.inter_tree = inter_tree
            else:
                # NET Default
                self.inter_tree = 28

            switch = alpha_latency.switch
            self.switch = switch if switch is not None else 10

            assert self.base_ring is not None
            assert self.base_tree is not None
            assert self.intra_ring is not None
            assert self.intra_tree is not None
            assert self.inter_ring is not None
            assert self.inter_tree is not None
            assert self.switch is not None

    def get_max_beta(self, ranks):
        # NOTE: Get beta by ring, even in the case of tree such as tree broadcast
        ranks = self.cluster.convert_rank_to_device_id(ranks)
        key = ','.join(map(str, sorted(ranks)))
        max_beta = None
        if key in self.beta:
            max_beta = self.beta[key]
        else:
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    forward_order_beta = self.cluster.get_beta(
                        ranks[i], ranks[j]
                    )
                    backward_order_beta = self.cluster.get_beta(
                        ranks[j], ranks[i]
                    )
                    beta = max(backward_order_beta, forward_order_beta)
                    if max_beta is None:
                        max_beta = beta
                    else:
                        if beta > max_beta:
                            max_beta = beta
            self.beta[key] = max_beta

        return max_beta

    def get_hops(self, ranks):
        key = ','.join(map(str, sorted(ranks)))
        hops = 0
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                hop = self.cluster.get_hop(ranks[i], ranks[j])
                hops += hop
        self.hops[key] = hops

        return hops


class Cost:
    def __init__(self, time=0, memory=0, flops=0):
        self.time = time
        self.memory = memory
        self.flops = flops

    def _check_time(self, val):
        assert val >= 0, "Time must be greater than or equal to 0."

    def _check_memory(self, val):
        assert (
            isinstance(val, int) and val >= 0
        ), "Memory must be int and greater than equal to 0."

    def _check_flops(self, val):
        assert (
            isinstance(val, int) and val >= 0
        ), "FLOPs must be int and greater than equal to 0."

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._check_time(val)
        self._time = val

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, val):
        self._check_memory(val)
        self._memory = val

    @property
    def flops(self):
        return self._flops

    @flops.setter
    def flops(self, val):
        self._check_flops(val)
        self._flops = val

    def __add__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time + rhs.time
        memory = self.memory + rhs.memory
        flops = self.flops + rhs.flops
        assert time >= 0 and memory >= 0 and flops >= 0
        return Cost(time, memory, flops)

    def __sub__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time - rhs.time
        memory = self.memory - rhs.memory
        flops = self.flops - rhs.flops
        assert time >= 0 and memory >= 0 and flops >= 0
        return Cost(time, memory, flops)


class OpCost:
    OP_TYPE = "op"

    def __init__(self, op=None, op_desc=None):
        self._op = op
        self._op_desc = op_desc
        self._cost = None

    @property
    def op(self):
        return self._op

    @property
    def op_desc(self):
        return self._op_desc

    @property
    def time(self):
        return self.cost.time

    @property
    def memory(self):
        return self.cost.memory

    @property
    def flops(self):
        return self.cost.flops

    @property
    def cost(self):
        return self._cost

    def calc_time(self):
        return 0

    def calc_memory(self):
        return 0

    def calc_flops(self):
        return 0

    def calc_cost(self):
        time = self.calc_time()
        memory = self.calc_memory()
        flops = self.calc_flops()
        cost = Cost(time, memory, flops)
        return cost

    def __add__(self, rhs):
        assert isinstance(rhs, (OpCost, Cost))
        time = 0
        memory = 0
        flops = 0
        if isinstance(rhs, OpCost):
            time = self.cost.time + rhs.cost.time
            memory = self.cost.memory + rhs.cost.memory
            flops = self.cost.flops + rhs.cost.flops
            assert time >= 0 and memory >= 0 and flops >= 0
        elif isinstance(rhs, Cost):
            time = self.time + rhs.time
            memory = self.memory + rhs.memory
            flops = self.flops + rhs.flops
            assert time >= 0 and memory >= 0 and flops >= 0
        return Cost(time, memory, flops)

    def __sub__(self, rhs):
        assert isinstance(rhs, (OpCost, Cost))
        time = 0
        memory = 0
        flops = 0
        if isinstance(rhs, OpCost):
            time = self.cost.time - rhs.cost.time
            memory = self.cost.memory - rhs.cost.memory
            flops = self.cost.flops - rhs.cost.flops
            assert time >= 0 and memory >= 0 and flops >= 0
        elif isinstance(rhs, Cost):
            time = self.time - rhs.time
            memory = self.memory - rhs.memory
            flops = self.flops - rhs.flops
            assert time >= 0 and memory >= 0 and flops >= 0
        return Cost(time, memory, flops)


class CommOpCost(OpCost):
    OP_TYPE = "COMM"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc)
        self._check_comm_op_type()
        self._comm_context = comm_context
        self._group_ranks = None
        self._comm_count = None
        self._hops = None
        self._rank_count = len(self.group_ranks)
        self._machine_count = None
        self._cost = self.calc_cost()

    @property
    def comm_context(self):
        return self._comm_context

    @property
    def comm_count(self):
        from ..reshard import get_var_with_recursion

        if self._comm_count is None:
            dtype = None
            shape = None
            if self.op is not None:
                vars = self.op.block.vars
                # NOTE: The tensor communicated input_name is "X" in default. Otherwise, this function should be overridden
                try:
                    var_name = self.op.input("X")[0]
                except:
                    var_name = self.op.output("Out")[0]
                var = get_var_with_recursion(
                    var_name, self.op.block, self.op.block.program
                )
                dtype = var.dtype
                shape = var.shape
            elif self.op_desc is not None:
                dtype = self.op_desc["inputs"]["X"][0][0]
                shape = self.op_desc["inputs"]["X"][0][1]

            factor = None
            if dtype == paddle.float32 or dtype == paddle.int32:
                factor = 4
            elif dtype == paddle.int64:
                factor = 8
            elif dtype == paddle.uint8:
                factor = 1
            elif dtype == paddle.float16:
                factor = 2
            elif dtype == paddle.bool:
                factor = 8
            else:
                raise ValueError(f"Unsupported comm dtype {dtype}")
            comm_count = int(np.prod(shape)) * factor
            self._comm_count = comm_count

        return self._comm_count

    @property
    def rank_count(self):
        return self._rank_count

    @property
    def machine_count(self):
        if self._machine_count is None:
            cluster = self._comm_context.cluster
            self._machine_count = cluster.get_involved_machine_count(
                self.group_ranks
            )
        return self._machine_count

    @property
    def hops(self):
        if self._hops is None:
            self._hops = self.comm_context.get_hops(self.group_ranks)
        return self._hops

    @property
    def group_ranks(self):
        if self._group_ranks is None:
            if self.op_desc is not None:
                self._group_ranks = self.op_desc["group_ranks"]
            elif self.op is not None:
                ring_id = self.op.attr("ring_id")
                process_group = get_process_group(ring_id)
                if process_group is None:
                    raise ValueError(
                        f"There not exists process group whose ring_id is {ring_id}."
                    )
                self._group_ranks = process_group.ranks
        return self._group_ranks

    @classmethod
    def _check_comm_op_type(cls):
        if cls.OP_TYPE != "COMM":
            if cls.OP_TYPE not in COMM_OP_TYPE:
                raise TypeError(
                    f"Please Check op type in {COMM_OP_TYPE}, but got {cls.OP_TYPE}."
                )


class CompOpCost(OpCost):
    OP_TYPE = "COMP"

    def __init__(self, op=None, op_desc=None, cluster=None, rank=None):
        super().__init__(op=op, op_desc=op_desc)
        self._check_comp_op_type()
        self.cluster = cluster
        self.rank = rank
        self._cost = self.calc_cost()

    @classmethod
    def _check_comp_op_type(cls):
        if cls.OP_TYPE != "COMP":
            if cls.OP_TYPE in NON_COMP_TYPE:
                raise TypeError(
                    f"Please Check op type not in {NON_COMP_TYPE}, but got {cls.OP_TYPE}."
                )

    def get_rank_gflops(self, rank, dtype):
        device = self.cluster.get_device(rank)
        gflops = 7800
        if dtype == paddle.float64:
            gflops = device.dp_gflops
        elif dtype == paddle.float32:
            gflops = device.sp_gflops
        elif dtype == paddle.float16 or dtype == paddle.bfloat16:
            gflops = device.hp_gflops
        return gflops

    def calc_flops(self):
        if not self.op_desc:
            return 0
        if "_grad" in self.__class__.OP_TYPE:
            op_type = self.__class__.OP_TYPE[: len(self.__class__.OP_TYPE) - 5]
            return 2 * flops(
                op_type, self.op_desc["inputs"], self.op_desc["attrs"]
            )
        return flops(
            self.__class__.OP_TYPE,
            self.op_desc["inputs"],
            self.op_desc["attrs"],
        )

    def calc_time(self):
        if self.rank is None or self.op_desc is None:
            device_gflops = 7800
        else:
            device_gflops = self.get_rank_gflops(
                self.rank, self.op_desc["dtype"]
            )
        flops_count = self.calc_flops()
        utilization_rate = 0.65
        return flops_count / (utilization_rate * device_gflops) * 1e-3


def register_op_cost(cls):
    op_type = cls.OP_TYPE

    def register(op_type):
        global _g_op_cost_factory
        _g_op_cost_factory[op_type] = cls

    register(op_type)
    return cls


def calc_time_by_modeling(op=None, desc=None, cluster=None):
    op_type = op.type if op is not None else desc["op"]
    if op_type in COMM_OP_TYPE:
        op_cost = _g_op_cost_factory[op_type](
            op=op, op_desc=desc, comm_context=CommContext(cluster)
        )
    elif op_type not in NON_COMP_TYPE:
        op_cost = _g_op_cost_factory[op_type](
            op=op, op_desc=desc, cluster=cluster
        )
    time = op_cost.calc_time()
    return time


def calc_time_by_cost_model(op, cluster=None):
    """Calc op time by cost model and the unit is microsecond."""
    if not isinstance(op, paddle.base.framework.Operator):
        raise TypeError(
            f"OP must be paddle.base.framework.Operator, but got {type(op)}."
        )
    if not cluster:
        cluster = get_default_cluster()

    assert cluster._gpu_model in [
        "V100",
        "A100",
    ], "Only A100 and V100 gpu has been supported currently."

    time = 0.0  # microsecond
    op_type = op.type
    # calc comp op time by flops
    if op_type not in NON_COMP_TYPE:
        attrs = op.all_attrs()
        # build comp op inputs desc to calc flops.
        # for example, a matmul op inputs desc will be {"X": [(1024, 1024)], "Y": [(1024, 1024)]}
        inputs = {}
        for input_name in op.input_names:
            var_names = op.input(input_name)
            inputs[input_name] = []
            for var_name in var_names:
                var = op.block._var_recursive(var_name)
                inputs[input_name].append(var.shape)

        # the time of grad operator is twice than its forward operator empirically
        if "_grad" in op_type:
            op_type = op_type[: len(op_type) - 5]
            flops_count = 2 * flops(op_type, inputs, attrs)
        else:
            flops_count = flops(op_type, inputs, attrs)

        # FIXME(Ruibiao): Need a better way to get dtype
        var_name = op.output_arg_names[0]
        dtype = op.block._var_recursive(var_name).dtype
        device = cluster.get_device(0)
        assert (
            device.type == DeviceType.GPU
        ), "Only GPU device is supported currently."

        gflops = 0.0
        if dtype == paddle.float64:
            gflops = device.dp_gflops
        elif dtype == paddle.float32:
            gflops = device.sp_gflops
        elif dtype == paddle.float16 or dtype == paddle.bfloat16:
            gflops = device.hp_gflops
        else:
            raise ValueError(
                f"Unsupported modeling compute time for dtype: {dtype}."
            )

        utilization_rate = 0.98
        time = flops_count / (utilization_rate * gflops) * 1e-3

    # calc comm op time by communication modeling formula
    elif op_type in COMM_OP_TYPE:
        op_cost = _g_op_cost_factory[op_type](
            op=op, comm_context=CommContext(cluster)
        )
        time = op_cost.calc_time()

    else:
        raise ValueError(f"The {op_type} has not been supported now.")

    return time
