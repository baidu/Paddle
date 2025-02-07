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
from functools import reduce

import paddle
from paddle.distributed.fleet.meta_optimizers.common import OpRole

from ..dist_tensor import DistributedTensor
from ..operators.common import get_distributed_operator_impl_container
from .base_cost import Cost


class CostEstimator:
    _special_op_type = ["fused_attention", "fused_feedforward"]

    def __init__(
        self, program, cluster, mode="modeling", rank=None, loop_count=10
    ):
        self._program = program
        self._cluster = cluster
        self._check_mode(mode)
        self._mode = mode
        self._rank = rank if rank is not None else paddle.distributed.get_rank()
        self._loop_count = loop_count
        self._global_cost = Cost()
        self._local_cost_mapping = {}
        self._detailed_cost = (
            OrderedDict()
        )  # {`op_id`: {"reshard": [], "dist_op": [], "local_cost": local_cost}}}
        self._bubble_time_mapping = {}
        self._ordered_ops = []
        self.max_memories = {}
        self.max_memory = None

    @property
    def loop_count(self):
        return self._loop_count

    @property
    def detailed_cost(self):
        return self._detailed_cost

    @property
    def program(self):
        return self._program

    @property
    def rank(self):
        return self._rank

    @property
    def dist_context(self):
        return self._dist_context

    @property
    def cluster(self):
        return self._cluster

    @property
    def mode(self):
        return self._mode

    @property
    def global_cost(self):
        max_time = 0
        memory = 0
        flops = 0
        for rank in self._local_cost_mapping:
            cost = self._local_cost_mapping[rank]
            if cost.time > max_time:
                max_time = cost.time
            memory += cost.memory
            flops += cost.flops
        self._global_cost.time = max_time
        self._global_cost.memory = memory
        self._global_cost.flops = flops
        return self._global_cost

    def local_cost(self, rank=None):
        rank = self.rank if rank is None else rank
        if rank not in self._local_cost_mapping:
            self._local_cost_mapping[rank] = Cost()

        return self._local_cost_mapping[rank]

    def local_bubble_time(self, rank=None):
        rank = self.rank if rank is None else rank
        return self._bubble_time_mapping[rank]

    def _check_mode(self, mode):
        if mode not in ["modeling", "profiling"]:
            raise ValueError(
                f"Just support modeling and profiling, but got {mode}"
            )

    def _is_special_var_name(self, var_name):
        special_var_name = ["lod_tensor_blocking_queue_0"]
        if var_name in special_var_name:
            return True
        return False

    def _estimate_core(self, dist_context, resharder, block):
        from ..reshard import get_var_with_recursion

        ops = block.ops
        loop_count = None
        if block.desc.id != self.program.global_block().desc.id:
            loop_count = self.loop_count
        else:
            loop_count = 1
        for i in range(loop_count):
            for op in ops:
                self._detailed_cost[op.desc.id()] = OrderedDict()
                # If in the while sub block, the detail of cost is the last cost
                detail = self._detailed_cost[op.desc.id()]
                detail["reshard_cost"] = OrderedDict()  #
                detail["dist_op_cost"] = []
                if int(op.attr('op_role')) == int(OpRole.Optimize):
                    continue
                if op.type in [
                    "create_py_reader",
                    "create_double_buffer_reader",
                    "read",
                ]:
                    continue

                # NOTE: It does not support nested loop and just supports while op when op has sub block now.
                if op.type == "while":
                    while_block = self.program.blocks[op.attr("sub_block").id]
                    self._estimate_core(dist_context, resharder, while_block)
                    continue

                for var_name in op.input_arg_names:
                    if self._is_special_var_name(var_name):
                        continue
                    var = get_var_with_recursion(var_name, block, self.program)
                    reshard_cost = resharder.get_cost(op, var, self.cluster)

                    # Calc reshard cost
                    if reshard_cost is not None:
                        detail["reshard_cost"][var_name] = reshard_cost

                        comm_costs = reshard_cost[0]
                        local_comp_cost = reshard_cost[1]
                        for comm_cost in comm_costs:
                            # Time is cumulative in global cost and local cost, but memory and flops just are cumulative in global cost.
                            # Comm sync
                            for item in comm_cost:
                                group_ranks, cost = item
                                max_time = None
                                cost_time = {}
                                for rank in group_ranks:
                                    rank_cost = self.local_cost(rank)
                                    cost_time[rank] = rank_cost.time
                                    if max_time is None:
                                        max_time = rank_cost.time
                                    else:
                                        if max_time < rank_cost.time:
                                            max_time = rank_cost.time

                                for rank in group_ranks:
                                    self.local_cost(rank).time = (
                                        max_time + cost.time
                                    )

                                    if rank not in self._bubble_time_mapping:
                                        self._bubble_time_mapping[rank] = 0

                                    self._bubble_time_mapping[rank] += (
                                        max_time - cost_time[rank]
                                    )

                        for rank in local_comp_cost:
                            for comp_cost in local_comp_cost[rank]:
                                self.local_cost(rank).time += comp_cost.time

                # Calc dist op cost
                dist_op = dist_context.get_dist_op_for_program(op)
                if not dist_op:
                    continue

                op_dist_attr = dist_op.dist_attr
                processes = op_dist_attr.process_mesh.process_ids

                container = get_distributed_operator_impl_container(
                    op_dist_attr.impl_type
                )
                dist_impl = container.impls[op_dist_attr.impl_idx]

                dist_op_cost = dist_impl.calc_cost(
                    op.attr('op_role'), dist_op, dist_context, self.cluster
                )
                detail["dist_op_cost"] = dist_op_cost

                if dist_op_cost is None:
                    assert (
                        dist_op.serial_op.type in CostEstimator._special_op_type
                    )
                    continue
                for item in dist_op_cost:
                    if isinstance(item, list):
                        # Comm sync
                        for comm_op_cost in item:
                            max_time = None
                            cost_time = {}
                            group_ranks = comm_op_cost.group_ranks
                            for rank in comm_op_cost.group_ranks:
                                rank_cost = self.local_cost(rank)
                                cost_time[rank] = rank_cost.time
                                if max_time is None:
                                    max_time = rank_cost.time
                                else:
                                    if max_time < rank_cost.time:
                                        max_time = rank_cost.time
                            for rank in group_ranks:
                                self.local_cost(rank).time = (
                                    max_time + comm_op_cost.time
                                    if op.attr('op_role') != OpRole.Backward
                                    else max_time + 0.9 * comm_op_cost.time
                                )
                                if rank not in self._bubble_time_mapping:
                                    self._bubble_time_mapping[rank] = 0
                                self._bubble_time_mapping[rank] += (
                                    max_time - cost_time[rank]
                                )
                    elif isinstance(item, dict):
                        # Op just one
                        for rank in processes:
                            # DP+PP+MP
                            if rank not in item:
                                continue
                            self.local_cost(rank).time += item[rank].time

    def prepare(self):
        self._global_cost = Cost()
        self._local_cost_mapping = {}
        self._detailed_cost = OrderedDict()
        self._bubble_time_mapping = {}

    def _calculate_bytes(self, sizes, dtype):
        if sizes:
            total_count = reduce(lambda x, y: x * y, sizes, 1)
        else:
            total_count = 0

        if dtype == paddle.float64 or dtype == paddle.int64:
            dtype_factor = 8
        elif dtype == paddle.float32 or dtype == paddle.int32:
            dtype_factor = 4
        elif (
            dtype == paddle.float16
            or dtype == paddle.bfloat16
            or dtype == paddle.int16
        ):
            dtype_factor = 2
        elif dtype == paddle.int8 or dtype == paddle.uint8:
            dtype_factor = 1
        else:
            dtype_factor = 8

        memory = total_count * dtype_factor
        return memory

    def _estimate_max_memory_by_dist_op(self, dist_context):
        # This estimation will be improved, now reshard and inplace are not considered.
        # Persist var is not free.
        def _convert_pm_and_dm_to_str(process_mesh, dims_mapping):
            processes = ",".join([str(x) for x in process_mesh.process_ids])
            topology = ",".join([str(x) for x in process_mesh.shape])
            dims_mapping = ",".join([str(x) for x in dims_mapping])
            result = processes + topology + dims_mapping
            return result

        memories = {}
        self.max_memories = {}
        var_info = (
            {}
        )  # var_name: [[process_mesh, dims_mapping], [id]], [[process_mesh, dims_mapping], [id]]}

        for block in self.program.blocks:
            for op in block.ops:
                self._ordered_ops.append([op.desc.id(), op])
        self._ordered_ops.sort(key=lambda x: x[0])

        parameters = set()
        for op_id, op in self._ordered_ops:
            if op.type in [
                "create_py_reader",
                "create_double_buffer_reader",
                "read",
            ]:
                continue
            dist_op = dist_context.get_dist_op_for_program(op)
            if not dist_op:
                continue
            process_mesh = dist_op.dist_attr.process_mesh
            for var_name in op.input_arg_names:
                input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(
                    var_name
                )

                if var_name not in var_info:
                    var_info[var_name] = {}
                key = _convert_pm_and_dm_to_str(
                    process_mesh, input_dims_mapping
                )
                if key not in var_info[var_name]:
                    var_info[var_name][key] = {}
                # It is even partition now
                if "position" not in var_info[var_name][key]:
                    var_info[var_name][key]["position"] = []
                var_info[var_name][key]["position"].append(op_id)

                if "memory" not in var_info[var_name][key]:
                    var = dist_op.get_serial_input(var_name)
                    if var is None:
                        continue
                    global_sizes = var.shape
                    dtype = var.dtype
                    sizes = DistributedTensor.get_local_sizes(
                        global_sizes,
                        input_dims_mapping,
                        process_mesh.shape,
                        process_mesh.process_ids,
                    )
                    var_info[var_name][key]["memory"] = self._calculate_bytes(
                        sizes, dtype
                    )
                    if var.persistable:
                        name = var_name + key
                        if name not in parameters:
                            parameters.add(name)
                            for process in process_mesh.process_ids:
                                if process not in memories:
                                    memories[process] = 0
                                memories[process] += var_info[var_name][key][
                                    "memory"
                                ]

            for var_name in op.output_arg_names:
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(
                    var_name
                )
                if var_name not in var_info:
                    var_info[var_name] = {}
                key = _convert_pm_and_dm_to_str(
                    process_mesh, output_dims_mapping
                )
                if key not in var_info[var_name]:
                    var_info[var_name][key] = {}
                if "position" not in var_info[var_name][key]:
                    var_info[var_name][key]["position"] = []
                var_info[var_name][key]["position"].append(op_id)

                if "memory" not in var_info[var_name][key]:
                    var = dist_op.get_serial_output(var_name)
                    global_sizes = var.shape
                    dtype = var.dtype
                    sizes = DistributedTensor.get_local_sizes(
                        global_sizes,
                        output_dims_mapping,
                        process_mesh.shape,
                        process_mesh.process_ids,
                    )
                    var_info[var_name][key]["memory"] = self._calculate_bytes(
                        sizes, dtype
                    )
                    if var.persistable:
                        name = var_name + key
                        if name not in parameters:
                            parameters.add(name)
                            for process in process_mesh.process_ids:
                                if process not in memories:
                                    memories[process] = 0
                                memories[process] += var_info[var_name][key][
                                    "memory"
                                ]

        has_used_vars = set()
        not_calc_vars = set()
        for op_id, op in self._ordered_ops:
            if op.type in [
                "create_py_reader",
                "create_double_buffer_reader",
                "read",
            ]:
                continue
            can_free_memories = {}
            can_free_vars = set()
            dist_op = dist_context.get_dist_op_for_program(op)
            if not dist_op:
                continue
            process_mesh = dist_op.dist_attr.process_mesh
            for var_name in op.input_arg_names:
                input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(
                    var_name
                )
                key = _convert_pm_and_dm_to_str(
                    process_mesh, input_dims_mapping
                )
                has_used_var = var_name + key
                var = dist_op.get_serial_input(var_name)
                if var is None:
                    continue
                # Not used
                if (
                    has_used_var not in has_used_vars
                    and has_used_var not in parameters
                ):
                    if has_used_var in not_calc_vars:
                        continue
                    has_used_vars.add(has_used_var)
                    for process in process_mesh.process_ids:
                        if process not in memories:
                            memories[process] = 0
                        memories[process] += var_info[var_name][key]["memory"]
                # Used
                if op_id == var_info[var_name][key]["position"][-1]:
                    if (
                        has_used_var not in can_free_vars
                        and not var.persistable
                    ):
                        can_free_vars.add(has_used_var)
                        for process in process_mesh.process_ids:
                            if process not in can_free_memories:
                                can_free_memories[process] = 0
                            can_free_memories[process] += var_info[var_name][
                                key
                            ]["memory"]

            for var_name in op.output_arg_names:
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(
                    var_name
                )
                key = _convert_pm_and_dm_to_str(
                    process_mesh, output_dims_mapping
                )
                has_used_var = var_name + key
                var = dist_op.get_serial_output(var_name)
                if (
                    op.type == "reshape2"
                    or op.type == "transpose2"
                    or op.type == "elementwise_add"
                ):
                    not_calc_vars.add(has_used_var)
                    continue
                # Not used
                if (
                    has_used_var not in has_used_vars
                    and has_used_var not in parameters
                ):
                    has_used_vars.add(has_used_var)
                    for process in process_mesh.process_ids:
                        if process not in memories:
                            memories[process] = 0
                        memories[process] += var_info[var_name][key]["memory"]
                # Used
                if op_id == var_info[var_name][key]["position"][-1]:
                    if (
                        has_used_var not in can_free_vars
                        and not var.persistable
                    ):
                        can_free_vars.add(has_used_var)
                        for process in process_mesh.process_ids:
                            if process not in can_free_memories:
                                can_free_memories[process] = 0
                            can_free_memories[process] += var_info[var_name][
                                key
                            ]["memory"]

            # Calc peak memory
            for process in memories:
                if process not in self.max_memories:
                    self.max_memories[process] = memories[process]
                else:
                    if memories[process] > self.max_memories[process]:
                        self.max_memories[process] = memories[process]
            # Free memory
            for process in can_free_memories:
                if process in memories:
                    memories[process] -= can_free_memories[process]

        # Calculate the max memory in all ranks
        max_memory = max(self.max_memories.values())
        self.max_memory = max_memory

        return max_memory

    def estimate(self, dist_context, resharder=None):
        self.prepare()
        from ..reshard import Resharder

        resharder = (
            Resharder(self.program, None, self.rank, dist_context, [])
            if resharder is None
            else resharder
        )

        block = self.program.global_block()
        self._estimate_core(dist_context, resharder, block)

        return self.global_cost

    def _print_tag(self, max_len, length):
        tag = "+" + "-" * max_len
        for i in range(length):
            print(tag, end="")
            if i == length - 1:
                print("+")

    def _print_vals(self, vals, max_len):
        for idx, val in enumerate(vals):
            s = "|" + str(val).center(max_len)
            print(s, end="")
            if idx == len(vals) - 1:
                print("|")

    def _pretty_print_memory_cost(self):
        """Print memory of every rank prettily."""
        if not self.max_memories or not self.max_memory:
            raise ValueError("Please calculate memory cost before print.")

        # Padding automatically
        max_len = 0
        header = ["Rank", "Memory(MiB)"]
        memories = [
            int(item // 1e6) for item in list(self.max_memories.values())
        ]
        for memory in memories + header:
            if len(str(memory)) > max_len:
                max_len = len(str(memory))
        max_len += 4  # for pretty print of center

        # Print tag
        self._print_tag(max_len, len(header))

        # Print header
        self._print_vals(header, max_len)

        # Print tag
        self._print_tag(max_len, len(header))

        # Print rank and its memory
        for i in range(len(self.max_memories)):
            memory = memories[i]
            vals = [i, memory]
            self._print_vals(vals, max_len)
            self._print_tag(max_len, len(header))

    def _pretty_print_global(self):
        """Print global execution time and max memory prettily."""
        if not self.max_memories or not self.max_memory:
            raise ValueError("Please calculate cost before print.")

        # Padding automatically
        max_len = 0
        header = ["Execution Time(us)", "Max Memory(MiB)"]
        vals = [round(self.global_cost.time, 3), int(self.max_memory // 1e6)]
        for memory in vals + header:
            if len(str(memory)) > max_len:
                max_len = len(str(memory))
        max_len += 4  # for pretty print of center

        # Print tag
        self._print_tag(max_len, len(header))

        # Print header
        self._print_vals(header, max_len)

        # Print tag
        self._print_tag(max_len, len(header))

        # Print exec time and max memory
        self._print_vals(vals, max_len)

        # Print tag
        self._print_tag(max_len, len(header))

    def pretty_print_cost(self):
        """Print cost prettily."""
        print("The global execution time and max memory are as follows:")
        self._pretty_print_global()
        print("The memory of every rank is as follows:")
        self._pretty_print_memory_cost()


def get_cost_from_engine(engine, mode):
    import copy

    from ..utils import to_list

    # Construct cost estimator by original main program
    serial_main_prog = (
        engine._fwd_main_progs[mode].clone()
        if mode in engine._fwd_main_progs
        else engine._orig_main_prog.clone()
    )

    serial_startup_prog = (
        engine._fwd_dist_contexts[mode]._original_serial_main_program.clone()
        if mode in engine._fwd_dist_contexts
        else engine._orig_startup_prog.clone()
    )
    losses = (
        to_list(engine._loss)
        if (
            not isinstance(engine._loss, paddle.nn.Layer)
            and not callable(engine._loss)
        )
        else engine._losses
    )
    serial_optimizer = copy.deepcopy(engine._orig_optimizer)
    if mode in engine._fwd_dist_contexts:
        dist_context = copy.deepcopy(engine._fwd_dist_contexts[mode])
    else:
        from ..dist_context import DistributedContext

        dist_context = DistributedContext(
            serial_main_prog,
            serial_startup_prog,
            serial_optimizer,
            losses,
            {},
            {"loss": losses},
            engine._cluster,
            engine._strategy,
        )
    from ..completion import Completer

    completer = Completer(dist_context)
    completer.complete_forward_annotation()
    dist_context.block_state.parse_forward_blocks(
        dist_context.serial_main_program
    )

    if mode == "eval" or mode == "predict":
        cost_estimator = CostEstimator(serial_main_prog, engine._cluster)
    elif mode == "train":
        from ..parallelizer_v2 import Parallelizer

        # Get serial main program with backward
        parallelizer = Parallelizer(mode, completer, dist_context)
        # Generate backward
        loss_name = dist_context.serial_loss.name
        serial_loss = serial_main_prog.global_block()._var_recursive(loss_name)
        params_grads = parallelizer._generate_backward(
            serial_main_prog, serial_startup_prog, serial_loss
        )

        # Generate optimizer
        optimizer_ops = parallelizer._generate_optimizer(
            serial_main_prog,
            serial_startup_prog,
            serial_optimizer,
            params_grads,
        )
        cost_estimator = CostEstimator(serial_main_prog, engine._cluster)

    # Estimate global_cost and  max memory
    global_cost = cost_estimator.estimate(dist_context)
    max_memory = cost_estimator._estimate_max_memory_by_dist_op(dist_context)

    # Print the cost
    cost_estimator.pretty_print_cost()

    return global_cost, max_memory
