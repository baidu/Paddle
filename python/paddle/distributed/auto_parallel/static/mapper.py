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
# limitations under the License

import functools
import operator
import os
from collections import deque

import paddle
import paddle.distributed as dist

from .cluster import DeviceType
from .graph import Graph
from .process_group import get_process_group


def is_collective_comm_op(op):
    comm_list = [
        "c_allreduce_sum",
        "c_allreduce_min",
        "c_allreduce_max",
        "c_allreduce_prod",
        "all_gather",
        "all_reduce",
        "broadcast",
    ]
    reduce_type = [
        dist.ReduceOp.SUM,
        dist.ReduceOp.MIN,
        dist.ReduceOp.MAX,
        dist.ReduceOp.PROD,
    ]
    if op.type == "reduce" and op.attr("reduce_type") in reduce_type:
        return True
    if op.type in comm_list:
        return True
    else:
        return False


def is_p2p_comm_op(op):
    comm_list = ["send_v2", "recv_v2"]
    if op.type in comm_list:
        return True
    else:
        return False


def get_dtype_bytes(dtype):
    num_bytes = 0
    if dtype == paddle.float64:
        num_bytes = 8
    elif dtype == paddle.float32:
        num_bytes = 4
    elif dtype == paddle.float16:
        num_bytes = 2
    elif dtype == paddle.bfloat16:
        num_bytes = 2
    elif dtype == paddle.int64:
        num_bytes = 8
    elif dtype == paddle.int32:
        num_bytes = 4
    elif dtype == paddle.int16:
        num_bytes = 2
    elif dtype == paddle.int8:
        num_bytes = 1
    elif dtype == paddle.uint8:
        num_bytes = 1
    else:
        raise ValueError(f"Unrecognized dtype {dtype}.")
    return num_bytes


def get_comm_volume(comm_op, src_rank, tgt_rank):
    comm_volume = None
    if src_rank == tgt_rank:
        return comm_volume
    comm_op_type = comm_op.type
    if comm_op_type != "recv_v2":
        tensor_name = comm_op.input_arg_names[0]
    else:
        tensor_name = comm_op.output_arg_names[0]
    tensor = comm_op.block._find_var_recursive(tensor_name)
    assert tensor is not None
    tensor_shape = tensor.shape
    # Skip the batch dim
    new_tensor_shape = []
    for val in tensor_shape:
        if val == -1:
            print("Warning: -1 in the tensor shape.")
            new_tensor_shape.append(1)
        else:
            new_tensor_shape.append(val)
    tensor_size = functools.reduce(operator.mul, new_tensor_shape, 1)
    tensor_bytes = tensor_size * get_dtype_bytes(tensor.dtype)
    if "c_allreduce" in comm_op_type:
        comm_volume = 2 * tensor_bytes
    elif "all_gather" in comm_op_type:
        comm_volume = tensor_bytes
    elif "broadcast" in comm_op_type:
        if comm_op.attr("root") == src_rank:
            comm_volume = tensor_bytes
        else:
            comm_volume = None
    elif "c_reduce" in comm_op_type:
        if comm_op.attr("root_id") == src_rank:
            comm_volume = None
        else:
            comm_volume = tensor_bytes
    elif "reduce" == comm_op_type:
        if comm_op.attr("root_id") == src_rank:
            comm_volume = None
        else:
            comm_volume = tensor_bytes
    elif "send_v2" in comm_op_type:
        if comm_op.attr("peer") == tgt_rank:
            comm_volume = tensor_bytes
        else:
            comm_volume = None
    elif "recv_v2" in comm_op_type:
        comm_volume = None
    else:
        raise ValueError("Unrecognized communication operator.")
    return comm_volume


def analyze_comm_requirements_from_op(op, rank, g_process_group_map):
    comm_requirements_to_ranks = {}
    if is_collective_comm_op(op):
        process_group_id = op.attr("ring_id")
        process_group = get_process_group(process_group_id, g_process_group_map)
        if rank not in process_group.ranks:
            return comm_requirements_to_ranks
        for tgt_rank in process_group.ranks:
            comm_volume = get_comm_volume(op, rank, tgt_rank)
            if comm_volume is not None:
                comm_requirements_to_ranks[tgt_rank] = {}
                comm_requirements_to_ranks[tgt_rank][
                    "comm_volume"
                ] = comm_volume
    elif is_p2p_comm_op(op):
        tgt_rank = op.attr("peer")
        comm_volume = get_comm_volume(op, rank, tgt_rank)
        if comm_volume is not None:
            comm_requirements_to_ranks[tgt_rank] = {}
            comm_requirements_to_ranks[tgt_rank]["comm_volume"] = comm_volume
    else:
        comm_requirements_to_ranks = {}
    return comm_requirements_to_ranks


def analyze_requirements_for_program(src_info, rank):
    program = src_info[0]
    g_process_group_map = src_info[1]
    resource_requirements = {}
    comm_requirements_to_ranks = {}
    # only support device_type and only support GPU for now
    resource_requirements["device_type"] = DeviceType.GPU
    for block in program.blocks:
        for op in block.ops:
            cur_comm_requirements_to_ranks = analyze_comm_requirements_from_op(
                op, rank, g_process_group_map
            )
            for tgt_rank, link_info in cur_comm_requirements_to_ranks.items():
                if tgt_rank in comm_requirements_to_ranks:
                    comm_requirements_to_ranks[tgt_rank][
                        "comm_volume"
                    ] += link_info["comm_volume"]
                else:
                    comm_requirements_to_ranks[tgt_rank] = {}
                    comm_requirements_to_ranks[tgt_rank]["comm_volume"] = (
                        link_info["comm_volume"]
                    )
    return resource_requirements, comm_requirements_to_ranks


def build_process_graph(distributed_program):
    graph = Graph()
    for src_rank, src_info in distributed_program.items():
        (
            resource_requirements,
            comm_requirements_to_ranks,
        ) = analyze_requirements_for_program(src_info, src_rank)
        graph.add_node(src_rank, resource_requirements=resource_requirements)
        for tgt_rank, comm_requirements in comm_requirements_to_ranks.items():
            graph.add_edge(
                src_rank, tgt_rank, comm_requirements=comm_requirements
            )
    return graph


def build_cluster_graph(cluster):
    graph = Graph()
    cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES")
    cuda_visible_devices = []
    if cuda_visible_devices_env is not None and cuda_visible_devices_env != "":
        cuda_visible_devices = [
            int(d.strip()) for d in cuda_visible_devices_env.split(",")
        ]
    for machine in cluster.machines.values():
        for device in machine.devices.values():
            graph.add_node(device.global_id, device=device)
            if (
                cuda_visible_devices
                and device.local_id not in cuda_visible_devices
            ):
                graph.nodes[device.global_id]["occupied"] = True
            else:
                graph.nodes[device.global_id]["occupied"] = False
        for link in machine.links.values():
            graph.add_edge(
                link.source.global_id, link.target.global_id, link=link
            )
    return graph


def mapping(distributed_program, cluster):
    # A very simple mapping algorithm only for GPUs.
    # Here we assume one process will be mapped to one GPU.
    # In the future, more mapping configurations and algorithms will be supported.
    process_graph = build_process_graph(distributed_program)

    cluster_graph = build_cluster_graph(cluster)

    for cur_rank_node in process_graph:
        cur_rank_node["visited"] = False

    def sort_by_comm_volume(rank_edge):
        return rank_edge["comm_requirements"]["comm_volume"]

    def sort_by_comm_bandwidth(device_edge):
        return device_edge["link"].bandwidth

    def select_unvisited_rank_node(rank_node_list):
        selected_rank_node = None
        for rank_node in rank_node_list:
            if rank_node["visited"] is False:
                selected_rank_node = rank_node
        return selected_rank_node

    queue = deque()
    root_rank_node = select_unvisited_rank_node(
        list(process_graph.nodes.values())
    )
    while root_rank_node is not None:
        queue.append(root_rank_node)
        while queue:
            cur_rank_node = queue.popleft()
            if cur_rank_node["visited"]:
                continue
            device_type = cur_rank_node["resource_requirements"]["device_type"]
            cur_device_node = None
            for device_node in cluster_graph.nodes.values():
                if (device_node["device"].type == device_type) and (
                    not device_node["occupied"]
                ):
                    device_node["occupied"] = True
                    cur_rank_node["visited"] = True
                    cur_rank_node["device"] = device_node["device"]
                    cur_device_node = device_node
                    break
            assert (
                cur_device_node
            ), "Cannot find a device to satisfy the requirement."

            nbr_rank_edges = []
            for nbr_rank_node_id, nbr_rank_edge in process_graph.adjs[
                cur_rank_node.id
            ].items():
                assert (
                    nbr_rank_edge.src_id == cur_rank_node.id
                    and nbr_rank_edge.tgt_id == nbr_rank_node_id
                )
                queue.append(process_graph.nodes[nbr_rank_node_id])
                nbr_rank_edges.append(nbr_rank_edge)
            nbr_rank_edges.sort(key=sort_by_comm_volume)

            nbr_device_edges = []
            for nbr_device_edge in cluster_graph.adjs[
                cur_device_node.id
            ].values():
                nbr_device_edges.append(nbr_device_edge)
            nbr_device_edges.sort(key=sort_by_comm_bandwidth)

            for nbr_rank_edge in nbr_rank_edges:
                src_rank_node = process_graph.nodes[nbr_rank_edge.src_id][
                    "visited"
                ]
                if src_rank_node:
                    continue
                device_type = src_rank_node["resource_requirements"][
                    "device_type"
                ]
                nbr_rank_node = process_graph.nodes[nbr_rank_edge.tgt_id]
                for nbr_device_edge in nbr_device_edges:
                    nbr_device_node = cluster_graph.nodes[
                        nbr_device_edge.tgt_id
                    ]
                    if (nbr_device_node["device"].type == device_type) and (
                        not nbr_device_node["occupied"]
                    ):
                        nbr_device_node["occupied"] = True
                        nbr_rank_node["visited"] = True
                        nbr_rank_node["device"] = nbr_device_node["device"]
                        break
        root_rank_node = select_unvisited_rank_node(
            list(process_graph.nodes.values())
        )

    rank_mapping = {}
    for rank, rank_node in process_graph.nodes.items():
        device = rank_node["device"]
        machine = device.machine
        if machine.id in rank_mapping:
            rank_mapping[machine.id]["hostname"] = machine.hostname
            rank_mapping[machine.id]["addr"] = machine.addr
            rank_mapping[machine.id]["port"] = machine.port
            if rank not in rank_mapping[machine.id]["ranks"]:
                rank_mapping[machine.id]["ranks"][rank] = []
                rank_mapping[machine.id]["ranks"][rank].append(device.local_id)
            else:
                rank_mapping[machine.id]["ranks"][rank].append(device.local_id)
        else:
            rank_mapping[machine.id] = {}
            rank_mapping[machine.id]["hostname"] = machine.hostname
            rank_mapping[machine.id]["addr"] = machine.addr
            rank_mapping[machine.id]["port"] = machine.port
            rank_mapping[machine.id]["ranks"] = {}
            rank_mapping[machine.id]["ranks"][rank] = []
            rank_mapping[machine.id]["ranks"][rank].append(device.local_id)
    for machine_mapping in rank_mapping.values():
        for rank_devices in machine_mapping["ranks"].values():
            rank_devices.sort()

    return rank_mapping
