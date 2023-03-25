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
# limitations under the License.

import copy
import logging
import math
import os
from abc import abstractmethod
from collections import OrderedDict

import paddle
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.dist_attribute import (
    OperatorDistAttr,
    TensorDistAttr,
)
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed.auto_parallel.dist_tensor import DistributedTensor
from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Parameter, unique_name

from ...utils.log_utils import get_logger
from ..graph import Graph

_PATTERNS = {}


def register_pattern(cls):
    """Register pattern for rule-based tuner."""

    def register():
        global _PATTERNS
        pattern = cls()
        _PATTERNS[pattern.name] = pattern
        # sort patterns according to the number of sharded tensors
        # set its dist attr by the fisrt one when a tensor can be matched by multiple patterns.
        _PATTERNS = dict(
            sorted(
                _PATTERNS.items(), key=lambda x: -x[1].attrs["sharded_tensors"]
            )
        )

    register()

    return cls


class BasePattern(Graph):
    """
    Base class of pattern.
    The BasePattern inherits the Graph, two important differences are shard_spec and sharded_tensors.
    For shard_spec, it indicates the shard specification of tensor node in this pattern under different parallelism.
    For sharded_tensors, it represents the number of tensors which sharded.
    """

    _name = "base"

    def __init__(self):
        """Every pattern has its own name and build method."""
        super().__init__()
        self.build()

    @property
    def name(self):
        return self.__class__._name

    @abstractmethod
    def build(self):
        pass


@register_pattern
class QKVPattern(BasePattern):
    """The QKV pattern defined by GPT model in PaddleFleetX."""

    name = "qkv"

    def __init__(self):
        super().__init__()

    def build(self):
        query = self.add_node(0, **{"type": "var"})

        # define q, k, v weight
        q_weight = self.add_node(1, **{"dim": 2, "type": "param"})
        k_weight = self.add_node(2, **{"dim": 2, "type": "param"})
        v_weight = self.add_node(3, **{"dim": 2, "type": "param"})
        # define q, k, v matmul_v2
        q_matmul_v2 = self.add_node(4, **{"type": "matmul_v2"})
        k_matmul_v2 = self.add_node(5, **{"type": "matmul_v2"})
        v_matmul_v2 = self.add_node(6, **{"type": "matmul_v2"})
        # define input edge
        q_x_edge = self.add_edge(
            query.id, q_matmul_v2.id, **{"input_name": "X"}
        )
        k_x_edge = self.add_edge(
            query.id, k_matmul_v2.id, **{"input_name": "X"}
        )
        v_x_edge = self.add_edge(
            query.id, v_matmul_v2.id, **{"input_name": "X"}
        )
        q_y_edge = self.add_edge(
            q_weight.id, q_matmul_v2.id, **{"input_name": "Y"}
        )
        k_y_edge = self.add_edge(
            k_weight.id, k_matmul_v2.id, **{"input_name": "Y"}
        )
        v_y_edge = self.add_edge(
            v_weight.id, v_matmul_v2.id, **{"input_name": "Y"}
        )
        # define q, k, v matmul_v2 output
        q = self.add_node(7, **{"type": "var"})
        k = self.add_node(8, **{"type": "var"})
        v = self.add_node(9, **{"type": "var"})

        # define output edge
        q_out_edge = self.add_edge(
            q_matmul_v2.id, q.id, **{"output_name": "Out"}
        )
        k_out_edge = self.add_edge(
            k_matmul_v2.id, k.id, **{"output_name": "Out"}
        )
        v_out_edge = self.add_edge(
            v_matmul_v2.id, v.id, **{"output_name": "Out"}
        )

        # define shard_spec
        shard_spec = {
            "dp_mp": {
                0: [0, -1, -1],
                1: [-1, 1],
                2: [-1, 1],
                3: [-1, 1],
            },
            "mp_dp": {
                0: [1, -1, -1],
                1: [-1, 0],
                2: [-1, 0],
                3: [-1, 0],
            },
            "mp": {0: [-1, -1, -1], 1: [-1, 0], 2: [-1, 0], 3: [-1, 0]},
            "dp": {
                0: [0, -1, -1],
                1: [-1, -1],
                2: [-1, -1],
                3: [-1, -1],
            },
        }
        self.attrs["shard_spec"] = shard_spec
        # define sharded_tensors
        self.attrs["sharded_tensors"] = 4


@register_pattern
class RowMatmulPattern(BasePattern):
    """Row matmul pattern defined by GPT model in PaddleFleetX."""

    name = "row_matmul"

    def __init__(self):
        super().__init__()

    def build(self):
        # define reshape input
        input = self.add_node(0, **{"type": "var"})

        # define reshape
        reshape = self.add_node(1, **{"type": "reshape2"})

        # define reshape input egde
        x_edge = self.add_edge(input.id, reshape.id, **{"input_name": "X"})

        # define reshape out
        output = self.add_node(2, **{"type": "var"})

        # define reshape output edge
        out_edge = self.add_edge(
            reshape.id, output.id, **{"output_name": "Out"}
        )

        # define matmul_v2 weight
        weight = self.add_node(3, **{"dim": 2, "type": "param"})

        # define matmul_v2
        matmul_v2 = self.add_node(4, **{"type": "matmul_v2"})

        # define input edge
        x_edge = self.add_edge(output.id, matmul_v2.id, **{"input_name": "X"})
        y_edge = self.add_edge(weight.id, matmul_v2.id, **{"input_name": "Y"})

        # define q, k, v matmul_v2 output
        output = self.add_node(5, **{"type": "var"})

        # define output edge
        out_edge = self.add_edge(
            matmul_v2.id, output.id, **{"output_name": "Out"}
        )

        # define shard_spec
        shard_spec = {
            "dp_mp": {
                3: [1, -1],
            },
            "mp_dp": {
                3: [0, -1],
            },
            "mp": {3: [0, -1]},
            "dp": {
                3: [-1, -1],
            },
        }
        self.attrs["shard_spec"] = shard_spec

        # define sharded_tensors
        self.attrs["sharded_tensors"] = 1


@register_pattern
class FFNPattrern(BasePattern):
    """FFN pattern defined by GPT model in PaddleFleetX."""

    name = "ffn"

    def __init__(self):
        super().__init__()

    def build(self):
        x = self.add_node(0, **{"type": "var"})

        w1_weight = self.add_node(1, **{"dim": 2, "type": "param"})
        w1_matmul = self.add_node(2, **{"type": "matmul_v2"})

        w1_x = self.add_edge(0, 2, **{"input_name": "X"})
        w1_y = self.add_edge(1, 2, **{"input_name": "Y"})

        out1 = self.add_node(3, **{"type": "var"})
        w1_out = self.add_edge(2, 3, **{"output_name": "Out"})

        w1_b = self.add_node(4, **{"dim": 1, "type": "param"})
        add1 = self.add_node(5, **{"type": "elementwise_add"})

        add1_x = self.add_edge(3, 5, **{"input_name": "X"})
        add1_y = self.add_edge(4, 5, **{"input_name": "Y"})

        out2 = self.add_node(6, **{"type": "var"})
        add1_out = self.add_edge(5, 6, **{"output_name": "Out"})

        gelu = self.add_node(7, **{"type": "gelu"})

        gelu_x = self.add_edge(6, 7, **{"input_name": "X"})
        out3 = self.add_node(8, **{"type": "var"})
        gelu_out = self.add_edge(7, 8, **{"output_name": "Out"})

        w2_weight = self.add_node(9, **{"dim": 2, "type": "param"})
        w2_matmul = self.add_node(10, **{"type": "matmul_v2"})

        w1_x = self.add_edge(8, 10, **{"input_name": "X"})
        w1_y = self.add_edge(9, 10, **{"input_name": "Y"})

        out4 = self.add_node(11, **{"type": "var"})
        w2_out = self.add_edge(10, 11, **{"output_name": "Out"})

        w2_b = self.add_node(12, **{"dim": 1, "type": "param"})
        add2 = self.add_node(13, **{"type": "elementwise_add"})

        add2_x = self.add_edge(11, 13, **{"input_name": "X"})
        add2_y = self.add_edge(12, 13, **{"input_name": "Y"})

        out5 = self.add_node(14, **{"type": "var"})
        add2_out = self.add_edge(13, 14, **{"output_name": "Out"})

        # define shard_spec
        shard_spec = {
            "dp_mp": {0: [0, -1, -1], 1: [-1, 1], 9: [1, -1]},
            "mp_dp": {0: [1, -1, -1], 1: [-1, 0], 9: [0, -1]},
            "mp": {1: [-1, 0], 9: [0, -1]},
            "dp": {0: [0, -1, -1], 1: [-1, -1], 9: [-1, -1]},
        }
        self.attrs["shard_spec"] = shard_spec

        # define sharded_tensors
        self.attrs["sharded_tensors"] = 2


@register_pattern
class SharedWordEmbeddingPattern(BasePattern):
    """Sharded word embedding pattern defined by GPT model in PaddleFleetX."""

    name = "shared_word_embedding"

    def __init__(self):
        super().__init__()

    def build(self):
        # define embedding input
        tokens = self.add_node(0, **{"type": "data"})
        word_embeddings = self.add_node(1, **{"dim": 2, "type": "param"})

        # define embedding
        embedding = self.add_node(2, **{"type": "lookup_table_v2"})

        # define embedding input edge
        ids = self.add_edge(0, 2, **{"input_name": "Ids"})
        w = self.add_edge(1, 2, **{"input_name": "W"})

        # define embedding output
        out = self.add_node(3, **{"type": "var"})

        # define embedding output edge
        out_edge = self.add_edge(2, 3, **{"output_name": "Out"})

        # define matmul_v2 input
        x = self.add_node(4, **{"type": "var"})

        # define matmul_v2
        matmul = self.add_node(5, **{"type": "matmul_v2"})

        # define matmul_v2 input edge
        x_edge = self.add_edge(4, 5, **{"input_name": "X"})
        y_edge = self.add_edge(1, 5, **{"input_name": "Y"})

        # define matmul_v2 output
        out = self.add_node(6, **{"type": "var"})

        # define matmul_v2 output edge
        out_edge = self.add_edge(5, 6, **{"output_name": "Out"})

        # define shard_spec
        shard_spec = {
            "dp_mp": {0: [0, -1], 1: [1, -1], 4: [0, -1, -1]},
            "mp_dp": {0: [1, -1], 1: [0, -1], 4: [1, -1, -1]},
            "mp": {0: [-1, -1], 1: [0, -1], 4: [-1, -1, -1]},
            "dp": {0: [0, -1], 1: [-1, -1], 4: [0, -1, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["sharded_tensors"] = 3


@register_pattern
class PositionEmbeddingPattern(BasePattern):
    """Position embedding pattern defined by GPT model in PaddleFleetX."""

    name = "position_embedding"

    def __init__(self):
        super().__init__()

    def build(self):
        # define embedding input
        tokens = self.add_node(0, **{"type": "data"})
        word_embeddings = self.add_node(1, **{"dim": 2, "type": "param"})

        # define embedding
        embedding = self.add_node(2, **{"type": "lookup_table_v2"})

        # define embedding input edge
        ids = self.add_edge(0, 2, **{"input_name": "Ids"})
        w = self.add_edge(1, 2, **{"input_name": "W"})

        # define embedding output
        out = self.add_node(3, **{"type": "var"})

        # define embedding output edge
        out_edge = self.add_edge(2, 3, **{"output_name": "Out"})

        # define shard_spec
        shard_spec = {
            "dp_mp": {0: [0, -1], 1: [-1, -1], 3: [-1, -1, -1]},
            "mp_dp": {0: [1, -1], 1: [-1, -1], 3: [1, -1, -1]},
            "mp": {0: [-1, -1], 1: [-1, -1], 3: [-1, -1, -1]},
            "dp": {0: [0, -1], 1: [-1, -1], 3: [0, -1, -1]},
        }
        self.attrs["shard_spec"] = shard_spec

        # define sharded_tensors
        self.attrs["sharded_tensors"] = 1


@register_pattern
class UnsqueezeDataPattern(BasePattern):
    """Unsqueeze data pattern defined by GPT model in the PaddleFleetX."""

    name = "unsqueeze_data"

    def __init__(self):
        super().__init__()

    def build(self):
        # define unsequeeze input
        tokens = self.add_node(0, **{"type": "data"})
        # define unsequeeze
        unsqueeze = self.add_node(1, **{"type": "unsqueeze2"})
        # define unsequeeze input edge
        x_edge = self.add_edge(0, 1, **{"input_name": "X"})
        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {0: [0, -1]},
            "mp_dp": {0: [1, -1]},
            "mp": {0: [-1, -1]},
            "dp": {0: [0, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["sharded_tensors"] = 1


@register_pattern
class ReshapeDataPattern(BasePattern):
    """Reshape data pattern defined by GPT model in PaddleFleetX."""

    name = "reshape_data"

    def __init__(self):
        super().__init__()

    def build(self):
        # define unsequeeze input
        data = self.add_node(0, **{"type": "data"})

        # define unsequeeze
        reshape = self.add_node(1, **{"type": "reshape2"})

        # define unsequeeze input edge
        x_edge = self.add_edge(0, 1, **{"input_name": "X"})

        # define shard_spec
        shard_spec = {
            "dp_mp": {0: [0, -1]},
            "mp_dp": {0: [1, -1]},
            "mp": {0: [-1, -1]},
            "dp": {0: [0, -1]},
        }
        self.attrs["shard_spec"] = shard_spec

        # define sharded_tensors
        self.attrs["sharded_tensors"] = 1


class GraphUtil:
    """Graph util is used to convert ops to graph or match pattern for graph."""

    @staticmethod
    def convert_to_graph(block):
        """Convert ops to graph."""
        graph = Graph()
        graph.attrs["var_to_id"] = {}  # {var_name: node_id}
        graph.attrs["id_to_var_desc_id"] = {}  # {node_id: var_desc_id}
        graph.attrs["id_to_var_name"] = {}
        graph.attrs["op_to_id"] = {}  # {op_id: node_id}
        graph.attrs["id_to_op"] = {}  # {node_id: op}

        ops = block.ops
        node_id = -1
        for op in ops:
            attrs = op.all_attrs()
            attrs["type"] = op.type
            node_id += 1

            # create op node
            op_node = graph.add_node(node_id, **attrs)
            graph.attrs["op_to_id"][op.desc.id()] = op_node.id
            graph.attrs["id_to_op"][op_node.id] = op
            graph._attr_to_nodes[op_node.id] = {}
            for input_name in op.input_names:
                graph._attr_to_nodes[op_node.id][input_name] = []
                for var_name in op.input(input_name):
                    if var_name not in graph.attrs["var_to_id"]:
                        # create var node
                        node_id += 1
                        var_node = graph.add_node(node_id)
                        var = block._var_recursive(var_name)
                        if var.is_parameter:
                            var_node.attrs["type"] = "param"
                            var_node.attrs["dim"] = len(var.shape)
                        elif var.is_data:
                            var_node.attrs["type"] = "data"
                            var_node.attrs["dim"] = len(var.shape)
                        else:
                            var_node.attrs["type"] = "var"
                        graph.attrs["var_to_id"][var_name] = var_node.id
                        graph.attrs["id_to_var_desc_id"][
                            var_node.id
                        ] = var.desc.original_id()
                        graph.attrs["id_to_var_name"][var_node.id] = var_name
                    else:
                        var_node_id = graph.attrs["var_to_id"][var_name]
                        var_node = graph._nodes[var_node_id]

                    # create edge that input -> op
                    input_edge = graph.add_edge(var_node.id, op_node.id)
                    input_edge.attrs["input_name"] = input_name
                    graph._attr_to_nodes[op_node.id][input_name].append(
                        var_node
                    )

                for output_name in op.output_names:
                    graph._attr_to_nodes[op_node.id][output_name] = []
                    for var_name in op.output(output_name):
                        if var_name not in graph.attrs["var_to_id"]:
                            # create var node
                            node_id += 1
                            var_node = graph.add_node(node_id)
                            var = block._var_recursive(var_name)
                            if var.is_parameter:
                                var_node.attrs["type"] = "param"
                            else:
                                var_node.attrs["type"] = "var"
                            graph.attrs["var_to_id"][var_name] = var_node.id
                            graph.attrs["id_to_var_desc_id"][
                                var_node.id
                            ] = var.desc.original_id()
                            graph.attrs["id_to_var_name"][
                                var_node.id
                            ] = var_name
                        else:
                            var_node_id = graph.attrs["var_to_id"][var_name]
                            var_node = graph._nodes[var_node_id]

                        # create edge that op -> output
                        output_edge = graph.add_edge(op_node.id, var_node.id)
                        output_edge.attrs["output_name"] = output_name

                        graph._attr_to_nodes[op_node.id][output_name].append(
                            var_node
                        )

        return graph

    @staticmethod
    def match_pattern(pattern, graph):
        def _is_op_node(node):
            """Judge whether node is op node."""
            if node.attrs["type"] not in ["var", "param", "data"]:
                return True

            return False

        def _compare_op_node(src, tgt):
            """Compare whether two op nodes are equivalent."""
            if src.attrs["type"] != tgt.attrs["type"]:
                return False

            return True

        def _compare_var_node(src, tgt):
            """Compare whether two var nodes are equivalent."""
            for key in src.attrs:
                if key not in tgt.attrs:
                    return False
                if src.attrs[key] != tgt.attrs[key]:
                    return False

            return True

        def _match_core(src_node, tgt_node):
            nonlocal not_matched

            # not support one input name or output name corresponding to multiple vars
            if not_matched:
                return

            if _is_op_node(src_node):
                # compare op node whether equal
                if not _compare_op_node(src_node, tgt_node):
                    not_matched = True
                    return

                result[src_node.id] = tgt_node.id

                # input var nodes
                src_input_nodes = src_reverse_adjs[src_node.id]
                for node in src_input_nodes:
                    # has visited
                    if node.id in result:
                        continue
                    edge = src_edges[node.id][src_node.id]
                    input_name = edge.attrs["input_name"]

                    # NOTE: do not support one input name or output name corresponding to multiple vars
                    compare_nodes = tgt_attr_to_nodes[tgt_node.id].get(
                        input_name, None
                    )
                    if not compare_nodes:
                        not_matched = True
                        return
                    _match_core(node, compare_nodes[0])

                # output var nodes
                src_output_node_ids = src_edges[src_node.id].keys()
                for node_id in src_output_node_ids:
                    # has visited
                    if node_id in result:
                        continue
                    node = src_nodes[node_id]
                    edge = src_edges[src_node.id][node_id]
                    output_name = edge.attrs["output_name"]

                    # NOTE: do not support one input name or output name corresponding to multiple vars
                    compare_nodes = tgt_attr_to_nodes[tgt_node.id].get(
                        output_name, None
                    )
                    if not compare_nodes:
                        not_matched = True
                        return
                    _match_core(node, compare_nodes[0])

            else:
                # compare var nodes whether equal
                if not _compare_var_node(src_node, tgt_node):
                    not_matched = True
                    return

                result[src_node.id] = tgt_node.id

                # as input for op node
                src_as_input_node_ids = src_edges[src_node.id].keys()
                for node_id in src_as_input_node_ids:
                    if node_id in result:
                        continue

                    src_edge = src_edges[src_node.id][node_id]
                    input_name = src_edge.attrs["input_name"]
                    compare_node_ids = tgt_edges[tgt_node.id].keys()

                    compare_node = None
                    for compare_node_id in compare_node_ids:
                        edge = tgt_edges[tgt_node.id][compare_node_id]
                        if (
                            edge.attrs["input_name"] == input_name
                            and compare_node_id not in result.values()
                        ):
                            compare_node = tgt_nodes[compare_node_id]
                            break

                    if not compare_node:
                        not_matched = True
                        return
                    _match_core(src_nodes[node_id], compare_node)

                # as output for op node
                src_as_output_nodes = src_reverse_adjs[src_node.id]
                for node in src_as_output_nodes:
                    if node.id in result:
                        continue

                    src_edge = src_edges[node.id][src_node.id]
                    output_name = src_edge.attrs["output_name"]

                    compare_nodes = tgt_reverse_adjs[tgt_node.id]

                    compare_node = None
                    for item in compare_nodes:
                        node_id = item.id
                        edge = tgt_edges[node_id][tgt_node.id]
                        if edge.attrs["output_name"] == output_name:
                            compare_node = tgt_nodes[node_id]
                            break
                    if not compare_node:
                        not_matched = True
                        return
                    _match_core(src_nodes[node.id], compare_node)

        results = []
        matched_ids = set()
        matched_op_node_ids = set()
        result = {}
        src_nodes = pattern.nodes
        src_edges = pattern._adjs
        src_reverse_adjs = pattern._reverse_adjs

        tgt_nodes = graph.nodes
        tgt_edges = graph._adjs
        tgt_reverse_adjs = graph._reverse_adjs
        tgt_attr_to_nodes = graph._attr_to_nodes

        # starts with a op node
        src_start_node = None
        for node_id in src_nodes:
            node = src_nodes[node_id]
            if node.attrs["type"] not in ["var", "param", "data"]:
                src_start_node = node
                break
        assert src_start_node is not None

        for node_id in tgt_nodes:
            node = tgt_nodes[node_id]
            if node.attrs["type"] == src_start_node.attrs["type"]:
                not_matched = False
                _match_core(src_start_node, node)
                if not not_matched:
                    need_to_append = True
                    for value in result.values():
                        if value in matched_op_node_ids:
                            result = {}
                            need_to_append = False
                            break
                    if need_to_append:
                        results.append(result)
                        for value in result.values():
                            matched_ids.add(value)
                            if value in graph.attrs["id_to_op"].keys():
                                matched_op_node_ids.add(value)
                        result = {}
                else:
                    not_matched = False
                    result = {}
        return results, matched_ids

    @staticmethod
    def match_all_patterns(graph):
        # matched_results maps pattern_name to list which contains pattern node id to graph node id mapping,
        # such as {"pattern_name": [{pattern_node_id: graph_node}, ]}
        matched_results = {}
        matched_ids = set()
        for pattern_name in _PATTERNS:
            pattern = _PATTERNS[pattern_name]
            results, matched = GraphUtil.match_pattern(pattern, graph)
            for result in results:
                has_matched = False
                for id in result:
                    if result[id] in matched_ids:
                        has_matched = True
                        break
                if not has_matched:
                    for item in result:
                        matched_ids.add(result[id])
                    if pattern.name not in matched_results:
                        matched_results[pattern.name] = []
                    matched_results[pattern.name].append(result)

        return matched_results


class OperatorClusteringUtil:
    """Operator clustering util is used to cluster operators to layers."""

    common_starts = ["layer_norm", "matmul_v2", "matmul"]

    @staticmethod
    def get_ranks(seq):
        """Get rank array of the given seq by doubled algorithm."""
        ordered_seq = sorted(list(set(seq)))
        item_to_rank = {item: idx for idx, item in enumerate(ordered_seq)}
        inter_ranks = [item_to_rank[item] for item in seq]

        length = len(inter_ranks)
        power = 0
        interval = 2**power
        while interval < length:
            for idx, item in enumerate(inter_ranks):
                if idx + interval >= length:
                    inter_ranks[idx] = [item, -1]
                else:
                    inter_ranks[idx] = [item, inter_ranks[idx + interval]]

            tmp = []
            for item in inter_ranks:
                if item not in tmp:
                    tmp.append(item)
            tmp.sort(key=lambda x: (x[0], x[1]))
            item_to_rank = {}
            for idx, val in enumerate(tmp):
                key = ",".join(str(item) for item in val)
                item_to_rank[key] = idx

            inter_ranks = [
                item_to_rank[",".join(str(val) for val in item)]
                for item in inter_ranks
            ]
            power += 1
            interval = 2**power

        return inter_ranks

    @staticmethod
    def get_suffixes(ranks):
        """Get suffix array by the given rank array."""
        suffixes = [0 for idx in range(len(ranks))]
        for idx, item in enumerate(ranks):
            suffixes[item] = idx
        return suffixes

    @staticmethod
    def get_heights(suffixes, seq):
        """Get height array by the suffix array and seq"""
        heights = [-1 for i in range(len(suffixes))]
        for i in range(1, len(seq)):
            x = seq[suffixes[i - 1] :]
            y = seq[suffixes[i] :]
            max_len = len(x) if len(x) > len(y) else len(y)
            same_count = 0
            for j in range(max_len):
                if j >= len(x) or j >= len(y):
                    break
                else:
                    if x[j] == y[j]:
                        same_count += 1
                    else:
                        break
            heights[i] = same_count

        return heights

    @staticmethod
    def get_longest_repeated_sub_seq(suffixes, heights, seq):
        """Get longest repeated sub sequence by suffix array algorithm."""
        length = len(seq)
        if length <= 1:
            return None
        k = length // 2
        height_groups = []
        longest_sub_seq = None
        longest_sub_seqs = []

        while k >= 2:
            height_group = []
            for i in range(1, len(heights)):
                if heights[i] >= k:
                    if i == 1:
                        height_group.append(0)
                    height_group.append(i)
                else:
                    if i == 1:
                        height_groups.append([0])
                        height_group = [i]
                    else:
                        height_groups.append(height_group)
                        height_group = [i]

            if height_group:
                height_groups.append(height_group)

            for height_group in height_groups:
                suffix_group = []
                index_group = []
                for idx in height_group:
                    suffix_group.append(idx)
                    index_group.append(suffixes[idx])

                max_index = max(index_group)
                min_index = min(index_group)
                if max_index - min_index >= k:
                    longest_sub_seq = seq[min_index : min_index + k]
                    if (
                        longest_sub_seq[0]
                        in OperatorClusteringUtil.common_starts
                    ):
                        return longest_sub_seq
            if longest_sub_seq is not None:
                return longest_sub_seq

            k -= 1
            height_groups = []

        return longest_sub_seq

    @staticmethod
    def get_decomposed_sub_seq(seq):
        """Get decomposed sub seq s by seq S such as s * R = S."""
        if not seq:
            return seq

        decomposed_sub_seq = seq
        seq_len = len(seq)
        if seq_len == 1:
            return decomposed_sub_seq
        else:
            for interval in range(2, seq_len + 1):
                if seq_len % interval == 0:
                    repeated_times = seq_len // interval
                    decomposed_sub_seq = seq[0:interval]
                    decomposed = True
                    for j in range(1, repeated_times + 1):
                        sub_seq = seq[interval * (j - 1) : interval * j]
                        if sub_seq != decomposed_sub_seq:
                            decomposed = False
                            break
                    if decomposed:
                        return decomposed_sub_seq

        return decomposed_sub_seq

    @staticmethod
    def replace_by_decomposed_seq(sub_seq, seq):
        """Replace seq by sub seq."""
        if not sub_seq:
            return seq

        result = []
        sub_seq_len = len(sub_seq)
        i = 0
        while i < len(seq):
            if seq[i : i + sub_seq_len] == sub_seq:
                result.append(seq[i : i + sub_seq_len])
                i += sub_seq_len
            else:
                result.append(seq[i])
                i += 1

        return result

    @staticmethod
    def stop_replace(seq):
        for item in seq:
            if not isinstance(item, list):
                return False
        return True


class ClusterPartitionUtil:
    """Cluster partition util is used to get device meshes and process meshes."""

    @staticmethod
    def factorization(num):
        factors = []
        for i in range(1, int(math.floor(math.sqrt(num))) + 1):
            if num % i == 0:
                factors.append([i, int(num / i)])
        return factors

    @staticmethod
    def complete_meshes(partitions: list, num: int):
        if num == 2:
            return [[1, 2], [2, 1]]
        if num == 3:
            return [[1, 2], [2, 1], [1]]
        # special cases
        if len(partitions) == 1:
            partitions = ClusterPartitionUtil.factorization(num - 1)
            partitions.append([1])
        return partitions

    @staticmethod
    def partition_cluster(
        n: int,
        m: int,
        filter=[
            complete_meshes.__func__,
        ],
    ) -> list:
        """
        Partiton cluster into possible device meshes.
        Args:
            n (int): The number of nodes.
            m (int): The number of single devices on each node.
            filter (list): Functions for filtering useful meshes
        Returns:
            device_meshed (list) : The possible device meshes.
        """
        partition_result = ClusterPartitionUtil.factorization(n)
        for func in filter:
            partition_result = func(partition_result, n)
        device_meshes = []
        if n == 1:
            partition_result = ClusterPartitionUtil.factorization(m)
            for partition in partition_result:
                device_mesh = []
                for i in range(partition[0]):
                    device_mesh.append([1, partition[1]])
                device_meshes.append(device_mesh)
        else:
            incerement = 1 if partition_result[-1] == [1] else 0
            for partition in partition_result:
                if len(partition) < 2:
                    continue
                device_mesh = []
                for i in range(partition[0]):
                    device_mesh.append([partition[1], m])
                device_mesh[-1][0] += incerement
                device_meshes.append(device_mesh)

        return device_meshes


def convert_to_process_meshes(device_mesh: list) -> list:
    """
    Transfer device_meshes into possible process meshes.
    Args:
        device meshes (list): [n,m], one device mesh.
    Returns:
        process_meshes (list): Possible process_meshes
    """
    n, m = device_mesh[0], device_mesh[1]
    factors = (
        ClusterPartitionUtil.factorization(m)
        if n == 1
        else ClusterPartitionUtil.factorization(n)
    )
    process_meshes = []
    if n == 1:
        for factor in factors:
            if factor[0] == 1:
                process_meshes.append([factor[1]])
                continue
            process_meshes.append(factor)
    else:
        for factor in factors:
            mul1, mul2 = factor[0], factor[1]
            if mul1 == 1:
                process_meshes.append([m * mul2])
            elif mul1 != mul2:
                process_meshes.append([int(n / mul2), m * mul2])
            process_meshes.append([int(n / mul1), m * mul1])
    return process_meshes


class RuleBasedTuner:
    """
    A tuner based on rule from expert experience to search a good parallel strategy.
    Args:
        dist_context (DistributedContext): The distributed context.
        mode (str): The mode of current task, it can be train or eval. Default: train.
        level (str): The level of this tuner, it can be o1 or o2.
                     o2 level may find better strategy but need more time than o1.
                     If level is o1, it means all layers within same parallelism and place layers evenly when in pipeline parallism.
                     If level is o2, it means layers can has own parallelism and place layers may not evenly.
                     Default: o1.
    """

    def __init__(self, dist_context, mode="train", level="o1"):
        self._dist_context = dist_context
        self._cluster = self._dist_context.cluster
        self._mode = mode
        assert level in ["o1", "o2"]
        self._level = level
        self._logger = get_logger(logging.INFO)
        self._use_dp = False

        # forward sub program
        self.fwd_sub_programs = OrderedDict()

        # dist_context of sub program
        self.sub_programs_dist_context = OrderedDict()

        # graph of forward sub program
        self.fwd_sub_program_graphs = OrderedDict()

        # full main program
        self.full_main_program = None

        # full startup program
        self.full_startup_program = None

        # full main program dist context
        self.full_main_program_dist_context = None

        # tensor dist attribute from pattern setting
        self.tensor_dist_attrs = {}

        # op original id to op mapping
        self.op_original_id_to_op = {}

        # op original id to op idx in program
        self.op_original_id_to_idx = {}

        # op original id to grad op original id mapping
        self.op_original_id_to_grad_op_original_id = {}

        # all process meshes that the cluster can express
        self.process_meshes = []

        # all device meshes that the cluster can be partitioned
        self.device_meshes_list = []

        # the best cost of stage in a given device mesh
        self.stage_best_cost_of_dm = {}

        # the best cost of stage in a given process mesh
        self.stage_best_cost_of_pm = {}

        # the op clustering result
        self.layers = []

        self._is_run = True
        if os.getenv("PADDLE_AUTO_PARALLEL_STAGE") != "tuner":
            self._is_run = True
        else:
            self._is_run = False
        self._strategy_path = None
        if self._dist_context._json_config:
            try:
                self._strategy_path = self._dist_context._json_config[
                    "tuner_save_path"
                ]
            except:
                self._strategy_path = None

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
    def level(self):
        return self._level

    def convert_process_mesh_to_key(self, process_mesh):
        """Convert process mesh object to str."""
        processes = ",".join([str(x) for x in process_mesh._process_ids])
        topology = ",".join([str(x) for x in process_mesh._shape])
        key = processes + ";" + topology
        return key

    def gen_full_program(self):
        """Generate full program that contain backward and update phase program if mode is train."""
        self.full_main_program = self.dist_context.serial_main_program.clone()
        if self.mode == "train":
            self.full_startup_program = (
                self.dist_context.serial_startup_program.clone()
            )
            loss = self.full_main_program.global_block().vars[
                self.dist_context.serial_loss.name
            ]
            serial_optimizer = self._dist_context.serial_optimizer
            optimizer = copy.deepcopy(serial_optimizer)
            self.full_main_program_dist_context = DistributedContext(
                serial_main_prog=self.full_main_program,
                serial_startup_prog=self.full_startup_program,
                serial_loss=loss,
            )
            # if in train mode, generate backward and update program.
            with program_guard(
                self.full_main_program, self.full_startup_program
            ):
                params_grads = append_backward(
                    loss,
                    distop_context=self.full_main_program_dist_context.dist_op_context,
                )

            with program_guard(
                self.full_main_program, self.full_startup_program
            ):
                with unique_name.guard("opt_"):
                    optimizer_ops = optimizer.apply_gradients(params_grads)

            # op original id to grad op id
            for idx, op in enumerate(self.full_main_program.global_block().ops):
                self.op_original_id_to_op[op.desc.original_id()] = op
                self.op_original_id_to_idx[op.desc.original_id()] = idx

            grad_op_id_to_op_id = (
                self.full_main_program_dist_context.dist_op_context.grad_op_id_to_op_id
            )

            for grad_op_original_id in grad_op_id_to_op_id:
                op_id = grad_op_id_to_op_id[grad_op_original_id]
                self.op_original_id_to_grad_op_original_id[
                    op_id
                ] = grad_op_original_id

    def cluster_operators(self):
        """Group operators to layers."""
        ops = self._dist_context._serial_main_program.global_block().ops

        # clear op dist attr when user shard tensor or op but in the full auto parallel mode.
        for op in ops:
            op.dist_attr = OperatorDistAttr(op.desc)

        vars = self._dist_context._serial_main_program.global_block().vars
        for var_name in vars:
            vars[var_name].dist_attr = TensorDistAttr(vars[var_name].desc)

        seq = [op.type for op in ops]

        while not OperatorClusteringUtil.stop_replace(seq):
            to_replace_seq = []
            to_replace_idxes = []
            has_append = False
            for idx, item in enumerate(seq):
                if not isinstance(item, list):
                    has_append = True
                    to_replace_seq.append(item)
                    to_replace_idxes.append(idx)
                elif isinstance(seq, list) and not has_append:
                    continue
                elif isinstance(seq, list) and has_append:
                    break

            ranks = OperatorClusteringUtil.get_ranks(to_replace_seq)
            suffixes = OperatorClusteringUtil.get_suffixes(ranks)
            heights = OperatorClusteringUtil.get_heights(
                suffixes, to_replace_seq
            )
            longest_sub_seq = (
                OperatorClusteringUtil.get_longest_repeated_sub_seq(
                    suffixes, heights, to_replace_seq
                )
            )
            has_merged = False
            if longest_sub_seq is None:
                for i in range(to_replace_idxes[-1] + 1, len(seq)):
                    if isinstance(seq[i], list):
                        seq[i] = to_replace_seq + seq[i]
                        has_merged = True
                        break
                if not has_merged:
                    for i in range(to_replace_idxes[0] - 1, -1, -1):
                        if isinstance(seq[i], list):
                            seq[i].extend(to_replace_seq)
                            has_merged = True
                            break
                if not has_merged:
                    seq = [to_replace_seq]
                    break

            decomposed_sub_seq = OperatorClusteringUtil.get_decomposed_sub_seq(
                longest_sub_seq
            )
            to_replace_seq = OperatorClusteringUtil.replace_by_decomposed_seq(
                decomposed_sub_seq, to_replace_seq
            )

            result = seq[: to_replace_idxes[0]]
            if not has_merged:
                result.extend(to_replace_seq)
            result.extend(seq[to_replace_idxes[-1] + 1 :])
            seq = result

        layers = []
        idx = 0
        for groups in seq:
            layer = []
            for op in groups:
                layer.append(ops[idx])
                idx += 1
            layers.append(layer)

        return layers

    def match_program(self, program):
        """Use patterns to match the program and get tensor shard spec when pattern matched."""
        graph = GraphUtil.convert_to_graph(program.global_block())
        results = GraphUtil.match_all_patterns(graph)
        if results:
            for pattern_name in results.keys():
                pattern = _PATTERNS[pattern_name]
                for parallelism in pattern.attrs["shard_spec"].keys():
                    shard_spec = pattern.attrs["shard_spec"][parallelism]
                    for pattern_node_id in shard_spec.keys():
                        for item in results[pattern_name]:
                            var_id = item[pattern_node_id]
                            var_desc_id = graph.attrs["id_to_var_desc_id"][
                                var_id
                            ]
                            if var_desc_id not in self.tensor_dist_attrs:
                                self.tensor_dist_attrs[var_desc_id] = {}
                            self.tensor_dist_attrs[var_desc_id][
                                parallelism
                            ] = shard_spec[pattern_node_id]
                            tensor_name = graph.attrs["id_to_var_name"][var_id]
                            self._logger.info(
                                "{}'s shard_spec may be {} when under {} parallelism.".format(
                                    tensor_name,
                                    shard_spec[pattern_node_id],
                                    parallelism,
                                )
                            )
        else:
            self._logger.info(
                "No pattern has be matched by this program. Currently, only the transformer-based models are supported. Data parallelism will be used."
            )
            self._use_dp = True

    def gen_fwd_sub_programs_by_clone(self):
        """Generate all forward sub programs by cloned from the original program."""
        for idx, layer in enumerate(self.layers):
            sub_fwd_program = self._gen_fwd_sub_program_by_clone(layer)
            self.fwd_sub_programs[idx] = sub_fwd_program

    def _gen_fwd_sub_program_by_clone(self, ops):
        """Generate the forward sub program of the given ops."""
        program = paddle.static.Program()
        block = ops[0].block
        vars = block.vars
        target_block = program.global_block()
        with paddle.static.program_guard(program):
            has_cloned_vars = set()
            for op in ops:
                new_op_desc = target_block.desc.append_op()
                new_op_desc.copy_from(op.desc)
                for var_name in op.input_arg_names:
                    if var_name not in has_cloned_vars:
                        if vars[var_name].is_parameter:
                            src_var = vars[var_name]
                            copied_kwargs = {}
                            copied_kwargs['trainable'] = src_var.trainable
                            copied_kwargs[
                                'optimize_attr'
                            ] = src_var.optimize_attr
                            copied_kwargs['regularizer'] = src_var.regularizer
                            copied_kwargs[
                                'do_model_average'
                            ] = src_var.do_model_average
                            copied_kwargs['need_clip'] = src_var.need_clip

                            param = Parameter(
                                block=target_block,
                                type=src_var.type,
                                name=src_var.name,
                                shape=src_var.shape,
                                dtype=src_var.dtype,
                                lod_level=src_var.lod_level,
                                error_clip=src_var.error_clip,
                                stop_gradient=src_var.stop_gradient,
                                is_data=src_var.is_data,
                                belong_to_optimizer=src_var.belong_to_optimizer,
                                **copied_kwargs
                            )
                        else:
                            target_block._clone_variable(vars[var_name])
                            target_block.vars[var_name].persistable = vars[
                                var_name
                            ].persistable
                        target_block.vars[var_name].desc.set_original_id(
                            vars[var_name].desc.original_id()
                        )
                        has_cloned_vars.add(var_name)

                for var_name in op.output_arg_names:
                    if var_name not in has_cloned_vars:
                        target_block._clone_variable(vars[var_name])
                        target_block.vars[var_name].persistable = vars[
                            var_name
                        ].persistable
                        target_block.vars[var_name].desc.set_original_id(
                            vars[var_name].desc.original_id()
                        )
                        has_cloned_vars.add(var_name)

        target_block._sync_with_cpp()

        return program

    def _compelte_sub_fwd_program(self, idx, sub_fwd_program, process_mesh):
        """Compelete forward sub  program."""
        selective_parallelisms = (
            ["dp", "mp"] if len(process_mesh.shape) == 1 else ["dp_mp", "mp_dp"]
        )
        for parallelism in selective_parallelisms:
            has_set_tensor_count = 0
            dist_context = DistributedContext(sub_fwd_program)
            has_set_dist_attr_tensors = set()
            dist_context.process_meshes = []
            dist_context.add_process_mesh(process_mesh)
            vars = sub_fwd_program.global_block().vars

            # clear op dist attr
            ops = sub_fwd_program.global_block().ops
            for op in ops:
                op.dist_attr = OperatorDistAttr(op.desc)
            # clear tensor dist attr
            for var_name in vars:
                vars[var_name].dist_attr = TensorDistAttr(vars[var_name].desc)

            for var_name in vars:
                var_id = vars[var_name].desc.original_id()
                if var_id in self.tensor_dist_attrs:
                    if parallelism in self.tensor_dist_attrs[var_id]:
                        dims_mapping = self.tensor_dist_attrs[var_id][
                            parallelism
                        ]
                        dist_tensor = DistributedTensor(vars[var_name])
                        dist_tensor.dist_attr.process_mesh = process_mesh
                        dist_tensor.dist_attr.dims_mapping = dims_mapping
                        dist_tensor.dist_attr.mark_annotated("dims_mapping")
                        dist_tensor.dist_attr.mark_annotated("process_mesh")
                        dist_context.add_dist_tensor_for_program(dist_tensor)
                        has_set_tensor_count += 1
                        has_set_dist_attr_tensors.add(var_id)

            # check whether no dist attr in dist context
            if has_set_tensor_count > 0:
                dist_context.initialize(no_default=True)
                completer = Completer(dist_context)
                completer.complete_forward_annotation()
                if parallelism not in self.sub_programs_dist_context[idx]:
                    self.sub_programs_dist_context[idx][parallelism] = {}
                key = self.convert_process_mesh_to_key(process_mesh)
                self.sub_programs_dist_context[idx][parallelism][
                    key
                ] = dist_context
            else:
                self._logger.info(
                    "No pattern has be matched under {} parallelism whe sub program is {}.".format(
                        parallelism, sub_fwd_program
                    )
                )

    def complete_sub_fwd_programs(self, process_mesh):
        """Complete all forward sub programs."""
        for idx in self.fwd_sub_programs.keys():
            sub_fwd_program = self.fwd_sub_programs[idx]
            if idx not in self.sub_programs_dist_context:
                self.sub_programs_dist_context[idx] = {}
            self._compelte_sub_fwd_program(idx, sub_fwd_program, process_mesh)

    def _complete_sub_bwd_program(self, sub_program_dist_context):
        """
        Complete the backward OP according to the forward OP.
        Most of the logic is the same as the backward completion in the completer.
        The difference is that find the backward OP according to the forward OP,
        while find the forward OP according to the backward OP in the completer.
        """

        def _is_grad_var_name(name):
            if "@GRAD" in name:
                return True
            return False

        sub_fwd_program = sub_program_dist_context.serial_main_program
        block = sub_fwd_program.global_block()
        vars = self.full_main_program.global_block().vars
        ops = self.full_main_program.global_block().ops
        grad_var_to_var = (
            self.full_main_program_dist_context.dist_op_context.grad_var_to_var[
                1
            ]
        )
        for forward_op in block.ops:
            if (
                forward_op.desc.original_id()
                not in self.op_original_id_to_grad_op_original_id
            ):
                continue
            grad_op_id = self.op_original_id_to_grad_op_original_id[
                forward_op.desc.original_id()
            ]
            # for unsqueeze2 op in gpt, it has no grad op
            # or for no need to bwd
            if grad_op_id not in self.op_original_id_to_op:
                continue
            grad_op = self.op_original_id_to_op[grad_op_id]
            if grad_op.type == "concat" and forward_op.type == "split":
                forward_op_dist_attr = (
                    sub_program_dist_context.get_op_dist_attr_for_program(
                        forward_op
                    )
                )
                output_var = vars[grad_op.desc.output('Out')[0]]
                split_input_var_name = forward_op.input("X")[0]
                ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                    split_input_var_name
                )
                ref_mesh = forward_op_dist_attr.process_mesh

                grad_op_dist_attr = OperatorDistAttr()
                for input_name in grad_op.input_arg_names:
                    grad_op_dist_attr.set_input_dims_mapping(
                        input_name, ref_dims_mapping
                    )

                output_var_dist_attr = TensorDistAttr()
                output_var_dist_attr.dims_mapping = ref_dims_mapping
                output_var_dist_attr.process_mesh = ref_mesh
                sub_program_dist_context.set_tensor_dist_attr_for_program(
                    output_var, output_var_dist_attr
                )

                grad_op_dist_attr.set_output_dims_mapping(
                    output_var.name, ref_dims_mapping
                )
                grad_op_dist_attr.process_mesh = ref_mesh
                sub_program_dist_context.set_op_dist_attr_for_program(
                    grad_op, grad_op_dist_attr
                )
                grad_op_dist_attr.impl_type = (
                    fwd_op_dist_attr.impl_type  # noqa: F821
                )
                grad_op_dist_attr.impl_idx = (
                    fwd_op_dist_attr.impl_idx  # noqa: F821
                )
                continue

            fwd_op_dist_attr = (
                sub_program_dist_context.get_op_dist_attr_for_program(
                    forward_op
                )
            )
            fwd_op_process_mesh = fwd_op_dist_attr.process_mesh
            grad_op_dist_attr = OperatorDistAttr()
            grad_op_dist_attr.process_mesh = fwd_op_process_mesh

            for input_name in grad_op.input_arg_names:
                if (
                    input_name not in forward_op.input_arg_names
                    and input_name not in forward_op.output_arg_names
                ):
                    if input_name in grad_var_to_var.keys():
                        fwd_name = grad_var_to_var[input_name]
                        ref_dims_mapping = (
                            fwd_op_dist_attr.get_output_dims_mapping(fwd_name)
                        )
                    else:
                        input_var = vars[input_name]
                        ref_dims_mapping = sub_program_dist_context.get_tensor_dist_attr_for_program(
                            input_var
                        ).dims_mapping
                else:
                    if input_name in forward_op.input_arg_names:
                        ref_dims_mapping = (
                            fwd_op_dist_attr.get_input_dims_mapping(input_name)
                        )
                    else:
                        ref_dims_mapping = (
                            fwd_op_dist_attr.get_output_dims_mapping(input_name)
                        )
                assert (
                    ref_dims_mapping is not None
                ), "[{}] 's dims mapping is NONE".format(input_name)
                grad_op_dist_attr.set_input_dims_mapping(
                    input_name, ref_dims_mapping
                )

            for output_name in grad_op.output_arg_names:
                assert output_name in grad_var_to_var
                fwd_name = grad_var_to_var[output_name]
                ref_dims_mapping = fwd_op_dist_attr.get_input_dims_mapping(
                    fwd_name
                )
                # var
                output_var = vars[output_name]
                tensor_dist_attr = TensorDistAttr()
                tensor_dist_attr.dims_mapping = ref_dims_mapping
                tensor_dist_attr.process_mesh = fwd_op_process_mesh
                sub_program_dist_context.set_tensor_dist_attr_for_program(
                    output_var, tensor_dist_attr
                )
                # op
                grad_op_dist_attr.set_output_dims_mapping(
                    output_name, ref_dims_mapping
                )

            grad_op_dist_attr.impl_type = fwd_op_dist_attr.impl_type
            grad_op_dist_attr.impl_idx = fwd_op_dist_attr.impl_idx
            sub_program_dist_context.set_op_dist_attr_for_program(
                grad_op, grad_op_dist_attr
            )

            grad_op_idx = self.op_original_id_to_idx[grad_op_id]
            if grad_op_idx + 1 < len(ops):
                grad_op_next_op = ops[grad_op_idx + 1]
                if grad_op_next_op.type == "sum":
                    assert all(
                        map(_is_grad_var_name, grad_op_next_op.input_arg_names)
                    )
                    output_name = grad_op_next_op.output_arg_names[0]
                    assert (
                        output_name in grad_var_to_var
                    ), "sum op's output '{}' has no corresponding var".format(
                        output_name
                    )
                    ref_fwd_var_name = grad_var_to_var[output_name]
                    ref_fwd_var = vars[ref_fwd_var_name]
                    ref_fwd_dist_attr = sub_program_dist_context.get_tensor_dist_attr_for_program(
                        ref_fwd_var
                    )
                    ref_fwd_dims_mapping = ref_fwd_dist_attr.dims_mapping
                    ref_fwd_process_mesh = ref_fwd_dist_attr.process_mesh

                    # output
                    tensor_dist_attr = TensorDistAttr()
                    tensor_dist_attr.dims_mapping = ref_fwd_dims_mapping
                    tensor_dist_attr.process_mesh = ref_fwd_process_mesh
                    output_var = vars[output_name]
                    sub_program_dist_context.set_tensor_dist_attr_for_program(
                        output_var, tensor_dist_attr
                    )

                    # op
                    grad_op_dist_attr = OperatorDistAttr()
                    grad_op_dist_attr.process_mesh = ref_fwd_process_mesh

                    for var_name in grad_op_next_op.input_arg_names:
                        grad_op_dist_attr.set_input_dims_mapping(
                            var_name, ref_fwd_dims_mapping
                        )
                    grad_op_dist_attr.set_output_dims_mapping(
                        output_name, ref_fwd_dims_mapping
                    )
                    grad_op_dist_attr.impl_type = "default"
                    grad_op_dist_attr.impl_idx = 0

                    sub_program_dist_context.set_op_dist_attr_for_program(
                        grad_op_next_op, grad_op_dist_attr
                    )

    def complete_sub_bwd_programs(self):
        for idx in self.sub_programs_dist_context:
            for parallelism in self.sub_programs_dist_context[idx]:
                for key in self.sub_programs_dist_context[idx][parallelism]:
                    sub_program_dist_context = self.sub_programs_dist_context[
                        idx
                    ][parallelism][key]
                    self._complete_sub_bwd_program(sub_program_dist_context)
