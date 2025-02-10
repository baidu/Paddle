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
# limitations under the License.

import unittest

from paddle.distributed.auto_parallel.static.graph import Graph


class TestAutoParallelGraph(unittest.TestCase):
    def test_graph(self):
        graph = Graph(name="foo")
        self.assertEqual(graph.attrs["name"], "foo")

        graph.add_node(1, weight=0)
        # Override the existing node attribute
        graph.add_node(1, weight=1)
        graph.add_node(2, weight=2)
        graph.add_node(3, weight=3)

        node = graph.nodes[1]
        node["info"] = "is a node"
        self.assertTrue(node.id, 1)
        self.assertTrue("weight" in node)
        self.assertTrue("info" in node)
        for node_attr in node.attrs:
            self.assertTrue(node_attr in ["weight", "info"])

        self.assertTrue(1 in graph)
        self.assertTrue(2 in graph)
        self.assertTrue(3 in graph)
        self.assertEqual(len(graph), 3)
        self.assertEqual(graph.nodes[1].id, 1)
        self.assertEqual(graph.nodes[2].id, 2)
        self.assertEqual(graph.nodes[3].id, 3)
        for node in graph:
            if node.id == 1:
                self.assertEqual(node["weight"], 1)
            if node.id == 2:
                self.assertEqual(node["weight"], 2)
            if node.id == 3:
                self.assertEqual(node["weight"], 3)

        graph.add_edge(1, 2, weight=0.1)
        graph.add_edge(1, 3, weight=0.2)
        graph.add_edge(2, 3, weight=0.3)
        graph.add_edge(4, 5, weight=0.4)

        edge = graph[1][2]
        edge["info"] = "is a edge"
        self.assertTrue(edge.src_id, 1)
        self.assertTrue(edge.tgt_id, 2)
        self.assertTrue("weight" in edge)
        self.assertTrue("info" in edge)
        for edge_attr in edge.attrs:
            self.assertTrue(edge_attr in ["weight", "info"])

        self.assertEqual(graph[1][2]["weight"], 0.1)
        self.assertEqual(graph[1][3]["weight"], 0.2)
        self.assertEqual(graph[2][3]["weight"], 0.3)

        self.assertEqual(graph[4][5]["weight"], 0.4)

        str = f"{graph}"
        self.assertIsNotNone(str)

        self.assertRaises(TypeError, 6 in graph)
        self.assertRaises(TypeError, "unknown_attr" in graph.nodes[1])
        self.assertRaises(TypeError, "unknown_attr" in graph[1][2])
        self.assertRaises(ValueError, graph.add_node, None)
        self.assertRaises(ValueError, graph.add_edge, 3, None)
        self.assertRaises(ValueError, graph.add_edge, None, 3)


if __name__ == '__main__':
    unittest.main()
