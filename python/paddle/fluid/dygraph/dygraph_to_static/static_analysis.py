#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import ast
import six
import warnings

__all__ = ['AstNodeWrapper', 'NodeVarType', 'StaticAnalysisVisitor']


class NodeVarType(object):
    """
    Enum class of python variable types. We have to know some variable types
    during compile time to transfer AST. For example, a string variable and a
    tensor variable in if clause may lead to different conversion from dygraph
    to static graph.
    """
    UNKNOWN = 0  # Reserve for AST nodes have not known the type
    STATEMENT = 1  # For nodes representing statement (non-variable type)
    CALLABLE = 2

    # python data types
    NONE = 100
    BOOLEAN = 101
    INT = 102
    FLOAT = 103
    STRING = 104
    TENSOR = 105

    # python collections
    LIST = 200
    SET = 201
    DICT = 202

    PADDLE_DYGRAPH_API = 300
    PADDLE_CONTROL_IF = 301
    PADDLE_CONTROL_WHILE = 302
    PADDLE_CONTROL_FOR = 303

    @staticmethod
    def binary_op_output_type(in_type1, in_type2):
        if in_type1 == in_type2:
            return in_type1

        if in_type1 == NodeVarType.UNKNOWN:
            return in_type2
        if in_type2 == NodeVarType.UNKNOWN:
            return in_type1

        supported_types = [
            NodeVarType.BOOLEAN, NodeVarType.INT, NodeVarType.FLOAT
        ]
        if in_type1 not in supported_types:
            warnings.warn("Binary Op on un supported in_type1 = %d " %
                          (in_type1))
            return NodeVarType.UNKNOWN
        if in_type2 not in supported_types:
            warnings.warn("Binary Op on un supported in_type2 = %d " %
                          (in_type2))
            return NodeVarType.UNKNOWN
        return max(in_type1, in_type2)


class AstNodeWrapper(object):
    """
    Wrapper for python ast.node. We need a node wrapper because ast.node
    doesn't store all required information when we are transforming AST.
    We should collect additional information which the actual transformation
    needs.
    """

    def __init__(self, node):
        self.node = node
        self.parent = None
        self.children = []
        self.node_var_type = NodeVarType.UNKNOWN


class AstVarScope(object):
    """
    AstVarScope is a class holding the map from current scope variable to its
    type. 
    """

    def __init__(self, parent_scope=None):
        self.sub_scopes = []
        self.name_to_id = {}
        self.id_to_type = {}
        self.cur_id = 0
        self.parent_scope = parent_scope
        if parent_scope is not None:
            parent_scope.sub_scopes.append(self)

    def set_var_type(self, var_name, node_var_type):
        if var_name in self.name_to_id:
            num_id = self.name_to_id[var_name]
        else:
            num_id = self.cur_id
            self.cur_id += 1
            self.name_to_id[var_name] = num_id
        self.id_to_type[num_id] = node_var_type

    def get_var_type(self, var_name):
        if var_name in self.name_to_id:
            num_id = self.name_to_id[var_name]
            return self.id_to_type[num_id]
        if self.parent_scope is None:
            return NodeVarType.UNKNOWN
        return self.parent_scope.get_var_type(var_name)


class AstVarEnv(object):
    """
    A class maintains scopes and mapping from variable name to type.
    """

    def __init__(self):
        self.cur_scope = AstVarScope()

    def enter_scope(self):
        self.cur_scope = AstVarScope(parent_scope=self.cur_scope)
        return self.cur_scope

    def exit_scope(self):
        assert self.cur_scope.parent_scope is not None, "Call exit_scope in "\
            "AstVarEnv when current scope doens't have parent scope."
        self.cur_scope = self.cur_scope.parent_scope
        return self.cur_scope

    def set_var_type(self, var_name, node_var_type):
        self.cur_scope.set_var_type(var_name, node_var_type)

    def get_var_type(self, var_name):
        return self.cur_scope.get_var_type(var_name)

    def get_scope_var_type(self):
        '''
        Returns a dict mapping from variable name to type. Used for debug and
        test.
        '''
        cur_scope_dict = {}
        for name in self.cur_scope.name_to_id:
            node_var_type = self.cur_scope.get_var_type(name)
            cur_scope_dict[name] = node_var_type
        return cur_scope_dict


class StaticAnalysisVisitor(object):
    """
    A class that does static analysis
    """

    def __init__(self, ast_root=None):
        if ast_root is not None:
            self.run(ast_root)

    def run(self, ast_root):
        self.node_wrapper_root = None
        self.ancestor_wrappers = []
        self.node_to_wrapper_map = {}
        self.var_env = AstVarEnv()

        self.dfs_visit(ast_root)

    def dfs_visit(self, node):
        # AST reuses some ast.nodes, such as Param node of expr_context
        if node not in self.node_to_wrapper_map:
            cur_wrapper = AstNodeWrapper(node)
            self.node_to_wrapper_map[node] = cur_wrapper
        else:
            cur_wrapper = self.node_to_wrapper_map[node]

        if self.node_wrapper_root is None:
            self.node_wrapper_root = cur_wrapper

        if len(self.ancestor_wrappers) != 0:
            last_wrapper = self.ancestor_wrappers[-1]
            last_wrapper.children.append(cur_wrapper)
            cur_wrapper.parent = last_wrapper

        self.ancestor_wrappers.append(cur_wrapper)
        for child in ast.iter_child_nodes(node):
            self.dfs_visit(child)
        self.ancestor_wrappers.pop()

        cur_wrapper.node_var_type = self._get_node_var_type(cur_wrapper)
        return cur_wrapper.node_var_type

    def get_node_wrapper_root(self):
        return self.node_wrapper_root

    def get_node_to_wrapper_map(self):
        return self.node_to_wrapper_map

    def get_var_env(self):
        return self.var_env

    def _get_node_var_type(self, cur_wrapper):
        node = cur_wrapper.node
        if isinstance(node, ast.Num):
            if node.n is None:
                return NodeVarType.NONE
            if isinstance(node.n, int):
                return NodeVarType.INT
            if isinstance(node.n, float):
                return NodeVarType.FLOAT
        if isinstance(node, ast.Str):
            return NodeVarType.STRING

        if six.PY3:
            # NameConstant are Py3 grammar
            if isinstance(node, ast.NameConstant):
                # singleton: None, True or False
                if node.value is None:
                    return NodeVarType.NONE
                else:
                    return NodeVarType.BOOLEAN

        if isinstance(node, ast.BoolOp):
            return NodeVarType.BOOLEAN
        if isinstance(node, ast.Compare):
            return NodeVarType.BOOLEAN

        if isinstance(node, ast.Dict):
            return NodeVarType.DICT
        if isinstance(node, ast.Set):
            return NodeVarType.SET

        if isinstance(node, ast.UnaryOp):
            return self.node_to_wrapper_map[node.operand].node_var_type

        if isinstance(node, ast.BinOp):
            left_type = self.node_to_wrapper_map[node.left].node_var_type
            right_type = self.node_to_wrapper_map[node.right].node_var_type
            return NodeVarType.binary_op_output_type(left_type, right_type)

        if isinstance(node, ast.Assign):
            ret_type = self.node_to_wrapper_map[node.value].node_var_type
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.node_to_wrapper_map[target].node_var_type = ret_type
                    self.var_env.set_var_type(target.id, ret_type)
            return ret_type

        if isinstance(node, ast.Name):
            return self.var_env.get_var_type(node.id)

        return NodeVarType.STATEMENT
