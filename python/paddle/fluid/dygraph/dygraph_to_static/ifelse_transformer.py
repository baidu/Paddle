#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from collections import defaultdict

# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/
import gast
import six
from paddle.fluid import unique_name

from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import create_funcDef_node
from paddle.fluid.dygraph.dygraph_to_static.utils import create_assign_node
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType

TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'
LOGIC_AND_PREFIX = 'logic_and'
LOGIC_OR_PREFIX = 'logic_or'
PLAIN_TENSOR_PREFIX = 'bool_tensor'


class IfElseTransformer(gast.NodeTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
        self.root = wrapper_root.node
        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.new_func_nodes = {}

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)
        self.after_visit(self.root)

    def visit_If(self, node):
        assert isinstance(node, gast.If)
        if_condition_visitor = IfConditionVisitor(node.test,
                                                  self.static_analysis_visitor)
        need_transform = if_condition_visitor.is_control_flow()
        self.generic_visit(node)
        if need_transform:
            pred_node, new_assign_nodes = if_condition_visitor.transform()
            true_func_node, false_func_node, return_name_ids = transform_if_else(
                node, self.root)
            # create layers.cond
            new_node = create_cond_node(return_name_ids, pred_node,
                                        true_func_node, false_func_node)
            self.new_func_nodes[new_node] = [true_func_node, false_func_node
                                             ] + new_assign_nodes
            return new_node
        else:
            return node

    def visit_Call(self, node):
        # Remove `numpy()` statement, like `Tensor.numpy()[i]` -> `Tensor[i]`
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        return node

    def after_visit(self, node):
        """
        This function will add some postprocessing operations with node.
        It can be used to add the created `true_fn/false_fn` in front of
        the node.body before they are called in cond layer.
        """
        self._insert_func_nodes(node)

    def _insert_func_nodes(self, node):
        """
        Defined `true_func` and `false_func` will be inserted in front of corresponding
        `layers.cond` statement instead of inserting them all into body of parent node.
        Because private variables of class or other external scope will be modified.
        For example, `self.var_dict["key"]`. In this case, nested structure of newly
        defined functions is easier to understand.
        """
        if not self.new_func_nodes:
            return
        idx = -1
        if isinstance(node, list):
            idx = len(node) - 1
        elif isinstance(node, gast.AST):
            for _, child in gast.iter_fields(node):
                self._insert_func_nodes(child)
        while idx >= 0:
            child_node = node[idx]
            if child_node in self.new_func_nodes:
                node[idx:idx] = self.new_func_nodes[child_node]
                idx = idx + len(self.new_func_nodes[child_node]) - 1
                del self.new_func_nodes[child_node]
            else:
                self._insert_func_nodes(child_node)
                idx = idx - 1

    def get_new_func_nodes(self):
        return self.new_func_nodes


def is_candidate_node(node):
    """
    Nodes with specified type will be dependent on tensor.
    """
    return isinstance(node, (gast.Compare, gast.BoolOp))


def compare_with_none(node):
    """
    Whether the comparator of `gast.Compare` node is `None`.
    """
    if isinstance(node, gast.Compare):
        for child in [node.left, node.comparators]:
            # node.comparators is a list.
            if isinstance(child, list):
                child = child[0]
            if (isinstance(child, gast.Constant) and child.value is None) or (
                    isinstance(child, gast.Name) and child.id == 'None'):
                return True
    return False


class IsControlFlowVisitor(gast.NodeVisitor):
    """
    Judge whether the node.test from Dygraph code dependent on paddle Tensor.
    If does, it should satisfy:
        1. must involve at least one var whose type is Tensor.
        2. the Tensor var should call `.numpy()[]` interface or Tensor.shape is [1].
        3. involve Tensor.shape[i] and the shape[i] is unknown in compile time.
    The following examples should not be considered as control_flow_if:
        1. `if Tensor_var` or `if Tensor_var is None`
        2. if Tensor.shape[i] is determined with fixed value (not -1 or None)

    Note: pred in ConditionalBlock require variable, which means all vars should be Tensor
          or transformed into Tensor, like fill_constant(shape=[1], dtype='int32', value=Tensor.shape[i]).

    TODO: 1. need to deal with `tensor.shape[i]` which need to eval the data of shape[i],
             because reshape_op may be called before this statement.
    """

    def __init__(self,
                 ast_node,
                 static_analysis_visitor=None,
                 node_var_type_map=None):
        assert isinstance(
            ast_node, gast.AST
        ), "Type of input node should be gast.AST, but received %s." % type(
            ast_node)
        self.ast_root = ast_node
        if static_analysis_visitor is None:
            static_analysis_visitor = StaticAnalysisVisitor(ast_node)
        self.static_analysis_visitor = static_analysis_visitor
        self.node_var_type_map = node_var_type_map

        self.is_control_flow_num = 0
        self._compare_node_tenor_set = set()

    def transform(self):
        node = self.ast_root
        if is_candidate_node(node):
            self.visit(node)
        return self.is_control_flow_num > 0

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            if is_candidate_node(child):
                self.visit(child)
        return node

    def visit_Compare(self, node):
        # Ignores child node with `if x` or `if x is None`
        # TODO(Aurelius84): `if tensor` will be supported in dygraph
        # and should be considered as is_control_flow.
        pre_control_flow_num = self.is_control_flow_num
        if not compare_with_none(node):
            self.generic_visit(node)
            for child in gast.walk(node):
                if isinstance(child, gast.Subscript):
                    self._visit_Subscript(child)
        if self.is_control_flow_num > pre_control_flow_num:
            self._compare_node_tenor_set.add(node)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if hasattr(node, 'value') and isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            if attr_node.attr == 'numpy':
                self.is_control_flow_num += 1

    def visit_Call(self, node):
        if is_paddle_api(node):
            self.is_control_flow_num += 1
        return node

    def visit_Name(self, node):
        if self._is_node_with_tensor(node, node.id):
            self.is_control_flow_num += 1
        return node

    def visit_Constant(self, node):
        if self._is_node_with_tensor(node, node.value):
            self.is_control_flow_num += 1
        return node

    def _is_node_with_tensor(self, node, name_id):
        tensor_types = set(
            [NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES])
        # Look up the node_var_type_map by name_id.
        if self.node_var_type_map:
            if name_id and isinstance(name_id, six.string_types):
                var_type = self.node_var_type_map.get(name_id, None)
                if var_type and var_type & tensor_types:
                    return True
        # if not found, look up the node_to_wrapper_map by node.
        node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        wrapper_node = node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & tensor_types:
                return True

        return False

    def get_compare_nodes_with_tensor(self):
        return self._compare_node_tenor_set


class NodeTestTransformer(gast.NodeTransformer):
    def __init__(self, ast_node, compare_nodes_with_tensor=set()):
        self.ast_root = ast_node
        self._compare_nodes_with_tensor = compare_nodes_with_tensor
        self._new_assign_nodes = []

    def transform(self):
        return self.visit(self.ast_root)

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            if not is_candidate_node(child):
                node.values[i] = self._create_bool_node(child)
                continue
        self.generic_visit(node)
        new_node = self._create_logic_node(node)
        return new_node

    def visit_Compare(self, node):
        if compare_with_none(
                node) or node not in self._compare_nodes_with_tensor:
            return self._create_bool_node(node)
        return node

    def _create_bool_node(self, node):
        node_code = ast_to_source_code(node)
        new_node_str = "fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool({}))".format(
            node_code)
        # gast.parse return Module(body=[expr(value=...)])
        new_node = gast.parse(new_node_str).body[0].value
        bool_tensor_name = unique_name.generate(PLAIN_TENSOR_PREFIX)
        assign_name, assign_node = create_assign_node(bool_tensor_name,
                                                      new_node)

        self._new_assign_nodes.append(assign_node)

        return assign_name

    def _create_logic_node(self, node):
        def _create_node(nodes, api_type):
            assert len(
                nodes
            ) > 1, "The length of BoolOp should be at least 2, but received {}.".format(
                len(nodes))
            if len(nodes) > 2:
                # Creates logic_and/logic_or node recursively.
                pre_assign_node = _create_node(nodes[:2], api_type)
                nodes = [pre_assign_node] + nodes[2:]
            args = [ast_to_source_code(child) for child in nodes]
            new_node_str = "fluid.layers.logical_{}(x={}, y={})".format(
                api_type, args[0], args[1])
            # gast.parse return Module(body=[expr(value=...)])
            new_node = gast.parse(new_node_str).body[0].value
            logic_tensor_name = unique_name.generate(
                LOGIC_AND_PREFIX if 'and' in api_type else LOGIC_OR_PREFIX)
            assign_name, assign_node = create_assign_node(logic_tensor_name,
                                                          new_node)
            self._new_assign_nodes.append(assign_node)

            return assign_name

        if isinstance(node.op, gast.And):
            node = _create_node(node.values, 'and')
        elif isinstance(node.op, gast.Or):
            node = _create_node(node.values, 'or')
        else:
            raise TypeError(
                "Only supports and/or syntax in control flow if statement.")
        return node

    def get_new_assign_nodes(self):
        return self._new_assign_nodes

    def set_compare_nodes_with_tensor(self, nodes_set):
        self._compare_nodes_with_tensor = set(nodes_set)
        return self._compare_nodes_with_tensor


class IfConditionVisitor(object):
    def __init__(self,
                 node,
                 static_analysis_visitor=None,
                 node_var_type_map=None):
        self.node = node
        self.static_analysis_visitor = static_analysis_visitor
        self.visitor = IsControlFlowVisitor(node, static_analysis_visitor,
                                            node_var_type_map)
        self.transformer = NodeTestTransformer(node)
        self.compare_nodes_with_tensor = set()
        self._is_control_flow_if = False

    def is_control_flow(self):
        """
        Determine whether the node is a plain python `if statement` or
        control flow in Paddle.
        """
        self._is_control_flow_if = self.visitor.transform()
        return self._is_control_flow_if

    def transform(self):
        if not self._is_control_flow_if:
            return self.node, []
        else:
            self.compare_nodes_with_tensor = self.visitor.get_compare_nodes_with_tensor(
            )
            self.transformer.set_compare_nodes_with_tensor(
                self.compare_nodes_with_tensor)
            new_node = self.transformer.transform()
            new_assign_nodes = self.transformer.get_new_assign_nodes()
            return new_node, new_assign_nodes


class NameVisitor(gast.NodeVisitor):
    def __init__(self, node_black_set=None):
        # Set of nodes that will not be visited.
        self.node_black_set = node_black_set or set()
        # Dict to store the names and ctxs of vars.
        self.name_ids = defaultdict(list)
        # List of current visited nodes
        self.ancestor_nodes = []
        # Available only when node_black_set is set.
        self._is_finished = False
        self._candidate_ctxs = (gast.Store, gast.Load, gast.Param)

    def visit(self, node):
        """Visit a node."""
        if node in self.node_black_set or self._is_finished:
            self._is_finished = True
            return

        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()

        return ret

    def visit_If(self, node):
        """
        For nested `if/else`, the created vars are not always visible for parent node.
        In addition, the vars created in `if.body` are not visible for `if.orelse`.

        Case 1:
            x = 1
            if m > 1:
                res = new_tensor
            res = res + 1   # Error, `res` is not visible here.

        Case 2:
            if x_tensor > 0:
                res = new_tensor
            else:
                res = res + 1   # Error, `res` is not visible here.

        In above two cases, we should consider to manage the scope of vars to parsing
        the arguments and returned vars correctly.
        """
        before_if_name_ids = copy.deepcopy(self.name_ids)
        body_name_ids = self._visit_child(node.body)
        # If the traversal process stops early, just return the name_ids that have been seen.
        if self._is_finished:
            for name_id, ctxs in before_if_name_ids.items():
                self.name_ids[name_id] = ctxs + self.name_ids[name_id]
        # Blocks the vars in `if.body` and only inserts the vars both created in 'if/else' branch
        # into name_ids.
        else:
            else_name_ids = self._visit_child(node.orelse)
            new_name_ids = self._find_new_name_ids(body_name_ids, else_name_ids)
            for new_name_id in new_name_ids:
                before_if_name_ids[new_name_id].append(gast.Store())

            self.name_ids = before_if_name_ids

    def visit_Attribute(self, node):
        if not self._is_call_func_name_node(node):
            self.generic_visit(node)

    def visit_Name(self, node):
        if not self._is_call_func_name_node(node):
            if isinstance(node.ctx, self._candidate_ctxs):
                self.name_ids[node.id].append(node.ctx)

    def visit_Assign(self, node):
        # Visit `value` firstly.
        node._fields = ('value', 'targets')
        self.generic_visit(node)

    def visit_Return(self, node):
        # Ignore the vars in return
        return

    def _visit_child(self, node):
        self.name_ids = defaultdict(list)
        if isinstance(node, list):
            for item in node:
                if isinstance(item, gast.AST):
                    self.visit(item)
        elif isinstance(node, gast.AST):
            self.visit(node)

        return copy.deepcopy(self.name_ids)

    def _find_new_name_ids(self, body_name_ids, else_name_ids):
        def is_required_ctx(ctxs, required_ctx):
            for ctx in ctxs:
                if isinstance(ctx, required_ctx):
                    return True
            return False

        candidate_name_ids = set(body_name_ids.keys()) & set(else_name_ids.keys(
        ))
        store_ctx = gast.Store
        new_name_ids = set()
        for name_id in candidate_name_ids:
            if is_required_ctx(body_name_ids[name_id],
                               store_ctx) and is_required_ctx(
                                   else_name_ids[name_id], store_ctx):
                new_name_ids.add(name_id)

        return new_name_ids

    def _is_call_func_name_node(self, node):
        if len(self.ancestor_nodes) > 1:
            assert self.ancestor_nodes[-1] == node
            parent_node = self.ancestor_nodes[-2]
            if isinstance(parent_node, gast.Call) and parent_node.func == node:
                return True
        return False


def get_name_ids(nodes, node_black_set=None):
    """
    Return all ast.Name.id of python variable in nodes.
    """
    name_visitor = NameVisitor(node_black_set)
    for node in nodes:
        name_visitor.visit(node)
    return name_visitor.name_ids


def parse_cond_args(var_ids_dict, return_ids=None, ctx=gast.Load):
    """
    Find out the ast.Name.id list of input by analyzing node's AST information.
    """

    name_ids = [
        var_id for var_id, var_ctx in var_ids_dict.items()
        if isinstance(var_ctx[0], ctx)
    ]
    if return_ids:
        new_args = set(return_ids) - set(name_ids)
        name_ids.extend(list(new_args))
    name_ids.sort()
    args = [
        gast.Name(
            id=name_id, ctx=gast.Load(), annotation=None, type_comment=None)
        for name_id in name_ids
    ]
    arguments = gast.arguments(
        args=args,
        posonlyargs=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=None,
        kwarg=None,
        defaults=[])
    return arguments


def parse_cond_return(parent_vars_dict, if_vars_dict, else_vars_dict):
    """
    Find out the ast.Name list of output by analyzing node's AST information.
    Following conditions should be satisfied while determining whether a variable is a return value:
    1. the var in parent scope is modified in if/else node.
    2. new var is both created in if and else node.

    If different var is modified in if and else node, it should add the var in return_ids
    of different node.
    For example:
            x, y = 5, 10
            if x > 4:
                x = x+1
                z = x*x
            else:
                y = y - 1
                z = y*y

    The return_ids should be (x, y, z) for `if` and `else`node.
    """

    def _is_return_var(ctxs):
        for ctx in ctxs:
            if isinstance(ctx, (gast.Store, gast.Param)):
                return True
        return False

    def _vars_with_store(ids_dict):
        vars = []
        for k, ctxs in ids_dict.items():
            if _is_return_var(ctxs):
                vars.append(k)
        return vars

    def _candidate_vars(child_dict, parent_dict):
        return set([
            var for var in _vars_with_store(child_dict) if var in parent_dict
        ])

    # 1. the var in parent_ids is modified in if/else node.
    if_candidate_vars = _candidate_vars(if_vars_dict, parent_vars_dict)
    else_candidate_vars = _candidate_vars(else_vars_dict, parent_vars_dict)

    # 2. new var is both created in if and else node.
    if_new_vars = set([
        var for var in _vars_with_store(if_vars_dict)
        if var not in parent_vars_dict
    ])
    else_new_vars = set([
        var for var in _vars_with_store(else_vars_dict)
        if var not in parent_vars_dict
    ])
    new_vars = if_new_vars & else_new_vars

    # generate return_ids of if/else node.
    modified_vars = if_candidate_vars | else_candidate_vars
    return_ids = list(modified_vars | new_vars)
    return_ids.sort()

    return return_ids, list(modified_vars - new_vars)


def transform_if_else(node, root):
    """
    Transform ast.If into control flow statement of Paddle static graph.
    """
    parent_name_ids = get_name_ids([root], node_black_set=[node])
    if_name_ids = get_name_ids(node.body)
    else_name_ids = get_name_ids(node.orelse)

    return_name_ids, modified_name_ids = parse_cond_return(
        parent_name_ids, if_name_ids, else_name_ids)

    true_func_node = create_funcDef_node(
        node.body,
        name=unique_name.generate(TRUE_FUNC_PREFIX),
        input_args=parse_cond_args(if_name_ids, modified_name_ids),
        return_name_ids=return_name_ids)
    false_func_node = create_funcDef_node(
        node.orelse,
        name=unique_name.generate(FALSE_FUNC_PREFIX),
        input_args=parse_cond_args(else_name_ids, modified_name_ids),
        return_name_ids=return_name_ids)

    return true_func_node, false_func_node, return_name_ids


def create_cond_node(return_name_ids, pred, true_func, false_func):
    """
    Create `fluid.layers.cond(pred, true_fn, false_fn)` to replace
    original `python if/else` statement.
    """
    cond_api = gast.parse('fluid.layers.cond').body[0].value
    true_func_lambda = gast.Lambda(
        args=gast.arguments(
            args=[],
            posonlyargs=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=None,
            kwarg=None,
            defaults=[]),
        body=gast.Call(
            func=gast.Name(
                id=true_func.name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None),
            args=[true_func.args],
            keywords=[]))
    false_func_lambda = gast.Lambda(
        args=gast.arguments(
            args=[],
            posonlyargs=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=None,
            kwarg=None,
            defaults=[]),
        body=gast.Call(
            func=gast.Name(
                id=false_func.name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None),
            args=[false_func.args],
            keywords=[]))
    cond_layer = gast.Call(
        func=cond_api,
        args=[pred, true_func_lambda, false_func_lambda],
        keywords=[])
    if return_name_ids:
        _, cond_node = create_assign_node(return_name_ids, cond_layer)
    else:  # No variables can be returned if no assign statement in if.body.
        cond_node = gast.Expr(value=cond_layer)

    return cond_node
