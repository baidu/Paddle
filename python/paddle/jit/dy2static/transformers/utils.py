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

from __future__ import annotations

import copy
import textwrap
import warnings

import numpy as np

from paddle.base import unique_name
from paddle.jit.dy2static.ast_utils import ast_to_source_code
from paddle.utils import gast

from ..utils import PADDLE_MODULE_PREFIX, is_api_in_module_helper

GET_ARGS_FUNC_PREFIX = 'get_args'
SET_ARGS_FUNC_PREFIX = 'set_args'
ARGS_NAME = '__args'

TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'

FOR_ITER_INDEX_PREFIX = '__for_loop_var_index'
FOR_ITER_TUPLE_PREFIX = '__for_loop_iter_tuple'
FOR_ITER_TARGET_PREFIX = '__for_loop_iter_target'
FOR_ITER_ITERATOR_PREFIX = '__for_loop_iter_iterator'
FOR_ITER_TUPLE_INDEX_PREFIX = '__for_loop_iter_tuple_index'
FOR_ITER_VAR_LEN_PREFIX = '__for_loop_var_len'
FOR_ITER_VAR_NAME_PREFIX = '__for_loop_iter_var'
FOR_ITER_ZIP_TO_LIST_PREFIX = '__for_loop_iter_zip'

WHILE_CONDITION_PREFIX = 'while_condition'
WHILE_BODY_PREFIX = 'while_body'
FOR_CONDITION_PREFIX = 'for_loop_condition'
FOR_BODY_PREFIX = 'for_loop_body'


def index_in_list(array_list, item):
    try:
        return array_list.index(item)
    except ValueError:
        # Item not in array_list
        return -1


class BaseNodeVisitor(gast.NodeVisitor):
    """
    Implement customized NodeVisitor inherited from gast.NodeVisitor.
    Ancestor nodes are traced to easily support more operations of currently
    visited node.
    """

    def __init__(self):
        self.ancestor_nodes = []

    def visit(self, node):
        """Visit a node."""
        self.ancestor_nodes.append(node)

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret


def create_undefined_var(name):
    func_code = f"{name} = _jst.UndefinedVar('{name}')"
    return gast.parse(func_code).body[0]


def create_bool_node(name, value):
    '''
    Create a assign stmt for name = value .
    '''
    assert isinstance(value, bool)
    node = f"{name} = {value}"
    return gast.parse(node).body[0]


def get_parent_mapping(root):
    to_parent: dict[gast.AST, gast.AST] = {}
    for node in gast.walk(root):
        for child in gast.iter_child_nodes(node):
            to_parent[child] = node
    return to_parent


def create_name_str(name_ids):
    """
    Return "('x', 'y')" for [x, y]
    """
    if not name_ids:
        return 'None'

    names_str = ["'{}'".format(name.replace("'", "\\'")) for name in name_ids]
    return "({}, )".format(','.join(names_str))


def create_function_def_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    if return_name_ids:
        nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    else:
        nodes.append(gast.Return(value=None))
    func_def_node = gast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    return func_def_node


def create_assign_node(name, node):
    """
    Creates a `gast.Assign` node by given name_id as target and node as value.
    """
    targets = generate_name_node(name, ctx=gast.Store())
    assign_node = gast.Assign(
        targets=[targets],
        value=node,
        type_comment=None,
    )
    return targets, assign_node


def create_get_args_node(names):
    """
    Create get_args function as follows:

        def get_args_0():
            nonlocal x, y
            return x, y
    """

    def empty_node():
        func_def = f"""
        def {unique_name.generate(GET_ARGS_FUNC_PREFIX)}():
            return
        """
        return gast.parse(textwrap.dedent(func_def)).body[0]

    assert isinstance(names, (list, tuple))
    node = create_nonlocal_stmt_nodes(names)
    if not names:
        return empty_node()
    if node == []:
        nonlocal_vars = "\n"
    else:
        nonlocal_vars = ast_to_source_code(node[0])
    template = """
    def {func_name}():
        {nonlocal_vars}
        return {vars},
    """
    func_def = template.format(
        func_name=unique_name.generate(GET_ARGS_FUNC_PREFIX),
        nonlocal_vars=nonlocal_vars,
        vars=",".join(names),
    )
    return gast.parse(textwrap.dedent(func_def)).body[0]


def create_set_args_node(names):
    """
    Create set_args function as follows:

        def set_args_0(__args):
            nonlocal x, y
            x, y = __args
    """

    def empty_node():
        func_def = f"""
        def {unique_name.generate(SET_ARGS_FUNC_PREFIX)}({ARGS_NAME}):
            pass
        """
        return gast.parse(textwrap.dedent(func_def)).body[0]

    assert isinstance(names, (list, tuple))
    node = create_nonlocal_stmt_nodes(names)
    if not names:
        return empty_node()
    if node == []:
        nonlocal_vars = "\n"
    else:
        nonlocal_vars = ast_to_source_code(node[0])
    template = """
    def {func_name}({args}):
        {nonlocal_vars}
        {vars}, = {args}
    """
    func_def = template.format(
        func_name=unique_name.generate(SET_ARGS_FUNC_PREFIX),
        args=ARGS_NAME,
        nonlocal_vars=nonlocal_vars,
        vars=",".join(names),
    )
    return gast.parse(textwrap.dedent(func_def)).body[0]


def create_nonlocal_stmt_nodes(names):
    assert isinstance(names, (list, tuple))

    mapped = list(filter(lambda n: '.' not in n, names))
    mapped = list(filter(lambda n: '[' not in n, mapped))
    names = sorted(
        mapped, key=mapped.index
    )  # to keep the order, we can't use set() to unique
    if not names:
        return []
    func_code = "nonlocal {}".format(','.join(names))
    return [gast.parse(func_code).body[0]]


def generate_name_node(name_ids, ctx=gast.Load(), gen_tuple_if_single=False):
    """
    If name_ids is list or tuple or set with multiple strings, this function
    generates gast.Tuple of gast.Name.
    If the name_ids is single string or contains only 1 string, this function
    returns gast.Name if gen_tuple_if_single==False else returns gast.Tuple
    with only one gast.Name

    This function is used at several gast.Return statements.
    """
    if isinstance(name_ids, str):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError(
            f'name_ids must be list or tuple or set, but received {type(name_ids)}'
        )

    def create_node_for_name(name):
        if '.' not in name:
            return gast.Name(
                id=name, ctx=ctx, annotation=None, type_comment=None
            )
        return gast.parse(name).body[0].value

    gast_names = [create_node_for_name(name_id) for name_id in name_ids]
    if len(gast_names) == 1 and not gen_tuple_if_single:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def get_attribute_full_name(node):
    assert isinstance(
        node, gast.Attribute
    ), "Input non-Attribute node to get attribute full name"
    return ast_to_source_code(node).strip()


def is_api_in_module(node, module_prefix):
    assert isinstance(
        node, gast.Call
    ), "Input non-Call node for is_api_in_module"

    # Python can have gast.Call as function, for example: convert_call(func)(x)
    # We only check the most outside function
    func_node = node.func
    while isinstance(func_node, gast.Call):
        func_node = func_node.func

    func_str = ast_to_source_code(func_node).strip()
    try:
        import paddle
        import paddle.jit.dy2static as _jst
        from paddle import to_tensor

        globals = {
            'np': np,
            'paddle': paddle,
            '_jst': _jst,
            'to_tensor': to_tensor,
        }

        fn = eval(func_str, globals)
        return is_api_in_module_helper(fn, module_prefix)
    except Exception:
        return False


def is_paddle_api(node):
    return is_api_in_module(node, PADDLE_MODULE_PREFIX)


class NameScope:
    def __init__(self):
        """
        A NameScope is a object which manager all the variable names.
        only FunctionDef and Controlflow node will have a namescope property.

        type can be "function" and "controlflow"

        we don't analyze the read only variable because they don't affect the analysis.
        """
        self.globals = set()
        self.nonlocals = set()
        self.args = set()
        self.father = None  # point to the nearest function name scope.
        self.w_vars = set()  # all qualified + normal names been stored
        self.created = set()  # useful for control flow compatibility
        # only valid in control_flow nodes
        # may be remove later.
        self.push_pop_vars = set()  # we call push and pop in the vars

    def set_father(self, father):
        self.father = father

    def existed_vars(self):
        """vars existing in current scope.
        they must not contain qualified names.
        """
        local_vars = self.w_vars - self.globals - self.nonlocals - self.args
        return set(filter(lambda x: '.' not in x, local_vars))

    def created_vars(self):
        return self.created

    def modified_vars(self):
        # may be globals / non-locals / args / qualified names and created_vars
        return self.w_vars

    def variadic_length_vars(self):
        """
        At present, we do not support global append, such as

        import numpy as np
        a = []
        def func():
            a.append() # global names `a`, we will raise a warning.
            p.append(a, 1) # global names `np`, we will raise a warning.
        """
        non_global_push_pop_names = []
        for var in self.push_pop_vars:
            if self._is_simple_name(var) and self.is_global_var(var):
                warnings.warn(
                    f"Find variable `{var}` defined in global scope"
                    f" and call `{var}.append() or {var}.pop()`"
                    f", which will be ignored and never be transferred into"
                    f" tensor array."
                )
            else:
                non_global_push_pop_names.append(var)
        return set(non_global_push_pop_names)

    def control_flow_vars(self):
        valid_names = self.w_vars
        tmp = (self.father.global_vars & valid_names,)
        return {"global": tmp, "nonlocal": self.w_vars - tmp}

    def _is_simple_name(self, name):
        if '.' in name or '[' in name:
            return False
        return True

    def is_global_var(self, name):
        """
        Return whether the name is a var created in global scope.
        Search from bottom to top. If it is not created or modified,
        it means global vars; otherwise, it means local vars.
        Only valid after FunctionNameLivenessAnalysis visitor.
        """
        assert self._is_simple_name(
            name
        ), "is_global_var accept a simple name, but get `{name}`."
        ancestor = self
        while ancestor is not None:
            if name in ancestor.globals:
                return True
            if name in (ancestor.nonlocals | ancestor.w_vars):
                return False
            ancestor = ancestor.father
        return True

    def is_local_var(self, name):
        return not self.is_global_var(name)

    def merge_from(self, name_scope):
        self.globals |= name_scope.globals
        self.nonlocals |= name_scope.nonlocals
        self.args |= name_scope.args
        self.w_vars |= name_scope.w_vars
        self.push_pop_vars |= name_scope.push_pop_vars


class FunctionNameLivenessAnalysis(gast.NodeVisitor):
    """analyze the liveness of a function.

    every variables stored in this scope will be collected,
    in addition with global/nonlocal information and
    push_pop information.

    1. global variable is stored in node.var_globals.
    2. nonlocal variable is stored in node.var_nonlocals.
    3. arguments is stored in node.var_args.
    4. if a variable's push and pop attribute is called,
       it will be collected in push_pop_vars. They are
       used for transformation to tensor_array.
       NOTE: push_pop_vars **may not** in w_vars.
       a.push(0) don't modify the variable a, but the content
       of a.

    For example:

    def func(*args, **kargs):
        a = 12
        global i,j
        nonlocal x,y
        print(a)
        i = k
        b = []
        c = [1,2,3]
        for m in range(10):
            q = 12
            b.push(1)
            c.pop()

    After this visitor we have:
    # node is the FunctionDef node with name: "func"
    node.pd_scope = NameScope(
        globals = ['i', 'j'],
        nonlocals = ['x', 'y'],
        args = ['args', 'kargs'],
        wr_vars = ['a', 'i', 'q', 'm', 'c', 'b']
        push_pop_vars = ['b', 'c']
    )
    """

    def __init__(self, root_node):
        self.scope_node_stack = []  # controlflow, functiondef node
        self.visit(root_node)

    def _reset_name_scope(self, node):
        # always reset the node as empty namescope.
        node.pd_scope = NameScope()

    def _get_name_scope(self, node):
        if not hasattr(node, "pd_scope"):
            node.pd_scope = NameScope()
        return node.pd_scope

    def _current_name_scope(self):
        return self._get_name_scope(self.scope_node_stack[-1])

    def _father_name_scope(self):
        if len(self.scope_node_stack) == 1:
            return None
        return self._get_name_scope(self.scope_node_stack[-2])

    def _nearest_function_scope(self):
        if len(self.scope_node_stack) == 1:
            return None
        for node in self.scope_node_stack[-2::-1]:
            if isinstance(node, gast.FunctionDef):
                return self._get_name_scope(node)

    def visit_ListComp(self, node):
        """[ i for i in range(10) ]
        In this case, `i` will not created in FunctionScope.
        We don't collect `i` by not calling generic_visit.
        """
        pass

    def visit_DictComp(self, node):
        """the same as ListComp."""
        pass

    def visit_Name(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            self._current_name_scope().w_vars.add(node.id)

    def visit_FunctionDef(self, node):
        def pre_func():
            self._current_name_scope().args |= set(
                self._get_argument_names(node)
            )

        def post_func():
            """NOTE: why we need merge w_vars and push_pop_vars here ?
            because we do ifelse_transformer after loop_transformer. Loops will changed into functions. but we know this function will be called in if. so we add w_vars to father function scope.
            """
            control_flow_function_def = [
                WHILE_BODY_PREFIX,
                WHILE_BODY_PREFIX,
                FOR_CONDITION_PREFIX,
                FOR_BODY_PREFIX,
                TRUE_FUNC_PREFIX,
                FALSE_FUNC_PREFIX,
            ]

            def is_control_flow_def_node():
                for prefix in control_flow_function_def:
                    if node.name.startswith(prefix):
                        return True
                return False

            if self._father_name_scope() and is_control_flow_def_node():
                self._father_name_scope().w_vars |= (
                    self._current_name_scope().w_vars
                )
                self._father_name_scope().push_pop_vars |= (
                    self._current_name_scope().push_pop_vars
                )

        self._visit_scope_node(node, pre_func, post_func)

    def _visit_scope_node(self, node, pre_func, post_func):
        """scope node main visit logic.
        pre_func and post_func is callbacks
        """
        self._reset_name_scope(node)
        self.scope_node_stack.append(node)
        self._current_name_scope().set_father(self._nearest_function_scope())
        if pre_func:
            pre_func()
        self.generic_visit(node)
        if post_func:
            post_func()
        self.scope_node_stack.pop()

    def _visit_controlflow_node(self, node):
        def post_func():
            self._father_name_scope().merge_from(self._current_name_scope())
            self._nearest_function_scope().merge_from(
                self._current_name_scope()
            )
            self._current_name_scope().created = (
                self._nearest_function_scope().existed_vars()
                - node.before_created
            )
            # gather created vars into father and used in CreateUndefinedVarTransform
            self._nearest_function_scope().created |= (
                self._current_name_scope().created
            )

        def pre_func():
            node.before_created = self._nearest_function_scope().existed_vars()

        self._visit_scope_node(node, pre_func, post_func)

    def visit_For(self, node):
        self._visit_controlflow_node(node)

    def visit_While(self, node):
        self._visit_controlflow_node(node)

    def visit_If(self, node):
        self._visit_controlflow_node(node)

    def visit_Global(self, node):
        self._current_name_scope().globals |= set(node.names)

    def visit_Nonlocal(self, node):
        self._current_name_scope().nonlocals |= set(node.names)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            name = ast_to_source_code(node).strip()
            self._current_name_scope().w_vars.add(name)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            while isinstance(node.value, gast.Subscript):
                node = node.value
            if isinstance(node.value, gast.Name):
                self._current_name_scope().w_vars.add(node.value.id)

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, gast.Attribute):
            return
        variadic_length_method = ['append', 'pop']
        if node.func.attr not in variadic_length_method:
            return
        # we don't treat push and pop as a write operator. such as a[i]=10 is not modify a.
        name = ast_to_source_code(node.func.value).strip()
        self._current_name_scope().push_pop_vars.add(name)

    def _get_argument_names(self, node):
        """get all arguments name in the functiondef node.
        this node is local to the function and shouldn't
        be created.
        """
        assert isinstance(
            node, gast.FunctionDef
        ), "Input node is not function define node"
        names = list(node.args.args)
        names.append(node.args.vararg)
        names.append(node.args.kwarg)
        names = [i.id for i in names if i is not None]
        return names
