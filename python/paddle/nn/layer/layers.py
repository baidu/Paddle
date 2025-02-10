# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import inspect
import re
import typing
import warnings
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

import numpy as np
from typing_extensions import Self

import paddle
from paddle import Tensor, nn, profiler
from paddle.autograd.backward_utils import ValueSet
from paddle.base import core, framework, unique_name
from paddle.base.core import VarDesc
from paddle.base.dygraph import no_grad
from paddle.base.dygraph.base import (
    _convert_into_variable,
    in_declarative_mode,  # noqa: F401
    in_sot_simulation_mode,
    in_to_static_mode,
)
from paddle.base.dygraph_utils import _append_activation_in_dygraph
from paddle.base.executor import Executor, global_scope
from paddle.base.framework import (
    Parameter,
    Program,
    _current_expected_place as _get_device,
    convert_np_dtype_to_dtype_,
    default_main_program,
    in_dygraph_mode,
    in_pir_mode,
    name_struct,
    paddle_type_to_proto_type,
)
from paddle.base.layer_helper_base import LayerHelperBase
from paddle.framework import ParamAttr
from paddle.profiler.utils import in_profiler_mode
from paddle.utils import deprecated

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from paddle._typing import DTypeLike, ParamAttrLike, PlaceLike, ShapeLike
    from paddle.nn.initializer import Initializer


__all__ = []


_ForwardPreHook = Callable[
    ["Layer", Tensor], Tensor
]  # (layer, input) -> transformed_input
_ForwardPostHook = Callable[
    ["Layer", Tensor, Tensor], Tensor
]  # (layer, input, output) -> transformed_output
_StateDict = Union[Dict[str, Tensor], typing.OrderedDict[str, Tensor]]
_StateDictHook = Callable[[_StateDict], None]

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z])([A-Z])')


def record_program_ops_pre_hook(layer, inputs):
    """
    A pre-hook to mark op numbers before enter layer.forward.
    """
    if not in_dygraph_mode():
        if layer._op_recorder.start < 0:
            layer._op_recorder.start = len(
                default_main_program().current_block().ops
            )
            layer._op_recorder.is_valid = True
        else:
            layer._op_recorder.is_valid = False
            warnings.warn(
                f"{layer._full_name} has recorded the op information before. Please check whether you call this layer twice."
            )


def set_op_customized_attrs_post_hook(layer, inputs, outputs):
    """
    A post-hook to append customized attributes into all operators generated in current layer.
    """
    if not in_dygraph_mode() and layer._op_recorder.is_valid:
        start = layer._op_recorder.start
        end = len(default_main_program().current_block().ops)
        assert start >= 0 and end >= start
        ops = default_main_program().current_block().ops[start:end]

        layer._op_recorder.end = end
        layer._op_recorder.ops = ops

        for op in ops:
            for attr_name, val in layer._customized_attrs.items():
                op._set_attr(attr_name, val)

        # remove pre-hook and post-hook
        for hook_helper in layer._op_recorder.hooks:
            hook_helper.remove()


def _scope_dist2single(dist_scope):
    mapping = {
        "row_parallel_linear": "linear",
        "column_parallel_linear": "linear",
        "vocab_parallel_embedding": "embedding",
        # "parallel_cross_entropy": "cross_entropy", while mp_layer has parallel_cross_entropy,
        # but there is no parameters so the mapping of parallel_cross_entropy is not necessary.
    }
    return mapping.get(dist_scope, dist_scope)


def _convert_camel_to_snake(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    return _all_cap_re.sub(r'\1_\2', s1).lower()


def _addindent(string, indent):
    s1 = string.split('\n')
    if len(s1) == 1:
        return string
    s2 = []
    for idx, line in enumerate(s1):
        if idx > 0:
            s2.append(str((indent * ' ') + line))
    return s1[0] + '\n' + '\n'.join(s2)


def _layer_trans_dtype(layer, dtype, excluded_layers):
    if type(layer) in excluded_layers:
        return

    layer._to_impl(dtype=dtype, floating_only=True, include_sublayers=False)


class LayerObjectHelper(LayerHelperBase):
    def __init__(self, name):
        super().__init__(name, layer_type=name)

    def append_op(
        self,
        type=None,
        inputs=None,
        outputs=None,
        attrs=None,
        stop_gradient=None,
    ):
        """append an operator for this layer object.

           Args:
               type: operator type
               inputs: input variable of the operator
               dtype: data type of this parameter
               is_bias: if this is a bias parameter
               default_initializer: set the default initializer for this parameter

        Returns created parameter Variable.
        """
        return self.main_program.current_block().append_op(
            type=type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=stop_gradient,
        )

    def _multiple_input(self, inputs_in):
        inputs = inputs_in
        ret = []
        if isinstance(inputs, (list, tuple)):
            for inp in inputs:
                ret.append(self.to_variable(inp))
        else:
            ret.append(self.to_variable(inputs))
        return ret

    # TODO: make it public when we need it
    def _input(self, inputs_in):
        inputs = self._multiple_input(inputs_in)
        if len(inputs) != 1:
            raise f"{self.layer_type} layer only takes one input in"
        return inputs[0]

    def _multiple_param_attr(self, length, param_attr_in=None):
        param_attr = param_attr_in
        if isinstance(param_attr, ParamAttr):
            param_attr = [param_attr]

        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError(f"parameter number mismatch in {self.name}")
        elif len(param_attr) == 1 and length != 1:
            tmp = [None] * length
            for i in range(length):
                tmp[i] = copy.deepcopy(param_attr[0])
            param_attr = tmp
        return param_attr

    def iter_inputs_and_params(self, inputs_in, param_attr_in=None):
        """Access all inputs and params one by one

           Args:
               inputs_in: inputs to be iter
               param_attr_in: param_attr to be iter

        Returns input, param_attr
        """
        param_attr_in = ParamAttr._to_attr(param_attr_in)
        if isinstance(param_attr_in, bool):
            raise ValueError(f'Param_attr should not be False in {self.name}')
        inputs = inputs_in if (inputs_in is not None) else []
        inputs = self._multiple_input(inputs)
        param_attrs = self._multiple_param_attr(len(inputs), param_attr_in)
        yield from zip(inputs, param_attrs)

    def input_dtype(self, inputs_in):
        """Get input data type

           Args:
               inputs_in: inputs wanted know the data type

        Returns dtype of the input
        """
        inputs_in = inputs_in if (inputs_in is not None) else []
        inputs = self._multiple_input(inputs_in)
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.dtype
            elif dtype != each.dtype:
                raise ValueError(
                    f"Data Type mismatch: {dtype} to {each.dtype} in {self.name}"
                )
        return dtype

    def get_parameter(self, name):
        """Get parameter specifically

           Args:
               name: parameter's name

        Returns target parameter
        """
        param = self.main_program.global_block().var(name)
        if not isinstance(param, Parameter):
            raise ValueError(f"no Parameter name {name} found in {self.name}")
        return param

    # TODO: this should not be called anymore after all activation func move to Layers
    def append_activation(self, input_var, act=None, use_cudnn=None):
        """Append activation

            Args:
                input_var: the input variable. The len(input_var.shape) is
                larger or equal than 2.
                act: activation type
                use_cudnn: if use cudnn

        Return the Variable of after append activation
        """
        act = act
        if act is None:
            return input_var
        if isinstance(act, str):
            act = {'type': act}
        else:
            raise TypeError(f"{act} should be unicode or str in {self.name}")

        if (use_cudnn is not None) and use_cudnn:
            act['use_cudnn'] = use_cudnn
        act_type = act.pop('type')
        if in_dygraph_mode():
            res = _append_activation_in_dygraph(input_var, act_type, use_cudnn)
            return res
        else:
            tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
            self.append_op(
                type=act_type,
                inputs={"X": [input_var]},
                outputs={"Out": [tmp]},
                attrs=act,
            )
            return tmp

    def is_instance(self, param, cls):
        """Check if the input parameter is instance of input class

            Args:
                param: parameter to be check
                cls: class of the parameter

        Return result of the check (True or False)
        """
        param = param
        if not isinstance(param, cls):
            raise TypeError(
                "The input {0} parameter of method {1} must be {2}, in layer {3}",
                param,
                self.layer_type,
                cls.__name__,
                self.name,
            )


class LayerOpsRecorder:
    """
    Record generated operators information in nn.Layer.
    """

    def __init__(self, start=-1, end=-1, ops=None, is_valid=False, hooks=None):
        self.start = start
        self.end = end
        self.ops = ops
        self.is_valid = is_valid
        self.hooks = hooks


class HookRemoveHelper:
    """A HookRemoveHelper that can be used to remove hook."""

    next_hook_id: int = 0

    def __init__(
        self, hooks: typing.OrderedDict[int, Callable[..., Any]]
    ) -> None:
        self._hooks_ref = weakref.ref(hooks)
        self._hook_id = HookRemoveHelper.next_hook_id
        HookRemoveHelper.next_hook_id += 1

    def remove(self) -> None:
        hooks = self._hooks_ref()
        if hooks is not None and self._hook_id in hooks:
            del hooks[self._hook_id]


class Layer:
    """
    Dynamic graph Layer based on OOD, includes the parameters of the layer, the structure of the forward graph and so on.

    Parameters:
        name_scope (str, optional): prefix name used by the layer to name parameters.
            If prefix is "my_layer", parameter name in MyLayer
            can be "my_layer_0.w_n", where "w" is the parameter
            base name and "n" is an unique suffix auto-generated.
            If None, prefix name will be snake cased class name. Default: None.
        dtype(str, optional): data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16".
                Default: "float32"

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> class MyLayer(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = paddle.nn.Linear(1, 1)
            ...         self._dropout = paddle.nn.Dropout(p=0.5)
            ...
            ...     def forward(self, input):
            ...         temp = self._linear(input)
            ...         temp = self._dropout(temp)
            ...         return temp
            ...
            >>> x = paddle.randn([10, 1], 'float32')
            >>> mylayer = MyLayer()
            >>> mylayer.eval()  # set mylayer._dropout to eval mode
            >>> out = mylayer(x)
            >>> mylayer.train()  # set mylayer._dropout to train mode
            >>> out = mylayer(x)
            >>> print(out)
            Tensor(shape=[10, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[-3.44879317],
             [ 0.        ],
             [ 0.        ],
             [-0.73825276],
             [ 0.        ],
             [ 0.        ],
             [ 0.64444798],
             [-3.22185946],
             [ 0.        ],
             [-0.68077987]])
    """

    training: bool

    def __init__(
        self, name_scope: str | None = None, dtype: DTypeLike = "float32"
    ) -> None:
        self.training = True
        if name_scope is None:
            name_scope = _convert_camel_to_snake(self.__class__.__name__)
            name_scope = _scope_dist2single(name_scope)
        self._full_name = unique_name.generate(name_scope)
        self._helper = LayerObjectHelper(self._full_name)
        self._built = False
        self._dtype = dtype
        self._init_in_dynamic_mode = in_dygraph_mode()

        self._parameters = OrderedDict()
        # Buffers the variable (not parameter) created in layer
        self._buffers = OrderedDict()
        self._non_persistable_buffer_names_set = set()
        self._sub_layers = OrderedDict()
        self._loaddict_holder = OrderedDict()

        # Record generated op_descs in this layer
        self._op_recorder = LayerOpsRecorder(ops=[], hooks=[])
        self._customized_attrs = {}

        self._forward_pre_hooks: typing.OrderedDict[int, _ForwardPreHook] = (
            OrderedDict()
        )
        self._forward_post_hooks: typing.OrderedDict[int, _ForwardPostHook] = (
            OrderedDict()
        )

        # only used in AMP Training
        self._cast_to_low_precision = True

        self._state_dict_hooks: typing.OrderedDict[int, _StateDictHook] = (
            OrderedDict()
        )
        # Records original functions after @to_static to support to rollback
        self._original_funcs = OrderedDict()

    def train(self) -> None:
        """

        Sets this Layer and all its sublayers to training mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(100)

                >>> class MyLayer(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = paddle.nn.Linear(1, 1)
                ...         self._dropout = paddle.nn.Dropout(p=0.5)
                ...
                ...     def forward(self, input):
                ...         temp = self._linear(input)
                ...         temp = self._dropout(temp)
                ...         return temp
                ...
                >>> x = paddle.randn([10, 1], 'float32')
                >>> mylayer = MyLayer()
                >>> mylayer.eval()  # set mylayer._dropout to eval mode
                >>> out = mylayer(x)
                >>> mylayer.train()  # set mylayer._dropout to train mode
                >>> out = mylayer(x)
                >>> print(out)
                Tensor(shape=[10, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[-3.44879317],
                 [ 0.        ],
                 [ 0.        ],
                 [-0.73825276],
                 [ 0.        ],
                 [ 0.        ],
                 [ 0.64444798],
                 [-3.22185946],
                 [ 0.        ],
                 [-0.68077987]])

        """
        # global setting in dygraph
        # NOTE(chenweihang): nn.Layer also can be used in static mode,
        # but _dygraph_tracer() can not be called in static mode
        if in_dygraph_mode():
            framework._dygraph_tracer().train_mode()
        # Layer-level setting
        self.training = True
        for layer in self.sublayers():
            layer.training = True

    def eval(self) -> None:
        """
        Sets this Layer and all its sublayers to evaluation mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            None

        Example::
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(100)
                >>> class MyLayer(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = paddle.nn.Linear(1, 1)
                ...         self._dropout = paddle.nn.Dropout(p=0.5)
                ...
                ...     def forward(self, input):
                ...         temp = self._linear(input)
                ...         temp = self._dropout(temp)
                ...         return temp
                ...
                >>> x = paddle.randn([10, 1], 'float32')
                >>> mylayer = MyLayer()
                >>> mylayer.eval()  # set mylayer._dropout to eval mode
                >>> out = mylayer(x)
                >>> print(out)
                Tensor(shape=[10, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[-1.72439659],
                 [ 0.31532824],
                 [ 0.01192369],
                 [-0.36912638],
                 [-1.63426113],
                 [-0.93169814],
                 [ 0.32222399],
                 [-1.61092973],
                 [ 0.77209264],
                 [-0.34038994]])

        """
        # global setting in dygraph
        # NOTE(chenweihang): nn.Layer also can be used in static mode,
        # but _dygraph_tracer() can not be called in static mode
        if in_dygraph_mode():
            framework._dygraph_tracer().eval_mode()
        # Layer-level setting
        self.training = False
        for layer in self.sublayers():
            layer.training = False

    def apply(self, fn: Callable[[Self], None]) -> Self:
        """

        Applies ``fn`` recursively to every sublayer (as returned by ``.sublayers()``)
        as well as self. Typical use includes initializing the parameters of a model.

        Parameters:
            fn (function): a function to be applied to each sublayer

        Returns:
            Layer, self

        Example::
            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> paddle.seed(2023)

                >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

                >>> def init_weights(layer):
                ...     if type(layer) == nn.Linear:
                ...         print('before init weight:', layer.weight.numpy())
                ...         new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
                ...         layer.weight.set_value(new_weight)
                ...         print('after init weight:', layer.weight.numpy())
                ...
                >>> net.apply(init_weights)

                >>> print(net.state_dict())
                before init weight: [[ 0.89611185  0.04935038]
                                     [-0.5888344   0.99266374]]
                after init weight: [[0.9 0.9]
                                    [0.9 0.9]]
                before init weight: [[-0.18615901 -0.22924072]
                                     [ 1.1517721   0.59859073]]
                after init weight: [[0.9 0.9]
                                    [0.9 0.9]]
                OrderedDict([('0.weight', Parameter containing:
                Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[0.89999998, 0.89999998],
                 [0.89999998, 0.89999998]])), ('0.bias', Parameter containing:
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0., 0.])), ('1.weight', Parameter containing:
                Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[0.89999998, 0.89999998],
                 [0.89999998, 0.89999998]])), ('1.bias', Parameter containing:
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0., 0.]))])
        """
        for layer in self.children():
            layer.apply(fn)

        fn(self)

        return self

    def full_name(self) -> str:
        """

        Full name for this layer, composed by name_scope + "/" + MyLayer.__class__.__name__

        Returns:
            str, full name of this layer.

        Example::
            .. code-block:: python

                >>> import paddle

                >>> class LinearNet(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__(name_scope = "demo_linear_net")
                ...         self._linear = paddle.nn.Linear(1, 1)
                ...
                ...     def forward(self, x):
                ...         return self._linear(x)
                ...
                >>> linear_net = LinearNet()
                >>> print(linear_net.full_name())
                demo_linear_net_0

        """
        return self._full_name

    def register_forward_post_hook(
        self, hook: _ForwardPostHook
    ) -> HookRemoveHelper:
        """

        Register a forward post-hook for Layer. The hook will be called after `forward` function has been computed.

        It should have the following form, `input` and `output` of the `hook` is `input` and `output` of the `Layer` respectively.
        User can use forward post-hook to change the output of the Layer or perform information statistics tasks on the Layer.

        hook(Layer, input, output) -> None or modified output

        Parameters:
            hook(function): a function registered as a forward post-hook

        Returns:
            HookRemoveHelper, a HookRemoveHelper object that can be used to remove the added hook by calling `hook_remove_helper.remove()` .

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> # the forward_post_hook change the output of the layer: output = output * 2
                >>> def forward_post_hook(layer, input, output):
                ...     # user can use layer, input and output for information statistics tasks
                ...
                ...     # change the output
                ...     return output * 2
                ...
                >>> linear = paddle.nn.Linear(13, 5)

                >>> # register the hook
                >>> forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

                >>> value1 = np.arange(26).reshape(2, 13).astype("float32")
                >>> in1 = paddle.to_tensor(value1)

                >>> out0 = linear(in1)

                >>> # remove the hook
                >>> forward_post_hook_handle.remove()

                >>> out1 = linear(in1)

                >>> # hook change the linear's output to output * 2, so out0 is equal to out1 * 2.
                >>> assert (out0.numpy() == (out1.numpy()) * 2).any()

        """
        hook_remove_helper = HookRemoveHelper(self._forward_post_hooks)
        self._forward_post_hooks[hook_remove_helper._hook_id] = hook
        return hook_remove_helper

    def register_forward_pre_hook(
        self, hook: _ForwardPreHook
    ) -> HookRemoveHelper:
        """

        Register a forward pre-hook for Layer. The hook will be called before `forward` function has been computed.

        It should have the following form, `input` of the `hook` is `input` of the `Layer`,
        hook can either return a tuple or a single modified value in the hook. We will wrap the value into a tuple if
        a single value is returned(unless that value is already a tuple).
        User can use forward pre-hook to change the input of the Layer or perform information statistics tasks on the Layer.

        hook(Layer, input) -> None or modified input

        Parameters:
            hook(function): a function registered as a forward pre-hook

        Returns:
            HookRemoveHelper, a HookRemoveHelper object that can be used to remove the added hook by calling `hook_remove_helper.remove()` .

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> # the forward_pre_hook change the input of the layer: input = input * 2
                >>> def forward_pre_hook(layer, input):
                ...     # user can use layer and input for information statistics tasks
                ...
                ...     # change the input
                ...     input_return = (input[0] * 2)
                ...     return input_return
                ...
                >>> linear = paddle.nn.Linear(13, 5)

                >>> # register the hook
                >>> forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

                >>> value0 = np.arange(26).reshape(2, 13).astype("float32")
                >>> in0 = paddle.to_tensor(value0)
                >>> out0 = linear(in0)

                >>> # remove the hook
                >>> forward_pre_hook_handle.remove()

                >>> value1 = value0 * 2
                >>> in1 = paddle.to_tensor(value1)
                >>> out1 = linear(in1)

                >>> # hook change the linear's input to input * 2, so out0 is equal to out1.
                >>> assert (out0.numpy() == out1.numpy()).any()
        """
        hook_remove_helper = HookRemoveHelper(self._forward_pre_hooks)
        self._forward_pre_hooks[hook_remove_helper._hook_id] = hook
        return hook_remove_helper

    def create_parameter(
        self,
        shape: ShapeLike,
        attr: ParamAttrLike | None = None,
        dtype: DTypeLike | None = None,
        is_bias: bool = False,
        default_initializer: Initializer | None = None,
    ) -> Tensor:
        """Create parameters for this layer.

        Parameters:
            shape(list): Shape of the parameter. The data type in the list must be int.
            attr(ParamAttr, optional): Parameter attribute of weight. Please refer to :ref:`api_paddle_ParamAttr`. Default: None.
            dtype(str, optional): Data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16". Default: "float32".
            is_bias(bool, optional): if this is a bias parameter. Default: False.
            default_initializer(Initializer, optional): the default initializer for this parameter.
                If set None, default initializer will be set to paddle.nn.initializer.Xavier and paddle.nn.initializer.Constant
                for non-bias and bias parameter, respectively. Default: None.

        Returns:
            :Tensor, created parameter.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(2023)

                >>> class MyLayer(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = paddle.nn.Linear(1, 1)
                ...         w_tmp = self.create_parameter([1,1])
                ...         self.add_parameter("w_tmp", w_tmp)
                ...
                ...     def forward(self, input):
                ...         return self._linear(input)
                ...
                >>> mylayer = MyLayer()
                >>> for name, param in mylayer.named_parameters():
                ...     print(name, param)      # will print w_tmp,_linear.weight,_linear.bias
                w_tmp Parameter containing:
                Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[0.06979191]])
                _linear.weight Parameter containing:
                Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[1.26729357]])
                _linear.bias Parameter containing:
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0.])
        """
        temp_attr = copy.deepcopy(attr)
        if isinstance(temp_attr, str) and temp_attr == "":
            temp_attr = None
        return self._helper.create_parameter(
            temp_attr, shape, dtype, is_bias, default_initializer
        )

    @deprecated(
        since="2.0.0",
        update_to="paddle.nn.Layer.create_tensor",
        reason="New api in create_tensor, easier to use.",
    )
    def create_variable(
        self,
        name: str | None = None,
        persistable: bool | None = None,
        dtype: DTypeLike | None = None,
    ) -> Tensor:
        """

        Create Tensor for this layer.

        Parameters:
            name(str, optional): name of the tensor. Please refer to :ref:`api_guide_Name` . Default: None

            persistable(bool, optional): if set this tensor persistable. Default: False

            dtype(str, optional): data type of this parameter. If set str, it can be "bool", "float16", "float32", "float64","int8", "int16", "int32", "int64", "uint8" or "uint16". If set None, it will be "float32". Default: None

        Returns:
            Tensor, created Tensor.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> class MyLinear(paddle.nn.Layer):
                ...     def __init__(self,
                ...                 in_features,
                ...                 out_features):
                ...         super().__init__()
                ...         self.linear = paddle.nn.Linear( 10, 10)
                ...
                ...         self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)
                ...
                ...     def forward(self, input):
                ...         out = self.linear(input)
                ...         paddle.assign( out, self.back_var)
                ...
                ...         return out

        """
        if name is not None:
            var_name = ".".join([self._full_name, name])
        else:
            var_name = unique_name.generate(
                ".".join([self._full_name, "_generated_var"])
            )

        return self._helper.main_program.current_block().create_var(
            name=var_name,
            persistable=persistable,
            dtype=dtype,
            type=core.VarDesc.VarType.DENSE_TENSOR,
        )

    # TODO: Add more parameter list when we need them
    def create_tensor(
        self,
        name: str | None = None,
        persistable: bool | None = None,
        dtype: DTypeLike | None = None,
    ) -> Tensor:
        """

        Create Tensor for this layer.

        Parameters:
            name(str, optional): name of the tensor. Please refer to :ref:`api_guide_Name` . Default: None.
            persistable(bool, optional): if set this tensor persistable. Default: False.
            dtype(str, optional): data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16".
                If set None, it will be "float32". Default: None.

        Returns:
            Tensor, created Tensor.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> class MyLinear(paddle.nn.Layer):
                ...     def __init__(self,
                ...                  in_features,
                ...                  out_features):
                ...         super().__init__()
                ...         self.linear = paddle.nn.Linear(10, 10)
                ...
                ...         self.back_var = self.create_tensor(name = "linear_tmp_0", dtype=self._dtype)
                ...
                ...     def forward(self, input):
                ...         out = self.linear(input)
                ...         paddle.assign(out, self.back_var)
                ...
                ...         return out

        """
        if name is not None:
            var_name = ".".join([self._full_name, name])
        else:
            var_name = unique_name.generate(
                ".".join([self._full_name, "_generated_var"])
            )

        return self._helper.main_program.current_block().create_var(
            name=var_name,
            persistable=persistable,
            dtype=dtype,
            type=core.VarDesc.VarType.DENSE_TENSOR,
        )

    def parameters(self, include_sublayers: bool = True) -> list[Tensor]:
        """

        Returns a list of all Parameters from current layer and its sub-layers.

        Parameters:
            include_sublayers (bool, optional): Whether to return the parameters of the sublayer.
                If True, the returned list contains the parameters of the sublayer.
                Default: True.

        Returns:
            list, list of Tensor, a list of Parameters.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(100)

                >>> linear = paddle.nn.Linear(1, 1)
                >>> print(linear.parameters())
                [Parameter containing:
                Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[0.18551230]]), Parameter containing:
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0.])]

        """
        ret = [
            param
            for _, param in self.named_parameters(
                include_sublayers=include_sublayers
            )
        ]
        return ret

    def astype(self, dtype: DTypeLike | None = None) -> Self:
        """

        Casts all parameters and buffers to dtype and then return the Layer.

        Parameters:
            dtype(str|paddle.dtype|numpy.dtype): target data type of layer.
                If set str, it can be "bool", "bfloat16", "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8", "complex64", "complex128".
                Default: None

        Returns:
            Layer, self

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> weight_attr = paddle.ParamAttr(name="weight",initializer=paddle.nn.initializer.Constant(value=1.5))
                >>> bias_attr = paddle.ParamAttr(name="bias",initializer=paddle.nn.initializer.Constant(value=2.5))

                >>> linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr).to(device="cpu",dtype="float32")
                >>> print(linear)
                Linear(in_features=2, out_features=2, dtype=float32)
                >>> print(linear.parameters())
                [Parameter containing:
                Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
                    [[1.50000000, 1.50000000],
                        [1.50000000, 1.50000000]]), Parameter containing:
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=False,
                    [2.50000000, 2.50000000])]

                >>> linear=linear.astype("int8")
                >>> print(linear)
                Linear(in_features=2, out_features=2, dtype=paddle.int8)
                >>> print(linear.parameters())
                [Parameter containing:
                Tensor(shape=[2, 2], dtype=int8, place=Place(cpu), stop_gradient=False,
                    [[1, 1],
                        [1, 1]]), Parameter containing:
                Tensor(shape=[2], dtype=int8, place=Place(cpu), stop_gradient=False,
                    [2, 2])]

        """
        valid_dtypes = [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "complex64",
            "complex128",
            "bool",
        ]
        if (
            isinstance(dtype, (paddle.dtype, np.dtype))
            or type(dtype) is str
            and dtype in valid_dtypes
        ):
            if isinstance(dtype, (str, np.dtype)):
                dtype = framework.convert_np_dtype_to_dtype_(dtype)
            self._dtype = dtype
            for layer in self.sublayers():
                layer._dtype = dtype
            for _, param in self.named_parameters(include_sublayers=True):
                param._to(None, dtype)
            for _, buffer in self.named_buffers(include_sublayers=True):
                buffer.to(None, dtype)
            return self
        else:
            raise ValueError(
                "dtype value error, must be 'bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128', 'bool', or paddle.dtype, numpy.dtype, but receive "
                + str(dtype)
            )

    def children(self) -> Iterable[Layer]:
        """

        Returns an iterator over immediate children layers.

        Yields:
            Layer: a child layer

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> linear1 = paddle.nn.Linear(10, 3)
                >>> linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
                >>> model = paddle.nn.Sequential(linear1, linear2)

                >>> layer_list = list(model.children())

                >>> print(layer_list)
                [Linear(in_features=10, out_features=3, dtype=float32), Linear(in_features=3, out_features=10, dtype=float32)]

        """
        for _, layer in self.named_children():
            yield layer

    def named_children(self) -> Iterable[tuple[str, Layer]]:
        """Returns an iterator over immediate children layers, yielding both
        the name of the layer as well as the layer itself.

        Yields:
            (string, Layer): Tuple containing a name and child layer

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> linear1 = paddle.nn.Linear(10, 3)
                >>> linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
                >>> model = paddle.nn.Sequential(linear1, linear2)
                >>> for prefix, layer in model.named_children():
                ...     print(prefix, layer)
                0 Linear(in_features=10, out_features=3, dtype=float32)
                1 Linear(in_features=3, out_features=10, dtype=float32)
        """
        memo = set()
        for name, layer in self._sub_layers.items():
            if layer is not None and layer not in memo:
                memo.add(layer)
                yield name, layer

    def sublayers(self, include_self: bool = False) -> list[Layer]:
        """

        Returns a list of sub layers.

        Parameters:
            include_self(bool, optional): Whether return self as sublayers. Default: False.

        Returns:
            list of Layer, a list of sub layers.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> class MyLayer(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = paddle.nn.Linear(1, 1)
                ...         self._dropout = paddle.nn.Dropout(p=0.5)
                ...
                ...     def forward(self, input):
                ...         temp = self._linear(input)
                ...         temp = self._dropout(temp)
                ...         return temp
                ...
                >>> mylayer = MyLayer()
                >>> print(mylayer.sublayers())
                [Linear(in_features=1, out_features=1, dtype=float32), Dropout(p=0.5, axis=None, mode=upscale_in_train)]

        """
        ret = [
            layer
            for _, layer in self.named_sublayers(include_self=include_self)
        ]
        return ret

    def named_parameters(
        self,
        prefix: str = '',
        include_sublayers: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, Tensor]]:
        """
        Returns an iterator over all parameters in the Layer, yielding tuple of name and parameter.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            include_sublayers(bool, optional): Whether include the parameters of sublayers.
                If True, also include the named parameters from sublayers. Default: True.
            remove_duplicate(bool, optional): Whether to remove duplicated parameters in the result.
                Default: True.

        Yields:
            (string, Parameter): Tuple of name and Parameter

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(100)

                >>> fc1 = paddle.nn.Linear(10, 3)
                >>> fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
                >>> model = paddle.nn.Sequential(fc1, fc2)
                >>> for name, param in model.named_parameters():
                ...     print(name, param)
                0.weight Parameter containing:
                Tensor(shape=[10, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[ 0.07276392, -0.39791510, -0.66356444],
                 [ 0.02143478, -0.18519843, -0.32485050],
                 [-0.42249614,  0.08450919, -0.66838276],
                 [ 0.38208580, -0.24303678,  0.55127048],
                 [ 0.47745085,  0.62117910, -0.08336520],
                 [-0.28653207,  0.47237599, -0.05868882],
                 [-0.14385653,  0.29945642,  0.12832761],
                 [-0.21237159,  0.38539791, -0.62760031],
                 [ 0.02637231,  0.20621127,  0.43255770],
                 [-0.19984481, -0.26259184, -0.29696006]])
                0.bias Parameter containing:
                Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0., 0., 0.])
                1.weight Parameter containing:
                Tensor(shape=[3, 10], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[ 0.01985580, -0.40268910,  0.41172385, -0.47249708, -0.09002256,
                 -0.00533628, -0.52048630,  0.62360322,  0.20848787, -0.02033746],
                 [ 0.58281910,  0.12841827,  0.12907702,  0.02325618, -0.07746267,
                 0.31950659, -0.37924835, -0.59209681, -0.11732036, -0.58378261],
                 [-0.62100595,  0.22293305,  0.28229684, -0.03687060, -0.59323978,
                 0.08411229,  0.53275704,  0.40431368,  0.03171402, -0.17922515]])
        """
        params_set = (
            ValueSet() if in_pir_mode() and not in_to_static_mode() else set()
        )
        named_sublayers = (
            self.named_sublayers(
                prefix=prefix,
                include_self=True,
                remove_duplicate=remove_duplicate,
            )
            if include_sublayers
            else zip([prefix], [self])
        )
        for layer_prefix, sublayer in named_sublayers:
            params = sublayer._parameters.items()
            for key, param in params:
                if param is None or param in params_set:
                    continue
                if remove_duplicate:
                    params_set.add(param)
                name = layer_prefix + ('.' if layer_prefix else '') + key
                yield name, param

    def named_sublayers(
        self,
        prefix: str = '',
        include_self: bool = False,
        layers_set: set[Layer] | None = None,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, Layer]]:
        """
        Returns an iterator over all sublayers in the Layer, yielding tuple of name and sublayer.
        The duplicate sublayer will only be yielded once.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            include_self(bool, optional): Whether include the Layer itself. Default: False.
            layers_set(set, optional): The set to record duplicate sublayers. Default: None.
            remove_duplicate(bool, optional): Whether to remove duplicated sublayers in the result.
                Default: True.

        Yields:
            (string, Layer): Tuple of name and Layer

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> fc1 = paddle.nn.Linear(10, 3)
                >>> fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
                >>> model = paddle.nn.Sequential(fc1, fc2)
                >>> for prefix, layer in model.named_sublayers():
                ...     print(prefix, layer)
                0 Linear(in_features=10, out_features=3, dtype=float32)
                1 Linear(in_features=3, out_features=10, dtype=float32)

                >>> l = paddle.nn.Linear(10, 3)
                >>> model = paddle.nn.Sequential(l, l)
                >>> for prefix, layer in model.named_sublayers(include_self=True, remove_duplicate=True):
                ...     print(prefix, layer)
                 Sequential(
                  (0): Linear(in_features=10, out_features=3, dtype=float32)
                  (1): Linear(in_features=10, out_features=3, dtype=float32)
                )
                0 Linear(in_features=10, out_features=3, dtype=float32)

                >>> l = paddle.nn.Linear(10, 3)
                >>> model = paddle.nn.Sequential(l, l)
                >>> for prefix, layer in model.named_sublayers(include_self=True, remove_duplicate=False):
                ...     print(prefix, layer)
                 Sequential(
                  (0): Linear(in_features=10, out_features=3, dtype=float32)
                  (1): Linear(in_features=10, out_features=3, dtype=float32)
                )
                0 Linear(in_features=10, out_features=3, dtype=float32)
                1 Linear(in_features=10, out_features=3, dtype=float32)

        """
        if layers_set is None:
            layers_set = set()
        if include_self and self not in layers_set:
            if remove_duplicate:
                layers_set.add(self)
            yield prefix, self
        for key, layer in self._sub_layers.items():
            if layer is None:
                continue
            layer_prefix = prefix + ('.' if prefix else '') + key
            yield from layer.named_sublayers(
                prefix=layer_prefix,
                include_self=True,
                layers_set=layers_set,
                remove_duplicate=remove_duplicate,
            )

    def register_buffer(
        self, name: str, tensor: Tensor, persistable: bool = True
    ) -> None:
        """
        Registers a tensor as buffer into the layer.

        `buffer` is a non-trainable tensor and will not be updated by optimizer,
        but is necessary for evaluation and inference. For example, the mean and variance in BatchNorm layers.
        The registered buffer is persistable by default, and will be saved into
        `state_dict` alongside parameters. If set persistable=False, it registers
        a non-persistable buffer, so that it will not be a part of `state_dict` .

        Buffers can be accessed as attributes using given names.

        Parameters:
            name (string): name of the buffer. The buffer can be accessed
                from this layer using the given name
            tensor (Tensor): the tensor to be registered as buffer.
            persistable (bool): whether the buffer is part of this layer's
                state_dict.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> import paddle

                >>> linear = paddle.nn.Linear(10, 3)
                >>> value = np.array([0]).astype("float32")
                >>> buffer = paddle.to_tensor(value)
                >>> linear.register_buffer("buf_name", buffer, persistable=True)

                >>> # get the buffer by attribute.
                >>> print(linear.buf_name)
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.])

        """

        if '_buffers' not in self.__dict__:
            raise ValueError("super().__init__() should be called first")
        elif not isinstance(name, str):
            raise TypeError(
                f"The name of buffer should be a string, but received {type(name).__name__}."
            )
        elif '.' in name:
            raise KeyError(
                "The name of buffer can not contain `.`, "
                "because when you access the newly added buffer in the "
                "form of `self.**.**`, it will cause AttributeError."
            )
        elif name == '':
            raise KeyError("The name of buffer can not be empty.")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists.")
        elif tensor is not None and not (type(tensor) == core.eager.Tensor):
            raise TypeError(
                f"The registered buffer should be a Paddle.Tensor, but received {type(tensor).__name__}."
            )
        else:
            self._buffers[name] = tensor
            if persistable:
                self._non_persistable_buffer_names_set.discard(name)
            else:
                self._non_persistable_buffer_names_set.add(name)

    def buffers(self, include_sublayers: bool = True) -> list[Tensor]:
        """

        Returns a list of all buffers from current layer and its sub-layers.

        Parameters:
            include_sublayers(bool, optional): Whether include the buffers of sublayers. If True, also include the buffers from sublayers. Default: True.

        Returns:
            list of Tensor, a list of buffers.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> import paddle

                >>> linear = paddle.nn.Linear(10, 3)
                >>> value = np.array([0]).astype("float32")
                >>> buffer = paddle.to_tensor(value)
                >>> linear.register_buffer("buf_name", buffer, persistable=True)

                >>> print(linear.buffers())
                [Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.])]

        """
        ret = [
            buffer
            for _, buffer in self.named_buffers(
                include_sublayers=include_sublayers
            )
        ]
        return ret

    def named_buffers(
        self,
        prefix: str = '',
        include_sublayers: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, Tensor]]:
        """
        Returns an iterator over all buffers in the Layer, yielding tuple of name and Tensor.

        Parameters:
            prefix(str, optional): Prefix to prepend to all buffer names. Default: ''.
            include_sublayers(bool, optional): Whether include the buffers of sublayers.
                If True, also include the named buffers from sublayers. Default: True.
            remove_duplicate(bool, optional): Whether to remove duplicated buffers in the result.
                Default: True.

        Yields:
            (string, Tensor): Tuple of name and tensor

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> import paddle

                >>> fc1 = paddle.nn.Linear(10, 3)
                >>> buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))
                >>> # register a tensor as buffer by specific `persistable`
                >>> fc1.register_buffer("buf_name_1", buffer1, persistable=True)

                >>> fc2 = paddle.nn.Linear(3, 10)
                >>> buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))
                >>> # register a buffer by assigning an attribute with Tensor.
                >>> # The `persistable` can only be False by this way.
                >>> fc2.buf_name_2 = buffer2

                >>> model = paddle.nn.Sequential(fc1, fc2)

                >>> # get all named buffers
                >>> for name, buffer in model.named_buffers():
                ...     print(name, buffer)
                0.buf_name_1 Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.])
                1.buf_name_2 Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [1.])
        """
        buffers_set = set()
        named_sublayers = (
            self.named_sublayers(
                prefix=prefix,
                include_self=True,
                remove_duplicate=remove_duplicate,
            )
            if include_sublayers
            else zip([prefix], [self])
        )
        for layer_prefix, sublayer in named_sublayers:
            buffers = sublayer._buffers.items()
            for key, buffer in buffers:
                if buffer is None or buffer in buffers_set:
                    continue
                if remove_duplicate:
                    buffers_set.add(buffer)
                name = layer_prefix + ('.' if layer_prefix else '') + key
                yield name, buffer

    def clear_gradients(self, set_to_zero: bool = True) -> None:
        """
        Clear the gradients of all parameters for this layer.

        Args:
            set_to_zero (bool, optional): Whether to set the trainable parameters'
                gradients to zero or None. Default is True.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> value = np.arange(26).reshape(2, 13).astype("float32")
                >>> a = paddle.to_tensor(value)
                >>> linear = paddle.nn.Linear(13, 5)
                >>> adam = paddle.optimizer.Adam(learning_rate=0.01,
                ...                              parameters=linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> adam.step()
                >>> linear.clear_gradients()

        """
        for p in self.parameters():
            if p.trainable:
                p.clear_gradient(set_to_zero)

    def _build_once(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _dygraph_call_func(self, *inputs: Any, **kwargs: Any) -> Any:
        for forward_pre_hook in self._forward_pre_hooks.values():
            hook_result = forward_pre_hook(self, inputs)
            if hook_result is not None:
                if not isinstance(hook_result, tuple):
                    hook_result = (hook_result,)
                inputs = hook_result

        if not self._built:
            self._build_once(*inputs, **kwargs)

            self._built = True

        if in_profiler_mode():
            with profiler.RecordEvent(
                self.__class__.__name__, profiler.TracerEventType.Forward
            ):
                outputs = self.forward(*inputs, **kwargs)
        else:
            with name_struct(self.__class__.__name__):
                outputs = self.forward(*inputs, **kwargs)

        for forward_post_hook in self._forward_post_hooks.values():
            hook_result = forward_post_hook(self, inputs, outputs)
            if hook_result is not None:
                outputs = hook_result

        return outputs

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        if (
            (not in_to_static_mode())
            and (not self._forward_pre_hooks)
            and (not self._forward_post_hooks)
            and (self.__class__._build_once is Layer._build_once or self._built)
            and in_dygraph_mode()
            and (not in_profiler_mode() or in_sot_simulation_mode())
        ):
            return self.forward(*inputs, **kwargs)
        else:
            return self._dygraph_call_func(*inputs, **kwargs)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        Parameters:
            *inputs(tuple): unpacked tuple arguments
            **kwargs(dict): unpacked dict arguments
        """
        raise NotImplementedError

    def backward(self, *inputs: Any) -> Any:
        raise ValueError("Layer shouldn't implement backward")

    def add_sublayer(self, name: str, sublayer: Layer) -> Layer:
        """

        Adds a sub Layer instance.

        Added sublayer can be accessed by self.name

        Parameters:
            name(str): name of this sublayer.
            sublayer(Layer): an instance of Layer.
        Returns:
            Layer, the sublayer passed in.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> class MySequential(paddle.nn.Layer):
                ...     def __init__(self, *layers):
                ...         super().__init__()
                ...         if len(layers) > 0 and isinstance(layers[0], tuple):
                ...             for name, layer in layers:
                ...                 self.add_sublayer(name, layer)
                ...         else:
                ...             for idx, layer in enumerate(layers):
                ...                 self.add_sublayer(str(idx), layer)
                ...
                ...     def forward(self, input):
                ...         for layer in self._sub_layers.values():
                ...             input = layer(input)
                ...         return input
                ...
                >>> fc1 = paddle.nn.Linear(10, 3)
                >>> fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
                >>> model = MySequential(fc1, fc2)
                >>> for prefix, layer in model.named_sublayers():
                ...     print(prefix, layer)
                0 Linear(in_features=10, out_features=3, dtype=float32)
                1 Linear(in_features=3, out_features=10, dtype=float32)
        """
        assert isinstance(sublayer, Layer) or sublayer is None

        self._sub_layers[name] = sublayer
        return sublayer

    def add_parameter(self, name: str, parameter: Tensor) -> Tensor:
        """Adds a Parameter instance.

        Added parameter can be accessed by self.name

        Parameters:
            name(str): name of this sublayer.
            parameter(Parameter): an instance of Parameter.
        Returns:
            Parameter, the parameter passed in.
        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(100)

                >>> class MyLayer(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = paddle.nn.Linear(1, 1)
                ...         w_tmp = self.create_parameter([1,1])
                ...         self.add_parameter("w_tmp", w_tmp)
                ...
                ...     def forward(self, input):
                ...         return self._linear(input)
                ...
                >>> mylayer = MyLayer()
                >>> for name, param in mylayer.named_parameters():
                ...     print(name, param)
                w_tmp Parameter containing:
                Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[-1.01448846]])
                _linear.weight Parameter containing:
                Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[0.18551230]])
                _linear.bias Parameter containing:
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0.])
        """
        if '_parameters' not in self.__dict__:
            raise RuntimeError("super().__init__() should be called firstly.")
        elif not isinstance(name, str):
            raise TypeError(
                f"The name of parameter should be a string, but received {type(name).__name__}."
            )
        elif '.' in name:
            raise KeyError(
                "The name of parameter can not contain `.`, "
                "because when you access the newly added parameter in the "
                "form of `self.**.**`, it will cause AttributeError."
            )
        elif name == '':
            raise KeyError("The name of parameter can not be empty.")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"The parameter '{name}' already exists.")
        elif parameter is not None and not isinstance(
            parameter, (framework.Parameter, paddle.pir.Value)
        ):
            raise TypeError(
                f"The parameter to be added should be a Parameter, but received {type(parameter).__name__}."
            )
        else:
            if parameter is None:
                self._parameters[name] = None

            if len(self._loaddict_holder) > 0:
                assert (
                    parameter.name in self._loaddict_holder
                ), f"Parameter not found, Can't not find [ {parameter.name} ] in state_dict"

                parameter.set_value(self._loaddict_holder[parameter.name])

            self._parameters[name] = parameter
        return parameter

    def _set_op_attrs(self, attrs):
        """
        Add customized attribute while append_op. In case of quantization, we want to save
        some attributes into op_desc while exporting inference model by @to_static.

        Arguments:
            attrs(dict): customized attributes that will be added into op_descs.

        NOTE: The interface is only exposed to developers.
        """

        def is_already_registered(is_pre_hook):
            layers_hooks = (
                self._forward_pre_hooks
                if is_pre_hook
                else self._forward_post_hooks
            )
            candidate_hook = (
                record_program_ops_pre_hook
                if is_pre_hook
                else set_op_customized_attrs_post_hook
            )

            already_registered = False
            if layers_hooks:
                last_key = next(reversed(layers_hooks))
                already_registered = layers_hooks[last_key] == candidate_hook

            return already_registered

        if not isinstance(attrs, dict):
            raise TypeError(
                f"attrs should be type(dict), but received {type(attrs).__name__}"
            )

        # NOTE: Overwrite behavior for same key.
        self._customized_attrs.update(attrs)

        if not is_already_registered(is_pre_hook=True):
            pre_hook_helper = self.register_forward_pre_hook(
                record_program_ops_pre_hook
            )
            assert len(self._op_recorder.hooks) == 0
            self._op_recorder.hooks = [pre_hook_helper]

        # manually register post_hook to ensure it is inserted into the head.
        if not is_already_registered(is_pre_hook=False):
            post_hook_helper = self.register_forward_post_hook(
                set_op_customized_attrs_post_hook
            )
            if len(self._forward_post_hooks) > 1:
                self._forward_post_hooks.move_to_end(
                    post_hook_helper._hook_id, last=False
                )

            assert len(self._op_recorder.hooks) == 1

            # hooks that need to be removed once we finish executing them.
            self._op_recorder.hooks.append(post_hook_helper)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in self._parameters:
                if in_to_static_mode():
                    return _convert_into_variable(self._parameters[name])
                return self._parameters[name]
        if '_sub_layers' in self.__dict__:
            _sub_layers = self.__dict__['_sub_layers']
            if name in self._sub_layers:
                return self._sub_layers[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                if in_to_static_mode():
                    return _convert_into_variable(_buffers[name])
                return _buffers[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        def _remove_if_exist(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(
            value, paddle.jit.dy2static.program_translator.StaticFunction
        ):
            object.__setattr__(self, name, value)
            value._patched_name = name
            return
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
        params = self.__dict__.get('_parameters', None)
        if isinstance(value, framework.Parameter):
            if params is None:
                raise ValueError("super().__init__() should be called first")
            if len(self._loaddict_holder) > 0:
                assert (
                    value.name in self._loaddict_holder
                ), f"Parameter not found, Can't not find [ {value.name} ] in state_dict"

                value.set_value(self._loaddict_holder[value.name])

            _remove_if_exist(self.__dict__, self._buffers, self._sub_layers)
            params[name] = value
        elif (
            isinstance(value, paddle.pir.Value)
            and value.get_defining_op().name() == 'builtin.parameter'
        ):
            if params is None:
                raise ValueError("super().__init__() should be called first")
            _remove_if_exist(self.__dict__, self._buffers, self._sub_layers)
            params[name] = value
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"assignment to parameter '{name}' should be of type Parameter or None, but got '{type(value).__name__}'"
                )
            params[name] = None
        else:
            layers = self.__dict__.get('_sub_layers', None)
            if isinstance(value, Layer):
                if layers is None:
                    raise ValueError(
                        "super().__init__() should be called first"
                    )

                _remove_if_exist(self.__dict__, self._parameters, self._buffers)
                layers[name] = value
            elif layers is not None and name in layers:
                if value is not None:
                    raise TypeError(
                        f"assignment to sublayer '{name}' should be of type Layer or None, but got '{type(value).__name__}'"
                    )
                layers[name] = None
            else:
                _buffers = self.__dict__.get('_buffers', None)
                if isinstance(value, core.eager.Tensor):
                    if _buffers is None:
                        raise ValueError(
                            "super().__init__() should be called first"
                        )
                    _remove_if_exist(
                        self.__dict__, self._parameters, self._sub_layers
                    )
                    # Set persistable=False by default. Only `register_buffer` can
                    # add a persistable buffer.
                    if name not in self._buffers:
                        self._non_persistable_buffer_names_set.add(name)
                    if not value.name:
                        value.name = unique_name.generate('_buffers_' + name)
                    _buffers[name] = value
                elif _buffers is not None and name in _buffers:
                    # Note(Aurelius84): In Dy2stat, the value of the Buffer may be modified in
                    # decorated function, such as `self.buffer = new_tensor`. So we update its
                    # value via `assign`.
                    if type(value) == framework.Variable or isinstance(
                        value, paddle.pir.Value
                    ):
                        from paddle import assign

                        # Note(zhhsplendid): the condition below happens in PaddleGan model,
                        # but should all non-Variable _buffers[name] be re-assign? We
                        # should consider it in the future. I current wrote this as
                        # conservative code.
                        if in_to_static_mode() and _buffers[name] is None:
                            raise RuntimeError(
                                f'In Dy2stat, self.{name} is a buffer and self.{name} is '
                                f'not allowed to be set to Variable when self.{name} is None.'
                            )
                        elif (
                            _buffers[name] is None
                            or type(getattr(self, name)) == core.eager.Tensor
                        ):
                            _buffers[name] = assign(value)
                        else:
                            assign(value, getattr(self, name))
                    elif value is not None:
                        raise TypeError(
                            f"assignment to buffers '{name}' should be of type core.DenseTensor or None, but got '{type(value).__name__}'"
                        )
                    else:
                        # Assigning None will remove the buffer, but if re-assign a new varBase to it,
                        # it will be remarked as a buffer with same `persistable` attribute.
                        _buffers[name] = None
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._sub_layers:
            del self._sub_layers[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistable_buffer_names_set.discard(name)
        else:
            object.__delattr__(self, name)

    def __dir__(self) -> list[str]:
        """
        Return a list. Get all parameters, buffers(non-parameter tensors), sublayers, method and attr of Layer.

        Examples:
            .. code-block:: python
                >>> import paddle
                >>> import numpy as np

                >>> class Mylayer(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.linear1 = paddle.nn.Linear(10, 10)
                ...         self.linear2 = paddle.nn.Linear(5, 5)
                ...         self.conv2d = paddle.nn.Conv2D(3, 2, 3)
                ...         self.embedding = paddle.nn.Embedding(128, 16)
                ...         self.h_0 = paddle.to_tensor(np.zeros([10, 10]).astype('float32'))
                ...
                >>> mylayer = Mylayer()
                >>> print(dir(mylayer))
                ['__call__', '__class__', '__delattr__', '__dict__', ..., 'training']
        """
        method = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        sublayers = list(self._sub_layers.keys())
        buffers = list(self._buffers.keys())

        keys = method + attrs + parameters + sublayers + buffers

        return keys

    def extra_repr(self) -> str:
        """
        Extra representation of this layer, you can have custom implementation
        of your own layer.
        """
        return ''

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()
        extra_lines = extra_repr.split('\n')
        sublayer_lines = []
        for name, layer in self._sub_layers.items():
            sublayer_str = repr(layer)
            sublayer_str = _addindent(sublayer_str, 2)
            sublayer_lines.append('(' + name + '): ' + sublayer_str)

        final_str = self.__class__.__name__ + '('
        if extra_lines:
            if len(extra_lines) > 1:
                final_str += '\n  ' + '\n  '.join(extra_lines) + '\n'
            elif len(extra_lines) == 1:
                final_str += extra_lines[0]
        if sublayer_lines:
            final_str += '\n  ' + '\n  '.join(sublayer_lines) + '\n'

        final_str += ')'
        return final_str

    def register_state_dict_hook(
        self, hook: _StateDictHook
    ) -> HookRemoveHelper:
        hook_remove_helper = HookRemoveHelper(self._state_dict_hooks)
        self._state_dict_hooks[hook_remove_helper._hook_id] = hook
        return hook_remove_helper

    def _obtain_parameters_buffers(
        self,
        destination: _StateDict | None = None,
        include_sublayers: bool = True,
        structured_name_prefix: str = "",
    ) -> _StateDict:
        """
        The difference from state_dict() is that state_dict_hook will not be called,
        but the original types of parameters and buffers will be maintained.
        """
        if destination is None:
            destination = OrderedDict()
        for name, data in self._parameters.items():
            if data is not None:
                destination[structured_name_prefix + name] = data
        for name, buffer in self._buffers.items():
            if (
                buffer is not None
                and name not in self._non_persistable_buffer_names_set
            ):
                destination[structured_name_prefix + name] = buffer

        if include_sublayers:
            for layer_name, layer_item in self._sub_layers.items():
                if layer_item is not None:
                    destination_temp = destination.copy()
                    destination_temp.update(
                        layer_item._obtain_parameters_buffers(
                            destination_temp,
                            include_sublayers,
                            structured_name_prefix + layer_name + ".",
                        )
                    )
                    destination = destination_temp
        return destination

    def _state_dict_impl(
        self,
        destination: _StateDict | None = None,
        include_sublayers: bool = True,
        structured_name_prefix: str = "",
        include_non_persistable_buffer: bool = False,
        use_hook: bool = True,
        keep_vars: bool = True,
    ) -> _StateDict:
        """
        Get all parameters and persistable buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None.
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True.
            include_non_persistable_buffer(bool, optional): If true, include non persistable buffers of current layer and its sub-layers, it is used in pure fp16 and jit.save. Default: False.
            use_hook(bool, optional) : If true, the operations contained in _state_dict_hooks will be appended to the destination. Default: True.
            keep_vars(bool, optional) : If false, the returned tensors in the state dict are detached from autograd. Default: True.
        """

        if destination is None:
            destination = OrderedDict()
        for name, data in self._parameters.items():
            if data is not None:
                destination[structured_name_prefix + name] = (
                    data if keep_vars else data.detach()
                )
        for name, buffer in self._buffers.items():
            if not include_non_persistable_buffer:
                if (
                    buffer is not None
                    and name not in self._non_persistable_buffer_names_set
                ):
                    destination[structured_name_prefix + name] = (
                        buffer if keep_vars else buffer.detach()
                    )
            else:
                if buffer is not None:
                    destination[structured_name_prefix + name] = (
                        buffer if keep_vars else buffer.detach()
                    )

        if include_sublayers:
            for layer_name, layer_item in self._sub_layers.items():
                if layer_item is not None:
                    destination_temp = destination.copy()
                    destination_temp.update(
                        layer_item._state_dict_impl(
                            destination_temp,
                            include_sublayers,
                            structured_name_prefix + layer_name + ".",
                            include_non_persistable_buffer,
                            use_hook,
                            keep_vars,
                        )
                    )
                    destination = destination_temp
        if use_hook:
            for state_dict_hook in self._state_dict_hooks.values():
                hook_result = state_dict_hook(destination)
                if hook_result is not None:
                    destination = hook_result

        return destination

    def to_static_state_dict(
        self,
        destination: _StateDict | None = None,
        include_sublayers: bool = True,
        structured_name_prefix: str = "",
        use_hook: bool = True,
        keep_vars: bool = True,
    ) -> _StateDict:
        '''

        Get all parameters and buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None.
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True.
            use_hook(bool, optional) : If true, the operations contained in _state_dict_hooks will be appended to the destination. Default: True.
            keep_vars(bool, optional) : If false, the returned tensors in the state dict are detached from autograd. Default: True.

        Returns:
            dict, a dict contains all the parameters and persistable buffers.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> emb = paddle.nn.Embedding(10, 10)

                >>> state_dict = emb.to_static_state_dict()
                >>> paddle.save( state_dict, "paddle_dy.pdparams")

        '''
        return self._state_dict_impl(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
            include_non_persistable_buffer=True,
            use_hook=use_hook,
            keep_vars=keep_vars,
        )

    def state_dict(
        self,
        destination: _StateDict | None = None,
        include_sublayers: bool = True,
        structured_name_prefix: str = "",
        use_hook: bool = True,
        keep_vars: bool = True,
    ) -> _StateDict:
        '''
        Get all parameters and persistable buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None.
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True.
            use_hook(bool, optional) : If true, the operations contained in _state_dict_hooks will be appended to the destination. Default: True.
            keep_vars(bool, optional) : If false, the returned tensors in the state dict are detached from autograd. Default: True.

        Returns:
            dict: a dict contains all the parameters and persistable buffers.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> emb = paddle.nn.Embedding(10, 10)

                >>> state_dict = emb.state_dict()
                >>> paddle.save(state_dict, "paddle_dy.pdparams")

        '''
        return self._state_dict_impl(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
            include_non_persistable_buffer=False,
            use_hook=use_hook,
            keep_vars=keep_vars,
        )

    @framework.deprecate_stat_dict
    def set_state_dict(
        self,
        state_dict: _StateDict,
        use_structured_name: bool = True,
    ) -> tuple[list[str], list[str]]:
        '''
        Set parameters and persistable buffers from state_dict. All the parameters and buffers will be reset by the tensor in the state_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters and persistable buffers.
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter or buffer name as key.
                                                  Default: True.
        Returns:
            missing_keys(list):A list of str containing the missing keys
            unexpected_keys(list):A list of str containing the unexpected keys

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> emb = paddle.nn.Embedding(10, 10)

                >>> state_dict = emb.state_dict()
                >>> paddle.save(state_dict, "paddle_dy.pdparams")
                >>> para_state_dict = paddle.load("paddle_dy.pdparams")
                >>> emb.set_state_dict(para_state_dict)

        '''
        missing_keys = []
        match_keys = set()
        unexpected_keys = []

        def _check_match(key, param):
            state = state_dict.get(key, None)
            if state is None:
                missing_keys.append(key)
                raise ValueError(f"{key} is not found in the provided dict.")
            if isinstance(state, (dict, list)):
                if len(state) != len(param):
                    missing_keys.append(key)
                    raise ValueError(
                        f"{key} receives the length of {len(state)}, "
                        f"but the expected shape is {len(param)}"
                    )
                else:
                    match_keys.add(key)
                    return param, state
            else:
                state_shape = (
                    state.shape()
                    if inspect.ismethod(state.shape)
                    else state.shape
                )

                if list(state_shape) != list(param.shape):
                    missing_keys.append(key)
                    raise ValueError(
                        f"{key} receives a shape {list(state_shape)}, but the expected shape is {list(param.shape)}."
                    )
                match_keys.add(key)
                return param, state

        matched_param_state = []
        for key, param in self._state_dict_impl(use_hook=False).items():
            if isinstance(param, paddle.Tensor) and not param._is_initialized():
                continue
            key_name = key if use_structured_name else param.name
            try:
                match_res = _check_match(key_name, param)
                matched_param_state.append(match_res)
            except ValueError as err:
                warnings.warn(f"Skip loading for {key}. " + str(err))
        for key in state_dict.keys():
            if key not in match_keys:
                unexpected_keys.append(key)
        if in_dygraph_mode():
            for param, state in matched_param_state:
                param.set_value(state)
        else:

            def _set_var(var, ndarray):
                t = global_scope().find_var(var.name).get_tensor()
                p = t._place()
                if p.is_cpu_place():
                    place = core.CPUPlace()
                elif p.is_cuda_pinned_place():
                    place = core.CUDAPinnedPlace()
                elif p.is_xpu_place():
                    p = core.Place()
                    p.set_place(t._place())
                    place = core.XPUPlace(p.xpu_device_id())
                elif p.is_custom_place():
                    p = core.Place()
                    p.set_place(t._place())
                    place = core.CustomPlace(
                        paddle.device.get_device().split(':')[0],
                        p.custom_device_id(),
                    )
                else:
                    p = core.Place()
                    p.set_place(t._place())
                    place = core.CUDAPlace(p.gpu_device_id())
                t.set(ndarray, place)

            try:
                # restore parameter states
                if in_pir_mode():
                    executor = Executor(
                        paddle.base.framework._current_expected_place_()
                    )._default_executor
                    paddle.base.libpaddle.pir.create_loaded_parameter(
                        [param for param, state in matched_param_state],
                        global_scope(),
                        executor,
                    )
                else:
                    executor = Executor(_get_device())._default_executor
                    core._create_loaded_parameter(
                        [param for param, state in matched_param_state],
                        global_scope(),
                        executor,
                    )
                for param, state in matched_param_state:
                    _set_var(param, state)
            except ValueError as e:
                raise ValueError(
                    "This error might happens in dy2static, while calling 'set_state_dict' dynamically in 'forward', which is not supported. If you only need call 'set_state_dict' once, move it to '__init__'."
                )
            except TypeError as e:
                raise ValueError(
                    "This error might happens in dy2static, while calling 'set_state_dict' dynamically in 'forward', which is not supported. If you only need call 'set_state_dict' once, move it to '__init__'."
                )

        return missing_keys, unexpected_keys

    def to(
        self,
        device: PlaceLike | None = None,
        dtype: DTypeLike | None = None,
        blocking: bool | None = None,
    ) -> Self:
        '''
        Cast the parameters and buffers of Layer by the give device, dtype and blocking.

        Parameters:
            device(str|paddle.CPUPlace()|paddle.CUDAPlace()|paddle.CUDAPinnedPlace()|paddle.XPUPlace()|None, optional): The device of the Layer which want to be stored.
            If None, the device is the same with the original Tensor. If device is string, it can be ``cpu``, ``gpu:x`` and ``xpu:x``, where ``x`` is the
            index of the GPUs or XPUs. Default: None.

            dtype(str|numpy.dtype|paddle.dtype|None, optional): The type of the data. If None, the dtype is the same with the original Tensor. Default: None.

            blocking(bool|None, optional): If False and the source is in pinned memory, the copy will be
              asynchronous with respect to the host. Otherwise, the argument has no effect. If None, the blocking is set True. Default: None.

        Returns:
            self

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(2023)

                >>> linear=paddle.nn.Linear(2, 2)
                >>> linear.weight
                >>> print(linear.weight)
                Parameter containing:
                Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[ 0.89611185,  0.04935038],
                 [-0.58883440,  0.99266374]])

                >>> linear.to(dtype='float64')
                >>> linear.weight
                >>> print(linear.weight)
                Parameter containing:
                Tensor(shape=[2, 2], dtype=float64, place=Place(gpu:0), stop_gradient=False,
                [[ 0.89611185,  0.04935038],
                 [-0.58883440,  0.99266374]])

                >>> linear.to(device='cpu')
                >>> linear.weight
                >>> print(linear.weight)
                Parameter containing:
                Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
                [[ 0.89611185,  0.04935038],
                 [-0.58883440,  0.99266374]])

                >>> # doctest: +REQUIRES(env:GPU)
                >>> linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
                >>> linear.weight
                >>> print(linear.weight)
                Parameter containing:
                Tensor(shape=[2, 2], dtype=float64, place=Place(gpu_pinned), stop_gradient=False,
                [[ 0.89611185,  0.04935038],
                 [-0.58883440,  0.99266374]])

        '''
        return self._to_impl(
            device=device,
            dtype=dtype,
            blocking=blocking,
            include_sublayers=True,
            floating_only=False,
        )

    def _apply(
        self,
        func: Callable[
            [Tensor, PlaceLike | None, DTypeLike | None, bool | None], None
        ],
        device: PlaceLike | None,
        dtype: DTypeLike | None,
        blocking: bool | None,
        include_sublayers: bool = True,
    ) -> None:
        if include_sublayers:
            for layer in self.children():
                layer._apply(func, device, dtype, blocking, include_sublayers)

        for key, param in self._parameters.items():
            if param is not None:
                with no_grad():
                    param_applied = func(param, device, dtype, blocking)

                if param.grad is not None:
                    with no_grad():
                        grad_applied = func(
                            param._grad_ivar(), device, dtype, blocking
                        )

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = func(buf, device, dtype, blocking)

        self._dtype = dtype

    def _transform(
        self,
        t: Tensor,
        device: PlaceLike | None,
        dtype: DTypeLike | None,
        blocking: bool | None,
    ) -> Tensor:
        if device is None:
            device = t.place
        if dtype is None:
            dtype = t.dtype

        if not isinstance(dtype, (VarDesc.VarType, core.DataType)):
            dtype = convert_np_dtype_to_dtype_(dtype)

        # 1. gpu place need to determine whether the memory is sufficient for allocation:
        if t.place.is_gpu_place():
            # for gpu, minimum memory allocation unit is 256 bytes.
            proto_dtype = (
                paddle_type_to_proto_type[dtype]
                if isinstance(dtype, core.DataType)
                else dtype
            )
            size_dtype = core.size_of_dtype(proto_dtype)
            # Note(zhangbo): Paddle GPU minimum memory allocation unit is 256 bytes, waiting_alloc_memory will compute ‘t’ occupied memory space.
            # Coefficient 1.2 is used to avoid OOM that may occur in this critical state when the memory is just enough.
            waiting_alloc_memory = (
                ((np.prod(t.shape) * size_dtype) / 256 + 1) * 256 * 1.2
            )
            gpu_memory_available = core.gpu_memory_available()
            if gpu_memory_available < waiting_alloc_memory:
                # Copy param / Tensor to cpu
                t_used = t._copy_to(
                    paddle.CPUPlace(), blocking
                )  # k-v type will error
                # Release mem of t
                t.value().get_tensor()._clear()
            else:
                t_used = t
        else:
            t_used = t

        # 2. cast param / Tensor to dtype
        if dtype is not None and dtype != t_used.dtype:
            with paddle.base.framework._dygraph_place_guard(place=t_used.place):
                t_casted = t_used.cast(dtype=dtype)
        else:
            t_casted = t_used

        # 3. Copy casted cpu param / Tensor to device
        if device is not None and not t_casted.place._equals(device):
            new_t = t_casted._copy_to(device, blocking)
        else:
            new_t = t_casted

        # 4. share Tensor to origin param / Tensor
        dst_tensor = t.value().get_tensor()
        src_tensor = new_t.value().get_tensor()
        if t._is_initialized():
            dst_tensor._share_data_with(src_tensor)
        else:
            # If the tensor is not initialized, we can't check the memory size.
            dst_tensor._share_data_nocheck_with(src_tensor)

        return t

    def _to_impl(
        self,
        device: PlaceLike | None = None,
        dtype: DTypeLike | None = None,
        blocking: bool | None = None,
        include_sublayers: bool = True,
        floating_only: bool = False,
    ):
        '''
        Cast the parameters and buffers of Layer by the give device, dtype and blocking.

        Parameters:
            device(str|paddle.CPUPlace()|paddle.CUDAPlace()|paddle.CUDAPinnedPlace()|paddle.XPUPlace()|None, optional): The device of the Layer which want to be stored.
            If None, the device is the same with the original Tensor. If device is string, it can be ``cpu``, ``gpu:x`` and ``xpu:x``, where ``x`` is the
            index of the GPUs or XPUs. Default: None.

            dtype(str|numpy.dtype|paddle.dtype|None, optional): The type of the data. If None, the dtype is the same with the original Tensor. Default: None.

            blocking(bool|None, optional): If False and the source is in pinned memory, the copy will be
              asynchronous with respect to the host. Otherwise, the argument has no effect. If None, the blocking is set True. Default: None.

            include_sublayers(bool, optional): If True, deal with self and all sublayers parameters and buffers, if not only deal with self parameters and buffers. Default: True.

            floating_only(bool, optional): If True, only cast all floating point parameters and buffers of Layer by the give device, dtype and blocking.

        Returns:
            self

        '''

        if device is None and dtype is None and blocking is None:
            return self

        if device is not None:
            if isinstance(device, str):
                device = paddle.device._convert_to_place(device)
            elif isinstance(
                device,
                (
                    core.CPUPlace,
                    core.CUDAPlace,
                    core.CUDAPinnedPlace,
                    core.XPUPlace,
                ),
            ):
                pass
            else:
                raise ValueError(
                    "device value error, must be str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace() or paddle.XPUPlace(), but the type of device is "
                    + type(device).__name__
                )

        if blocking is None:
            blocking = True
        else:
            assert isinstance(
                blocking, bool
            ), "blocking value error, must be the True, False or None"

        def transform(t, device, dtype, blocking):
            if floating_only and (not paddle.is_floating_point(t)):
                return t
            return self._transform(t, device, dtype, blocking)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._apply(transform, device, dtype, blocking, include_sublayers)

        self._dtype = dtype
        return self

    def _startup_program(self) -> Program:
        """
        Return startup program containing initialization operations of all parameters.

        NOTE(dev): This is a very low level API and only for inner developer.
        """
        startup_program = paddle.base.Program()
        main_program = paddle.base.Program()
        with paddle.base.program_guard(main_program, startup_program):
            for param in self.parameters():
                param._create_init_op(startup_program.global_block())
        if paddle.framework.use_pir_api():
            return main_program
        else:
            return startup_program

    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict

    def float(
        self, excluded_layers: Layer | Sequence[Layer] | None = None
    ) -> Self:
        '''
        Casts all floating point parameters and buffers to ``float`` data type.

        Parameters:
            excluded_layers(nn.Layer|list|tuple|None, optional): Specify the layers that need to be kept original data type. if excluded_layers is None, casts all floating point parameters and buffers. Default: None.

        Returns:
            Layer: self

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> class Model(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.linear = paddle.nn.Linear(1, 1)
                ...         self.dropout = paddle.nn.Dropout(p=0.5)
                ...
                ...     def forward(self, input):
                ...         out = self.linear(input)
                ...         out = self.dropout(out)
                ...         return out
                ...
                >>> model = Model()
                >>> model.float()
                Model(
                    (linear): Linear(in_features=1, out_features=1, dtype=paddle.float32)
                    (dropout): Dropout(p=0.5, axis=None, mode=upscale_in_train)
                )
        '''

        excluded_layers = [] if excluded_layers is None else excluded_layers

        if isinstance(excluded_layers, type):
            excluded_layers = [excluded_layers]
        elif isinstance(excluded_layers, (list, tuple)):
            excluded_layers = list(excluded_layers)
        else:
            raise TypeError(
                f"excluded_layers should be type nn.Layer or list, but got {type(excluded_layers).__name__}.",
            )

        def layer_trans(layer):
            _layer_trans_dtype(layer, paddle.float32, excluded_layers)

        return self.apply(layer_trans)

    def float16(
        self, excluded_layers: Layer | Sequence[Layer] | None = None
    ) -> Self:
        '''
        Casts all floating point parameters and buffers to ``float16`` data type.


        .. note::
            ``nn.BatchNorm`` does not support ``bfloat16`` weights, so it would not be converted by default.


        Parameters:
           excluded_layers(nn.Layer|list|tuple|None, optional): Specify the layers that need to be kept original data type. if excluded_layers is None, casts all floating point parameters and buffers except ``nn.BatchNorm``. Default: None.

        Returns:
            Layer: self

        Examples:
            .. code-block:: python

                >>> # doctest: +SKIP('Paddle compiled by the user does not support float16, so keep original data type.')
                >>> import paddle

                >>> class Model(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.linear = paddle.nn.Linear(1, 1)
                ...         self.dropout = paddle.nn.Dropout(p=0.5)
                ...
                ...     def forward(self, input):
                ...         out = self.linear(input)
                ...         out = self.dropout(out)
                ...         return out
                ...
                >>> model = Model()
                >>> model.float16()
                Model(
                    (linear): Linear(in_features=1, out_features=1, dtype=float32)
                    (dropout): Dropout(p=0.5, axis=None, mode=upscale_in_train)
                )
        '''

        if paddle.amp.is_float16_supported() is False:
            warnings.warn(
                "Paddle compiled by the user does not support float16, so keep original data type."
            )
            return self

        excluded_layers = (
            [nn.BatchNorm] if excluded_layers is None else excluded_layers
        )

        if isinstance(excluded_layers, type):
            excluded_layers = [excluded_layers]
        elif isinstance(excluded_layers, (list, tuple)):
            excluded_layers = list(excluded_layers)
        else:
            raise TypeError(
                f"excluded_layers should be type nn.Layer or list, but got {type(excluded_layers).__name__}.",
            )

        def layer_trans(layer):
            _layer_trans_dtype(layer, paddle.float16, excluded_layers)

        return self.apply(layer_trans)

    def bfloat16(
        self, excluded_layers: Layer | Sequence[Layer] | None = None
    ) -> Self:
        '''
        Casts all floating point parameters and buffers to ``bfloat16`` data type.


        .. note::
            ``nn.BatchNorm`` does not support ``bfloat16`` weights, so it would not be converted by default.


        Parameters:
            excluded_layers(nn.Layer|list|tuple|None, optional): Specify the layers that need to be kept original data type. if excluded_layers is None, casts all floating point parameters and buffers except ``nn.BatchNorm``. Default: None.

        Returns:
            Layer: self

        Examples:
            .. code-block:: python

                >>> # doctest: +SKIP('bfloat need V100 compile')
                >>> import paddle

                >>> class Model(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.linear = paddle.nn.Linear(1, 1)
                ...         self.dropout = paddle.nn.Dropout(p=0.5)
                ...
                ...     def forward(self, input):
                ...         out = self.linear(input)
                ...         out = self.dropout(out)
                ...         return out
                ...
                >>> model = Model()
                >>> model.bfloat16()
                >>> #UserWarning: Paddle compiled by the user does not support bfloat16, so keep original data type.
                Model(
                    (linear): Linear(in_features=1, out_features=1, dtype=float32)
                    (dropout): Dropout(p=0.5, axis=None, mode=upscale_in_train)
                )
        '''

        if paddle.amp.is_bfloat16_supported() is False:
            warnings.warn(
                "Paddle compiled by the user does not support bfloat16, so keep original data type."
            )
            return self

        excluded_layers = (
            [nn.BatchNorm] if excluded_layers is None else excluded_layers
        )

        if isinstance(excluded_layers, type):
            excluded_layers = [excluded_layers]
        elif isinstance(excluded_layers, (list, tuple)):
            excluded_layers = list(excluded_layers)
        else:
            raise TypeError(
                f"excluded_layers should be type nn.Layer or list, but got {type(excluded_layers).__name__}.",
            )

        def layer_trans(layer):
            _layer_trans_dtype(layer, paddle.bfloat16, excluded_layers)

        return self.apply(layer_trans)
