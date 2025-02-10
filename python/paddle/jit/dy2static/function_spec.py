# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import inspect
import weakref

import numpy as np

import paddle
import paddle.pir.core as ir_static
from paddle.base import core
from paddle.base.data_feeder import convert_dtype
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.distributed.auto_parallel.placement_type import (
    to_placements,
)
from paddle.jit.pir_translated_layer import PirTranslatedLayer
from paddle.jit.translated_layer import TranslatedLayer
from paddle.nn.layer import layers

from . import logging_utils
from .utils import (
    func_to_source_code,
    parse_arg_and_kwargs,
    parse_varargs_name,
    type_name,
)

__all__ = []


class FunctionSpec:
    """
    Wrapper class for a function for class method.
    """

    def __init__(self, function, input_spec=None):
        if inspect.ismethod(function):
            self._dygraph_function = function.__func__
            self._class_instance = weakref.ref(function.__self__)
        else:
            self._dygraph_function = function
            self._class_instance = None
        if input_spec is None:
            self._input_spec = None
            self._flat_input_spec = None
        else:
            self._input_spec = self._verify_input_spec(input_spec)
            self._flat_input_spec = paddle.utils.flatten(self._input_spec)

        # parse full argument names list.
        self._arg_names, self._default_kwargs = parse_arg_and_kwargs(function)
        # parse *args
        self.varargs_name = parse_varargs_name(function)
        if self.varargs_name is not None and isinstance(
            getattr(function, '__self__', None),
            (TranslatedLayer, PirTranslatedLayer),
        ):
            self._arg_names += function.__self__._input_args_names

    def unified_args_and_kwargs(self, args, kwargs):
        """
        Moves kwargs with default value into arguments list to keep `args` contain the same length
        value as function definition.

        For example:

            Given function definition: `def foo(x, a=1, b=2)`,
            when calling it by `foo(23)`, the args is `[23]`, kwargs is `{a=1, b=2}`.
            In this function, it will return args with `[23, 1, 2]`, kwargs with `{}`

        Args:
            args(tuple): tuple of input arguments value of decorated function.
            kwargs(dict): dict of input keyword arguments value of decorated function.

        Return:
            New arguments tuple containing default kwargs value.
        """
        if len(self._arg_names) < len(args):
            error_msg = f"The decorated function `{self.dygraph_function.__name__}` requires {len(self._arg_names)} arguments: {self._arg_names}, but received {len(args)} with {args}."
            if args and inspect.isclass(args[0]):
                error_msg += "\n\tMaybe the function has more than one decorator, we don't support this for now."
                raise NotImplementedError(error_msg)
            else:
                raise ValueError(error_msg)

        args = list(args)

        for i in range(len(args), len(self._arg_names)):
            arg_name = self._arg_names[i]
            if arg_name in kwargs:
                args.append(kwargs[arg_name])
                del kwargs[arg_name]
            else:
                if arg_name not in self._default_kwargs:
                    raise ValueError(
                        f"`{self.dygraph_function.__name__}()` requires `{arg_name}` arguments, but not found in input `args`: {args} and `kwargs`: {kwargs}."
                    )
                args.append(self._default_kwargs[arg_name])

        return tuple(args), kwargs

    def args_to_input_spec(self, args, kwargs):
        """
        Converts input arguments into InputSpec.

        1. If specific input_spec, use them to construct feed layers.
        2. If input_spec is None, consider all Tensor and Numpy.ndarray as feed layers

        Args:
            args(tuple): tuple of input arguments value of function containing default kwargs value.
            kwargs(dict): kwargs arguments received by **kwargs.

        Return:
            Same nest structure with args and kwargs by replacing value with InputSpec.
        """

        args_with_spec = []
        kwargs_with_spec = []
        if self._input_spec is not None:
            # Note: Because the value type and length of `kwargs` is uncertain.
            # So we don't support to deal this case while specifying `input_spec` currently.
            if kwargs:
                raise ValueError(
                    f"{self.dygraph_function.__name__} got unexpected keyword arguments: {kwargs}. Cannot trace the function when `input_spec` is specified."
                )

            # Note: The length of `input_spec` can be greater than `args`,
            # because `args` may contains non-tensor value merged form `kwargs`
            # after `unified_args_and_kwargs`.
            if len(args) < len(self._input_spec):
                raise ValueError(
                    f"Requires len(arguments) >= len(input_spec), but received len(args):{len(args)} < len(InputSpec): {len(self._input_spec)}"
                )

            # replace argument with corresponding InputSpec.
            args_with_spec = convert_to_input_spec(args, self._input_spec)
        else:
            args_with_spec = _replace_to_input_spec_with_new_name(
                args, self._arg_names
            )
            kwarg_names = ["kwargs." + key for key in kwargs.keys()]
            kwargs_list_with_spec = _replace_to_input_spec_with_new_name(
                list(kwargs.values()), kwarg_names
            )
            kwargs_with_spec = {
                key: kwargs_list_with_spec[idx]
                for idx, key in enumerate(kwargs)
            }

        # If without specifying name in input_spec, add default name
        # according to argument name from decorated function.
        args_with_spec = replace_spec_empty_name(
            self._arg_names, args_with_spec
        )

        return args_with_spec, kwargs_with_spec

    @switch_to_static_graph
    def pir_to_static_inputs_with_spec(self, input_with_spec, main_program):
        """
        Constructs feed layer by inputs with InputSpec information for main program.

        Args:
            input_with_spec(tuple): input arguments by replacing argument with InputSpec.
            main_program(Program): main program for inserting feed layer.
        """
        flat_input_spec = paddle.utils.flatten(input_with_spec)

        # NOTE(zhangbo): Why do we need function_args and program_inputs: The primary function of this module is to construct the corresponding DataOp based on the inputSpec. The output of DataOp serves as the input to the Program and will also become the arguments for subsequent Static functions to construct the static graph. When the input is a DistributedInputSpec, the shard_tensor operation will be performed on the output of DataOp to obtain the corresponding distributed Value. The input to the Program will still be the output of DataOp, but in this case, the arguments for the Static functions will be the output of shard_tensor.
        function_args = []
        program_inputs = []
        with ir_static.program_guard(main_program):
            for i, var_spec in enumerate(flat_input_spec):
                if isinstance(var_spec, paddle.static.InputSpec):
                    stop_gradient = getattr(var_spec, 'stop_gradient', False)
                    feed_value = paddle.static.input.data(
                        name=var_spec.name or f"feed_{i}",
                        shape=var_spec.shape,
                        dtype=convert_dtype(var_spec.dtype),
                    )
                    feed_value.stop_gradient = stop_gradient

                    # warp dist tensor
                    from paddle.distributed.auto_parallel.static.dist_input_spec import (
                        DistributedInputSpec,
                    )

                    if isinstance(var_spec, DistributedInputSpec):
                        placements = to_placements(
                            var_spec.dims_mapping, var_spec
                        )
                        dist_feed_value = paddle._pir_ops.shard_tensor(
                            feed_value, var_spec.mesh, placements
                        )
                        function_args.append(dist_feed_value)
                    else:
                        function_args.append(feed_value)
                else:
                    feed_value = var_spec
                    function_args.append(feed_value)

                program_inputs.append(feed_value)

        return paddle.utils.pack_sequence_as(
            input_with_spec, function_args
        ), paddle.utils.pack_sequence_as(input_with_spec, program_inputs)

    @switch_to_static_graph
    def to_static_inputs_with_spec(self, input_with_spec, main_program):
        """
        Constructs feed layer by inputs with InputSpec information for main program.

        Args:
            input_with_spec(tuple): input arguments by replacing argument with InputSpec.
            main_program(Program): main program for inserting feed layer.
        """
        flat_input_spec = paddle.utils.flatten(input_with_spec)

        inputs = []
        block = main_program.global_block()
        for i, var_spec in enumerate(flat_input_spec):
            if isinstance(var_spec, paddle.static.InputSpec):
                stop_gradient = getattr(var_spec, 'stop_gradient', False)
                feed_layer = block.create_var(
                    # TODO(Aurelius84): consider a more elegant way to name this
                    name=var_spec.name or f"feed_{i}",
                    shape=var_spec.shape,
                    dtype=var_spec.dtype,
                    is_data=True,
                    need_check_feed=False,
                    stop_gradient=stop_gradient,
                )
                # warp dist tensor
                from paddle.distributed.auto_parallel.static.dist_input_spec import (
                    DistributedInputSpec,
                )
                from paddle.distributed.auto_parallel.static.dist_tensor import (
                    DistributedTensor,
                )

                if isinstance(var_spec, DistributedInputSpec):
                    from paddle.distributed.auto_parallel.static.dist_context import (
                        get_default_distributed_context,
                    )

                    default_dist_ctx = get_default_distributed_context()
                    dist_tensor = DistributedTensor(feed_layer)
                    dist_tensor.dist_attr.process_mesh = var_spec.mesh
                    dist_tensor.dist_attr.dims_mapping = var_spec.dims_mapping
                    dist_tensor.dist_attr.mark_annotated("process_mesh")
                    dist_tensor.dist_attr.mark_annotated("dims_mapping")
                    default_dist_ctx.add_dist_tensor_for_program(dist_tensor)
            else:
                feed_layer = var_spec

            inputs.append(feed_layer)

        return paddle.utils.pack_sequence_as(input_with_spec, inputs)

    def _verify_input_spec(self, input_spec):
        """
        Verifies the `input_spec` and its element type is valid.
        """
        if not isinstance(input_spec, (tuple, list)):
            raise TypeError(
                f"The type(input_spec) should be one of (tuple, list), but received {type_name(input_spec)}."
            )

        return tuple(input_spec)

    def __repr__(self):
        return "function: {}({}), input_spec: {}".format(
            self.dygraph_function.__name__,
            ','.join(self._arg_names),
            self._input_spec,
        )

    @property
    def class_instance(self):
        if self._class_instance is None:
            return None
        if self._class_instance() is None:
            raise RuntimeError(
                "The instance of class has been deleted, please re-create the instance."
            )
        return self._class_instance()

    @property
    def dygraph_function(self):
        if self.class_instance is not None:
            return self._dygraph_function.__get__(self.class_instance)
        else:
            return self._dygraph_function

    @property
    def args_name(self):
        return self._arg_names

    @property
    def input_spec(self):
        return self._input_spec

    @property
    def flat_input_spec(self):
        return self._flat_input_spec

    @property
    def code(self):
        return func_to_source_code(self.dygraph_function)


def get_parameters(layer_instance, include_sublayer=True):
    """
    Returns parameters of decorated layers. If set `include_sublayer` True,
    the parameters created in sub layers will be added.
    """
    params = collections.OrderedDict()
    if layer_instance is not None:
        if isinstance(layer_instance, layers.Layer):
            if include_sublayer:
                params = layer_instance.parameters()
                names = [p.name for p in params]
                params = collections.OrderedDict(zip(names, params))
            else:
                params = layer_instance._parameters
        else:
            raise TypeError(
                f"Type of `layer_instance` should be nn.Layer, but received {type_name(layer_instance)}"
            )

    return params


def get_buffers(layer_instance, include_sublayer=True):
    """
    Returns Variable buffers of decorated layers. If set `include_sublayer` True,
    the Variable buffers created in sub layers will be added.
    """
    buffers = collections.OrderedDict()
    if layer_instance is not None:
        if isinstance(layer_instance, layers.Layer):
            if include_sublayer:
                buffers = layer_instance.buffers()
                names = [buffer.name for buffer in buffers]
                buffers = collections.OrderedDict(zip(names, buffers))
            else:
                buffers = layer_instance._buffers
        else:
            raise TypeError(
                f"Type of `layer_instance` should be nn.Layer, but received {type_name(layer_instance)}"
            )
    return buffers


def _replace_value_with_input_spec(args):
    from paddle.distributed.auto_parallel.static.dist_input_spec import (
        DistributedInputSpec,
    )

    args_with_spec = []
    for idx, input_var in enumerate(paddle.utils.flatten(args)):
        if isinstance(input_var, np.ndarray):
            input_var = paddle.static.InputSpec.from_numpy(input_var)
            input_var.stop_gradient = True
        elif isinstance(input_var, core.eager.Tensor):
            stop_gradient = input_var.stop_gradient
            if input_var.is_dist():
                input_var = DistributedInputSpec.from_dtensor(input_var)
            else:
                input_var = paddle.static.InputSpec.from_tensor(input_var)
            input_var.stop_gradient = stop_gradient
        elif isinstance(
            input_var, (paddle.base.framework.Variable, paddle.pir.Value)
        ):
            stop_gradient = input_var.stop_gradient
            if input_var.is_dist():
                mesh = input_var.dist_attr().process_mesh
                placements = to_placements(
                    input_var.dist_attr().dims_mapping, mesh
                )
                input_var = DistributedInputSpec(
                    input_var.shape,
                    dtype=input_var.dtype,
                    name=input_var.name,
                    mesh=mesh,
                    placements=placements,
                    local_shape=input_var._local_shape,
                )
            else:
                input_var = paddle.static.InputSpec(
                    input_var.shape, input_var.dtype, input_var.name
                )
            input_var.stop_gradient = stop_gradient

        args_with_spec.append(input_var)
    args_with_spec = paddle.utils.pack_sequence_as(args, args_with_spec)
    return args_with_spec


def _replace_to_input_spec_with_new_name(args, arg_names):
    assert len(args) == len(arg_names)
    order_digit = len(str(len(arg_names) - 1))
    args_with_spec = []
    for order, (arg, name_prefix) in enumerate(zip(args, arg_names)):
        index = 0
        for idx, origin_input in enumerate(paddle.utils.flatten(arg)):
            if isinstance(origin_input, np.ndarray):
                input_var = paddle.static.InputSpec.from_numpy(origin_input)
                input_var.stop_gradient = True
            elif isinstance(origin_input, core.eager.Tensor):
                stop_gradient = origin_input.stop_gradient
                input_var = paddle.static.InputSpec.from_tensor(origin_input)
                input_var.stop_gradient = stop_gradient
            elif isinstance(origin_input, paddle.base.framework.Variable):
                stop_gradient = origin_input.stop_gradient
                input_var = paddle.static.InputSpec(
                    origin_input.shape, origin_input.dtype, origin_input.name
                )
                input_var.stop_gradient = stop_gradient
            else:
                input_var = origin_input

            if isinstance(
                origin_input,
                (
                    np.ndarray,
                    core.eager.Tensor,
                    paddle.base.framework.Variable,
                ),
            ):
                input_var.name = f"_jst.{str(order).zfill(order_digit)}.{name_prefix}.{index}"
                index += 1
            args_with_spec.append(input_var)
    args_with_spec = paddle.utils.pack_sequence_as(args, args_with_spec)
    return args_with_spec


def convert_to_input_spec(inputs, input_spec):
    """
    Replaces tensor in structured `inputs` by InputSpec in `input_spec`.

    Args:
        inputs(list|dict): nested structure list or dict.
        input_spec(list|dict): same nested structure list or dict as inputs.


    Return:
        Same structure with inputs by replacing the element with specified InputSpec.
    """

    def check_type_and_len(input, spec, check_length=False):
        if type(input) is not type(spec):
            raise TypeError(
                f'type(input) should be {type(spec)}, but received {type(input)}.'
            )
        if check_length and len(input) < len(spec):
            raise ValueError(
                f'Requires len(inputs) >= len(input_spec), but received len(inputs):{len(inputs)} < len(input_spec):{len(input_spec)}'
            )

    if isinstance(input_spec, (tuple, list)):
        input_with_spec = []
        check_type_and_len(inputs, input_spec, True)

        for i, spec in enumerate(input_spec):
            out_spec = convert_to_input_spec(inputs[i], spec)
            input_with_spec.append(out_spec)

        # Note: If the rest inputs contain tensor or numpy.ndarray
        # without specific InputSpec, raise warning.
        if len(inputs) > len(input_spec):
            for rest_input in inputs[len(input_spec) :]:
                if isinstance(rest_input, (core.eager.Tensor, np.ndarray)):
                    logging_utils.warn(
                        f"The inputs contain `{type_name(rest_input)}` without specifying InputSpec, its shape and dtype will be treated immutable. "
                        "Please specific InputSpec information in `@to_static` if you expect them as mutable inputs."
                    )
        input_with_spec.extend(inputs[len(input_spec) :])

        return input_with_spec
    elif isinstance(input_spec, dict):
        input_with_spec = {}
        check_type_and_len(inputs, input_spec, True)
        for name, input in inputs.items():
            if name in input_spec:
                input_with_spec[name] = convert_to_input_spec(
                    input, input_spec[name]
                )
            else:
                input_with_spec[name] = input
        return input_with_spec
    elif isinstance(input_spec, paddle.static.InputSpec):
        """we compare input_spec with real_input_spec constructed from arguments."""
        real_spec = _replace_value_with_input_spec([inputs])[0]
        if not isinstance(real_spec, paddle.static.InputSpec):
            raise RuntimeError(
                f"Give input spec into a non-tensorable arguments `{inputs}`."
            )
        real_spec.name = input_spec.name
        if spec_greater(input_spec, real_spec):
            # change shape but keep the others (stop_gradient / dtype) .
            real_spec.shape = input_spec.shape
        else:
            logging_utils.warn(
                f"input spec is not compatible with real inputs. input_spec: {input_spec} , real_spec: {real_spec} "
            )
        return real_spec
    else:
        # NOTE(Aurelius84): Support non-Tensor type as input spec info
        return input_spec


def replace_spec_empty_name(args_name, input_with_spec):
    """
    Adds default name according to argument name from decorated function
    if without specifying InputSpec.name

    The naming rule are as followed:
        1. If InputSpec.name is not None, do nothing.
        2. If each argument `x` corresponds to an InputSpec, using the argument name like `x`
        3. If the arguments `inputs` corresponds to a list(InputSpec), using name like `inputs_0`, `inputs_1`
        4. If the arguments `input_dic` corresponds to a dict(InputSpec), using key as name.

    For example:

        # case 1: foo(x, y)
        foo = to_static(foo, input_spec=[InputSpec([None, 10]), InputSpec([None])])
        print([in_var.name for in_var in foo.inputs])  # [x, y]

        # case 2: foo(inputs) where inputs is a list
        foo = to_static(foo, input_spec=[[InputSpec([None, 10]), InputSpec([None])]])
        print([in_var.name for in_var in foo.inputs])  # [inputs_0, inputs_1]

        # case 3: foo(inputs) where inputs is a dict
        foo = to_static(foo, input_spec=[{'x': InputSpec([None, 10]), 'y': InputSpec([None])}])
        print([in_var.name for in_var in foo.inputs])  # [x, y]
    """
    input_with_spec = list(input_with_spec)
    candidate_arg_names = args_name[: len(input_with_spec)]

    for i, arg_name in enumerate(candidate_arg_names):
        input_spec = input_with_spec[i]
        input_with_spec[i] = _replace_spec_name(arg_name, input_spec)

    return input_with_spec


def _replace_spec_name(name, input_spec):
    """
    Replaces InputSpec.name with given `name` while not specifying it.
    """
    if isinstance(input_spec, paddle.static.InputSpec):
        if input_spec.name is None:
            input_spec.name = name
        return input_spec
    elif isinstance(input_spec, (list, tuple)):
        processed_specs = []
        for i, spec in enumerate(input_spec):
            new_name = f"{name}_{i}"
            processed_specs.append(_replace_spec_name(new_name, spec))
        return processed_specs
    elif isinstance(input_spec, dict):
        processed_specs = {}
        for key, spec in input_spec.items():
            processed_specs[key] = _replace_spec_name(key, spec)
        return processed_specs
    else:
        return input_spec


def _hash_spec_names(args_specs, kwargs_specs):
    """
    Generator hash spec with args/kwargs InputSpec names.
    Consider the following InputSpecs with same shape/dtype except for name:
      1. [InputSpec([3,3], 'float32', 'x'), InputSpec([3,3], 'float32', 'x')]
      2. [InputSpec([3,3], 'float32', 'x'), InputSpec([3,3], 'float32', 'y')]
    Under @to_static, we should generate two different program not just one, because
    the former has one input ('x'), but the latter has two input ('x', 'y').
    """
    spec_names = [
        spec.name
        for spec in paddle.utils.flatten(args_specs)
        if isinstance(spec, paddle.static.InputSpec)
    ]
    spec_names += [
        spec.name
        for spec in paddle.utils.flatten(kwargs_specs)
        if isinstance(spec, paddle.static.InputSpec)
    ]
    i, name_ids = 0, {}

    def to_idx(name):
        nonlocal i
        if name not in name_ids:
            name_ids[name] = i
            i += 1
        return name_ids[name]

    value = [to_idx(name) for name in spec_names]

    return tuple(value)


def spec_greater(first, other):
    def _shape_greater(first_shape, second_shape):
        if len(first_shape) != len(second_shape):
            return False
        for first_n, second_n in zip(first_shape, second_shape):
            if first_n != -1 and first_n != second_n:
                return False
        return True

    return _shape_greater(first.shape, other.shape)
