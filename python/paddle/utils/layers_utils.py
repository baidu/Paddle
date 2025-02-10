#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import typing
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Callable, TypeVar
from uuid import uuid4
from weakref import WeakKeyDictionary

import numpy as np
from typing_extensions import TypeGuard

import paddle
from paddle.pir.core import convert_np_dtype_to_dtype_

from ..base.data_feeder import check_dtype, convert_dtype
from ..base.framework import (
    Block,
    Variable,
    in_dygraph_mode,
)
from ..pir import Value

if typing.TYPE_CHECKING:
    from paddle._typing import NestedStructure, ShapeLike

_T = TypeVar("_T")
_U = TypeVar("_U")


class NotSupportedTensorArgumentError(TypeError):
    def __init__(self, msg, name: str):
        super().__init__(msg)
        self.name = name


def convert_to_list(value, n, name, dtype=int):
    """
    Converts a single numerical type or iterable of numerical
    types into a numerical type list.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the list to be returned.
      name: The name of the argument being validated, e.g. "stride" or
        "filter_size". This is only used to format error messages.
      dtype: the numerical type of the element of the list to be returned.

    Returns:
      A list of n dtypes.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, dtype):
        return [value] * n
    else:
        if isinstance(value, (Variable, paddle.pir.Value)):
            raise NotSupportedTensorArgumentError(
                f"`{name}` required numerical type with `{dtype}`, but received Tensor.",
                name,
            )
        try:
            value_list = list(value)
        except TypeError:
            raise ValueError(
                f"The {name}'s type must be list or tuple. Received: {value}"
            )
        if len(value_list) != n:
            raise ValueError(
                f"The {name}'s length must be {n}. Received: {value}"
            )
        for single_value in value_list:
            if isinstance(single_value, (Variable, paddle.pir.Value)):
                raise NotSupportedTensorArgumentError(
                    f"`{name}` required numerical type with `{dtype}`, but received Tensor.",
                    name,
                )
            try:
                dtype(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"The {name}'s type must be a list or tuple of {n} {dtype}. "
                    + f"Received: {value} including element {single_value} of type {type(single_value)}"
                )
        return value_list


def is_sequence(seq: Any) -> TypeGuard[typing.Sequence[Any] | dict[str, Any]]:
    """
    Whether `seq` is an entry or nested structure
    """
    if isinstance(seq, dict):
        return True
    return isinstance(seq, Sequence) and not isinstance(seq, str)


class UniqueIdMap(WeakKeyDictionary):
    def __init__(self):
        super().__init__(self)
        self.data = defaultdict(uuid4)


uniqueidmap = UniqueIdMap()


def uniqueid(obj):
    if isinstance(obj, str):
        return (hash(obj),)
    elif isinstance(obj, list):
        return (id(obj),)
    else:
        return (uniqueidmap[obj].int,)


def _hash_with_id(*args):
    """
    Return int hash value calculated by id(arg) or tuple(id1,id2, ...).
    """
    assert len(args) > 0
    info = ()
    for v in args:
        info = info + uniqueid(v)
    return hash(info)


def _sorted(dict_):
    """
    Returns a sorted list of the dict keys, with error if keys not sortable.
    """
    try:
        return sorted(dict_.keys())
    except TypeError:
        raise TypeError("nest only supports dicts with sortable keys.")


def _yield_value(iterable):
    if isinstance(iterable, dict):
        for key in _sorted(iterable):
            yield iterable[key]
    else:
        yield from iterable


def _yield_flat_nest(nest):
    for n in _yield_value(nest):
        if is_sequence(n):
            yield from _yield_flat_nest(n)
        else:
            yield n


def to_sequence(nest):
    if is_sequence(nest):
        return nest
    else:
        return [nest]


def flatten(nest: NestedStructure[_T]) -> typing.Sequence[_T]:
    """
        :alias_main: paddle.flatten
        :alias: paddle.flatten,paddle.tensor.flatten,paddle.tensor.manipulation.flatten
        :old_api: paddle.base.layers.flatten

    Traverse all entries in the nested structure and put them into an list.
    """
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]


def _sequence_like(instance, args):
    """
    Convert the sequence `args` to the same type as `instance`.
    """
    if isinstance(instance, dict):
        result = dict(zip(_sorted(instance), args))
        return type(instance)((key, result[key]) for key in instance.keys())
    elif (
        isinstance(instance, tuple)
        and hasattr(instance, "_fields")
        and isinstance(instance._fields, Sequence)
        and all(isinstance(f, str) for f in instance._fields)
    ):
        # This is a namedtuple
        return type(instance)(*args)
    else:
        # Not a namedtuple
        return type(instance)(args)


def _packed_nest_with_indices(structure, flat, index):
    """
    Helper function for pack_sequence_as.
    """
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def pack_sequence_as(structure, flat_sequence):
    """
    Pack a given flattened sequence into a given structure.
    """
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")
    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                f"Structure is a scalar but len(flat_sequence) == {len(flat_sequence)} > 1"
            )
        return flat_sequence[0]
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(
            f"Could not pack sequence. Structure had {len(flat_structure)} elements, but flat_sequence "
            f"had {len(flat_sequence)} elements. Structure: {structure}, flat_sequence: {flat_sequence}."
        )
    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
    return _sequence_like(structure, packed)


def map_structure(
    func: Callable[[_T], _U], *structure: NestedStructure[_T]
) -> NestedStructure[_U]:
    """
    Apply `func` to each entry in `structure` and return a new structure.
    """
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure[0], [func(*x) for x in entries])


def hold_mutable_vars(structure):
    """
    Returns whether structure holds sequence like `list/dict`.
    """
    for s in structure:
        if is_sequence(s):
            return True
    return False


def copy_mutable_vars(structure):
    """
    Returns vars copied from sequence without mutable property.
    """
    flat_structure = copy.copy(flatten(structure))
    return pack_sequence_as(structure, flat_structure)


def _recursive_assert_same_structure(nest1, nest2, check_types, skip_if):
    """
    Helper function for `assert_same_structure`.
    """
    if skip_if is not None and (skip_if(nest1) or skip_if(nest2)):
        return
    is_sequence_nest1 = is_sequence(nest1)
    if is_sequence_nest1 != is_sequence(nest2):
        raise ValueError(
            "The two structures don't have the same nested structure.\n\n"
            f"First structure: {nest1}\n\nSecond structure: {nest2}."
        )
    if not is_sequence_nest1:
        return  # finished checking
    if check_types:
        type_nest1 = type(nest1)
        type_nest2 = type(nest2)
        if type_nest1 != type_nest2:
            raise TypeError(
                "The two structures don't have the same sequence type. First "
                f"structure has type {type_nest1}, while second structure has type {type_nest2}."
            )
        if isinstance(nest1, dict):
            keys1 = set(nest1.keys())
            keys2 = set(nest2.keys())
            if keys1 != keys2:
                raise ValueError(
                    "The two dictionaries don't have the same set of keys. First "
                    f"structure has keys {keys1}, while second structure has keys {keys2}."
                )
    nest1_as_sequence = list(_yield_value(nest1))
    nest2_as_sequence = list(_yield_value(nest2))
    if len(nest1_as_sequence) != len(nest2_as_sequence):
        raise ValueError(
            "The two structures don't have the same number of elements.\n\n"
            f"First structure ({len(nest1_as_sequence)} elements): {nest1}\n\n"
            f"Second structure ({len(nest2_as_sequence)} elements): {nest2}"
        )
    for n1, n2 in zip(nest1_as_sequence, nest2_as_sequence):
        _recursive_assert_same_structure(n1, n2, check_types, skip_if)


def padding_to_same_structure(nest1, nest2, obj=None):
    def _padding_to_same_structure_single(value, obj):
        def change_none_to_obj(x):
            if x is None:
                return obj
            return x

        if is_sequence(value):
            value = pack_sequence_as(
                value, [change_none_to_obj(item) for item in flatten(value)]
            )
        else:
            value = change_none_to_obj(value)
        return value

    nest1 = _padding_to_same_structure_single(nest1, obj)
    nest2 = _padding_to_same_structure_single(nest2, obj)
    return nest1, nest2


def assert_same_structure(nest1, nest2, check_types=True, skip_if=None):
    """
    Confirm two nested structures with the same structure.
    """
    if skip_if is not None and (skip_if(nest1) or skip_if(nest2)):
        return
    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2 and skip_if is None:
        raise ValueError(
            "The two structures don't have the same number of "
            f"elements.\n\nFirst structure ({len_nest1} elements): {nest1}\n\n"
            f"Second structure ({len_nest2} elements): {nest2}"
        )
    _recursive_assert_same_structure(nest1, nest2, check_types, skip_if)


def _is_symmetric_padding(padding, data_dim):
    """
    Check whether padding is symmetrical.
    """
    assert len(padding) == data_dim * 2 or len(padding) == data_dim
    is_sys = True
    if len(padding) == data_dim * 2:
        for i in range(data_dim):
            if padding[i * 2] != padding[i * 2 + 1]:
                is_sys = False
    return is_sys


def _contain_var(list_or_tuple):
    """
    Check whether list or tuple contains variable / Value.
    """
    for item in list_or_tuple:
        if isinstance(item, (Variable, paddle.pir.Value)):
            return True
    return False


def get_int_tensor_list(ele_list, default_dtype='int64'):
    int_tensor_list = []
    for ele in ele_list:
        if isinstance(ele, paddle.pir.Value):
            ele.stop_gradient = True
            if convert_dtype(ele.dtype) != default_dtype:
                ele = paddle.cast(x=ele, dtype=default_dtype)
            if ele.shape != []:
                ele = paddle.reshape(ele, [])
            int_tensor_list.append(ele)
        else:
            temp_out = paddle.tensor.fill_constant(
                shape=[],
                dtype=convert_np_dtype_to_dtype_(np.dtype(default_dtype)),
                value=ele,
                force_cpu=True,
            )
            int_tensor_list.append(temp_out)
    return int_tensor_list


def get_shape_tensor_inputs(inputs, attrs, shape, op_type):
    from paddle.tensor import fill_constant

    def _get_attr_shape(list_shape):
        attr_shape = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                attr_shape.append(-1)
            else:
                attr_shape.append(dim)
        return attr_shape

    def _get_shape_tensor(list_shape):
        shape_tensor_list = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                check_dtype(
                    dim.dtype,
                    'shape[' + str(idx) + ']',
                    ['int32', 'int64'],
                    op_type,
                    f'(When type of shape in {op_type} is list or tuple.)',
                )
                if convert_dtype(dim.dtype) == 'int64':
                    dim = paddle.cast(x=dim, dtype='int32')
                shape_tensor_list.append(dim)
            else:
                temp_out = fill_constant([], 'int32', dim, force_cpu=True)
                shape_tensor_list.append(temp_out)
        return shape_tensor_list

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        check_dtype(
            shape.dtype,
            'shape',
            ['int32', 'int64'],
            'fill_constant',
            f'(When type of shape in {op_type} is Variable.)',
        )
        if convert_dtype(shape.dtype) == 'int64':
            shape = paddle.cast(shape, 'int32')
        inputs["ShapeTensor"] = shape
    elif isinstance(shape, (list, tuple)):
        attrs["shape"] = _get_attr_shape(shape)
        if _contain_var(shape):
            inputs['ShapeTensorList'] = _get_shape_tensor(shape)
    else:
        raise TypeError("Shape only supports Variable, or list, or tuple.")


def _convert_to_tensor_list(old_list, dtype="int32"):
    """
    Converts all elements of a list to Variable / Value.
    """
    from paddle.tensor import fill_constant

    if _contain_var(old_list):
        for ele in old_list:
            if isinstance(ele, paddle.pir.Value):
                dtype = ele.dtype

    new_list_tensor = []
    for ele in old_list:
        if isinstance(ele, (Variable, paddle.pir.Value)):
            ele.stop_gradient = True
            new_list_tensor.append(ele)
        else:
            assert isinstance(ele, int)
            temp_out = fill_constant([1], dtype, ele, force_cpu=True)
            new_list_tensor.append(temp_out)
    return new_list_tensor


def convert_shape_to_list(shape):
    """
    Convert shape(list, tuple, variable) to list in imperative mode
    """
    if isinstance(shape, (list, tuple)):
        shape = [x.item(0) if isinstance(x, Variable) else x for x in shape]
    else:
        if in_dygraph_mode():
            shape = shape.astype(int).tolist()
    return shape


def check_shape(shape):
    """
    Check shape type and shape elements type before passing it to fill_constant
    """
    if isinstance(shape, (Variable, Value)):
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant')
    elif isinstance(shape, (list, tuple)):
        for ele in shape:
            if not isinstance(ele, (Variable, Value)):
                if ele < 0:
                    raise ValueError(
                        "All elements in ``shape`` must be positive when it's a list or tuple"
                    )
                if not isinstance(ele, int):
                    raise TypeError(
                        "All elements in ``shape`` must be integers when it's a list or tuple"
                    )
            else:
                check_dtype(
                    ele.dtype,
                    'element of shape',
                    ['int32', 'int64'],
                    'fill_constant',
                )


def try_get_constant_shape_from_tensor(shape_tensor):
    """Try to get shape from a tensor with constant value.

    For example,

    import paddle
    paddle.enable_static()
    data = paddle.static.data(name="x", shape=[-1, 2], dtype='float32')
    shape = paddle.shape(data)  # shape should be [-1, 2] instead of [-1, -1]
    x = paddle.uniform(shape)
    print(x.shape)
    # (-1, 2)

    """
    if not in_dygraph_mode():
        try:
            if shape_tensor.op is not None:
                generate_op = shape_tensor.op
                if generate_op.type == 'shape':
                    var = shape_tensor.block.vars[
                        generate_op.input_arg_names[0]
                    ]
                    return var.shape
        except:
            return None

        return None


def get_inputs_outputs_in_block(block):
    """
    Returns the inputs and outputs variable used in this block but not
    created in this block.
    """
    assert isinstance(
        block, Block
    ), "input non-Block argument for get_inputs_outputs_in_block."
    assert (
        block.parent_idx != -1
    ), "input block should be a sub-block, not main block."

    # Find input/output var names of all ops in block
    inner_inputs = set()
    inner_outputs = set()
    for op in block.ops:
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if not block.has_var(in_var_name):
                    # variable not created in this block
                    inner_inputs.add(in_var_name)
        for oname in op.output_names:
            for out_var_name in op.output(oname):
                if not block.has_var(out_var_name):
                    # variable not created in this block
                    inner_outputs.add(out_var_name)

    return inner_inputs, inner_outputs


def is_same_shape(shape1: ShapeLike, shape2: ShapeLike) -> bool:
    """
    Check whether two shapes are the same. Deal with the dynamic shape.
    """
    if paddle.in_dynamic_mode():
        return shape1 == shape2

    def is_tensor(x):
        return isinstance(x, (paddle.static.Variable, paddle.pir.Value))

    def is_dynamic_axis(axis):
        return is_tensor(axis) or axis == -1

    if is_tensor(shape1) or is_tensor(shape2):
        return True
    if len(shape1) != len(shape2):
        return False
    for s1, s2 in zip(shape1, shape2):
        if is_dynamic_axis(s1) or is_dynamic_axis(s2):
            continue
        if s1 != s2:
            return False
    return True
