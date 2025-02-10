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

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager

import paddle
from paddle.autograd.py_layer import PyLayerMeta
from paddle.base.data_feeder import convert_dtype
from paddle.base.dygraph.base import _convert_into_variable, in_to_static_mode
from paddle.base.framework import Variable, core, default_main_program
from paddle.framework import use_pir_api
from paddle.jit.utils import OrderedSet
from paddle.pir import Value
from paddle.static.amp.fp16_utils import AmpOptions
from paddle.utils import is_sequence, map_structure

from .py_layer import StaticPyLayer
from .utils import (
    RETURN_NO_VALUE_VAR_NAME,
    Dygraph2StaticException,
    GetterSetterHelper,
    UndefinedVar,
    create_undefined_variable,
)

__all__ = []


def to_static_variable(x, dtype=None):
    '''
    Translate a Python Tensor to PaddlePaddle static graph Tensor
    '''
    if isinstance(x, bool):
        dtype = 'bool' if dtype is None else dtype
        return paddle.full(shape=[], dtype=dtype, fill_value=x)
    if isinstance(x, float):
        dtype = 'float64' if dtype is None else dtype
        return paddle.full(shape=[], dtype=dtype, fill_value=x)
    if isinstance(x, int):
        dtype = 'int64' if dtype is None else dtype
        return paddle.full(shape=[], dtype=dtype, fill_value=x)
    if not use_pir_api() and (isinstance(x, UndefinedVar) or x is None):
        """
        for early return case, we need a variable to represent None, current we use data_layer_not_check.
        """
        return create_undefined_variable()
    if is_sequence(x):
        return map_structure(to_static_variable, x)
    return x


def convert_attr(x, attr):
    # TODO(cleanup-legacy-ir): In PIR mode, the size attr in
    # Value and Tensor are unified. So we don't need to transform
    # the size attr into a method call. The AttributeJstTransformer and
    # convert_attr can be safely removed.
    if (
        isinstance(x, Variable)
        and not isinstance(x, paddle.Tensor)
        and attr == "size"
    ):
        return x.size()
    else:
        return getattr(x, attr)


def convert_load(x):
    # convert dygraph `PyLayer` into StaticPyLayer
    if isinstance(x, PyLayerMeta):
        return StaticPyLayer(x)

    if in_to_static_mode():
        if isinstance(x, paddle.Tensor):
            """
            TODO:(@xiongkun) may run convert_load in dygraph mode, which should be fixed.
            """
            return _convert_into_variable(x)

        # get the new output of the var
        if isinstance(x, Value):

            from paddle.jit.pir_dy2static.parameter_recorder import (
                _global_inplace_map,
            )

            new_var = _global_inplace_map.get(
                paddle.static.default_main_program(), x
            )
            if new_var is not None:
                return new_var

        if isinstance(x, Variable):
            cur_block = default_main_program().current_block()

            from paddle.jit.dy2static.program_translator import (
                ProgramTranslator,
            )

            new_var = ProgramTranslator.get_instance()._inplace_map.get(
                cur_block.program, x.desc.id()
            )
            if new_var is not None:
                return new_var

        if x is paddle.amp.auto_cast and not use_pir_api():
            return convert_auto_cast

    return x


def indexable(x, code=None):
    if isinstance(x, (Variable, Value)):
        return x
    elif hasattr(x, '__iter__'):
        return list(x)
    elif hasattr(x, '__len__') and hasattr(
        x, '__getitem__'
    ):  # used for customed type and non-iterable type.
        return x
    else:
        raise RuntimeError("X can't be convert into indexable.")


def unpack_by_structure(target, structure):
    """unified unpack interface for paddle and python."""
    if isinstance(target, (Variable, Value)):
        return _unpack_by_structure_paddle(target, structure)
    else:
        return _unpack_by_structure_python(target, structure)


def _unpack_by_structure_python(target, structure):
    """TODO(xiongkun): analysis the differences between python and paddle unpack."""
    return _unpack_by_structure_paddle(target, structure)


def _unpack_by_structure_paddle(target, structure):
    if structure == 1:
        return target
    ret = []
    for idx, ele in enumerate(structure):
        if ele == 1:
            ret.append(target[idx])
            continue
        if isinstance(ele, list):
            ret.append(unpack_by_structure(target[idx], ele))
            continue
        raise AssertionError("structure element must be 1 or list")
    return ret


def convert_while_loop(
    cond, body, getter, setter, return_name_ids=None, push_pop_names=None
):
    """
    A function representation of a Python ``while`` statement.

    Args:
        cond(Callable): A callable object that returns a boolean variable to control whether to execute the loop body. It takes ``loop_vars`` as arguments.
        body(Callable): A callable object that returns a tuple or list of variables with the same arguments ``loops_vars`` as ``cond`` .
        get_args(callable): Get all arguments that needed in true_fn and false_fn.
        set_args(callable): Update arguments that modified in trure_fn and false_fn.
        return_name_ids(list[string], optional): the returned names.
        push_pop_names(list[string], optional): the names on which called .append() or .pop().

    Returns:
        A list or tuple of variables which returned by ``body``.
    """

    # NOTE: It may be slower if cond is very expensive, but usually cond is just O(1).
    # If loop_vars is changed during cond callable, then it causes bug, but current logical_and/logical_not/... doesn't change the loop_vars.
    pred = cond()
    if isinstance(pred, (Variable, Value)):
        _run_paddle_while(
            cond, body, getter, setter, return_name_ids, push_pop_names
        )
    else:
        _run_py_while(cond, body, getter, setter)


def _convert_tensor_array_if_necessary(setterhelper, push_pop_names):
    push_pop_vars = setterhelper.get(push_pop_names)
    if push_pop_vars is None:
        return

    def maybe_to_tensor_array(v):
        if isinstance(v, list):
            dtype = (
                paddle.base.libpaddle.DataType.UNDEFINED
                if use_pir_api()
                else "float32"
            )
            return paddle.tensor.create_array(dtype, initialized_list=v)
        else:
            return v

    setterhelper.set(
        push_pop_names, [maybe_to_tensor_array(v) for v in push_pop_vars]
    )


def _run_paddle_while(
    cond, body, getter, setter, return_name_ids, push_pop_names
):
    # NOTE: loop_vars of Paddle op `control_flow.while_loop` must be Paddle Tensors.
    helper = GetterSetterHelper(getter, setter, return_name_ids, push_pop_names)
    _convert_tensor_array_if_necessary(helper, push_pop_names)

    union_name = (
        OrderedSet(return_name_ids) if return_name_ids else OrderedSet()
    ) | (OrderedSet(push_pop_names) if push_pop_names else OrderedSet())
    union_name = list(union_name)

    def new_body_fn(*args):
        """wrap the body() and add return value for `while_loop`
        the args may be differ from getter().
        """
        mutable_loop_vars = args
        helper.set(union_name, mutable_loop_vars)
        body()
        return helper.get(union_name)

    def new_cond_fn(*args):
        """cond is a zero-args function, which is not
        compatible with `while_loop`.
        """
        return cond()

    # UndefinedVar will become data layer not check variable with value=NO_VALUE_MAGIC.
    loop_vars = [
        to_static_variable(var) if not isinstance(var, UndefinedVar) else var
        for var in helper.get(union_name)
    ]
    helper.set(union_name, loop_vars)  # change the non-local var to variable
    # variable maybe modified to inner var. change it into
    from paddle.static.nn import while_loop

    loop_vars = while_loop(new_cond_fn, new_body_fn, loop_vars)
    helper.set(union_name, loop_vars)
    return loop_vars


def _run_py_while(cond, body, getter, setter):
    while True:
        pred = cond()
        if isinstance(pred, (Variable, Value)):
            raise Dygraph2StaticException(
                "python while pred change from bool to variable."
            )
        if not pred:
            break
        body()


def convert_logical_and(x_func, y_func):
    """
    A function representation of a Python ``and`` statement.

    Args:
        x_func(callable): x_func() is the left hand operand of ``and`` operator. x_func() is bool or Tensor.
        y_func(callable): y_func() is the right hand operand of ``and`` operator.  y_func() is bool or Tensor.

    Returns:
        A python bool variable or a bool Tensor.

    NOTE(liym27):
        1) The operands are executed sequentially according to the running logic of Python. So here the arguments
        should be callable.
        2) If the left hand operand is False, the right hand operand should be executed.

        For example:
            a = x > 1 and y < 1
        Transformed code:
            a = paddle.jit.dy2static.convert_logical_and(lambda:x>1, lambda:y<1)

          In `convert_logical_and(lambda:x>1, lambda:y<1)`, `lambda:y<1` must be run after `lambda:x>1`. And
        if `x>1` is False, `y<1` should NOT be run.
    """
    x_value = x_func()
    if not isinstance(x_value, (Variable, Value)):
        return _run_py_logical_and(lambda: x_value, y_func)

    y_value = y_func()
    if not isinstance(y_value, (Variable, Value)):
        return _run_py_logical_and(lambda: y_value, lambda: x_value)

    return _run_paddle_logical_and(x_value, y_value)


def _run_paddle_logical_and(x, y):
    x = cast_bool_if_necessary(x)
    y = cast_bool_if_necessary(y)
    return paddle.logical_and(x, y)


def _run_py_logical_and(x_func, y_func):
    x_value = x_func()
    assert not isinstance(x_value, (Variable, Value))

    # NOTE(liym27):
    #  1. Returns y_func() if x_value is False;
    #  2. If x_value is False, y_func() should not be run.
    return x_value and y_func()


def convert_logical_or(x_func, y_func):
    """
    A function representation of a Python ``or`` statement.

    Args:
        x_func(callable): x_func() is the left hand operand of ``or`` operator. x_func() is bool or Tensor.
        y_func(callable): y_func() is the right hand operand of ``or`` operator.  y_func() is bool or Tensor.

    Returns:
        A python bool variable or a bool Tensor.

    NOTE(liym27):
        1) The operands are executed sequentially according to the running logic of Python. So here the arguments
        should be callable.
        2) If the left hand operand is True, the right hand operand should be executed.

        For example:
            a = x > 1 or y < 1
        Transformed code:
            a = paddle.jit.dy2static.convert_logical_or(lambda:x>1, lambda:y<1)

        In `convert_logical_or(lambda:x>1, lambda:y<1)`, `lambda:y<1` must be run after `lambda:x>1`. And
        if `x>1` is True, `y<1` should NOT be run.
    """
    x_value = x_func()
    if not isinstance(x_value, (Variable, Value)):
        return _run_py_logical_or(lambda: x_value, y_func)

    y_value = y_func()
    if not isinstance(y_value, (Variable, Value)):
        return _run_py_logical_or(lambda: y_value, lambda: x_value)

    return _run_paddle_logical_or(x_value, y_value)


def _run_paddle_logical_or(x, y):
    x = cast_bool_if_necessary(x)
    y = cast_bool_if_necessary(y)
    return paddle.logical_or(x, y)


def _run_py_logical_or(x_func, y_func):
    x_value = x_func()
    assert not isinstance(x_value, (Variable, Value))

    # NOTE(liym27):
    #  1. Returns y_func() if x_value is False;
    #  2. If x_value is True, y_func() should not be run.
    return x_value or y_func()


def convert_logical_not(x):
    """
    A function representation of a Python ``not`` statement.

    Args:
        x(bool|Tensor): Operand of ``not`` operator.

    Returns:
        A python bool variable or a bool Tensor.
    """

    if isinstance(x, (Variable, Value)):
        return _run_paddle_logical_not(x)
    else:
        return _run_py_logical_not(x)


def _run_paddle_logical_not(x):
    x = cast_bool_if_necessary(x)
    return paddle.logical_not(x)


def _run_py_logical_not(x):
    return not x


def convert_ifelse(
    pred,
    true_fn,
    false_fn,
    get_args,
    set_args,
    return_name_ids,
    push_pop_names=None,
):
    """
    A function representation of a Python ``if/else`` statement.

    Args:
        pred(bool|Tensor): A boolean Tensor which determines whether to return the result of ``true_fn`` or ``false_fn`` .
        true_fn(callable): A callable to be performed if ``pred`` is true.
        false_fn(callable): A callable to be performed if ``pred`` is false.
        get_args(callable): Get all arguments that needed in true_fn and false_fn.
        set_args(callable): Update arguments that modified in trure_fn and false_fn.
        return_name_ids(list[string], optional): the returned names.
        push_pop_names(list[string], optional): the names on which called .append() or .pop().

    Returns:
        ``true_fn()`` if the predicate ``pred`` is true else ``false_fn()`` .

    """
    if isinstance(pred, (Variable, Value)):
        out = _run_paddle_cond(
            pred,
            true_fn,
            false_fn,
            get_args,
            set_args,
            return_name_ids,
            push_pop_names,
        )
    else:
        out = _run_py_ifelse(
            pred, true_fn, false_fn, get_args, set_args, return_name_ids
        )

    return out


def _run_paddle_cond(
    pred, true_fn, false_fn, get_args, set_args, return_name_ids, push_pop_names
):
    """
    Paddle cond API will evaluate both true_fn and false_fn codes.
    """
    helper = GetterSetterHelper(
        get_args, set_args, return_name_ids, push_pop_names
    )
    _convert_tensor_array_if_necessary(helper, push_pop_names)
    pred = cast_bool_if_necessary(pred)
    init_args = helper.get(return_name_ids)
    from paddle.jit.dy2static.program_translator import ProgramTranslator
    from paddle.jit.pir_dy2static.parameter_recorder import _global_inplace_map

    if use_pir_api():
        inplace_map = _global_inplace_map
    else:
        inplace_map = ProgramTranslator.get_instance()._inplace_map
    union_name = None
    # TODO(@xiongkun) lambda can have push_pop_names, which will cause error.
    if return_name_ids is None and push_pop_names is None:
        union_name = None
    else:
        union_name = (
            OrderedSet(return_name_ids) if return_name_ids else OrderedSet()
        ) | (OrderedSet(push_pop_names) if push_pop_names else OrderedSet())
        union_name = list(union_name)

    def new_true_fn():
        nonlocal union_name
        # init args may contain mutable python container like [var, 2], we copy then like in while_loop
        inplace_map_checkpoint = inplace_map.save_checkpoint()
        helper.set(
            return_name_ids,
            paddle.utils.copy_mutable_vars(init_args),
        )
        ret = true_fn()
        # IfExpr will return a non-None return value, so we just return ret.
        # We assume normal return has no return value.
        if ret is None:
            ret = helper.get(union_name)
        inplace_map.restore_checkpoint(inplace_map_checkpoint)
        return ret

    def new_false_fn():
        nonlocal union_name
        # init args may contain mutable python container like [var, 2], we copy then like in while_loop
        inplace_map_checkpoint = inplace_map.save_checkpoint()
        helper.set(
            return_name_ids,
            paddle.utils.copy_mutable_vars(init_args),
        )
        ret = false_fn()
        if ret is None:
            ret = helper.get(union_name)
        inplace_map.restore_checkpoint(inplace_map_checkpoint)
        return ret

    try:
        cond_outs = paddle.static.nn.cond(
            pred, new_true_fn, new_false_fn, None, union_name
        )
    except Exception as e:
        if re.search(
            "Unsupported return type of true_fn and false_fn in cond", str(e)
        ):
            raise Dygraph2StaticException(
                f"Your if/else have different return type. TODO: add link to modify. {e}"
            )
        if re.search("Incompatible return values of", str(e)):
            raise Dygraph2StaticException(
                f"Your if/else have different number of return value. TODO: add link to modify. {e}"
            )
        raise e
    get_args = lambda: helper.get(union_name)
    set_args = lambda vs: helper.set(union_name, vs)
    return _recover_args_state(cond_outs, get_args, set_args, union_name)


def _run_py_ifelse(
    pred, true_fn, false_fn, get_args, set_args, return_name_ids
):
    """
    Evaluate python original branch function if-else.
    """
    py_outs = true_fn() if pred else false_fn()
    return py_outs


def _remove_no_value_return_var(out):
    if isinstance(out, tuple) and len(out) > 0:
        processed_out = out
        align_ret = out[0]
        if isinstance(align_ret, tuple):
            for index, item in enumerate(align_ret):
                if isinstance(item, (Variable, Value)) and (
                    RETURN_NO_VALUE_VAR_NAME in item.name
                ):
                    # return None
                    if index == 0:
                        processed_out = (None,) + out[1:]
                    elif index == 1:
                        processed_out = align_ret[:1] + out[1:]
                    else:
                        processed_out = (align_ret[:index],) + out[1:]
                    break

        for index, item in enumerate(processed_out):
            if isinstance(item, (Variable, Value)) and (
                RETURN_NO_VALUE_VAR_NAME in item.name
            ):
                processed_out = processed_out[:index]

        if not processed_out:
            return None
        elif len(processed_out) == 1:
            return processed_out[0]
        else:
            return processed_out

    else:
        return out


def _check_no_undefined_var(outs, names, branch_name):
    if names is None:
        return
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    for var, name in zip(list(outs), names):
        if isinstance(var, UndefinedVar):
            raise ValueError(
                f"Required '{name}' must be initialized both in if-else branch, but found it not initialized in '{branch_name}'."
            )


def _recover_args_state(outs, get_args, set_args, return_name_ids):
    """
    Currently we support variant length of early return statement by padding
    _no_return_value.

    # TODO(dev): We shall consider to evaluate whether should support this for Python if-else?
    """
    # IfExpr's return_name_ids maybe None
    if return_name_ids is None:
        return outs

    init_args = get_args()
    # recover args state
    num_outs = len(return_name_ids)
    num_args = len(init_args)
    assert num_outs <= num_args

    if num_args == 1:
        final_outs = (
            (outs,) if not isinstance(outs, (list, tuple)) else tuple(outs)
        )
    else:
        outs = (outs,) if num_outs == 1 else tuple(outs)
        final_outs = outs + init_args[num_outs:]

    set_args(final_outs)
    return final_outs


def convert_len(var):
    """
    Returns variable(length) from shape ops based on var.type

    Note: In addition to some ast transformations, some block-related
          operations are added in `len` transformation, such as appending
          `shape_op` in var.block.
    """
    if isinstance(var, Variable):
        assert var.ndim > 0, "len() of a 0-D tensor is wrong"
        if var.type in [
            core.VarDesc.VarType.DENSE_TENSOR,
            core.VarDesc.VarType.SELECTED_ROWS,
        ]:
            # Note: Length of var may be known ahead of time in dygraph,
            # but it probably represents batch size which can be variant.
            # so we return a variable dynamically inferred from var.shape.
            if (
                var.shape[0] > 0
                and var.type == core.VarDesc.VarType.DENSE_TENSOR
            ):
                return var.shape[0]
            return paddle.shape(var)[0]
        elif var.type == core.VarDesc.VarType.DENSE_TENSOR_ARRAY:
            return paddle.tensor.array_length(var)
        else:
            raise TypeError(
                f'len(var) only supports DenseTensor/DenseTensorArray/SelectedRows, but received {type(var)}.'
            )
    elif isinstance(var, Value):
        if var.is_dense_tensor_type() or var.is_selected_row_type():
            assert var.ndim > 0, "len() of a 0-D tensor is wrong"
            # Note: Length of var may be known ahead of time in dygraph,
            # but it probably represents batch size which can be variant.
            # so we return a variable dynamically inferred from var.shape.
            if var.shape[0] > 0 and var.is_dense_tensor_type():
                return var.shape[0]
            return paddle.shape(var)[0]
        elif var.is_dense_tensor_array_type():
            return paddle.tensor.array_length(var)
        else:
            raise TypeError(
                'len(var) only supports DenseTensor/DenseTensorArray/SelectedRows, '
                + f'but received {type(var)}.'
            )
    else:
        if isinstance(var, VariableTuple):
            return var.__len__()
        return len(var)


def convert_zip(*args):
    for i, arg in enumerate(args):
        if isinstance(arg, (Variable, Value)) and arg.shape[0] == -1:
            raise RuntimeError(
                "Not support zip(tensor, ...) when tensor.shape[0] == -1, "
                f"but found args[{i}].shape[0] == -1 in 'zip'"
            )
    return zip(*args)


def convert_super(super_fn):
    if super_fn is super:
        return super_fn
    return lambda cls, instance: super_fn()


# TODO(xiongkun): delete when list<variable> is ready.
class VariableTuple:
    """
    this class will cause enumerate can't be wrapped by other iterator change function.
    this will be fixed when list<Variable> is produced.
    VariableTuple can only deal with variables which is fixed.
    """

    def __init__(self, var, start=0):
        self.var = var
        self.len = convert_len(var)
        if isinstance(self.len, (Variable, Value)):
            self.rag = paddle.arange(start, start + self.len, 1, "int64")
        else:
            self.rag = range(start, start + self.len)

    def __getitem__(self, idx):
        return self.rag[idx], self.var[idx]

    def __len__(self):
        return self.len


def convert_enumerate(*args):
    has_variable = any(isinstance(x, (Variable, Value)) for x in args)
    if has_variable:
        return VariableTuple(*args)
    return enumerate(*args)


def convert_range(*args):
    has_variable = any(isinstance(x, (Variable, Value)) for x in args)
    if has_variable:
        if len(args) == 1:
            return paddle.arange(0, args[0], 1, "int64")
        if len(args) == 2:
            return paddle.arange(args[0], args[1], 1, "int64")
        if len(args) == 3:
            return paddle.arange(args[0], args[1], args[2], "int64")
    return range(*args)


def convert_shape(x):
    """
    A function representation of the shape of variable.
    """

    def has_negative(list_shape):
        return any(x < 0 for x in list_shape)

    # When `x` is Variable:
    #  (1) if x.shape contains -1, such as [2, -1, 64], returns [2, var, 64],
    #      where var = paddle.shape(x)[1]

    #  (2) if x.shape does not contains -1, return list(x.shape) directly

    if isinstance(x, (Variable, Value)):
        values = list(x.shape)
        if has_negative(values):
            shape_tensor = paddle.shape(x)
            for i, v in enumerate(values):
                if v is None or v < 0:
                    values[i] = shape_tensor[i]
        return values
    else:
        return x.shape


def cast_bool_if_necessary(var):
    assert isinstance(var, (Variable, Value))
    if convert_dtype(var.dtype) not in ['bool']:
        var = paddle.cast(var, dtype="bool")
    return var


def convert_var_dtype(var, dtype):
    if isinstance(var, (Variable, Value)):
        src_dtype = convert_dtype(var.dtype)
        assert src_dtype in [
            'bool',
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'uint8',
        ], f"The dtype of var {var.name} is {src_dtype}, which is not supported in the cast op."
        assert dtype in [
            'bool',
            'int',
            'float',
            'complex',
        ], f"The casted target dtype is {dtype}, which is not supported in type casting."
        cast_map = {
            'bool': 'bool',
            'int': 'int32',
            'float': 'float32',
            'complex': 'complex64',
        }
        return paddle.cast(var, dtype=cast_map[dtype])
    else:
        assert dtype in [
            'bool',
            'int',
            'float',
            'complex',
        ], f"The casted target dtype is {dtype}, which is not supported in type casting."
        return eval(dtype)(var)


def convert_assert(cond, message=""):
    """
    A function representation of a Python ``assert`` statement.
    """
    if isinstance(cond, (Variable, Value)):
        cond = paddle.cast(cond, "bool")
        # NOTE: message is not used because Paddle Assert has no corresponding parameter to use.
        from paddle.static.nn.control_flow import Assert

        return Assert(cond)
    else:
        assert cond, message


def convert_print(*objects, sep=' ', end='\n', file=None, flush=False):
    """
    A function representing Python ``print`` function. It will print all arguments
    at compile time and only print the Tensor values at runtime.
    """
    for obj in objects:
        if isinstance(obj, (Variable, Value)):
            paddle.static.Print(obj)
    print(*objects, sep=sep, end=end, file=file, flush=flush)


@contextmanager
def convert_auto_cast(
    enable=True,
    custom_white_list=None,
    custom_black_list=None,
    level='O1',
    dtype='float16',
    use_promote=True,
):
    from .program_translator import ProgramTranslator

    warnings.warn(
        "paddle.amp.auto_cast is an experimental features in auto parallel."
        + "This will take no effect in normal dy2static."
    )

    amp_records = ProgramTranslator.get_instance()._amp_records
    main_program = paddle.static.default_main_program()
    current_block_idx = main_program.current_block_idx
    current_block = main_program.current_block()
    start_op_idx = len(current_block.ops)
    amp_options = AmpOptions(
        enable, custom_white_list, custom_black_list, level, dtype, use_promote
    )
    yield
    end_op_idx = len(current_block.ops)
    if current_block_idx not in amp_records:
        amp_records[current_block_idx] = []
    amp_records[current_block_idx].append(
        (amp_options, start_op_idx, end_op_idx)
    )


def create_bool_as_type(x, value=True):
    '''
    Create a bool variable, which type is the same as x.
    '''
    if isinstance(x, (Variable, Value)):
        return paddle.full(shape=[], fill_value=value, dtype="bool")
    else:
        return value
