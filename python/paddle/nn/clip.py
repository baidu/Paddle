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

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import paddle
import paddle.autograd as imperative_base
import paddle.distributed as dist
from paddle import _C_ops
from paddle.base import core, framework, unique_name
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.libpaddle import DataType
from paddle.common_ops_import import Variable, check_type, default_main_program
from paddle.framework import (
    LayerHelper,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


def clip_by_norm(x, max_norm, name=None):
    r"""

    Limits the L2 norm of the input :math:`x` within :math:`max\_norm`.
    If the L2 norm of :math:`x` is less than or equal to :math:`max\_norm`, :math:`out` will be
    the same as :math:`x`. If the L2 norm of :math:`x` is greater than :math:`max\_norm`, :math:`x` will
    be linearly scaled to make the L2 norm of :math:`out` equal to :math:`max\_norm`, as
    shown in the following formula:

    .. math::

        out = \frac{max\_norm * x}{norm(x)}

    where :math:`norm(x)` represents the L2 norm of :math:`x`.

    Args:
        x(Tensor): The input of clip_by_norm and data type is float32.
            The number of dimensions must be between [1, 9].
        max_norm(float): The maximum norm value.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor: The output of clip_by_norm with shape as input.
            The data type is float32.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn import clip

            >>> input = paddle.to_tensor([[2.0, 2.0], [2.0, 2.0]], dtype='float32')
            >>> reward = clip.clip_by_norm(x=input, max_norm=1.0)
            >>> print(reward)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.50000000, 0.50000000],
             [0.50000000, 0.50000000]])
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.clip_by_norm(x, max_norm)

    helper = LayerHelper("clip_by_norm", **locals())
    check_variable_and_dtype(
        x, 'X', ['float16', 'float32', 'uint16'], 'clip_by_norm'
    )
    check_type(max_norm, 'max_norm', (float), 'clip_by_norm')

    if name is None:
        name = unique_name.generate_with_ignorable_key(
            ".".join([helper.name, 'tmp'])
        )

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False
    )

    helper.append_op(
        type="clip_by_norm",
        inputs={"X": x},
        attrs={"max_norm": max_norm},
        outputs={"Out": out},
    )

    return out


def merge_selected_rows(x, name=None):
    """
    Merge by adding duplicated rows in the input SelectedRows object.

    Args:
        x(Tensor): The input selected rows to be merge.
        name(basestring|None): Name of the output.

    Returns:
        Tensor, merged output.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.base as base

            >>> b = paddle.static.default_main_program().global_block()
            >>> var = b.create_var(
            ...     name="X", dtype="float32", persistable=True,
            ...     type=base.core.VarDesc.VarType.SELECTED_ROWS)
            >>> y = paddle.nn.clip.merge_selected_rows(var)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.merge_selected_rows(x)

    helper = LayerHelper("merge_selected_rows", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="merge_selected_rows",
        inputs={"X": x},
        attrs={},
        outputs={"Out": out},
    )
    return out


def get_tensor_from_selected_rows(x, name=None):
    """
    Get tensor data from input with SelectedRows type, and outputs a Tensor.

    .. code-block:: text

        input x is SelectedRows:
           x.rows = [0, 5, 5, 4, 19]
           x.height = 20
           x.value = [[1, 1] [2, 2] [2, 2] [3, 3] [6, 6]]

        Output is DenseTensor:
           out.shape = [5, 2]
           out.data = [[1, 1],
                       [2, 2],
                       [2, 2],
                       [3, 3],
                       [6, 6]]

    Args:
        x(SelectedRows): Input with SelectedRows type. The data type is float32, float64, int32 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: DenseTensor transformed from SelectedRows. The data type is same with input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.base as base
            >>> from paddle.base import core
            >>> paddle.enable_static()
            >>> scope = core.Scope()
            >>> block = paddle.static.default_main_program().global_block()
            >>> x_rows = [0, 5, 5, 4, 19]
            >>> height = 20
            >>> x = scope.var('X').get_selected_rows()
            >>> x.set_rows(x_rows)
            >>> x.set_height(height)
            >>> x = block.create_var(name="X", dtype="float32", persistable=True, type=base.core.VarDesc.VarType.SELECTED_ROWS)
            >>> z = paddle.nn.clip.get_tensor_from_selected_rows(x)
    """
    if in_pir_mode():
        return _C_ops.get_tensor_from_selected_rows(x)

    check_type(x, 'x', Variable, 'get_tensor_from_selected_rows')
    if x.type != core.VarDesc.VarType.SELECTED_ROWS:
        raise TypeError(
            "The type of 'x' in get_tensor_from_selected_rows must be SELECTED_ROWS."
        )
    helper = LayerHelper('get_tensor_from_selected_rows', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='get_tensor_from_selected_rows',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={},
    )
    return out


_clip_by_global_norm_using_mp_type_flag = False


def _clip_by_global_norm_using_mp_type(*args):
    global _clip_by_global_norm_using_mp_type_flag
    assert len(args) <= 1
    if len(args) == 1:
        assert isinstance(args[0], bool)
        old_value = _clip_by_global_norm_using_mp_type_flag
        _clip_by_global_norm_using_mp_type_flag = args[0]
        return old_value
    else:
        return _clip_by_global_norm_using_mp_type_flag


def _cast_to_mp_type_if_enabled(x):
    if (
        x.dtype == core.VarDesc.VarType.FP16
        or x.dtype == core.VarDesc.VarType.BF16
    ) and _clip_by_global_norm_using_mp_type():
        return x.astype(core.VarDesc.VarType.FP32)
    elif (
        x.dtype == DataType.FLOAT16 or x.dtype == DataType.BFLOAT16
    ) and _clip_by_global_norm_using_mp_type():
        return x.astype(DataType.FLOAT32)
    else:
        return x


def _squared_l2_norm(x):
    r"""
    Return the squared L2 norm of a tensor.
    """

    x = _cast_to_mp_type_if_enabled(x)

    if in_dynamic_or_pir_mode():
        return _C_ops.squared_l2_norm(x)

    op_type = 'squared_l2_norm'
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'float16', 'uint16'], op_type
    )
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(x.dtype)

    inputs = {"X": x}
    outputs = {'Out': out}
    helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
    return out


class BaseErrorClipAttr:
    def __str__(self):
        raise NotImplementedError

    def _append_clip_op(self, block, grad_name):
        raise NotImplementedError


class ErrorClipByValue(BaseErrorClipAttr):
    r"""
    Clip tensor values to the range [min, max].

    Given a tensor ``t`` (see Examples below), this operation clips its value \
    to ``min`` and ``max`` inplace.

    - Any values less than min are set to min.
    - Any values greater than max are set to max.

    Args:
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, \
        will be set to ``-max`` by framework.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()
            >>> BATCH_SIZE = 128
            >>> CLIP_MAX = 2e-6
            >>> CLIP_MIN = -1e-6
            >>> prog = paddle.static.Program()
            >>> with paddle.static.program_guard(main_program=prog):
            ...     image = paddle.static.data(name='x', shape=[None, 784], dtype='float32')
            ...     hidden1 = paddle.static.nn.fc(image, size=128, activation='relu')
            ...     hidden2 = paddle.static.nn.fc(hidden1, size=64, activation='relu')
            ...     predict = paddle.static.nn.fc(hidden2, size=10, activation='softmax')
            ...     label = paddle.static.data(name='y', shape=[1], dtype='int64')
            ...     cost = paddle.nn.functional.cross_entropy(input=predict, label=label)
            ...     avg_cost = paddle.mean(cost)
            >>> prog_clip = prog.clone()
            >>> prog_clip.block(0).var(hidden1.name)._set_error_clip(
            ...     paddle.nn.clip.ErrorClipByValue(
            ...         max=CLIP_MAX, min=CLIP_MIN))
    """

    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def __str__(self):
        return f"ByValue, min={self.min:f}, max={self.max:f}"

    def _append_clip_op(self, block, grad_name):
        clip_op_desc = block.desc.append_op()
        clip_op_desc.set_type("clip")
        clip_op_desc.set_input("X", [grad_name])
        clip_op_desc.set_output("Out", [grad_name])
        clip_op_desc._set_attr("min", self.min)
        clip_op_desc._set_attr("max", self.max)


def error_clip_callback(block, context):
    # the context is a grad_to_var map
    grad_to_var = context
    op_desc = block.desc.op(block.desc.op_size() - 1)
    for grad_n in [n for n in op_desc.output_arg_names() if n in grad_to_var]:
        fwd_var = block._var_recursive(grad_to_var[grad_n])
        error_clip = getattr(fwd_var, "error_clip", None)
        if not (
            error_clip is None or isinstance(error_clip, BaseErrorClipAttr)
        ):
            raise TypeError(
                "Variable's error_clip should be an instance of BaseErrorClipAttr or None."
            )
        if error_clip is not None:
            error_clip._append_clip_op(block, grad_n)


class ClipGradBase:
    def __init__(self):
        super().__init__()

    def __str__(self):
        raise NotImplementedError

    @imperative_base.no_grad()
    def _dygraph_clip(self, params_grads):
        raise NotImplementedError

    def _pir_clip(self, params_grads):
        raise NotImplementedError

    def _static_clip(self, params_grads):
        raise NotImplementedError

    def __call__(
        self, params_grads: list[tuple[Tensor, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        if in_dynamic_mode():
            return self._dygraph_clip(params_grads)
        elif in_pir_mode():
            return self._pir_clip(params_grads)
        else:
            for p, g in params_grads:
                if getattr(p, 'gradient_clip_attr', None) is not None:
                    warnings.warn(
                        "'set_gradient_clip' will be ineffective, because you have "
                        "set 'need_clip' in 'ParamAttr'. So, 'set_gradient_clip' "
                        "is redundant and you can remove it."
                    )
                    break
            return self._static_clip(params_grads)

    def _process_context(self, context, param, grad):
        raise NotImplementedError

    def _create_operators(self, param, grad):
        raise NotImplementedError


class ClipGradByValue(ClipGradBase):
    """
    Limit the value of multi-dimensional Tensor :math:`X` to the range [min, max].

    - Any values less than min are set to ``min``.

    - Any values greater than max are set to ``max``.

    The multi-dimensional Tensor :math:`X` is not passed from this class, but the gradients of all parameters set in ``optimizer``.
    If ``need_clip`` of specific param is ``False`` in its ``ParamAttr``, then the gradients of this param will not be clipped.

    Gradient clip will takes effect after being set in ``optimizer`` , see the document ``optimizer``
    (for example: :ref:`api_paddle_optimizer_SGD`).

    Note:
        ``need_clip`` of ``ClipGradByValue`` HAS BEEN DEPRECATED since 2.0.
        Please use ``need_clip`` in ``ParamAttr`` to specify the clip scope.

    Args:
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, it will be set to ``-max``
            automatically. In this case, ``max`` must be greater than :math:`0`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10,
            ...                           weight_attr=paddle.ParamAttr(need_clip=True),
            ...                           bias_attr=paddle.ParamAttr(need_clip=False))
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()

            >>> clip = paddle.nn.ClipGradByValue(min=-1, max=1)
            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
            >>> sdg.step()
    """

    max: float
    min: float

    def __init__(self, max: float, min: float | None = None) -> None:
        super().__init__()
        if min is None:
            assert max > 0.0
            min = -max
        self.max = float(max)
        self.min = float(min)

    def __str__(self) -> str:
        return f"Clip Gradient By Value, min = {self.min:f}, max={self.max:f}"

    @imperative_base.no_grad()
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            new_grad = paddle.clip(x=g, min=self.min, max=self.max)
            params_and_grads.append((p, new_grad))
        return params_and_grads

    def _static_clip(self, params_grads):
        params_and_grads = []
        param_new_grad_name_dict = {}
        with framework.name_scope('gradient_clip'):
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, 'need_clip', True) is False:
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = paddle.clip(x=g, min=self.min, max=self.max)
                params_and_grads.append((p, new_grad))
                param_new_grad_name_dict[p.name] = new_grad.name
        _correct_clip_op_role_var(params_and_grads, param_new_grad_name_dict)
        return params_and_grads

    def _process_context(self, context, param, grad):
        pass

    def _create_operators(self, param, grad):
        new_grad = paddle.clip(x=grad, min=self.min, max=self.max)
        return param, new_grad


class ClipGradByNorm(ClipGradBase):
    r"""
    Limit the l2 norm of multi-dimensional Tensor :math:`X` to ``clip_norm`` .

    - If the l2 norm of :math:`X` is greater than ``clip_norm`` , :math:`X` will be compressed by a ratio.

    - If the l2 norm of :math:`X` is less than or equal to ``clip_norm`` , nothing will be done.

    The multidimensional Tensor :math:`X` is not passed from this class, but the gradients of all parameters set in ``optimizer``.
    If ``need_clip`` of specific param is ``False`` in its ``ParamAttr``, then the gradients of this param will not be clipped.

    Gradient clip will takes effect after being set in ``optimizer`` , see the document ``optimizer``
    (for example: :ref:`api_paddle_optimizer_SGD`).

    The clipping formula is:

    .. math::
        Out =
        \left\{
            \begin{array}{ccl}
                X & & if (norm(X) \leq clip\_norm) \\
                \frac{clip\_norm*X}{norm(X)} & & if (norm(X) > clip\_norm) \\
        \end{array}
        \right.


    where :math:`norm(X)` represents the L2 norm of :math:`X`.

    .. math::
        norm(X) = ( \sum_{i=1}^{n}|x\_i|^2)^{ \frac{1}{2}}

    Note:
        ``need_clip`` of ``ClipGradByNorm`` HAS BEEN DEPRECATED since 2.0.
        Please use ``need_clip`` in ``ParamAttr`` to specify the clip scope.

    Args:
        clip_norm(float): The maximum norm value.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10,
            ...                           weight_attr=paddle.ParamAttr(need_clip=True),
            ...                           bias_attr=paddle.ParamAttr(need_clip=False))
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()

            >>> clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
            >>> sdg.step()
    """

    clip_norm: float

    def __init__(self, clip_norm: float) -> None:
        super().__init__()
        self.clip_norm = float(clip_norm)

    def __str__(self) -> str:
        return f"Gradient Clip By Norm, clip_norm={self.clip_norm:f}"

    def _clip_gradients(self, params_grads):
        params_and_grads = []
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            new_grad = clip_by_norm(x=g, max_norm=self.clip_norm)
            params_and_grads.append((p, new_grad))
        return params_and_grads

    @imperative_base.no_grad()
    def _dygraph_clip(self, params_grads):
        return self._clip_gradients(params_grads)

    def _pir_clip(self, params_grads):
        return self._clip_gradients(params_grads)

    def _static_clip(self, params_grads):
        params_and_grads = []
        with framework.name_scope('gradient_clip'):
            param_new_grad_name_dict = {}
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, 'need_clip', True) is False:
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_grad = clip_by_norm(x=g, max_norm=self.clip_norm)
                param_new_grad_name_dict[p.name] = new_grad.name
                params_and_grads.append((p, new_grad))
        _correct_clip_op_role_var(params_and_grads, param_new_grad_name_dict)
        return params_and_grads

    def _process_context(self, context, param, grad):
        pass

    def _create_operators(self, param, grad):
        new_grad = clip_by_norm(x=grad, max_norm=self.clip_norm)
        return param, new_grad


_allow_pure_fp16_global_norm_clip_flag = False


def _allow_pure_fp16_global_norm_clip(*args):
    global _allow_pure_fp16_global_norm_clip_flag
    if len(args) == 0:
        return _allow_pure_fp16_global_norm_clip_flag
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        old_value = _allow_pure_fp16_global_norm_clip_flag
        _allow_pure_fp16_global_norm_clip_flag = args[0]
        return old_value


_allow_pure_bf16_global_norm_clip_flag = False


def _allow_pure_bf16_global_norm_clip(*args):
    global _allow_pure_bf16_global_norm_clip_flag
    if len(args) == 0:
        return _allow_pure_bf16_global_norm_clip_flag
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        old_value = _allow_pure_bf16_global_norm_clip_flag
        _allow_pure_bf16_global_norm_clip_flag = args[0]
        return old_value


class ClipGradByGlobalNorm(ClipGradBase):
    r"""
    Given a list of Tensor :math:`t\_list` , calculate the global norm for the elements of all tensors in
    :math:`t\_list` , and limit it to ``clip_norm`` .

    - If the global norm is greater than ``clip_norm`` , all elements of :math:`t\_list` will be compressed by a ratio.

    - If the global norm is less than or equal to ``clip_norm`` , nothing will be done.

    The list of Tensor :math:`t\_list` is not passed from this class, but the gradients of all parameters set in ``optimizer``.
    If ``need_clip`` of specific param is ``False`` in its ``ParamAttr``, then the gradients of this param will not be clipped.

    Gradient clip will takes effect after being set in ``optimizer`` , see the document ``optimizer``
    (for example: :ref:`api_paddle_optimizer_SGD`).

    The clipping formula is:

    .. math::

        t\_list[i] = t\_list[i] * \frac{clip\_norm}{\max(global\_norm, clip\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    Note:
        ``need_clip`` of ``ClipGradyGlobalNorm`` HAS BEEN DEPRECATED since 2.0.
        Please use ``need_clip`` in ``ParamAttr`` to specify the clip scope.

    Args:
        clip_norm (float): The maximum norm value.
        group_name (str, optional): The group name for this clip. Default value is ``default_group``.
        auto_skip_clip (bool, optional): skip clipping gradient. Default value is ``False``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10,
            ...                           weight_attr=paddle.ParamAttr(need_clip=True),
            ...                           bias_attr=paddle.ParamAttr(need_clip=False))
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()

            >>> clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
            >>> sdg.step()
    """

    clip_norm: float
    group_name: str
    auto_skip_clip: bool

    def __init__(
        self,
        clip_norm: float,
        group_name: str = "default_group",
        auto_skip_clip: bool = False,
    ) -> None:
        super().__init__()
        self.clip_norm = float(clip_norm)
        self.group_name = group_name
        assert isinstance(auto_skip_clip, bool)
        self.auto_skip_clip = auto_skip_clip
        # TODO(zhiqiu): Now, in dygraph mode async_add_n is always used.
        # However, in static mode, it is only used in auto_parallel mode
        # by setting self._async_add_n to True. The reason is that there
        # are so many hard code depends on `add_n` in the legacy static
        # manual hybrid-parallel.
        self._async_add_n = None
        self.should_comm_on_shard_dim = False

    def __str__(self) -> str:
        return f"Gradient Clip By GlobalNorm, global_norm={self.clip_norm:f}"

    @imperative_base.no_grad()
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        if len(params_grads) > 0 and len(params_grads[0]) > 0:
            src_mesh = params_grads[0][0].process_mesh
        else:
            src_mesh = None

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g

            if in_dynamic_mode() and g.is_selected_rows():
                merge_grad = merge_selected_rows(g)
                merge_grad = merge_grad._get_tensor_from_selected_rows()

            elif g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = merge_selected_rows(g)
                merge_grad = get_tensor_from_selected_rows(merge_grad)

            sum_square = _squared_l2_norm(merge_grad)

            # if the gradient mesh is not equal to src mesh
            # do reshard to get the result of squared_l2 from other pp stage mesh
            if src_mesh is not None and g.process_mesh != src_mesh:
                sum_square = dist.reshard(
                    sum_square, src_mesh, sum_square.placements
                )

            if (
                sum_square.dtype == paddle.float16
                or sum_square.dtype == paddle.bfloat16
            ):
                sum_square_list_fp16.append(sum_square)
            elif sum_square.dtype == paddle.float32:
                sum_square_list_fp32.append(sum_square)
            else:
                sum_square_list.append(sum_square)

        # all parameters have been filtered out
        if (
            len(sum_square_list)
            + len(sum_square_list_fp16)
            + len(sum_square_list_fp32)
            == 0
        ):
            return params_grads

        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"
        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = async_add_n(sum_square_list_fp16)
            global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = async_add_n(sum_square_list_fp32)
            if sum_dtype == 'float32':
                global_norm_var.append(global_norm_var_fp32)
            else:
                global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_list) > 0:
            global_norm_var_fp64 = async_add_n(sum_square_list)
            global_norm_var.append(global_norm_var_fp64)

        global_norm_var = async_add_n(global_norm_var)
        global_norm_var = paddle.sqrt(global_norm_var)
        max_global_norm = paddle.full(
            shape=[], dtype=sum_dtype, fill_value=self.clip_norm
        )

        need_clip = False
        if not self.auto_skip_clip:  # always apply clip
            need_clip = True
            clip_var = paddle.divide(
                x=max_global_norm,
                y=paddle.maximum(x=global_norm_var, y=max_global_norm),
            )
        elif global_norm_var > max_global_norm:
            # only when global_norm_var > max_global_norm, grad need clip
            need_clip = True
            clip_var = paddle.divide(x=max_global_norm, y=global_norm_var)

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            if need_clip:
                clip_input = (
                    clip_var.astype(g.dtype)
                    if clip_var.dtype != g.dtype
                    else clip_var
                )
                if clip_input.process_mesh != g.process_mesh:
                    # TODO(pkuzyc): refine the reshard function between local
                    # and global mesh to avoid the following "_local_tensor()"
                    # operation.
                    if set(g.process_mesh.process_ids) < set(
                        clip_input.process_mesh.process_ids
                    ):
                        placements = clip_input.placements
                        is_replicate = True
                        for placement in placements:
                            if not placement.is_replicated():
                                is_replicate = False
                                break
                        if is_replicate:
                            clip_input = clip_input._local_value()
                        else:
                            raise NotImplementedError(
                                "Reshard a sharded tensor from a local mesh to a global mesh is not supported"
                            )
                    else:
                        clip_input = paddle.distributed.reshard(
                            clip_input, g.process_mesh, clip_input.placements
                        )
                new_grad = paddle.multiply(g, clip_input)
                params_and_grads.append((p, new_grad))
            else:
                params_and_grads.append((p, g))

        return params_and_grads

    def _pir_clip(self, params_grads):
        params_and_grads = []

        # no fusion grad
        no_fusion_sum_square = []
        no_fusion_sum_square_fp16 = []
        no_fusion_sum_square_fp32 = []

        # fusion grad need to commnuicate in dp&mp
        sum_square_dist = []
        sum_square_dist_fp16 = []
        sum_square_dist_fp32 = []

        # fusion grad only need to commnuicate in dp
        sum_square_not_dist = []
        sum_square_not_dist_fp16 = []
        sum_square_not_dist_fp32 = []

        auto_parallel_pp = False
        pp_meshes = set()
        pp_stage0_mesh = None
        for p, g in params_grads:
            if p.is_dist_dense_tensor_type():
                pp_meshes.add(p.dist_attr().process_mesh)
                if 0 in p.dist_attr().process_mesh.process_ids:
                    if pp_stage0_mesh is None:
                        pp_stage0_mesh = p.dist_attr().process_mesh
                    else:
                        p_mesh = p.dist_attr().process_mesh
                        if set(pp_stage0_mesh.process_ids) < set(
                            p_mesh.process_ids
                        ):
                            pp_stage0_mesh = p_mesh
                        assert set(p_mesh.process_ids) <= set(
                            pp_stage0_mesh.process_ids
                        )

        if len(pp_meshes) > 1:
            from paddle.distributed.auto_parallel.placement_type import (
                to_placements,
            )

            auto_parallel_pp = True
            assert pp_stage0_mesh is not None

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g

            if in_pir_mode() and g.is_selected_row_type():
                merge_grad = merge_selected_rows(g)
                merge_grad = get_tensor_from_selected_rows(merge_grad)

            sum_square = _squared_l2_norm(merge_grad)
            if (
                auto_parallel_pp
                and sum_square.dist_attr().process_mesh != pp_stage0_mesh
            ):
                sum_square = paddle.distributed.reshard(
                    sum_square,
                    pp_stage0_mesh,
                    to_placements(
                        sum_square.dist_attr().dims_mapping,
                        sum_square.dist_attr().process_mesh,
                        sum_square.dist_attr().partial_dims,
                    ),
                )
            if (
                not self.should_comm_on_shard_dim
                or p.optimize_attr["no_fusion"]
            ):
                if (
                    sum_square.dtype == DataType.FLOAT16
                    or sum_square.dtype == DataType.BFLOAT16
                ):
                    no_fusion_sum_square_fp16.append(sum_square)
                elif sum_square.dtype == DataType.FLOAT32:
                    no_fusion_sum_square_fp32.append(sum_square)
                else:
                    no_fusion_sum_square.append(sum_square)
            elif p.is_distributed:
                if (
                    sum_square.dtype == DataType.FLOAT16
                    or sum_square.dtype == DataType.BFLOAT16
                ):
                    sum_square_dist_fp16.append(sum_square)
                elif sum_square.dtype == DataType.FLOAT32:
                    sum_square_dist_fp32.append(sum_square)
                else:
                    sum_square_dist.append(sum_square)
            else:
                if (
                    sum_square.dtype == DataType.FLOAT16
                    or sum_square.dtype == DataType.BFLOAT16
                ):
                    sum_square_not_dist_fp16.append(sum_square)
                elif sum_square.dtype == DataType.FLOAT32:
                    sum_square_not_dist_fp32.append(sum_square)
                else:
                    sum_square_not_dist.append(sum_square)

        # all parameters have been filtered out
        if (
            len(no_fusion_sum_square)
            + len(no_fusion_sum_square_fp16)
            + len(no_fusion_sum_square_fp32)
            + len(sum_square_dist)
            + len(sum_square_dist_fp16)
            + len(sum_square_dist_fp32)
            + len(sum_square_not_dist)
            + len(sum_square_not_dist_fp16)
            + len(sum_square_not_dist_fp32)
            == 0
        ):
            return params_grads

        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        sum_dtype = (
            'float64'
            if len(no_fusion_sum_square)
            + len(sum_square_dist)
            + len(sum_square_not_dist)
            > 0
            else "float32"
        )
        no_fusion_global_norm = []
        global_norm_dist = []
        global_norm_not_dist = []
        if len(no_fusion_sum_square_fp16) > 0:
            global_norm_var_fp16 = async_add_n(no_fusion_sum_square_fp16)
            no_fusion_global_norm.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_dist_fp16) > 0:
            global_norm_var_fp16 = async_add_n(sum_square_dist_fp16)
            global_norm_dist.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_not_dist_fp16) > 0:
            global_norm_var_fp16 = async_add_n(sum_square_not_dist_fp16)
            global_norm_not_dist.append(global_norm_var_fp16.astype(sum_dtype))

        if len(no_fusion_sum_square_fp32) > 0:
            global_norm_var_fp32 = async_add_n(no_fusion_sum_square_fp32)
            if sum_dtype == 'float32':
                no_fusion_global_norm.append(global_norm_var_fp32)
            else:
                no_fusion_global_norm.append(
                    global_norm_var_fp32.astype(sum_dtype)
                )
        if len(sum_square_dist_fp32) > 0:
            global_norm_var_fp32 = async_add_n(sum_square_dist_fp32)
            if sum_dtype == 'float32':
                global_norm_dist.append(global_norm_var_fp32)
            else:
                global_norm_dist.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_not_dist_fp32) > 0:
            global_norm_var_fp32 = async_add_n(sum_square_not_dist_fp32)
            if sum_dtype == 'float32':
                global_norm_not_dist.append(global_norm_var_fp32)
            else:
                global_norm_not_dist.append(
                    global_norm_var_fp32.astype(sum_dtype)
                )
        if len(no_fusion_sum_square) > 0:
            global_norm_var_fp64 = async_add_n(no_fusion_sum_square)
            no_fusion_global_norm.append(global_norm_var_fp64)
        if len(sum_square_dist) > 0:
            global_norm_var_fp64 = async_add_n(sum_square_dist)
            global_norm_dist.append(global_norm_var_fp64)
        if len(sum_square_not_dist) > 0:
            global_norm_var_fp64 = async_add_n(sum_square_dist)
            global_norm_not_dist.append(global_norm_var_fp64)

        global_norm_var = None
        if len(no_fusion_global_norm) > 0:
            global_norm_var = async_add_n(no_fusion_global_norm)

        if len(global_norm_dist) > 0:
            global_norm_dist_var = async_add_n(global_norm_dist)
        elif self.should_comm_on_shard_dim and self.has_dist_param:
            global_norm_dist_var = paddle.full(
                shape=[1], dtype=sum_dtype, fill_value=0.0
            )

        if self.should_comm_on_shard_dim and self.has_dist_param:
            global_norm_dist_var = paddle._C_ops.c_allreduce_sum(
                global_norm_dist_var, self.sharding_group.id, True, False
            )
            global_norm_dist_var = paddle._C_ops.c_allreduce_sum(
                global_norm_dist_var, self.mp_group.id, True, False
            )
            if global_norm_var is None:
                global_norm_var = global_norm_dist_var
            else:
                global_norm_var = global_norm_var + global_norm_dist_var

        if len(global_norm_not_dist) > 0:
            global_norm_not_dist_var = async_add_n(global_norm_not_dist)
        elif self.should_comm_on_shard_dim and self.has_not_dist_param:
            global_norm_not_dist_var = paddle.full(
                shape=[1], dtype=sum_dtype, fill_value=0.0
            )
        if self.should_comm_on_shard_dim and self.has_not_dist_param:
            global_norm_not_dist_var = paddle._C_ops.c_allreduce_sum(
                global_norm_not_dist_var, self.sharding_group.id, True, False
            )
            if global_norm_var is None:
                global_norm_var = global_norm_not_dist_var
            else:
                global_norm_var = global_norm_var + global_norm_not_dist_var

        global_norm_var = paddle.sqrt(global_norm_var)
        max_global_norm = paddle.full(
            shape=[1], dtype=global_norm_var.dtype, fill_value=self.clip_norm
        )

        need_clip = False
        if not self.auto_skip_clip:  # always apply clip
            need_clip = True
            clip_var = paddle.divide(
                x=max_global_norm,
                y=paddle.maximum(x=global_norm_var, y=max_global_norm),
            )
        elif global_norm_var > max_global_norm:
            # only when global_norm_var > max_global_norm, grad need clip
            need_clip = True
            clip_var = paddle.divide(x=max_global_norm, y=global_norm_var)

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            if need_clip:
                clip_input = (
                    clip_var.astype(g.dtype)
                    if clip_var.dtype != g.dtype
                    else clip_var
                )
                if (
                    auto_parallel_pp
                    and clip_input.dist_attr().process_mesh
                    != g.dist_attr().process_mesh
                ):
                    clip_input = paddle.distributed.reshard(
                        clip_input,
                        g.dist_attr().process_mesh,
                        to_placements(
                            clip_input.dist_attr().dims_mapping,
                            clip_input.dist_attr().process_mesh,
                            clip_input.dist_attr().partial_dims,
                        ),
                    )

                new_grad = paddle.multiply(g, clip_input)
                params_and_grads.append((p, new_grad))
            else:
                params_and_grads.append((p, g))

        return params_and_grads

    def _static_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        sum_square_list_fp16 = []
        sum_square_list_bf16 = []
        sum_square_list_fp32 = []

        def _add_n(var_list):
            if self._async_add_n:
                return paddle.stack(var_list).sum()
            else:
                return paddle.add_n(var_list)

        with framework.name_scope('gradient_clip'):
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, 'need_clip', True) is False:
                    continue
                merge_grad = g
                with p.block.program._optimized_guard([p, g]):
                    if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                        merge_grad = merge_selected_rows(g)
                        merge_grad = get_tensor_from_selected_rows(merge_grad)
                    sum_square = _squared_l2_norm(merge_grad)
                    if sum_square.dtype == core.VarDesc.VarType.FP16:
                        sum_square_list_fp16.append(sum_square)
                    elif sum_square.dtype == core.VarDesc.VarType.BF16:
                        sum_square_list_bf16.append(sum_square)
                    elif sum_square.dtype == core.VarDesc.VarType.FP32:
                        sum_square_list_fp32.append(sum_square)
                    else:
                        sum_square_list.append(sum_square)

            if len(sum_square_list_fp16) > 0 and len(sum_square_list_bf16) > 0:
                raise NotImplementedError(
                    'FP16 and BF16 are not supported at the same time.'
                )

            # all parameters have been filtered out
            if (
                len(sum_square_list)
                + len(sum_square_list_fp16)
                + len(sum_square_list_fp32)
                == 0
            ) and (
                len(sum_square_list)
                + len(sum_square_list_bf16)
                + len(sum_square_list_fp32)
                == 0
            ):
                return params_grads

            with p.block.program._optimized_guard([p, g]):
                sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"

                global_norm_var = []
                if len(sum_square_list_fp16) > 0:
                    global_norm_var_fp16 = _add_n(sum_square_list_fp16)
                    if (
                        sum_square_list_fp32
                        or sum_square_list
                        or not _allow_pure_fp16_global_norm_clip()
                    ):
                        global_norm_var.append(
                            global_norm_var_fp16.astype(sum_dtype)
                        )
                    else:
                        global_norm_var.append(global_norm_var_fp16)
                if len(sum_square_list_bf16) > 0:
                    global_norm_var_bf16 = _add_n(sum_square_list_bf16)
                    if (
                        sum_square_list_fp32
                        or sum_square_list
                        or not _allow_pure_bf16_global_norm_clip()
                    ):
                        global_norm_var.append(
                            global_norm_var_bf16.astype(sum_dtype)
                        )
                    else:
                        global_norm_var.append(global_norm_var_bf16)
                if len(sum_square_list_fp32) > 0:
                    global_norm_var_fp32 = _add_n(sum_square_list_fp32)
                    if sum_dtype == 'float32':
                        global_norm_var.append(global_norm_var_fp32)
                    else:
                        global_norm_var.append(
                            global_norm_var_fp32.astype(sum_dtype)
                        )
                if len(sum_square_list) > 0:
                    # fp64
                    global_norm_var_other_dtype = _add_n(sum_square_list)
                    global_norm_var.append(global_norm_var_other_dtype)

                global_norm_var = (
                    _add_n(global_norm_var)
                    if len(global_norm_var) > 1
                    else global_norm_var[0]
                )
                global_norm_var = paddle.sqrt(x=global_norm_var)
                max_global_norm = paddle.full(
                    shape=[1],
                    dtype=global_norm_var.dtype,
                    fill_value=self.clip_norm,
                )
                scale_var = paddle.divide(
                    x=max_global_norm,
                    y=paddle.maximum(x=max_global_norm, y=global_norm_var),
                )
            param_new_grad_name_dict = {}
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, 'need_clip', True) is False:
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_g = _cast_to_mp_type_if_enabled(g)
                    # inplace
                    if (
                        new_g.dtype == core.VarDesc.VarType.FP16
                        and scale_var.dtype != core.VarDesc.VarType.FP16
                    ):
                        scale_input = scale_var.astype('float16')
                    elif (
                        new_g.dtype == core.VarDesc.VarType.BF16
                        and scale_var.dtype != core.VarDesc.VarType.BF16
                    ):
                        scale_input = scale_var.astype('bfloat16')
                    else:
                        scale_input = scale_var
                    # NOTE(Yuang Liu): For pure dp with gradient merge, the p and g
                    # will be in different blocks with the gradient clip related ops.
                    # We need to handle the correct block, otherwise will encounter
                    # a 'NotFoundError' during compile time.
                    block = default_main_program().current_block()
                    block.append_op(
                        type='elementwise_mul',
                        inputs={'X': new_g, 'Y': scale_input},
                        outputs={'Out': new_g},
                    )
                    if new_g is not g:
                        block.append_op(
                            type='cast',
                            inputs={'X': new_g},
                            outputs={'Out': g},
                            attrs={
                                'in_dtype': new_g.dtype,
                                'out_dtype': g.dtype,
                            },
                        )

                param_new_grad_name_dict[p.name] = g.name
                params_and_grads.append((p, g))

        _correct_clip_op_role_var(params_and_grads, param_new_grad_name_dict)
        return params_and_grads

    def _process_context(self, context, param, grad):
        if self.group_name not in context:
            context[self.group_name] = []
            context[self.group_name + "_clip_value"] = self.clip_norm
            context[self.group_name + "_clip"] = paddle.full(
                shape=[1], dtype=grad.dtype, fill_value=self.clip_norm
            )
        else:
            if not self.clip_norm == context[self.group_name + "_clip_value"]:
                raise ValueError(
                    "All parameters' 'clip_norm' of a same group should be the same"
                )

        merge_grad = grad
        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            merge_grad = merge_selected_rows(grad)
            merge_grad = get_tensor_from_selected_rows(merge_grad)
        elif in_pir_mode() and grad.is_selected_row_type():
            merge_grad = merge_selected_rows(grad)
            merge_grad = get_tensor_from_selected_rows(merge_grad)

        local_norm_var = _squared_l2_norm(merge_grad)
        context[self.group_name].append(local_norm_var)

        self.context = context

    def _create_operators(self, param, grad):
        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        group_scale_name = self.group_name + "_scale"
        if group_scale_name not in self.context:
            group_norm_var = async_add_n(self.context[self.group_name])
            group_norm_var = paddle.sqrt(x=group_norm_var)
            clip_var = self.context[self.group_name + "_clip"]
            group_scale_var = paddle.divide(
                x=clip_var,
                y=paddle.maximum(x=clip_var, y=group_norm_var),
            )
            assert group_scale_var.shape == (1,)
            self.context[group_scale_name] = group_scale_var

        if in_pir_mode():
            grad = paddle.multiply(grad, self.context[group_scale_name])
            return param, grad

        # inplace
        param.block.append_op(
            type='elementwise_mul',
            inputs={'X': grad, 'Y': self.context[group_scale_name]},
            outputs={'Out': grad},
        )

        return param, grad


@framework.dygraph_not_support
def set_gradient_clip(clip, param_list=None, program=None):
    """
    Warning:

        This API must be used after building network, and before ``minimize`` ,
        and it may be removed in future releases, so it is not recommended.
        It is recommended to set ``grad_clip`` when initializing the ``optimizer`` ,
        this is a better method to clip gradient. There are three clipping strategies:
         :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
         :ref:`api_paddle_nn_ClipGradByValue` .

    To specify parameters that require gradient clip.

    Args:
        grad_clip (GradientClipBase, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default value: None, and there is no
            gradient clipping.
        param_list (list(Variable), optional): Parameters that require gradient clip.
                It can be a list of parameter or a list of parameter's name.
                Default None, meaning that all parameters in the program will be included.
        program (Program, optional): The program where parameters are located.
                Default None, meaning that using :ref:`api_paddle_static_default_main_program` .

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()

            >>> def network():
            ...     image = paddle.static.data(name='image', shape=[
            ...                        None, 28], dtype='float32')
            ...     param_attr1 = paddle.ParamAttr("fc1_param")
            ...     fc1 = paddle.static.nn.fc(image, size=10, weight_attr=param_attr1)
            ...     param_attr2 = paddle.ParamAttr("fc2_param")
            ...     fc2 = paddle.static.nn.fc(fc1, size=10, weight_attr=param_attr2)
            ...     loss = paddle.mean(fc2)
            ...     return loss


            >>> # network 1: clip all parameter gradient
            >>> with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            ...     loss = network()
            ...     paddle.nn.clip.set_gradient_clip(
            ...         paddle.nn.ClipGradByGlobalNorm(clip_norm=2.0))
            ...     sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            ...     sgd.minimize(loss)

            >>> # network 2: clip parameter gradient by name
            >>> with paddle.static.program_guard(base.Program(), paddle.static.Program()):
            ...     loss = network()
            ...     paddle.nn.clip.set_gradient_clip(
            ...         paddle.nn.ClipGradByValue(min=-1.0, max=1.0),
            ...         param_list=["fc1_param", "fc2_param"])
            ...     sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            ...     sgd.minimize(loss)

            >>> # network 3: clip parameter gradient by value
            >>> with paddle.static.program_guard(base.Program(), paddle.static.Program()):
            ...     loss = network()
            ...     param_var1 = paddle.static.default_main_program().global_block().var("fc1_param")
            ...     param_var2 = paddle.static.default_main_program().global_block().var("fc2_param")
            ...     paddle.nn.clip.set_gradient_clip(
            ...         paddle.nn.ClipGradByValue(min=-1.0, max=1.0),
            ...         param_list=[param_var1, param_var2])
            ...     sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            ...     sgd.minimize(loss)

            >>> # network 4: use 'set_gradient_clip' and 'optimize(grad_clip=clip)' together
            >>> with paddle.static.program_guard(base.Program(), paddle.static.Program()):
            ...     loss = network()
            ...     clip1 = paddle.nn.ClipGradByValue(min=-1.0, max=1.0)
            ...     clip2 = paddle.nn.ClipGradByNorm(clip_norm=1.0)
            ...     # Set the gradient clipping strategy: clip1
            ...     paddle.nn.clip.set_gradient_clip(clip1)
            ...     # Set the gradient clipping strategy: clip2
            ...     sgd = paddle.optimizer.SGD(learning_rate=1e-3, grad_clip=clip2)
            ...     sgd.minimize(loss)
            ...     # 'set_gradient_clip' will not take effect when setting has a conflict,
            ...     # and the gradient clipping strategy will be 'clip2'


    """
    warnings.warn(
        "Caution! 'set_gradient_clip' is not recommended "
        "and may be deprecated in future! "
        "We recommend a new strategy: set 'grad_clip' "
        "when initializing the 'optimizer'. "
        "This method can reduce the mistakes, please "
        "refer to documentation of 'optimizer'."
    )

    if not isinstance(clip, ClipGradBase):
        raise TypeError(
            "'clip' should be an instance of ClipGradBase's derived class"
        )
    if program is None:
        program = framework.default_main_program()

    for op in program.block(0).ops:
        if 'op_namescope' in op.all_attrs() and "optimizer" in op.attr(
            "op_namescope"
        ):
            warnings.warn(
                "'minimize' has been invoked before, this will make 'set_gradient_clip' "
                "be ineffective! Please invoke 'set_gradient_clip' before 'minimize'."
            )
            break

    if param_list is None:
        param_list = program.block(0).all_parameters()
    if all(isinstance(elem, str) for elem in param_list):
        param_list = [program.block(0).var(elem) for elem in param_list]
    if not all(isinstance(elem, framework.Parameter) for elem in param_list):
        raise TypeError(
            "'param_list' should be a list of Parameter or basestring(parameter's name)."
        )

    for param in param_list:
        param.gradient_clip_attr = copy.deepcopy(clip)


def append_gradient_clip_ops(param_grads):
    context = {}
    for p, g in param_grads:
        if g is None:
            continue
        with p.block.program._optimized_guard([p, g]), framework.name_scope(
            'gradient_clip'
        ):
            clip_attr = getattr(p, 'gradient_clip_attr', None)
            if clip_attr is None:
                return param_grads
            if not isinstance(clip_attr, ClipGradBase):
                raise TypeError(
                    "clip attribute should be an instance of GradientClipBase"
                )

            clip_attr._process_context(context=context, param=p, grad=g)

    res = []
    param_new_grad_name_dict = {}
    for p, g in param_grads:
        if g is None:
            continue
        with p.block.program._optimized_guard([p, g]), framework.name_scope(
            'gradient_clip'
        ):
            param, new_grad = clip_attr._create_operators(param=p, grad=g)
            param_new_grad_name_dict[param.name] = new_grad.name
            res.append([param, new_grad])

    _correct_clip_op_role_var(res, param_new_grad_name_dict)
    return res


# change wrong mapping relation between param & grad in clip op
# Note: This function is sensitive to the time cost of the network with gradient clipping
# and should not be changed easily. If you must change, please test the time cost.
def _correct_clip_op_role_var(params_grads, param_new_grad_name_dict):
    block_id_list = []
    if len(param_new_grad_name_dict) == 0:
        return
    for param, grad in params_grads:
        if grad is None:
            continue
        block_id = param.block.idx
        if block_id in block_id_list:
            continue
        block_id_list.append(block_id)
        for op in param.block.program.global_block().ops:
            if (
                op.has_attr("op_namescope")
                and "gradient_clip" in op.attr("op_namescope")
                and op.attr('op_role_var')
            ):
                param_name = op.attr('op_role_var')[0]
                if param_name in param_new_grad_name_dict:
                    correct_p_g = [
                        param_name,
                        param_new_grad_name_dict[param_name],
                    ]
                    op._set_attr('op_role_var', correct_p_g)


GradientClipBase = ClipGradBase
GradientClipByValue = ClipGradByValue
GradientClipByNorm = ClipGradByNorm
GradientClipByGlobalNorm = ClipGradByGlobalNorm
