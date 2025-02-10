#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the initializers of Kaiming functions in neural network
import math
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops

from ...base import core, framework, unique_name
from ...base.framework import (
    _current_expected_place,
    in_dygraph_mode,
    in_pir_mode,
)
from .initializer import Initializer, calculate_gain

if TYPE_CHECKING:
    from .initializer import _NonLinearity

__all__ = []


class MSRAInitializer(Initializer):
    r"""Implements the MSRA initializer a.k.a. Kaiming Initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = gain \times \sqrt{\frac{3}{fan\_in}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \frac{gain}{\sqrt{{fan\_in}}}

    Args:
        uniform (bool, optional): whether to use uniform or normal distribution. Default is True.
        fan_in (float32|None, optional): fan_in (in_features) of trainable Tensor, If None, it will be inferred automatically. If you don't want to use in_features of the Tensor, you can set the value of 'fan_in' smartly by yourself. Default is None.
        seed (int32, optional): random seed. Default is 0.
        negative_slope (float, optional): negative_slope (only used with leaky_relu). Default is 0.0.
        nonlinearity(str, optional): the non-linear function. Default is relu.

    Note:
        It is recommended to set fan_in to None for most cases.

    """

    def __init__(
        self,
        uniform: bool = True,
        fan_in: float | None = None,
        seed: int = 0,
        negative_slope: float = 0,
        nonlinearity: _NonLinearity = 'relu',
    ) -> None:
        """Constructor for MSRAInitializer"""
        assert uniform is not None
        assert seed is not None
        super().__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._seed = seed
        self._negative_slope = negative_slope
        self._nonlinearity = nonlinearity

    def forward(
        self, var: paddle.Tensor, block: paddle.pir.Block | None = None
    ) -> paddle.Tensor | None:
        """Initialize the input tensor with MSRA initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block|None, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op.
        """
        assert not (
            isinstance(var, framework.EagerParamBase) and var.is_dist()
        ), "Currently, kaiming initializer not support lazy init for dist param."
        block = self._check_block(block)
        assert isinstance(
            var, (framework.Variable, paddle.pir.core.ParameterMeta)
        )
        assert isinstance(block, (framework.Block, paddle.pir.Block))
        f_in, f_out = self._compute_fans(var)

        # If fan_in is passed, use it
        fan_in = f_in if self._fan_in is None else self._fan_in

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initializers
        origin_dtype = var.dtype
        if origin_dtype == core.VarDesc.VarType.FP16 or (
            origin_dtype == core.VarDesc.VarType.BF16 and not self._uniform
        ):
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['masra_init', var.name, 'tmp'])
                ),
                shape=var.shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
            )
        elif (
            origin_dtype in (core.DataType.FLOAT16, core.DataType.BFLOAT16)
            and not self._uniform
        ):
            out_dtype = core.DataType.FLOAT32
            out_var = var
        else:
            out_dtype = origin_dtype
            out_var = var

        if in_dygraph_mode():
            if self._uniform:
                gain = calculate_gain(self._nonlinearity, self._negative_slope)
                limit = gain * math.sqrt(3.0 / float(fan_in))
                out_var = _C_ops.uniform(
                    var.shape,
                    out_dtype,
                    -limit,
                    limit,
                    self._seed,
                    _current_expected_place(),
                )
            else:
                gain = calculate_gain(self._nonlinearity, self._negative_slope)
                std = gain / math.sqrt(float(fan_in))
                place = _current_expected_place()
                out_var = _C_ops.gaussian(
                    out_var.shape, 0.0, std, self._seed, out_dtype, place
                )

            if origin_dtype == core.VarDesc.VarType.FP16 or (
                origin_dtype
                in [
                    core.VarDesc.VarType.BF16,
                    core.DataType.FLOAT16,
                    core.DataType.BFLOAT16,
                ]
                and not self._uniform
            ):
                var_tmp = _C_ops.cast(out_var, origin_dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        elif in_pir_mode():
            if self._uniform:
                gain = calculate_gain(self._nonlinearity, self._negative_slope)
                limit = gain * math.sqrt(3.0 / float(fan_in))
                out_var = _C_ops.uniform(
                    var.shape,
                    out_dtype,
                    -limit,
                    limit,
                    self._seed,
                    _current_expected_place(),
                )
            else:
                gain = calculate_gain(self._nonlinearity, self._negative_slope)
                std = gain / math.sqrt(float(fan_in))
                place = _current_expected_place()
                out_var = _C_ops.gaussian(
                    out_var.shape, 0.0, std, self._seed, out_dtype, place
                )

            if (
                origin_dtype in (core.DataType.FLOAT16, core.DataType.BFLOAT16)
                and not self._uniform
            ):
                return _C_ops.cast(out_var, origin_dtype)

            return out_var
        else:
            if self._uniform:
                gain = calculate_gain(self._nonlinearity, self._negative_slope)
                limit = gain * math.sqrt(3.0 / float(fan_in))
                op = block.append_op(
                    type="uniform_random",
                    inputs={},
                    outputs={"Out": out_var},
                    attrs={
                        "shape": out_var.shape,
                        "dtype": int(out_dtype),
                        "min": -limit,
                        "max": limit,
                        "seed": self._seed,
                    },
                    stop_gradient=True,
                )

            else:
                gain = calculate_gain(self._nonlinearity, self._negative_slope)
                std = gain / math.sqrt(float(fan_in))
                op = block.append_op(
                    type="gaussian_random",
                    outputs={"Out": out_var},
                    attrs={
                        "shape": out_var.shape,
                        "dtype": int(out_dtype),
                        "mean": 0.0,
                        "std": std,
                        "seed": self._seed,
                    },
                    stop_gradient=True,
                )

            if origin_dtype == core.VarDesc.VarType.FP16 or (
                origin_dtype == core.VarDesc.VarType.BF16 and not self._uniform
            ):
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={
                        "in_dtype": out_var.dtype,
                        "out_dtype": origin_dtype,
                    },
                )

            var.op = op
            return op


class KaimingNormal(MSRAInitializer):
    r"""Implements the Kaiming Normal initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \frac{gain}{\sqrt{{fan\_in}}}

    Args:
        fan_in (float32|None, optional): fan_in (in_features) of trainable Tensor, If None, it will be inferred automatically. If you don't want to use in_features of the Tensor, you can set the value of 'fan_in' smartly by yourself. Default is None.
        negative_slope (float, optional): negative_slope (only used with leaky_relu). Default is 0.0.
        nonlinearity(str, optional): the non-linear function. Default is relu.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> linear = nn.Linear(2, 4, weight_attr=nn.initializer.KaimingNormal())
            >>> data = paddle.rand([30, 10, 2], dtype='float32')
            >>> res = linear(data)

    """

    def __init__(
        self,
        fan_in: float | None = None,
        negative_slope: float = 0.0,
        nonlinearity: str = 'relu',
    ) -> None:
        super().__init__(
            uniform=False,
            fan_in=fan_in,
            seed=0,
            negative_slope=negative_slope,
            nonlinearity=nonlinearity,
        )


class KaimingUniform(MSRAInitializer):
    r"""Implements the Kaiming Uniform initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = gain \times \sqrt{\frac{3}{fan\_in}}

    Args:
        fan_in (float32|None, optional): fan_in (in_features) of trainable Tensor, If None, it will be inferred automatically. If you don't want to use in_features of the Tensor, you can set the value of 'fan_in' smartly by yourself. Default is None.
        negative_slope (float, optional): negative_slope (only used with leaky_relu). Default is 0.0.
        nonlinearity(str, optional): the non-linear function. Default is relu.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> linear = nn.Linear(2, 4, weight_attr=nn.initializer.KaimingUniform())
            >>> data = paddle.rand([30, 10, 2], dtype='float32')
            >>> res = linear(data)

    """

    def __init__(
        self,
        fan_in: float | None = None,
        negative_slope: float = 0.0,
        nonlinearity: str = 'relu',
    ) -> None:
        super().__init__(
            uniform=True,
            fan_in=fan_in,
            seed=0,
            negative_slope=negative_slope,
            nonlinearity=nonlinearity,
        )
