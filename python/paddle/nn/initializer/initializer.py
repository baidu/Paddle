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

import functools
import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import TypeAlias

from ...base.framework import (
    EagerParamBase,
    default_main_program,
    in_dygraph_mode,
    use_pir_api,
)
from .lazy_init import lazy_init_helper

if TYPE_CHECKING:
    import paddle

    _NonLinearity: TypeAlias = Literal[  # noqa: PYI047
        "sigmoid",
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv1d_transpose",
        "conv2d_transpose",
        "conv3d_transpose",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
    ]

__all__ = []


class Initializer:
    """Base class for parameter initializers

    Defines the common interface of parameter initializers.
    They add operations to the init program that are used
    to initialize parameter. Users should not use this class
    directly, but need to use one of its implementations.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self, param: paddle.Tensor, block: paddle.pir.Block | None = None
    ):
        if not lazy_init_helper().state:
            return self.forward(param, block)

        return self._lazy_init(param, block)

    def forward(
        self, param: paddle.Tensor, block: paddle.pir.Block | None = None
    ) -> paddle.Tensor | None:
        """Add corresponding initialization operations to the network."""
        raise NotImplementedError

    def _lazy_init(
        self, param: paddle.Tensor, block: paddle.pir.Block | None = None
    ):
        """
        Apply lazy initialization
        """
        assert in_dygraph_mode()

        def init_op_creator(
            forward, param: paddle.Tensor, block: paddle.pir.Block | None
        ):
            if use_pir_api():
                new_var = param
            else:
                new_var = param._to_static_var(True, block=block)
            # Record initializer operator
            with lazy_init_helper():
                forward(new_var, block)

        # Add hook function for initializing param in dygraph mode
        param.set_init_func(functools.partial(self.forward))
        param._init_op_creator = functools.partial(
            init_op_creator, self.forward
        )

        return param

    def _check_block(self, block: paddle.pir.Block | None) -> paddle.pir.Block:
        if block is None:
            block = default_main_program().global_block()

        return block

    def _compute_fans(self, var: paddle.Tensor) -> tuple[int, int]:
        """Compute the fan_in and the fan_out for layers

        This method computes the fan_in and the fan_out
        for neural network layers, if not specified. It is
        not possible to perfectly estimate fan_in and fan_out.
        This method will estimate it correctly for matrix multiply and
        convolutions.

        Args:
            var: variable for which fan_in and fan_out have to be computed.

        Returns:
            tuple of two integers (fan_in, fan_out).
        """
        shape = (
            var._local_shape
            if (isinstance(var, EagerParamBase) and var.is_dist())
            else var.shape
        )
        if not shape or len(shape) == 0:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assume this to be a convolutional kernel
            # In PaddlePaddle, the shape of the kernel is like:
            # [num_filters, num_filter_channels, ...] where the remaining
            # dimensions are the filter_size
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size

        return (fan_in, fan_out)


def calculate_gain(
    nonlinearity: str, param: bool | float | None = None
) -> float:
    """
    Get the recommended ``gain`` value of some nonlinearity function. ``gain`` value can be used in some
    ``paddle.nn.initializer`` api to adjust the initialization value.

    Args:
        nonlinearity(str): name of nonlinearity activation function. If it is a linear function, such as:
            `linear/conv1d/conv2d/conv3d/conv1d_transpose/conv2d_transpose/conv3d_transpose` , 1.0 will be returned.
        param(bool|int|float|None, optional): optional parameter for some nonlinearity function. Now, it only applies to
            'leaky_relu'. Default: None, it will be calculated as 0.01 in the formula.

    Returns:
        A float value, which is the recommended gain for this nonlinearity function.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> gain = paddle.nn.initializer.calculate_gain('tanh')
            >>> print(gain)
            1.6666666666666667
            >>> # 5.0 / 3
            >>> gain = paddle.nn.initializer.calculate_gain('leaky_relu', param=1.0)
            >>> print(gain)
            1.0
            >>> # math.sqrt(2.0 / (1+param^2))
            >>> initializer = paddle.nn.initializer.Orthogonal(gain)

    """
    if param is None:
        param = 0.01
    else:
        assert isinstance(param, (bool, int, float))
        param = float(param)
    recommended_gain = {
        'sigmoid': 1,
        'linear': 1,
        'conv1d': 1,
        'conv2d': 1,
        'conv3d': 1,
        'conv1d_transpose': 1,
        'conv2d_transpose': 1,
        'conv3d_transpose': 1,
        'tanh': 5.0 / 3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + param**2)),
        'selu': 3.0 / 4,
    }
    if nonlinearity in recommended_gain.keys():
        return recommended_gain[nonlinearity]
    else:
        raise ValueError(
            f"nonlinearity function {nonlinearity} is not supported now."
        )
