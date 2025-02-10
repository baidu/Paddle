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

from typing import TYPE_CHECKING

from .. import functional as F
from .layers import Layer

if TYPE_CHECKING:
    import paddle

__all__ = []


class PairwiseDistance(Layer):
    r"""

    It computes the pairwise distance between two vectors. The
    distance is calculated by p-order norm:

    .. math::

        \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right) ^ {1/p}.

    Parameters:
        p (float, optional): The order of norm. Default: :math:`2.0`.
        epsilon (float, optional): Add small value to avoid division by zero.
            Default: :math:`1e-6`.
        keepdim (bool, optional): Whether to reserve the reduced dimension
            in the output Tensor. The result tensor is one dimension less than
            the result of ``|x-y|`` unless :attr:`keepdim` is True. Default: False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`.
            Generally, no setting is required. Default: None.

    Shape:
        - x: :math:`[N, D]` or :math:`[D]`, where :math:`N` is batch size, :math:`D`
          is the dimension of the data. Available data type is float16, float32, float64.
        - y: :math:`[N, D]` or :math:`[D]`, y have the same dtype as x.
        - output: The same dtype as input tensor.
            - If :attr:`keepdim` is True, the output shape is :math:`[N, 1]` or :math:`[1]`,
              depending on whether the input has data shaped as :math:`[N, D]`.
            - If :attr:`keepdim` is False, the output shape is :math:`[N]` or :math:`[]`,
              depending on whether the input has data shaped as :math:`[N, D]`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1., 3.], [3., 5.]], dtype=paddle.float64)
            >>> y = paddle.to_tensor([[5., 6.], [7., 8.]], dtype=paddle.float64)
            >>> dist = paddle.nn.PairwiseDistance()
            >>> distance = dist(x, y)
            >>> print(distance)
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            [4.99999860, 4.99999860])
    """

    def __init__(
        self,
        p: float = 2.0,
        epsilon: float = 1e-6,
        keepdim: bool = False,
        name: str | None = None,
    ):
        super().__init__()
        self.p = p
        self.epsilon = epsilon
        self.keepdim = keepdim
        self.name = name

    def forward(self, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        return F.pairwise_distance(
            x, y, self.p, self.epsilon, self.keepdim, self.name
        )

    def extra_repr(self) -> str:
        main_str = 'p={p}'
        if self.epsilon != 1e-6:
            main_str += ', epsilon={epsilon}'
        if self.keepdim is not False:
            main_str += ', keepdim={keepdim}'
        if self.name is not None:
            main_str += ', name={name}'
        return main_str.format(**self.__dict__)
