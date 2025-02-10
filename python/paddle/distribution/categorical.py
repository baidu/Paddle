# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING, Union

import numpy as np

import paddle
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution import distribution
from paddle.framework import in_dynamic_mode
from paddle.tensor import multinomial

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import NestedSequence
    from paddle._typing.dtype_like import _DTypeLiteral

    _CategoricalBoundary: TypeAlias = Union[
        Sequence[float],
        NestedSequence[float],
        npt.NDArray[Union[np.float32, np.float64]],
        Tensor,
    ]


class Categorical(distribution.Distribution):
    r"""
    Categorical distribution is a discrete probability distribution that
    describes the possible results of a random variable that can take on
    one of K possible categories, with the probability of each category
    separately specified.

    The probability mass function (pmf) is:

    .. math::

        pmf(k; p_i) = \prod_{i=1}^{k} p_i^{[x=i]}

    In the above equation:

    * :math:`[x=i]` : it evaluates to 1 if :math:`x==i` , 0 otherwise.

    Args:
        logits(list|tuple|numpy.ndarray|Tensor): The logits input of categorical distribution. The data type is float32 or float64.
        name(str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Categorical

            >>> paddle.seed(100) # on CPU device
            >>> x = paddle.rand([6])
            >>> print(x)
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.55355281, 0.20714243, 0.01162981, 0.51577556, 0.36369765, 0.26091650])

            >>> paddle.seed(200) # on CPU device
            >>> y = paddle.rand([6])
            >>> print(y)
            Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.77663314, 0.90824795, 0.15685187, 0.04279523, 0.34468332, 0.79557180])

            >>> cat = Categorical(x)
            >>> cat2 = Categorical(y)

            >>> paddle.seed(1000) # on CPU device
            >>> print(cat.sample([2, 3]))
            Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [[0, 1, 5],
            [3, 4, 5]])

            >>> print(cat.entropy())
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.77528250)

            >>> print(cat.kl_divergence(cat2))
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.07195196])

            >>> value = paddle.to_tensor([2, 1, 3])
            >>> print(cat.probs(value))
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.00608027, 0.10829761, 0.26965630])

            >>> print(cat.log_prob(value))
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-5.10270691, -2.22287226, -1.31060708])
    """

    logits: Tensor
    dtype: _DTypeLiteral

    def __init__(
        self,
        logits: _CategoricalBoundary,
        name: str | None = None,
    ) -> None:
        """
        Args:
            logits(list|tuple|numpy.ndarray|Tensor): The logits input of categorical distribution. The data type is float32 or float64.
            name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        """
        if not in_dynamic_mode():
            check_type(
                logits,
                'logits',
                (np.ndarray, Variable, paddle.pir.Value, list, tuple),
                'Categorical',
            )

        self.name = name if name is not None else 'Categorical'
        self.dtype = 'float32'

        if self._validate_args(logits):
            self.logits = logits
            self.dtype = convert_dtype(logits.dtype)
        else:
            if isinstance(logits, np.ndarray) and str(logits.dtype) in [
                'float32',
                'float64',
            ]:
                self.dtype = convert_dtype(logits.dtype)
            self.logits = self._to_tensor(logits)[0]
            if self.dtype != convert_dtype(self.logits.dtype):
                self.logits = paddle.cast(self.logits, dtype=self.dtype)
        self._prob = paddle.nn.functional.softmax(self.logits, axis=-1)

    def sample(self, shape: Sequence[int] = []) -> Tensor:
        """Generate samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Shape of the generated samples.

        Returns:
            Tensor: A tensor with prepended dimensions shape.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Categorical

                >>> paddle.seed(100) # on CPU device
                >>> x = paddle.rand([6])
                >>> print(x)
                Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.55355281, 0.20714243, 0.01162981, 0.51577556, 0.36369765, 0.26091650])

                >>> # doctest: +SKIP('Random output')
                >>> cat = Categorical(x)
                >>> paddle.seed(1000) # on CPU device
                >>> print(cat.sample([2, 3]))
                Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[0, 1, 5],
                [3, 4, 5]])
        """
        name = self.name + '_sample'
        if not in_dynamic_mode():
            check_type(shape, 'shape', (list, tuple), 'sample')

        num_samples = np.prod(np.array(shape))

        logits_shape = list(self.logits.shape)
        if len(logits_shape) > 1:
            sample_shape = shape + logits_shape[:-1]
            logits = paddle.reshape(
                self.logits, [np.prod(logits_shape[:-1]), logits_shape[-1]]
            )
        else:
            sample_shape = shape
            logits = self.logits

        sample_index = multinomial(
            self._logits_to_probs(logits), num_samples, True
        )

        # multinomial sample shape is (logits.shape[:-1], num_samples), need to
        # transpose to (num_samples, logits.shape[:-1])
        permute = list(range(sample_index.dim()))
        permute.insert(0, permute.pop(-1))
        sample_index = sample_index.transpose(permute)

        return paddle.reshape(sample_index, sample_shape, name=name)

    def kl_divergence(self, other: Categorical) -> Tensor:
        """The KL-divergence between two Categorical distributions.

        Args:
            other (Categorical): instance of Categorical. The data type is float32.

        Returns:
            Tensor: kl-divergence between two Categorical distributions.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Categorical

                >>> paddle.seed(100) # on CPU device
                >>> x = paddle.rand([6])
                >>> print(x)
                Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.55355281, 0.20714243, 0.01162981, 0.51577556, 0.36369765, 0.26091650])

                >>> paddle.seed(200) # on CPU device
                >>> y = paddle.rand([6])
                >>> print(y)
                Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.77663314, 0.90824795, 0.15685187, 0.04279523, 0.34468332, 0.79557180])

                >>> cat = Categorical(x)
                >>> cat2 = Categorical(y)

                >>> print(cat.kl_divergence(cat2))
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.07195196])
        """
        name = self.name + '_kl_divergence'
        if not in_dynamic_mode():
            check_type(other, 'other', Categorical, 'kl_divergence')

        logits = self.logits - paddle.max(self.logits, axis=-1, keepdim=True)
        other_logits = other.logits - paddle.max(
            other.logits, axis=-1, keepdim=True
        )
        e_logits = paddle.exp(logits)
        other_e_logits = paddle.exp(other_logits)
        z = paddle.sum(e_logits, axis=-1, keepdim=True)
        other_z = paddle.sum(other_e_logits, axis=-1, keepdim=True)
        prob = e_logits / z
        kl = paddle.sum(
            prob
            * (logits - paddle.log(z) - other_logits + paddle.log(other_z)),
            axis=-1,
            keepdim=True,
            name=name,
        )

        return kl

    def entropy(self) -> Tensor:
        """Shannon entropy in nats.

        Returns:
            Tensor: Shannon entropy of Categorical distribution. The data type is float32.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Categorical

                >>> paddle.seed(100) # on CPU device
                >>> x = paddle.rand([6])
                >>> print(x)
                Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.55355281, 0.20714243, 0.01162981, 0.51577556, 0.36369765, 0.26091650])

                >>> cat = Categorical(x)

                >>> print(cat.entropy())
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                1.77528250)
        """
        name = self.name + '_entropy'
        logits = self.logits - paddle.max(self.logits, axis=-1, keepdim=True)
        e_logits = paddle.exp(logits)
        z = paddle.sum(e_logits, axis=-1, keepdim=True)
        prob = e_logits / z

        neg_entropy = paddle.sum(prob * (logits - paddle.log(z)), axis=-1)
        entropy = paddle.scale(neg_entropy, scale=-1.0, name=name)
        return entropy

    def probs(self, value: Tensor) -> Tensor:
        """Probabilities of the given category (``value``).

        If ``logits`` is 2-D or higher dimension, the last dimension will be regarded as
        category, and the others represents the different distributions.
        At the same time, if ``value`` is 1-D Tensor, ``value`` will be broadcast to the
        same number of distributions as ``logits``.
        If ``value`` is not 1-D Tensor, ``value`` should have the same number distributions
        with ``logits. That is, ``value[:-1] = logits[:-1]``.

        Args:
            value (Tensor): The input tensor represents the selected category index.

        Returns:
            Tensor: probability according to the category index.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Categorical

                >>> paddle.seed(100) # on CPU device
                >>> x = paddle.rand([6])
                >>> print(x)
                Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.55355281, 0.20714243, 0.01162981, 0.51577556, 0.36369765, 0.26091650])

                >>> cat = Categorical(x)

                >>> value = paddle.to_tensor([2, 1, 3])
                >>> print(cat.probs(value))
                Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.00608027, 0.10829761, 0.26965630])
        """
        name = self.name + '_probs'
        if len(self._prob.shape) == 1:  # batch_shape is empty
            return paddle.gather(
                self._prob, value.reshape([-1], name=name), name=name
            ).reshape(value.shape, name=name)
        else:
            if len(value.shape) == 1:
                return paddle.take_along_axis(
                    self._prob,
                    paddle.reshape(
                        value,
                        (len(self._prob.shape) - 1) * [1] + [-1],
                        name=name,
                    ),
                    axis=-1,
                )
            else:
                return paddle.take_along_axis(self._prob, value, axis=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probabilities of the given category. Refer to ``probs`` method.

        Args:
            value (Tensor): The input tensor represents the selected category index.

        Returns:
            Tensor: Log probability.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Categorical

                >>> paddle.seed(100) # on CPU device
                >>> x = paddle.rand([6])
                >>> print(x)
                Tensor(shape=[6], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.55355281, 0.20714243, 0.01162981, 0.51577556, 0.36369765, 0.26091650])

                >>> cat = Categorical(x)

                >>> value = paddle.to_tensor([2, 1, 3])
                >>> print(cat.log_prob(value))
                Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                [-5.10270691, -2.22287226, -1.31060708])
        """
        name = self.name + '_log_prob'

        return paddle.log(self.probs(value), name=name)
