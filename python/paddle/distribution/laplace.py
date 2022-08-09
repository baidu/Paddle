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

import numpy as np
import paddle
from paddle.distribution import distribution
try:
    from collections.abc import Iterable
except:
    from collections import Iterable


class Laplace(distribution.Distribution):
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Examples:
        .. code-block:: python
                        m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                        m.sample()  # Laplace distributed with loc=0, scale=1
                        # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                        # [3.68546247])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """

    def __init__(self, loc, scale, name=None):

        if type(loc) != type(scale):
            raise TypeError("type of loc and scale must be identical!")

        self.batch_size_unknown = False
        self.name = name if name is not None else 'Laplace'
        self.dtype = 'float32'

        if self._validate_args(loc, scale):
            # is_variable
            if paddle.is_integer(loc):
                loc = paddle.cast(loc, self.dtype)
            if paddle.is_integer(scale):
                scale = paddle.cast(scale, self.dtype)
        else:
            # is_number
            loc = paddle.to_tensor(loc, dtype=self.dtype)
            scale = paddle.to_tensor(scale, dtype=self.dtype)
            self.batch_size_unknown = True
        self.loc = loc
        self.scale = scale

        super(Laplace, self).__init__(self.loc.shape)

    @property
    def mean(self):
        """Mean of distribution
        Returns:
            Tensor: mean value.
        """
        return self.loc

    @property
    def stddev(self):
        """standard deviation
        Returns:
            Tensor: std value.
        """
        return (2**0.5) * self.scale

    @property
    def variance(self):
        """Variance of distribution
        Returns:
            Tensor: variance value.
        """
        return self.stddev.pow(2)

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is same with value.

        Examples:
            .. code-block:: python

                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            value = paddle.to_tensor([0.1])
                            m.log_prob(value) 
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [-0.79314721])

        """
        if value.dtype != self.scale.dtype:
            value = paddle.cast(value, self.scale.dtype)
        log_scale = -paddle.log(2 * self.scale)

        return (log_scale - paddle.abs(value - self.loc) / self.scale)

    def entropy(self):
        """Entropy of Laplace distribution.

        Returns:
            Entropy of distribution.

        Examples:
            .. code-block:: python
                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m.entropy()
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [1.69314718])
        """
        return 1 + paddle.log(2 * self.scale)

    def cdf(self, value):
        """Cumulative distribution function
        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: cumulative probability of value.
        
        Examples:
            .. code-block:: python

                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            value = paddle.to_tensor([0.1])
                            m.cdf(value)
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [0.54758132])
        """
        if value.dtype != self.scale.dtype:
            value = paddle.cast(value, self.scale.dtype)
        iterm = (0.5 * (value - self.loc).sign() *
                 paddle.expm1(-(value - self.loc).abs() / self.scale))
        return 0.5 - iterm

    def icdf(self, value):
        """Inverse Cumulative distribution function
        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: cumulative probability of value.

        Examples:
            .. code-block:: python

                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            value = paddle.to_tensor([0.1])
                            m.icdf(value)
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [-1.60943794])
        """
        if value.dtype != self.scale.dtype:
            value = paddle.cast(value, self.scale.dtype)
        term = value - 0.5
        return (self.loc - self.scale *
                (term).sign() * paddle.log1p(-2 * term.abs()))

    def sample(self, shape=(), seed=0):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        Examples:
            .. code-block:: python
                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m.sample()  # Laplace distributed with loc=0, scale=1
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [3.68546247])
        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        with paddle.no_grad():
            return self.rsample(shape, seed=seed)

    def rsample(self, shape, seed=0):
        """reparameterized sample
        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        Examples:
            .. code-block:: python

                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m.rsample([1])  # Laplace distributed with loc=0, scale=1
                            # Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [[0.04337667]])
        """

        eps = self.get_eps()
        if self.batch_size_unknown and len(shape) == 0:
            shape = (1, )
        shape = self._extend_shape(shape)
        u = paddle.uniform(shape=shape, min=eps - 1, max=1, seed=seed)

        return (self.loc - self.scale * u.sign() * paddle.log1p(-u.abs()))

    def get_eps(self):
        """
        Get eps of certain data type.

        Note: 
            Since paddle.finfo is temporarily unavailable, we 
            use hard-coding style to get eps value.

        Returns:
            Float: eps value by different data types.
        """
        if (self.loc.dtype == paddle.float16 or self.loc.dtype == paddle.float32
                or self.loc.dtype == paddle.complex64):
            eps = 1.19209e-07
        elif (self.loc.dtype == paddle.float64
              or self.loc.dtype == paddle.complex128):
            eps = 2.22045e-16
        else:
            raise TypeError("self.loc requires a floating point type")

        return eps

    def _extend_shape(self, sample_shape):
        """compute shape of the sample 

        Args:
            sample_shape (list or tuple): sample shape

        Returns:
            Tensor: generated sample data shape
        
        Examples:
            .. code-block:: python
                            m = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m._extend_shape([1, 2])
                            # [1, 2, 1]
        """

        if self.batch_size_unknown:
            self._batch_shape = ()

        expanded_shape = super(Laplace, self)._extend_shape(tuple(sample_shape))
        return expanded_shape

    def kl_divergence(self, other):
        """The KL-divergence between two laplace distributions.

        Args:
            other (Laplace): instance of Laplace.

        Returns:
            Tensor: kl-divergence between two laplace distributions.

        Examples:
            .. code-block:: python
                            m1 = Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m2 = Laplace(paddle.to_tensor([1.0]), paddle.to_tensor([0.5]))
                            m1.kl_divergence(m2)
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [1.04261160])
        """

        var_ratio = other.scale / self.scale
        t = paddle.abs(self.loc - other.loc)
        term1 = ((self.scale * paddle.exp(-t / self.scale) + t) / other.scale)
        term2 = paddle.log(var_ratio)
        return term1 + term2 - 1
