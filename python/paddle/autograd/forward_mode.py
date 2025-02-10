# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from collections import namedtuple
from contextlib import ContextDecorator

from paddle.framework import core

# TODO(alband): Once most of the formulas are implemented, these functions need to be added
# to the main doc to make them fully "public".

__all__ = [
    "UnpackedDualTensor",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
    "dual_level",
]


# Global variable used to make the python API simpler to use
_current_level = -1


def enter_dual_level():
    r"""Function that can be used to enter a new forward grad level.
    This level can be used to make and unpack dual Tensors to compute
    forward gradients.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    new_level = core._enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError(
            "Entering a new forward AD level but the current level "
            "is not valid. Make sure you did not modified it directly."
        )
    _current_level = new_level
    return new_level


def exit_dual_level(*, level=None):
    r"""Function that can be used to exit a forward grad level.
    This function deletes all the gradients associated with this
    level. Only deleting the latest entered level is allowed.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError(
            "Trying to exit a forward AD level that was not the last one "
            "that was created. This is not supported."
        )
    core._exit_dual_level(level)
    _current_level = level - 1


def make_dual(tensor, tangent, *, level=None):
    r"""Function that creates a "dual object" that can be used to compute forward AD gradients
    based on the given Tensor and its tangent. It returns a new Tensor that shares memory with
    :attr:`tensor` and the :attr:`tangent` is used as-is.

    This function is backward differentiable.

    Given a function `f` whose jacobian is `J`, it allows to compute the jacobian vector product,
    named `jvp`, between `J` and a given vector `v` as follows.

    Example::
        >>> inp = make_dual(x, v)
        >>> out = f(inp)
        >>> y, jvp = unpack_dual(out)

    """
    if level is None:
        level = _current_level

    if level < 0:
        raise RuntimeError(
            "Trying to create a dual Tensor for forward AD but no level "
            "exists, make sure to enter_dual_level() first."
        )

    return core.make_dual(tensor, tangent, level)


_UnpackedDualTensor = namedtuple("_UnpackedDualTensor", ["primal", "tangent"])


class UnpackedDualTensor(_UnpackedDualTensor):
    r"""Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.

    See :func:`unpack_dual` for more details.

    """


def unpack_dual(tensor, *, level=None):
    r"""Function that unpacks a "dual object" to recover two plain tensors, one representing
    the primal and the other the tangent (both are views of :attr:`tensor`. Neither of these
    tensors can be dual tensor of level :attr:`level`.

    This function is backward differentiable.
    """
    if level is None:
        level = _current_level

    if level < 0:
        return tensor, None

    return core.unpack_dual(tensor, level)


class dual_level(ContextDecorator):
    r"""Context-manager that controls the current forward ad level. It
    appropriately enters and exit the dual level.

    This function also updates the current level that is used by default
    by the other functions in this API.

    Example::

        >>> x = paddle.tensor([1])
        >>> x_t = paddle.tensor([1])
        >>> with dual_level():
        ...   inp = make_dual(x, x_t)
        ...   # Do computations with inp
        ...   out = your_fn(inp)
        ...   _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    """

    def __init__(self):
        super().__init__()

    def __enter__(self):
        return enter_dual_level()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        exit_dual_level()
