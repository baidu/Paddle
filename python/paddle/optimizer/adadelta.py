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
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops
from paddle.base.framework import in_dynamic_or_pir_mode

from ..base import framework
from ..base.dygraph import no_grad
from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import NotRequired

    from paddle import Tensor
    from paddle.nn.clip import GradientClipBase
    from paddle.regularizer import WeightDecayRegularizer

    from .lr import LRScheduler
    from .optimizer import _ParameterConfig

    class _AdadeltaParameterConfig(_ParameterConfig):
        epsilon: NotRequired[float]
        rho: NotRequired[float]


__all__ = []


class Adadelta(Optimizer):
    r"""
    **Notes: This API does not support sparse parameter optimization.**

    Adadelta Optimizer. Please refer to this for details:
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_.

    The update is done as follows:

    .. math::

        E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2

        learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }

        E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2

    Args:
        learning_rate (float|Tensor|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        epsilon (float): a small float number for numeric stability. Default 1.0e-6.
        rho (float): a floating point value indicating the decay rate. Default 0.95.
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in parameter groups \
            represents the scale of base learning_rate. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (int|float|WeightDecayRegularizer|None, optional): The strategy of regularization. \
            It can be a int or float value as coeff of L2 regularization or \
            :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_paddle_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str|None, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.uniform([10, 10], dtype="float32", min=-0.1, max=0.1)
            >>> linear = paddle.nn.Linear(10, 10)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
            >>> beta2 = paddle.to_tensor([0.99], dtype="float32")
            >>> adadelta = paddle.optimizer.Adadelta(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
            >>> back = out.backward()
            >>> adadelta.step()
            >>> adadelta.clear_grad()

            >>> # Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> adadelta = paddle.optimizer.Adadelta(
            ...     learning_rate=0.1,
            ...     parameters=[{  # type: ignore
            ...         'params': linear_1.parameters()
            ...     }, {
            ...         'params': linear_2.parameters(),
            ...         'weight_decay': 0.001,
            ...         'learning_rate': 0.1,
            ...     }],
            ...     weight_decay=0.01)
            >>> out.backward()
            >>> adadelta.step()
            >>> adadelta.clear_grad()

    """

    type: str
    _avg_squared_grad_acc_str = "_avg_squared_grad"
    _avg_squared_update_acc_str = "_avg_squared_update"

    def __init__(
        self,
        learning_rate: float | Tensor | LRScheduler = 0.001,
        epsilon: float = 1.0e-6,
        rho: float = 0.95,
        parameters: (
            Sequence[Tensor] | Sequence[_AdadeltaParameterConfig] | None
        ) = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        name: str | None = None,
    ) -> None:
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self._multi_precision = False
        self._master_weights = {}
        self.type = "adadelta"
        self._epsilon = epsilon
        self._rho = rho
        self._default_dict = {
            'epsilon': epsilon,
            'rho': rho,
        }

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, (framework.Block, paddle.pir.Block)):
            raise TypeError("block is not instance of framework.Block.")
        if isinstance(parameters, dict):
            parameters = parameters.get('params')

        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_accumulator(self._avg_squared_grad_acc_str, master_p)
                self._add_accumulator(
                    self._avg_squared_update_acc_str, master_p
                )
                self._already_create_accumulator.add(p.name)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Lars optimizer."
                )
            self._add_accumulator(self._avg_squared_grad_acc_str, p)
            self._add_accumulator(self._avg_squared_update_acc_str, p)
            self._already_create_accumulator.add(p.name)

    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        avg_squared_grad_acc = self._get_accumulator_master(
            self._avg_squared_grad_acc_str, param_and_grad[0]
        )
        avg_squared_update_acc = self._get_accumulator_master(
            self._avg_squared_update_acc_str, param_and_grad[0]
        )
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        if in_dynamic_or_pir_mode():
            with no_grad():
                _C_ops.adadelta_(
                    param_and_grad[0],
                    param_and_grad[1],
                    avg_squared_grad_acc,
                    avg_squared_update_acc,
                    self._create_param_lr(param_and_grad),
                    master_weight,
                    self._rho,
                    self._epsilon,
                    find_master,
                )
            return None
        else:
            if not isinstance(block, (framework.Block, paddle.pir.Block)):
                raise TypeError("block is not instance of framework.Block.")

            # Create the adadelta optimizer op
            inputs = {
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "AvgSquaredGrad": avg_squared_grad_acc,
                "AvgSquaredUpdate": avg_squared_update_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            }
            outputs = {
                "ParamOut": param_and_grad[0],
                "AvgSquaredGradOut": avg_squared_grad_acc,
                "AvgSquaredUpdateOut": avg_squared_update_acc,
            }
            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight
            adadelta_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs={
                    "epsilon": self._epsilon,
                    "rho": self._rho,
                    "multi_precision": find_master,
                },
                stop_gradient=True,
            )

            return adadelta_op

    def _update_param_group(self, parameters):
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._rho = parameters.get('rho', self._default_dict['rho'])
        parameters = parameters.get('params')
        return parameters
