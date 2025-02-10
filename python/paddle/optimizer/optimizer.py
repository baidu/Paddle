# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


# type: ignore
# Avoid mypy internal error

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.autograd as imperative_base
from paddle import _C_ops
from paddle._pir_ops import parameter, set_parameter
from paddle.autograd.backward_utils import ValueDict
from paddle.base import core
from paddle.base.framework import (
    Variable,
    _current_expected_place,
    default_main_program,
    device_guard,
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
    name_scope,
)
from paddle.regularizer import L2Decay, WeightDecayRegularizer

from ..base import framework, unique_name
from ..base.backward import (
    _get_no_grad_set_name,
    _get_no_grad_set_value,
    append_backward,
)
from ..base.framework import Parameter
from ..base.layer_helper import LayerHelper, LayerHelperBase
from .fusion_utils import FusionStorage
from .lr import LambdaDecay, LRScheduler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import NotRequired, TypedDict

    from paddle import Tensor
    from paddle.nn.clip import GradientClipBase

    from ..base.framework import Operator, Program

    class _ParameterConfig(TypedDict):
        params: Sequence[Tensor]
        weight_decay: NotRequired[float | WeightDecayRegularizer | None]
        learning_rate: NotRequired[float | Tensor | LRScheduler | None]


__all__ = []

g_shard_bypass_dygraph_optimizer = int(
    os.environ.get("FLAGS_shard_bypass_dygraph_optimizer", 0)
)


@framework.static_only
def append_backward_new(
    loss_list,
    parameter_list=None,
    no_grad_set=None,
    callbacks=None,
    checkpoints=None,
    distop_context=None,
):
    from paddle.incubate.autograd.primx import Transform, orig2prim

    program = default_main_program()
    assert (
        program.num_blocks == 1
    ), "The append_backward_new interface is designed to process only one block."
    block = program.current_block()
    for el in loss_list:
        assert (
            el.block == block
        ), 'variable in loss_list should be in current block of main program'

    orig2prim(block)
    ad = Transform(block)
    if parameter_list is None:
        parameter_list = program.global_block().all_parameters()
    param_dot, loss_dot = ad.linearize(parameter_list, loss_list)
    loss_bar, param_bar = ad.transpose(loss_dot, param_dot)

    # remove param_dot and their constructor ops
    op_indexes = []
    for var in param_dot:
        if var is not None:
            op_index = block.ops.index(var.op)
            assert op_index >= 0
            op_indexes.append(op_index)

    ad.erase_ops(sorted(op_indexes))
    ad.erase_dots(param_dot)

    if len(parameter_list) == 1:
        params_and_grads = [(parameter_list, param_bar)]
    else:
        params_and_grads = []
        for i, param in enumerate(parameter_list):
            params_and_grads.append((param, param_bar[i]))
    return params_and_grads


class Optimizer:
    r"""Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.

    Args:
        learning_rate (float|LRScheduler): The learning rate used to update ``Parameter``.
            It can be a float value or any subclass of ``LRScheduler`` .
        parameters (list|tuple|None, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``. \
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
        grad_clip (GradientClipBase|None, optional): Gradient clipping strategy, it's an instance of \
            some derived class of ``GradientClipBase`` . There are three clipping strategies \
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` , \
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str|None, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Returns:
       Base class for optimizer.

    Examples:
        .. code-block:: python

            >>> # Take the subclass adam as an example
            >>> import paddle
            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> adam = paddle.optimizer.Adam(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters()
            ... )
            >>> loss.backward()
            >>> adam.step()
            >>> adam.clear_grad()

            >>> #Take the subclass sgd as an example
            >>> #optimize parameters in linear_1 and linear2 in different options.
            >>> #Note that the learning_rate of linear_2 is 0.01.
            >>> linear_1 = paddle.nn.Linear(10, 10)
            >>> linear_2 = paddle.nn.Linear(10, 10)
            >>> inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            >>> out = linear_1(inp)
            >>> out = linear_2(out)
            >>> loss = paddle.mean(out)
            >>> sgd = paddle.optimizer.SGD(
            ...     learning_rate=0.1,
            ...     parameters=[{
            ...         'params': linear_1.parameters()
            ...     }, {
            ...         'params': linear_2.parameters(),
            ...         'weight_decay': 0.001,
            ...         'learning_rate': 0.1
            ...     }],
            ...     weight_decay=0.01)
            >>> loss.backward()
            >>> sgd.step()
            >>> sgd.clear_grad()

    """

    regularization: WeightDecayRegularizer | None
    helper: LayerHelperBase | None
    clear_gradients: Callable[[bool], None]

    @imperative_base.no_grad()
    def __init__(
        self,
        learning_rate: float | LRScheduler,
        parameters: Sequence[Tensor] | Sequence[_ParameterConfig] | None = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        name: str | None = None,
    ) -> None:
        if parameters is not None:
            # paddle.Tensor is also iterable, so here we don't check whether
            # the input is iterable, if the input is paddle.Tensor, the
            # list(paddle.Tensor) will be a error value
            if isinstance(parameters, paddle.Tensor):
                raise TypeError(
                    "`parameters` argument given to the optimizer should be "
                    f"an iterable of paddle Tensors, but got argument type is `{type(parameters)}`."
                )
            if isinstance(parameters, dict):
                raise TypeError(
                    "`parameters` argument should not get dict type, "
                    "if parameter groups is needed, please set `parameters`"
                    " as list of dict"
                )
            self._parameter_list = list(parameters)
        else:
            self._parameter_list = None

        self._name = name
        if framework.in_dygraph_mode():
            if self._parameter_list is None:
                raise AttributeError(
                    "parameters argument given to the Optimizer should not be None in dygraph mode."
                )
            if weight_decay is not None:
                if not isinstance(self._parameter_list[0], dict):
                    for param in self._parameter_list:
                        if (
                            hasattr(param, 'regularizer')
                            and param.regularizer is not None
                        ):
                            logging.info(
                                "If regularizer of a Parameter has been set by 'paddle.ParamAttr' or 'static.WeightNormParamAttr' already. "
                                f"The weight_decay[{weight_decay}] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            )
                            break

        if not isinstance(learning_rate, (float, LRScheduler)):
            raise TypeError(
                f"learning rate should be float or LRScheduler, got {type(learning_rate)} here"
            )
        if grad_clip is not None:
            if not isinstance(grad_clip, paddle.nn.clip.GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )
        if isinstance(weight_decay, float):
            self.regularization = L2Decay(weight_decay)
        elif isinstance(weight_decay, int):
            self.regularization = L2Decay(float(weight_decay))
        else:
            self.regularization = weight_decay
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate

        self._dtype = None
        # Infer the dtype form parameter
        if self._parameter_list:
            if isinstance(self._parameter_list[0], dict):
                for param_group in self._parameter_list:
                    assert (
                        'params' in param_group
                    ), 'params should be set in parameters if parameter groups are optimized in different options'
                self._dtype = self._parameter_list[0]['params'][0].dtype
            else:
                self._dtype = self._parameter_list[0].dtype

        # each program should have a independent learning rate
        # program -> tensor(learning_rate)
        self._learning_rate_map = {}
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra tensors associated with the parameters
        # to train. These tensors are called accumulators.
        # {accum_name : { parameter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: {})
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = {}
        self.clear_gradients = self.clear_grad
        self._default_dict = {
            'weight_decay': self.regularization,
            'grad_clip': self._grad_clip,
        }

        self._param_groups = []
        if self._parameter_list and isinstance(self._parameter_list[0], dict):
            for param_group in self._parameter_list:
                self._add_param_group(param_group.copy())
        else:
            self._param_groups = self._parameter_list

        # NOTE: Multi Tensor: Pass in all parameters and gradients to the op kernel of the Optimizer at one time for updating for dygraph mode.
        # Optimizer support list: [ paddle.optimizer.Momentum, paddle.optimizer.Adam].
        self._use_multi_tensor = None

        self._param_dict = self._create_multi_tensor_dict()
        self._auxiliary_vars = {}
        self._already_create_accumulator = set()

        self._master_weights = {}
        # create master gradients' states
        self._create_master_grad_states()

        # for fusion storage
        self._use_fusion_storage = False
        self._need_refuse = True
        self.fusion_storage = None
        self._fuse_buffer_version = 0
        self.merged_model_params = None

    def _create_master_grad_states(self):
        # master gradients states
        if in_pir_mode():
            self._master_grads = ValueDict()
        else:
            self._master_grads = {}
        self._master_grad = False

    def _set_auxiliary_var(self, key, val):
        self._auxiliary_vars[key] = val

    def _create_multi_tensor_dict(self):
        n = len(self._param_groups) if self._param_groups is not None else 1
        return {
            'FP32_LODTensor': [[] for _ in range(n)],
            'FP16_LODTensor': [[] for _ in range(n)],
        }

    def _get_auxiliary_var(self, key):
        return self._auxiliary_vars.get(key, None)

    def set_merged_model_params(self, merged_model_params):
        self.merged_model_params = merged_model_params
        self.need_refuse()

    @imperative_base.no_grad()
    def _maybe_refuse(self):
        # only support dygraph mode
        if not framework.in_dygraph_mode():
            return

        # TODO(@gexiao): support other optimizer if needed
        if self.__class__.__name__ != "AdamW":
            return

        if not self._need_refuse:
            return

        self.fusion_storage = FusionStorage(
            self._accumulators,
            self._master_weights,
            self.merged_model_params,
        )
        self._fuse_buffer_version += 1
        self.reset_need_refuse()

    @framework.dygraph_only
    def state_dict(self) -> dict[str, Tensor]:
        '''
        Get state dict information from optimizer. It contain all the tensor used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. If LRScheduler have been used, global_step will be include in state dict.
        If the optimizer never be called(minimize function), the state_dict is empty.


        Returns:
            dict[str,Tensor], dict contains all the Tensor used by optimizer

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> emb = paddle.nn.Embedding(10, 10)

                >>> adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
                >>> state_dict = adam.state_dict()

        '''
        state_dict = {}
        if len(self._accumulators) == 0 and len(self._accumulators_holder) > 0:
            for name, var in self._accumulators_holder.items():
                state_dict[name] = var
        else:
            for k, v in self._accumulators.items():
                for para_name, var_tmp in v.items():
                    state_dict[var_tmp.name] = var_tmp
                    # save scale value for xpu
                    if core.is_compiled_with_xpu():
                        xpu_adamw_moment_dtype = os.getenv(
                            "xpu_adamw_moment_dtype", default="fp32"
                        )
                        if xpu_adamw_moment_dtype == "fp16":
                            state_dict[var_tmp.name + ".SCALE_VALUE"] = (
                                var_tmp.get_tensor().get_xpu_scale_value()
                            )
        # if has master weight and then save master weight
        if hasattr(self, "_master_weights"):
            if len(self._master_weights) != 0:
                state_dict["master_weights"] = self._master_weights
        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        '''
        Load optimizer state dict. For Adam optimizer, contains beta1, beta2, momentum etc. If LRScheduler have been used, global_step will be changed.

        Args:
            state_dict(dict): Dict contains all the Tensor needed by optimizer

        Return:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> emb = paddle.nn.Embedding(10, 10)

                >>> layer_state_dict = emb.state_dict()
                >>> paddle.save(layer_state_dict, "emb.pdparams")

                >>> scheduler = paddle.optimizer.lr.NoamDecay(
                ...     d_model=100, warmup_steps=100, verbose=True)
                >>> adam = paddle.optimizer.Adam(
                ...     learning_rate=scheduler,
                ...     parameters=emb.parameters())
                >>> opt_state_dict = adam.state_dict()
                >>> paddle.save(opt_state_dict, "adam.pdopt")

                >>> opti_state_dict = paddle.load("adam.pdopt")
                >>> adam.set_state_dict(opti_state_dict)

        '''
        if isinstance(self._learning_rate, LRScheduler):
            lr_state_dict = state_dict.get("LR_Scheduler", None)
            if not isinstance(self._learning_rate, LambdaDecay):
                assert (
                    lr_state_dict is not None
                ), "LR_Scheduler state must be included in the state dict except LambdaDecay"
            if lr_state_dict:
                self._learning_rate.set_state_dict(lr_state_dict)

        # NOTE: exclude learning rate scheduler's state from
        # _accumulators_holder.
        state_dict = state_dict.copy()
        if "LR_Scheduler" in state_dict:
            state_dict.pop("LR_Scheduler")
        if "master_weights" in state_dict:
            if hasattr(self, "_master_weights"):
                self._master_weights = state_dict["master_weights"]
            state_dict.pop("master_weights")
        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert (
                    var_tmp.name in state_dict
                ), f"optimizer Tensor {var_tmp.name} not found"

                var = var_tmp.value()
                tensor = var.get_tensor()
                # load scale value for xpu
                if core.is_compiled_with_xpu():
                    xpu_adamw_moment_dtype = os.getenv(
                        "xpu_adamw_moment_dtype", default="fp32"
                    )
                    if xpu_adamw_moment_dtype == "fp16":
                        tensor.set_xpu_scale_value(
                            state_dict.get(var_tmp.name + ".SCALE_VALUE", -1.0)
                        )
                var.set_value(state_dict[var_tmp.name])

    def get_opti_var_name_list(self) -> list[str]:
        return self._opti_name_list

    def _create_global_learning_rate(self):
        def do_create():
            # lr var can't be float16 or bfloat16, for pure fp16 or bf16 training, should extra handle the dtype for lr
            _lr_dtype = (
                paddle.get_default_dtype()
                if self._dtype is None
                else self._dtype
            )
            _lr_dtype = (
                paddle.float32
                if (
                    (
                        paddle.get_default_dtype() != "float16"
                        and _lr_dtype == paddle.float16
                    )
                    or (
                        paddle.get_default_dtype() != "bfloat16"
                        and _lr_dtype == paddle.bfloat16
                    )
                )
                else _lr_dtype
            )
            if isinstance(self._learning_rate, LRScheduler):
                lr_var = self._global_learning_rate()
                # only create global lr_var once
                if in_pir_mode():
                    startup_program = paddle.static.default_startup_program()
                    main_program = paddle.static.default_main_program()

                    lr_name = unique_name.generate('learning_rate')
                    # startup program  insert && set_parameter
                    lr_value = float(self._learning_rate())
                    with paddle.static.program_guard(startup_program):
                        initializer = paddle.nn.initializer.Constant(
                            value=lr_value
                        )
                        parameter_meta = paddle.pir.core.ParameterMeta(
                            [], _lr_dtype
                        )
                        init_result = initializer(
                            parameter_meta, startup_program.global_block()
                        )
                        init_result.persistable = True
                        set_parameter(init_result, lr_name)
                    main_program.set_parameters_from(startup_program)

                    if not isinstance(lr_var, paddle.pir.Value):
                        self._learning_rate._var_name = lr_name
                        with paddle.static.program_guard(main_program):
                            param = parameter(lr_name, _lr_dtype, [])
                        param.stop_gradient = True
                        param.persistable = True
                        main_program.lr_scheduler = self._learning_rate
                        main_program.lr_var = param
                        main_program.lr_name = lr_name
                        self._learning_rate_map[main_program] = param

                else:
                    if not isinstance(lr_var, framework.Variable):
                        lr_name = unique_name.generate('learning_rate')
                        self._learning_rate._var_name = lr_name
                        lr_var = self.helper.create_global_variable(
                            name=lr_name,
                            shape=[],
                            persistable=True,
                            stop_gradient=True,
                            dtype=_lr_dtype,
                        )
                        main_prog = framework.default_main_program()
                        main_prog.lr_scheduler = self._learning_rate
                        main_prog.lr_var = lr_var

                        self._learning_rate_map[
                            framework.default_main_program()
                        ] = lr_var

                    lr_value = float(self._learning_rate())
                    self.helper.set_variable_initializer(
                        lr_var,
                        initializer=paddle.nn.initializer.Constant(
                            value=lr_value
                        ),
                    )
            elif isinstance(self._learning_rate, float):
                # only create global lr_var once
                lr = self._global_learning_rate()
                if in_pir_mode():
                    if isinstance(lr, paddle.pir.Value):
                        return
                    else:
                        place = _current_expected_place()
                        if not isinstance(_lr_dtype, paddle.base.core.DataType):
                            if isinstance(
                                _lr_dtype, paddle.base.libpaddle.VarDesc.VarType
                            ):
                                _lr_dtype = paddle.pir.core.vartype_to_datatype[
                                    _lr_dtype
                                ]
                            else:
                                _lr_dtype = (
                                    paddle.pir.core.convert_np_dtype_to_dtype_(
                                        _lr_dtype
                                    )
                                )
                        self._learning_rate_map[
                            paddle.static.default_main_program()
                        ] = paddle.pir.core.create_persistable_value(
                            dtype=_lr_dtype,
                            shape=[],
                            name=unique_name.generate("learning_rate"),
                            initializer=paddle.nn.initializer.ConstantInitializer(
                                value=float(self._learning_rate)
                            ),
                        )
                else:
                    if isinstance(lr, framework.Variable):
                        return
                    else:
                        self._learning_rate_map[
                            framework.default_main_program()
                        ] = paddle.static.create_global_var(
                            name=unique_name.generate("learning_rate"),
                            shape=[],
                            value=float(self._learning_rate),
                            dtype=_lr_dtype,
                            persistable=True,
                        )

        with paddle.base.framework.dygraph_guard_if_declarative():
            do_create()

    @framework.dygraph_only
    def set_lr(self, value: float) -> None:
        """
        :api_attr: imperative

        Set the value of the learning rate manually in the optimizer. If the optimizer use LRScheduler,
        this API cannot be invoked, because it will lead to conflict.

        Args:
            value (float): the value of learning rate.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> linear = paddle.nn.Linear(10, 10)

                >>> adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

                >>> # set learning rate manually by python float value
                >>> lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                >>> for i in range(5):
                ...     adam.set_lr(lr_list[i])
                ...     lr = adam.get_lr()
                ...     print("current lr is {}".format(lr))
                current lr is 0.2
                current lr is 0.3
                current lr is 0.4
                current lr is 0.5
                current lr is 0.6

        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"The type of 'value' in optimizer.set_lr must be float, but received {type(value)}."
            )
        if isinstance(self._learning_rate, LRScheduler):
            raise RuntimeError(
                "optimizer's learning rate can't be LRScheduler when invoke this API, because this will lead to conflict."
            )
        self._learning_rate = float(value)
        current_lr = self._global_learning_rate()
        if current_lr is not None:
            if in_dygraph_mode():
                place = _current_expected_place()
                _C_ops.full_(
                    current_lr,
                    list(current_lr.shape),
                    float(value),
                    current_lr.dtype,
                    place,
                )
            else:
                global_block = framework.default_main_program().global_block()
                global_block.append_op(
                    type='fill_constant',
                    outputs={'Out': [current_lr]},
                    attrs={
                        'dtype': current_lr.dtype,
                        'shape': list(current_lr.shape),
                        'value': float(value),
                    },
                    stop_gradient=True,
                )

    @framework.dygraph_only
    def set_lr_scheduler(self, scheduler: LRScheduler) -> None:
        """
        :api_attr: imperative

        Set the LRScheduler of the learning rate manually in the optimizer. If the optimizer already used LRScheduler previously,
        this API will set it be the new one.

        Args:
            scheduler (LRScheduler): the LRScheduler of learning rate

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> linear = paddle.nn.Linear(10, 10)

                >>> adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

                >>> # set learning rate manually by class LRScheduler
                >>> scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2,4,6], gamma=0.8)
                >>> adam.set_lr_scheduler(scheduler)
                >>> lr = adam.get_lr()
                >>> print("current lr is {}".format(lr))
                current lr is 0.5

                >>> # set learning rate manually by another LRScheduler
                >>> scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=5, gamma=0.6)
                >>> adam.set_lr_scheduler(scheduler)
                >>> lr = adam.get_lr()
                >>> print("current lr is {}".format(lr))
                current lr is 0.1

        """
        from paddle.optimizer.lr import LRScheduler

        if not isinstance(scheduler, LRScheduler):
            raise TypeError(
                f"The type of 'scheduler' in optimizer.set_lr_scheduler must be LRScheduler, but received {type(scheduler)}."
            )
        self._learning_rate = scheduler

    def get_lr(self) -> float:
        """
        Get current learning rate of optimizer.
        If 'LRScheduler' is not used, the return value is all the same.
        If 'LRScheduler' is used, the return value is the current scheduled learning rete.

        Returns:
            float, The current learning rate of optimizer.

        Examples:
            .. code-block:: python

                >>> # train on default dynamic graph mode
                >>> import paddle
                >>> import numpy as np
                >>> emb = paddle.nn.Embedding(10, 3)

                >>> ## example1: LRScheduler is not used, return the same value is all the same
                >>> adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
                >>> for batch in range(10):
                ...     input = paddle.randint(low=0, high=5, shape=[5])
                ...     out = emb(input)
                ...     out.backward()
                ...     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.01
                ...     adam.step()
                Learning rate of step0: 0.01
                Learning rate of step1: 0.01
                Learning rate of step2: 0.01
                Learning rate of step3: 0.01
                Learning rate of step4: 0.01
                Learning rate of step5: 0.01
                Learning rate of step6: 0.01
                Learning rate of step7: 0.01
                Learning rate of step8: 0.01
                Learning rate of step9: 0.01

                >>> ## example2: StepDecay is used, return the scheduled learning rate
                >>> scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
                >>> adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
                >>> for batch in range(10):
                ...     input = paddle.randint(low=0, high=5, shape=[5])
                ...     out = emb(input)
                ...     out.backward()
                ...     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.5->0.05...
                ...     adam.step()
                ...     scheduler.step()
                Learning rate of step0: 0.5
                Learning rate of step1: 0.5
                Learning rate of step2: 0.05
                Learning rate of step3: 0.05
                Learning rate of step4: 0.005000000000000001
                Learning rate of step5: 0.005000000000000001
                Learning rate of step6: 0.0005000000000000001
                Learning rate of step7: 0.0005000000000000001
                Learning rate of step8: 5.000000000000001e-05
                Learning rate of step9: 5.000000000000001e-05

                >>> # train on static graph mode
                >>> paddle.enable_static()
                >>> main_prog = paddle.static.Program()
                >>> start_prog = paddle.static.Program()
                >>> with paddle.static.program_guard(main_prog, start_prog):
                ...     x = paddle.static.data(name='x', shape=[None, 10])
                ...     z = paddle.static.nn.fc(x, 100)
                ...     loss = paddle.mean(z)
                ...     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
                ...     adam = paddle.optimizer.Adam(learning_rate=scheduler)
                ...     adam.minimize(loss)

                >>> exe = paddle.static.Executor()
                >>> exe.run(start_prog)
                >>> for batch in range(10):
                ...     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.5->0.05->0.005...
                ...     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
                ...     scheduler.step()
                Learning rate of step0: 0.5
                Learning rate of step1: 0.5
                Learning rate of step2: 0.05
                Learning rate of step3: 0.05
                Learning rate of step4: 0.005000000000000001
                Learning rate of step5: 0.005000000000000001
                Learning rate of step6: 0.0005000000000000001
                Learning rate of step7: 0.0005000000000000001
                Learning rate of step8: 5.000000000000001e-05
                Learning rate of step9: 5.000000000000001e-05
        """
        if isinstance(self._learning_rate, float):
            return self._learning_rate
        else:
            return self._learning_rate()

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            if in_dygraph_mode():
                program = framework.default_main_program()
            else:
                program = paddle.static.default_main_program()
        return self._learning_rate_map.get(program, None)

    def _append_optimize_op(self, block, param_and_grad):
        """append optimize operator to block and return all the added optimize_op"""
        raise NotImplementedError(
            'Class "Optimizer" cannot be used directly as an optimizer, please use its subclasses such as "Adam"'
        )

    def _create_param_lr(self, param_and_grad):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        if (
            hasattr(param, 'optimize_attr')
            and param.optimize_attr is not None
            and 'learning_rate' in param.optimize_attr
        ):
            param_lr = param.optimize_attr['learning_rate']
            if isinstance(param_lr, (Variable, paddle.pir.Value)):
                return param_lr
            else:
                if param_lr == 1.0:
                    return self._global_learning_rate()
                else:
                    with paddle.static.default_main_program()._lr_schedule_guard(
                        is_with_opt=True
                    ), framework.name_scope(
                        'scale_with_param_lr'
                    ):
                        return self._global_learning_rate() * param_lr
        else:
            return self._global_learning_rate()

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            var_name = self._gen_master_weight_var_name(param)
            if in_pir_mode():
                startup_program = paddle.static.default_startup_program()
                main_program = paddle.static.default_main_program()
                with paddle.static.program_guard(startup_program):

                    def get_param_from_startup(startup, name):
                        for op in startup.global_block().ops:
                            if (
                                op.name() == 'builtin.set_parameter'
                                and name == op.attrs()['parameter_name']
                            ):
                                return op.operand(0).source()
                        return None

                    startup_param = get_param_from_startup(
                        startup_program, param.name
                    )
                    startup_var = paddle.cast(startup_param, 'float32')
                    startup_var.persistable = True
                    paddle._pir_ops.set_persistable_value(startup_var, var_name)
                with paddle.static.program_guard(main_program):
                    paddle.pir.reset_insertion_point_to_start()
                    var = paddle.static.data(
                        var_name,
                        startup_var.shape,
                        startup_var.dtype,
                        core.Place(),
                    )
                    if startup_var.is_dist():
                        var.set_type(startup_var.type())
                        op_dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                startup_var.dist_attr().process_mesh,
                                [],
                                [startup_var.dist_attr()],
                            )
                        )
                        var.get_defining_op().dist_attr = op_dist_attr
                    var.persistable = True
            elif framework.in_dygraph_mode():
                var = paddle.cast(param, 'float32')
                var.name = var_name
            else:
                assert isinstance(self.helper, LayerHelper)
                var = paddle.static.create_global_var(
                    name=var_name,
                    shape=param.shape,
                    value=0,
                    dtype='float32',
                    persistable=True,
                )
                block = self.helper.startup_program.global_block()
                block.append_op(
                    type="cast",
                    inputs={"X": [param]},
                    outputs={"Out": [var]},
                    attrs={
                        "in_dtype": param.dtype,
                        "out_dtype": core.VarDesc.VarType.FP32,
                    },
                )
            self._master_weights[param.name] = var
        return var

    def _gen_master_weight_var_name(self, param):
        var_name = param.name + "_fp32_master"
        return unique_name.generate(var_name)

    def _create_master_grad(self, grad):
        assert self._is_dtype_fp16_or_bf16(grad.dtype)
        if in_pir_mode():
            if grad in self._master_grads:
                var = self._master_grads[grad]
            else:
                var = paddle.cast(grad, 'float32')
                var.get_defining_op().set_bool_attr('master_grad_cast', True)
                self._master_grads[grad] = var
        else:
            if grad.name in self._master_grads:
                var = self._master_grads[grad.name]
            else:
                var_name = grad.name + "_fp32_master"
                var_name = unique_name.generate(var_name)
                var = grad.block.create_var(
                    name=var_name,
                    shape=grad.shape,
                    value=0,
                    dtype='float32',
                    lod_level=grad.lod_level,
                    persistable=grad.persistable,
                    is_data=grad.is_data,
                )
                self._master_grads[grad.name] = var
        return var

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """
        pass

    def _finish_update(self, block, parameters_and_grads):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer

        Returns:
            None
        """
        pass

    def _add_accumulator(
        self,
        name,
        param,
        dtype=None,
        fill_value=0.0,
        shape=None,
        type=None,
        device=None,
    ):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss tensor is present
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be added
            dtype: data type of the accumulator tensor
            fill_value: value to initialize the accumulator tensor
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (
            name in self._accumulators
            and param.name in self._accumulators[name]
        ):
            if framework.in_dygraph_mode():
                return self._accumulators[name][param.name]
            raise Exception(
                f"Accumulator {name} already exists for parameter {param.name}"
            )
        else:
            # once master weights are created, accumulators must be created at the same time
            self.need_refuse()
        if shape is None:
            shape = param.shape

        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        if device is None:
            device = self._get_device_for_param(param.name)

        if in_pir_mode():
            if 'beta' not in var_name:
                var = paddle.pir.core.create_persistable_value(
                    dtype or param.dtype,
                    shape,
                    var_name,
                    initializer=paddle.nn.initializer.Constant(
                        value=float(fill_value)
                    ),
                    dist_attr=param.dist_attr(),
                )
            else:
                var = paddle.pir.core.create_persistable_value(
                    dtype or param.dtype,
                    shape,
                    var_name,
                    initializer=paddle.nn.initializer.Constant(
                        value=float(fill_value)
                    ),
                )
        else:
            if self.helper is None:
                self.helper = LayerHelper(self.__class__.__name__)
            assert isinstance(self.helper, LayerHelper)
            var = self.helper.create_global_variable(
                name=var_name,
                persistable=True,
                dtype=dtype or param.dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                shape=shape,
                belong_to_optimizer=True,
            )

            if in_dygraph_mode() and (
                device == 'cpu' or isinstance(device, core.CPUPlace)
            ):
                _C_ops.full_(
                    var,
                    var.shape,
                    str(float(fill_value)),
                    var.dtype,
                    core.CPUPlace(),
                )
            else:
                with device_guard(device):
                    self.helper.set_variable_initializer(
                        var,
                        initializer=paddle.nn.initializer.Constant(
                            value=float(fill_value)
                        ),
                    )

            if framework.in_dygraph_mode():
                if len(self._accumulators_holder) > 0:
                    assert (
                        var_name in self._accumulators_holder
                    ), f"Optimizer set error, {var_name} should in state dict"
                    var.set_value(self._accumulators_holder.pop(var_name))

                    # load scale value for xpu
                    if core.is_compiled_with_xpu():
                        xpu_adamw_moment_dtype = os.getenv(
                            "xpu_adamw_moment_dtype", default="fp32"
                        )
                        if xpu_adamw_moment_dtype == "fp16":
                            var.get_tensor().set_xpu_scale_value(
                                self._accumulators_holder.get(
                                    var_name + ".SCALE_VALUE", -1.0
                                )
                            )

        self._accumulators[name][param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be fetched

        Returns:
            accumulator tensor for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (
            name not in self._accumulators
            or param.name not in self._accumulators[name]
        ):
            raise Exception(
                f"Accumulator {name} does not exist for parameter {param.name}"
            )
        return self._accumulators[name][param.name]

    def _get_accumulator_master(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param.dtype
        )
        target_param = (
            self._master_weights[param.name] if find_master else param
        )
        target_name = target_param.name
        if (
            name not in self._accumulators
            or target_name not in self._accumulators[name]
        ):
            raise Exception(
                f"Accumulator {name} does not exist for parameter {target_name}"
            )
        return self._accumulators[name][target_name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].stop_gradient is False:
                param_name = param_and_grad[0].name
                ops = target_block.ops
                device_attr_name = (
                    core.op_proto_and_checker_maker.kOpDeviceAttrName()
                )
                for op in ops:
                    input_arg_names = op.input_arg_names
                    if param_name in input_arg_names:
                        self._param_device_map[param_name] = op.attr(
                            device_attr_name
                        )
                        break

    def _get_device_for_param(self, param_name):
        device = None
        if param_name in self._param_device_map:
            device = self._param_device_map[param_name]
        return device

    def _create_optimization_pass(
        self, parameters_and_grads, param_group_idx=0
    ):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Always called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert (
                current_block.backward_block_idx != -1
            ), "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx
            ]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)

        self._create_global_learning_rate()

        # NOTE: Multi Tensor support [ Momentum, Adam ] for dygraph mode
        if self._use_multi_tensor and self.__class__.__name__ in [
            'Momentum',
            'Adam',
        ]:
            if (
                len(self._param_dict['FP32_LODTensor'][param_group_idx]) == 0
                and len(self._param_dict['FP16_LODTensor'][param_group_idx])
                == 0
            ):
                if isinstance(parameters_and_grads, list):
                    assert param_group_idx == 0
                    self._multi_tensor_init(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads
                            if not p[0].stop_gradient
                        ],
                        param_group_idx,
                    )
                else:
                    self._update_param_group(parameters_and_grads)
                    self._multi_tensor_init(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads['params']
                            if not p[0].stop_gradient
                        ],
                        param_group_idx,
                    )
            if framework.in_dygraph_mode():
                self._append_optimize_multi_tensor_op(
                    target_block,
                    parameters_and_grads,
                    param_group_idx=param_group_idx,
                )
            else:
                self._update_param_device_map(
                    parameters_and_grads, target_block
                )
                # NOTE: Multi Tensor requires all parameters to be in the same device and program.
                # param_grad_list = [p_0,g_0,p_1,g_1,....]
                param_grad_list = []
                for param_and_grad in parameters_and_grads:
                    if (
                        not param_and_grad[0].stop_gradient
                        and param_and_grad[1] is not None
                    ):
                        param_grad_list.append(param_and_grad[0])
                        param_grad_list.append(param_and_grad[1])
                with param_grad_list[0].block.program._optimized_guard(
                    param_grad_list
                ), name_scope("optimizer"):
                    device = self._get_device_for_param(param_grad_list[0].name)
                    with device_guard(device):
                        self._append_optimize_multi_tensor_op(
                            target_block,
                            parameters_and_grads,
                            param_group_idx=param_group_idx,
                        )
        else:
            if not framework.in_dygraph_mode():
                params_grads_device_map = (
                    parameters_and_grads['params']
                    if isinstance(parameters_and_grads, dict)
                    else parameters_and_grads
                )
                self._update_param_device_map(
                    params_grads_device_map, target_block
                )

            if isinstance(parameters_and_grads, list):
                with paddle.base.framework.dygraph_guard_if_declarative():
                    self._create_accumulators(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads
                            if not p[0].stop_gradient
                        ],
                    )
            else:
                params_acc_dict = parameters_and_grads.copy()
                params_acc_dict['params'] = [
                    p[0]
                    for p in params_acc_dict['params']
                    if not p[0].stop_gradient
                ]
                with paddle.base.framework.dygraph_guard_if_declarative():
                    self._create_accumulators(target_block, params_acc_dict)

            if framework.in_dygraph_mode():
                found_inf = self._get_auxiliary_var('found_inf')
                if found_inf:
                    if isinstance(found_inf, core.eager.Tensor):
                        self._set_auxiliary_var('found_inf', True)
                else:
                    if isinstance(found_inf, core.eager.Tensor):
                        self._set_auxiliary_var('found_inf', False)
                    if isinstance(parameters_and_grads, list):
                        self._maybe_refuse()
                        for param_and_grad in parameters_and_grads:
                            # Parameters can be uninitialized in pipeline parallel of semi-auto parallel.
                            # Since gradient clip and parameters update mixed up in one interface, so we
                            # need to filter again here.
                            if (
                                param_and_grad[1] is None
                                or not param_and_grad[0]._is_initialized()
                            ):
                                continue
                            if param_and_grad[0].stop_gradient is False:
                                self._append_optimize_op(
                                    target_block, param_and_grad
                                )
                    else:
                        for param_and_grad in parameters_and_grads['params']:
                            if (
                                param_and_grad[1] is None
                                or not param_and_grad[0]._is_initialized()
                            ):
                                continue
                            if param_and_grad[0].stop_gradient is False:
                                param_grad_dict = {}
                                param_grad_dict['params'] = param_and_grad
                                param_grad_dict.update(
                                    {
                                        k: v
                                        for k, v in parameters_and_grads.items()
                                        if k != 'params'
                                    }
                                )
                                self._append_optimize_op(
                                    target_block, param_grad_dict
                                )
            else:
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad
                    ), name_scope("optimizer"):
                        if param_and_grad[0].stop_gradient is False:
                            device = self._get_device_for_param(
                                param_and_grad[0].name
                            )
                            with device_guard(device):
                                optimize_op = self._append_optimize_op(
                                    target_block, param_and_grad
                                )

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)
        paddle.base.core._set_warmup(False)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _pir_create_optimization_pass(
        self, parameters_and_grads, param_group_idx=0
    ):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """

        global_block = paddle.static.default_main_program().global_block()
        target_block = global_block

        last_op = target_block.ops[-1]

        self._create_global_learning_rate()

        # params_grads_device_map = (
        #     parameters_and_grads['params']
        #     if isinstance(parameters_and_grads, dict)
        #     else parameters_and_grads
        # )
        # self._update_param_device_map(params_grads_device_map, target_block)

        if isinstance(parameters_and_grads, list):
            self._create_accumulators(
                target_block,
                [p[0] for p in parameters_and_grads if not p[0].stop_gradient],
            )
        else:
            params_acc_dict = parameters_and_grads.copy()
            params_acc_dict['params'] = [
                p[0]
                for p in params_acc_dict['params']
                if not p[0].stop_gradient
            ]
            self._create_accumulators(target_block, params_acc_dict)

        if isinstance(parameters_and_grads, list):
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].stop_gradient is False:
                    self._append_optimize_op(target_block, param_and_grad)
        else:
            for param_and_grad in parameters_and_grads['params']:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].stop_gradient is False:
                    param_grad_dict = {}
                    param_grad_dict['params'] = param_and_grad
                    param_grad_dict.update(
                        {
                            k: v
                            for k, v in parameters_and_grads.items()
                            if k != 'params'
                        }
                    )
                    self._append_optimize_op(target_block, param_grad_dict)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)
        paddle.base.core._set_warmup(False)

        start_index = target_block.ops.index(last_op) + 1
        return target_block.ops[start_index:]

    def backward(
        self,
        loss: Tensor,
        startup_program: Program | None = None,
        parameters: list[Tensor] | list[str] | None = None,
        no_grad_set: set[Tensor] | set[str] | None = None,
        callbacks: list[Callable[..., None]] | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.

        Args:
            loss (Tensor): ``loss`` tensor to run optimizations.
            startup_program (Program|None, optional): :ref:`api_paddle_static_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_paddle_static_default_startup_program` will be used.
            parameters (list[Tensor]|list[str]|None, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set[Tensor]|set[str]|None, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.
            callbacks (list|None, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.

        Return:
            list[tuple[Tensor, Tensor]], list of (param, grad) tensor pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> x = paddle.arange(26, dtype="float32").reshape([2, 13])

                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> adam = paddle.optimizer.Adam(learning_rate = 0.01,
                ...                             parameters = linear.parameters())
                >>> out = linear(x)
                >>> out.backward()
                >>> adam.step()
                >>> adam.clear_grad()
        """
        act_no_grad_set = None
        if framework.in_dygraph_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        # Infer dtype by loss if None
        if self._dtype is None:
            self._dtype = loss.dtype

        if framework.in_dygraph_mode():
            parameter_list = parameters if parameters else self._parameter_list

            # It is very time-consuming to call c++ functions in a loop on the python side.
            # We put this part of the code on the c++ side to improve the speed in eager mode.
            params_grads = []
            grads = core.eager.get_all_grads(parameter_list)
            for index, grad in enumerate(grads):
                if grad is not None:
                    params_grads.append((parameter_list[index], grad))
        else:
            if callbacks is None:
                callbacks = [paddle.nn.clip.error_clip_callback]
            else:
                assert isinstance(callbacks, list)
            program = loss.block.program
            assert np.prod(loss.shape) == 1, (
                f"The number of elements of loss should be 1, but the current loss.shape is {loss.shape}, whose number of elements is not 1. "
                "Maybe that you should call paddle.mean to process the current loss."
            )
            parameter_list = parameters if parameters else self._parameter_list
            with paddle.static.program_guard(program, startup_program):
                if in_pir_mode():
                    if parameter_list is None:
                        # all parameters will be updated.
                        program_all_params = (
                            program.global_block().all_parameters()
                        )
                        parameter_list = [
                            param
                            for param in program_all_params
                            if param.stop_gradient is False
                        ]
                    params_grads = []
                    grads = paddle.autograd.ir_backward.grad(
                        loss, parameter_list, no_grad_vars=act_no_grad_set
                    )
                    for index, grad in enumerate(grads):
                        if grad is not None:
                            params_grads.append((parameter_list[index], grad))
                else:
                    from paddle.incubate.autograd.utils import prim_enabled

                    if prim_enabled():
                        params_grads = append_backward_new(
                            [loss], parameter_list, act_no_grad_set, callbacks
                        )
                    else:
                        params_grads = append_backward(
                            loss, parameter_list, act_no_grad_set, callbacks
                        )
        return params_grads

    def apply_gradients(
        self, params_grads: list[tuple[Tensor, Tensor]]
    ) -> list[Operator]:
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list[tuple[Tensor, Tensor]]): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> inp = paddle.uniform([10, 10], dtype="float32", min=-0.1, max=0.1)
                >>> linear = paddle.nn.Linear(10, 10)
                >>> out = linear(inp)
                >>> loss = paddle.mean(out)
                >>> optimizer = paddle.optimizer.Adam(learning_rate=0.1,
                ...         parameters=linear.parameters())
                >>> params_grads = optimizer.backward(loss)
                >>> optimizer.apply_gradients(params_grads)

        """

        # NOTE(zhaoyinglia): AutoParallel set '_sorted' attribute to skip the 'sorted' operator.
        if not hasattr(self, "_sorted"):
            params_grads = sorted(params_grads, key=lambda x: x[0].name)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:
            params_grads = paddle.nn.clip.append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = self.append_regularization_ops(
            params_grads, self.regularization
        )

        if in_pir_mode():
            optimize_ops = self._pir_create_optimization_pass(params_grads)
        else:
            optimize_ops = self._create_optimization_pass(params_grads)
        return optimize_ops

    def _apply_optimize(
        self, loss, startup_program, params_grads, param_group_idx=0
    ):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.
        Args:
            loss (Tensor): loss tensor to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameters`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Returns:
            list: A list of operators appended to the current program.
        """

        if framework.in_dygraph_mode() and g_shard_bypass_dygraph_optimizer:
            return

        if in_dynamic_or_pir_mode():
            with paddle.static.program_guard(
                paddle.static.default_main_program(),
                paddle.static.default_startup_program(),
            ):
                if isinstance(params_grads, list):
                    if self._grad_clip is not None:
                        params_grads = self._grad_clip(params_grads)
                    params_grads = self.append_regularization_ops(
                        params_grads, self.regularization
                    )
                else:
                    grad_clip = params_grads['grad_clip']
                    if grad_clip is not None:
                        params_grads['params'] = grad_clip(
                            params_grads['params']
                        )

                    params_grads['params'] = self.append_regularization_ops(
                        params_grads['params'], self.regularization
                    )
                if in_pir_mode():
                    optimize_ops = self._pir_create_optimization_pass(
                        params_grads, param_group_idx=param_group_idx
                    )
                else:
                    optimize_ops = self._create_optimization_pass(
                        params_grads, param_group_idx=param_group_idx
                    )
        else:
            assert param_group_idx == 0
            program = loss.block.program
            with paddle.static.program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        """Create and add backward regularization Operators

        Function helper of append_regularization_ops.
        """
        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or (
            (
                not hasattr(param, 'regularizer')
                or (hasattr(param, 'regularizer') and param.regularizer is None)
            )
            and regularization is None
        ):
            return grad
        regularization_term = None

        # when master_grad is true in amp training, grad will be fp32, but param maybe fp16.
        # we get master weight when master_grad is true to avoid type mismatch error.
        def get_target_param(param, grad):
            target_param = param
            if param.dtype != grad.dtype:
                find_master = (
                    self._multi_precision
                    and self._is_dtype_fp16_or_bf16(param.dtype)
                )
                if find_master and len(self._master_weights) != 0:
                    target_param = self._master_weights[param.name]
                else:
                    target_param = param.astype(grad.dtype)
            return target_param

        param = get_target_param(param, grad)
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)

        assert regularization_term is not None

        if in_dynamic_or_pir_mode():
            return _C_ops.add_n([grad, regularization_term])
        else:
            new_grad = grad
            if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
                # FIXME(zcd): If the grad is SELECTED_ROWS, after regularization,
                # the grad's type and name will be changed. But the gradient's name
                # is used in ParallelExecutor Reduce mode, so I add a flag for
                # the new_grad here.
                new_grad = grad.block.create_var(
                    name=grad.name + core.kNewGradSuffix(),
                    dtype=param.dtype,
                    shape=param.shape,
                    lod_level=param.lod_level,
                    type=core.VarDesc.VarType.DENSE_TENSOR,
                )

            inputs = {"X": [grad, regularization_term]}
            outputs = {"Out": [new_grad]}
            grad.block.append_op(type='sum', inputs=inputs, outputs=outputs)

            return new_grad

    def append_regularization_ops(
        self,
        parameters_and_grads: list[tuple[Tensor, Tensor]],
        regularization: WeightDecayRegularizer | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        r"""Create and add backward regularization Operators

        Creates and adds backward regularization operators in the BlockDesc.
        This will add gradients of the regularizer function to the gradients
        of the parameters and return these modified gradients. This is the
        same as implementing weight decay in optimizers for regularization.

        Args:
            parameters_and_grads (list[tuple[Tensor,Tensor]]): A list of (parameters, gradients) pairs
                that need to be regularized.
            regularization (WeightDecayRegularizer|None, optional): A global regularizer. If the parameter is not
                set. It will be applied with regularizer.

        Returns:
            list[tuple[Tensor,Tensor]]: list of (parameters, gradients) \
                pair with the regularized gradient

        Raises:
            Exception: Unknown regularization type
        """
        params_and_grads = []
        if framework.in_dygraph_mode() or in_pir_mode():
            for param, grad in parameters_and_grads:
                new_grad = self._create_regularization_of_grad(
                    param, grad, regularization
                )
                params_and_grads.append((param, new_grad))
        else:
            repeat_regularizer = False
            with framework.name_scope('regularization'):
                for param, grad in parameters_and_grads:
                    if (
                        not repeat_regularizer
                        and param.regularizer is not None
                        and regularization is not None
                    ):
                        repeat_regularizer = True
                        logging.info(
                            "If regularizer of a Parameter has been set by 'base.ParamAttr' or 'base.WeightNormParamAttr' already. "
                            f"The Regularization[{regularization}] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                        )
                    with param.block.program._optimized_guard([param, grad]):
                        new_grad = self._create_regularization_of_grad(
                            param, grad, regularization
                        )
                        params_and_grads.append((param, new_grad))
        return params_and_grads

    def _get_no_grad_set(self, loss, no_grad_set=None):
        if in_pir_mode():
            no_grad_set = _get_no_grad_set_value(no_grad_set)
            parameters = loss.block.program.global_block().all_parameters()
            param_no_trainable = [
                param for param in parameters if param.stop_gradient is True
            ]
            # If the parameter is no trainable, it should not have a gradient.
            no_grad_set.update(param_no_trainable)
            return no_grad_set
        else:
            no_grad_set = _get_no_grad_set_name(no_grad_set)
            parameters = loss.block.program.global_block().all_parameters()
            param_no_trainable = {
                param.name
                for param in parameters
                if param.stop_gradient is True
            }
            # If the parameter is no trainable, it should not have a gradient.
            no_grad_set.update(param_no_trainable)
            return no_grad_set

    @framework.non_static_only
    def clear_grad(self, set_to_zero: bool = True) -> None:
        """
        Clear the gradients of all optimized parameters for model.

        If not, new gradient will accumulat on previous gradient.

        There are two method to clear grad: set_to_zero or delete grad.

        Args:
            set_to_zero (bool, optional): If set grads to zero or not, default is True.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> a = paddle.arange(26, dtype="float32").reshape([2, 13])
                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> adam = paddle.optimizer.Adam(learning_rate = 0.01,
                ...                             parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> adam.step()
                >>> adam.clear_grad()

        """
        param_list = []
        if self._parameter_list is None or not isinstance(
            self._parameter_list[0], dict
        ):
            for p in self._parameter_list:
                if not p.stop_gradient:
                    param_list.append(p)
        else:
            for param_group in self._param_groups:
                for p in param_group['params']:
                    if not p.stop_gradient:
                        param_list.append(p)

        for p in param_list:
            p.clear_gradient(set_to_zero)

    @imperative_base.no_grad()
    def minimize(
        self,
        loss: Tensor,
        startup_program: Program | None = None,
        parameters: list[Tensor] | list[str] | None = None,
        no_grad_set: set[Tensor] | set[str] | None = None,
    ) -> tuple[list[Operator], list[tuple[Tensor, Tensor]]]:
        """
        Add operations to minimize ``loss`` by updating ``parameters``.

        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program|None, optional): :ref:`api_paddle_static_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_paddle_static_default_startup_program` will be used.
            parameters (list[Tensor]|list[str]|None, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set[Tensor]|set[str]|None, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple[list[Operator],list[tuple[Tensor, Tensor]]], A list of operators appended
                by minimize and a list of (param, grad) tensor pairs, param is
                ``Parameter``, grad is the gradient value corresponding to the parameter.
                In static graph mode, the returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to
                indicate program pruning. If so, the program will be pruned by ``feed`` and
                ``fetch_list`` before run, see details in ``Executor``.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> linear = paddle.nn.Linear(10, 10)
                >>> input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
                >>> out = linear(input)
                >>> loss = paddle.mean(out)

                >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
                >>> beta2 = paddle.to_tensor([0.99], dtype="float32")

                >>> adam = paddle.optimizer.Adam(learning_rate=0.1,
                ...         parameters=linear.parameters(),
                ...         weight_decay=0.01)
                >>> loss.backward()
                >>> adam.minimize(loss)
                >>> adam.clear_grad()

        """
        assert isinstance(
            loss, (Variable, paddle.pir.Value)
        ), "The loss should be an Tensor."

        parameter_list = parameters if parameters else self._parameter_list

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameters=parameter_list,
            no_grad_set=no_grad_set,
        )

        optimize_ops = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads

    def _declarative_step(self):
        """
        In declarative mode, we forward `call step` to `call apply_gradients`
        """
        params = (
            paddle.static.default_main_program().global_block().all_parameters()
        )
        assert not isinstance(
            self._parameter_list[0], dict
        ), "Only list of parameters is supported while using optimizer in @paddle.jit.static."
        selected_params = {param.name for param in self._parameter_list}
        parameters = [param for param in params if param.trainable]
        parameters = list(
            filter(
                lambda x: x.name in selected_params and hasattr(x, "grad"),
                parameters,
            )
        )
        params_grads = [(param, param.grad) for param in parameters]
        optimize_ops = self.apply_gradients(params_grads)

    @imperative_base.no_grad()
    @framework.non_static_only
    def step(self) -> None:
        """
        Execute the optimizer and update parameters once.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> a = paddle.arange(26, dtype="float32").reshape([2, 13])
                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> adam = paddle.optimizer.Adam(learning_rate = 0.01,
                ...                         parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> adam.step()
                >>> adam.clear_grad()
        """
        if paddle.base.dygraph.base.in_to_static_mode():
            self._declarative_step()
            return

        if not isinstance(self._param_groups[0], dict):
            params_grads = []
            for param in self._param_groups:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
                param_group_idx=0,
            )

        else:
            # optimize parameters in groups
            for idx, param_group in enumerate(self._param_groups):
                params_grads = defaultdict(lambda: [])
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                self._apply_optimize(
                    loss=None,
                    startup_program=None,
                    params_grads=params_grads,
                    param_group_idx=idx,
                )

    def _add_param_group(self, param_group):
        """
        Add a param group to parameter_list.

        Args:
            param_group (dict): The group of Tensors to be optimized with
            different optimization options.
        """
        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters should be in ordered collections,"
                "but received set, please use list instead."
            )
        else:
            param_group['params'] = list(params)

        # Update optimization options for each groups
        for k, v in self._default_dict.items():
            param_group.setdefault(k, v)

        param_set = set()
        for group in self._param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group"
            )

        for param in param_group['params']:
            weight_decay = param_group['weight_decay']
            if isinstance(weight_decay, float):
                regularization = L2Decay(weight_decay)
            else:
                regularization = weight_decay
            param.regularizer = regularization
            param.optimize_attr['learning_rate'] = param_group.get(
                'learning_rate', 1.0
            )

        self._param_groups.append(param_group)

    def _update_param_group(self, parameters):
        """
        Update the param group with new entry
        Args:
            parameters (dict): The extra group of Tensors to be optimized with
            different optimization options. Only used in child class.
        """
        pass

    @framework.dygraph_only
    def _multi_tensor_init(self, target_block, parameters, param_group_idx):
        """
        All parameters used for optimizer (such as: parameters, master_weight, velocity_acc for momentum) calculations are grouped into a python list by data type (float16, float32).
        This function will be overridden in the corresponding optimizer file.

        Args:
            target_block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """
        pass

    @framework.dygraph_only
    def _append_optimize_multi_tensor_op(
        self, target_block, parameters_and_grads, param_group_idx
    ):
        """
        For Multi Tensor, append optimize merged_operator to block.
        """
        pass

    def _is_dtype_fp16_or_bf16(self, dtype):
        """
        check the dtype is fp16 or the dtype is bf16
        :param dtype: instance of core.VarDesc.VarType
        :return: True if dtype is one of fp16 or bf16, False otherwise
        """
        assert isinstance(
            dtype, (core.VarDesc.VarType, core.DataType)
        ), "The dtype should be an instance of core.VarDesc.VarType or core.DataType."
        if isinstance(dtype, core.VarDesc.VarType):
            return (
                dtype == core.VarDesc.VarType.FP16
                or dtype == core.VarDesc.VarType.BF16
            )
        else:
            return (
                dtype == core.DataType.FLOAT16
                or dtype == core.DataType.BFLOAT16
            )

    def use_fusion_storage(self):
        self._use_fusion_storage = True

    def need_refuse(self):
        self._need_refuse = self._use_fusion_storage

    def reset_need_refuse(self):
        self._need_refuse = False

    @property
    def fused_buffer_version(self):
        return self._fuse_buffer_version

    @property
    def fused_states_buffer(self):
        if self.fusion_storage is None:
            return None
        return self.fusion_storage.buffer

    @property
    def fused_states_buffer_ipc_meta(self):
        if self.fusion_storage is None:
            return None
        return self.fusion_storage.buffer_ipc_meta

    @property
    def fused_states_accumulators_meta(self):
        if self.fusion_storage is None:
            return None
        return self.fusion_storage.accumulators_meta

    @property
    def fused_states_master_weights_meta(self):
        if self.fusion_storage is None:
            return None
        return self.fusion_storage.master_weights_meta
