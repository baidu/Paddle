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

import copy
import inspect
import logging
from collections import defaultdict

import paddle
from paddle import core
from paddle.jit import not_to_static, to_static
from paddle.jit.dy2static.program_translator import (
    ProgramTranslator,
    StaticFunction,
)
from paddle.jit.dy2static.utils import as_not_paddle_func
from paddle.nn import Layer
from paddle.static import Parameter, global_scope, program_guard
from paddle.static.amp.fp16_utils import (
    DEFAULT_AMP_OPTIONS,
    prepare_op_amp_options,
)

from .converter import Converter
from .dist_attribute import TensorDistAttr
from .process_group import get_world_process_group
from .utils import get_logger, to_list


class ProxyLayer(Layer):
    """
    ProxyLayer implements all logic for converting dygraph model into
    static Program IR. Meanwhile, it provides conventional interfaces for
    auto parallel to visit feed/fetch/loss/metric variables.
    """

    def __init__(self, layer, loss_func, metrics):
        super().__init__()
        # NOTE: All verify logics are finished in Engine.Prepare
        self.inner_layer = layer
        self.loss_func = loss_func
        self.metrics = metrics
        # train / eval / predict
        self.mode = None

        # generated program vars
        self._input_vars = defaultdict(list)
        self._label_vars = defaultdict(list)
        self._output_vars = defaultdict(list)
        self._loss_vars = defaultdict(list)
        self._loss_names = defaultdict(list)
        self._metric_vars = defaultdict(list)

        # Consider ProxyLayer as not Paddle inner function because it contains
        # user-defined layer.
        for fn_name in [
            "_train",
            "_eval",
            "_predict",
            "call_loss",
            "call_metrics",
        ]:
            as_not_paddle_func(
                f"{inspect.getmodule(ProxyLayer).__name__}.ProxyLayer.{fn_name}"
            )

    @paddle.jit.not_to_static
    def append_loss_to_shadow_output(self, mode):
        name = paddle.utils.unique_name.generate('loss')
        paddle._C_ops.set_persistable_value(self._loss_vars[mode], name)
        self._loss_names[mode] = name

    def _train(self, inputs, labels):
        """
        Train process of inner_layer with forward/loss/metric logic.
        """
        # step 1. save feed variables of Program
        mode = 'train'
        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels

        # step 2. call inner_layer.forward
        self._output_vars[mode] = self.inner_layer(*inputs)

        # step 3. calculate loss if needed
        new_inputs = self._prepare(self.output_vars, labels)
        self._loss_vars[mode] = self.call_loss(new_inputs)
        if paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            self.append_loss_to_shadow_output(mode)

        # step 4. calculate metrics if needed
        self._metric_vars[mode] = self.call_metrics(new_inputs)

    def _eval(self, inputs, labels):
        """
        Evaluate process of inner_layer with forward/loss/metric logic.
        """
        # TODO(dev): we can reuse codes with self._train after making
        # sure if they can.

        # step 1. save feed variables of Program
        mode = 'eval'
        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels

        # step 2. call inner_layer.forward
        self._output_vars[mode] = self.inner_layer(*inputs)

        # step 3. calculate loss if needed
        new_inputs = self._prepare(self.output_vars, labels)
        self._loss_vars[mode] = self.call_loss(new_inputs)
        if paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            self.append_loss_to_shadow_output(mode)

        # step 4. calculate metrics if needed
        self._metric_vars[mode] = self.call_metrics(new_inputs)

    def _predict(self, inputs, labels):
        """
        Predict process of inner_layer with forward logic.
        """
        # step 1. save feed variables of Program
        mode = 'predict'
        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels

        # step 2. call inner_layer.forward
        self._output_vars[mode] = self.inner_layer(*inputs)

    @not_to_static
    def _prepare(self, outputs, labels):
        """
        Concat outputs and labels as a single list

        NOTE(dev): We use @not_to_static to avoid AST Analysis.
        """
        return to_list(outputs) + to_list(labels)

    def call_loss(self, inputs):
        """
        Apply Loss Function on outputs and labels.

        Args:
            inputs: List[Variable]

        Returns: List[Variable]
        """
        res = []
        if self.loss_func is not None:
            res = self.loss_func(*inputs)
        return res

    def call_metrics(self, inputs):
        """
        Apply Metrics Function on outputs and labels.

        Args:
            inputs: List[Variable]

        Returns: List[Variable]
        """
        outs = []
        for metric in self.metrics:
            outs.append(to_list(metric.compute(*inputs)))

        return outs

    def set_mode(self, mode):
        self.mode = mode
        self.training = mode == 'train'

    def clone(self):
        return ProxyLayer(self.inner_layer, self.loss_func, self.metrics)

    @property
    def input_vars(self):
        return self._input_vars[self.mode]

    @property
    def label_vars(self):
        return self._label_vars[self.mode]

    @property
    def output_vars(self):
        return self._output_vars[self.mode]

    @property
    def loss_vars(self):
        return self._loss_vars[self.mode]

    @property
    def loss_names(self):
        return self._loss_names[self.mode]

    @property
    def metric_vars(self):
        return self._metric_vars[self.mode]

    @property
    def startup_program(self):
        return self.inner_layer._startup_program()


class BuildInfo:
    def __init__(self):
        self.clear()

    def has_cache(self, mode, update=False):
        is_cache = self.states[mode]
        if update:
            self.cache(mode)
        return is_cache

    def cache(self, mode):
        self.states[mode] = True

    def clear(self):
        self.states = defaultdict(bool)


class ProgramHelper:
    """
    A Helper class for Engine to provides different Program IR according specified 'mode'.
    """

    def __init__(self, layer, loss_func, metrics, inputs_spec, labels_spec):
        # original model config information
        # TODO(Aurelius84): Implement append_backward and optimizer in ProxyLayer
        # after distribute engine satisfy basic condition.
        self.proxy_layer = ProxyLayer(layer, loss_func, metrics)
        self.inputs_spec = inputs_spec
        self.labels_spec = labels_spec

        self.build_info = BuildInfo()
        self._logger = get_logger(logging.INFO)
        self.lazy_init = False
        self._all_params_dist_attr = {}

    def reset(self):
        """
        Reset all state of current Object.
        """
        self.build_info.clear()
        self.proxy_layer = self.proxy_layer.clone()

    def build_program(self, mode):
        """
        Convert dygraph model into static Program IR.
        """
        assert mode in ['train', 'eval', 'predict']
        self.proxy_layer.set_mode(mode)
        # skip if we has already built program.
        if self.build_info.has_cache(mode, True):
            self._logger.info(
                f"Already build program with mode = {mode}, use cached program."
            )
            return

        self._logger.info(f"start to build program for mode = {mode}.")
        input_spec = [self.inputs_spec, self.labels_spec]
        static_func = to_static(
            self.static_func(), input_spec=input_spec, full_graph=True
        )

        func_name = '_' + mode
        setattr(self.proxy_layer, func_name, static_func)

        # NOTE(dev): Because @to_static is a Lazy mechanism, so we explicitly call this to trigger
        # generating Program IR immediately.
        concrete_program = getattr(self.proxy_layer, func_name).concrete_program

        # TODO(zhiqiu): prepare_op_amp_options is not supported for PIR program
        # It will to use dynamic-static unified amp in pir program, and there is
        # no need to fit for prepare_op_amp_options
        if not paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            prepare_op_amp_options(
                concrete_program.main_program,
                ProgramTranslator.get_instance()._amp_records,
                DEFAULT_AMP_OPTIONS,
            )
        self._build_startup_program()

    def _build_startup_program(self):
        """
        Create and Sync parameters into startup program.
        """
        startup_program = self.startup_program
        if len(startup_program.global_block().ops) > 1:
            self.lazy_init = True
            return

        for param in self.concrete_program.parameters:
            Parameter(
                name=param.name,
                desc=param,
                type=param.type,
                shape=param.shape,
                dtype=param.dtype,
                stop_gradient=param.stop_gradient,
                block=startup_program.global_block(),
            )

    def apply_optimizer(self, optimizer):
        """
        Append backward and generate optimizer operations.
        """
        self._verify_optimizer(optimizer)
        self._logger.info(
            "start to apply optimizer: %s ", type(optimizer).__name__
        )
        # clear optimizer parameters
        original_params = optimizer._parameter_list
        optimizer._parameter_list = None
        with program_guard(self.main_program, self.startup_program):
            res = optimizer.minimize(self.loss_vars[0])

        # restore optimizer parameters
        optimizer._parameter_list = original_params
        return res

    def _verify_optimizer(self, optimizer):
        assert optimizer is not None
        assert hasattr(
            optimizer, "minimize"
        ), "Optimizer must have minimize() method."
        assert (
            self.proxy_layer.mode == 'train'
        ), f"Required mode == 'train', but received '{self.proxy_layer.mode}'"
        assert (
            len(self.loss_vars) == 1
        ), f"Required len(loss_vars) == 1, but received len(loss_vars) = {len(self.loss_vars)}"

    def to(self, mode):
        """
        Switch underly proxy layer mode into target mode.
        """
        assert mode in ['train', 'eval', 'predict']
        func = getattr(self.proxy_layer, '_' + mode)
        assert isinstance(
            func, StaticFunction
        ), "Please call build_program(mode) firstly."
        self.proxy_layer.set_mode(mode)

    def static_func(self):
        """
        Return StaticFunction instance with underly target mode.
        """
        assert self.proxy_layer.mode in [
            'train',
            'eval',
            'predict',
        ], "Please call build_program(mode) firstly."
        func_name = '_' + self.proxy_layer.mode
        return getattr(self.proxy_layer, func_name)

    def init_pir(self, main_program, place):
        # collect all params in current dist program
        param_values = main_program.global_block().all_parameters()
        value_name_to_value = {}
        dy_param_name_to_pir_param_name = {}
        for value in param_values:
            value_name_to_value[value.name] = value

        dy_params = self.concrete_program.parameters[0]
        pir_param = self.concrete_program.parameters[1]

        for i in range(len(pir_param)):
            if pir_param[i].name in value_name_to_value:
                dy_param_name_to_pir_param_name[dy_params[i].name] = pir_param[
                    i
                ].name

        is_comm = False
        for param in dy_params:
            if param.is_dist():
                process_mesh, dims_mapping = self._all_params_dist_attr[
                    param.name
                ]
                var_dist_attr = TensorDistAttr()
                var_dist_attr.process_mesh = process_mesh
                var_dist_attr.dims_mapping = dims_mapping
                is_comm = True
                with paddle.no_grad():
                    tmp = paddle.base.core.reshard(param, var_dist_attr)
                if tmp._is_initialized():
                    param.get_tensor()._share_data_with(tmp.get_tensor())
                else:
                    # Only setting the "param" to "None" can't release the memory
                    param.get_tensor()._clear()
                    param = None

            # create var in scope and share parameters to scope
            if param is None:
                continue
            if param.name not in dy_param_name_to_pir_param_name:
                # Release the redundant params
                param.get_tensor()._clear()
                continue
            if not param._is_initialized():
                continue
            if param.is_dense():
                value_name = dy_param_name_to_pir_param_name[param.name]
                value = value_name_to_value[value_name]
                # get param_var's dist_attr
                assert (
                    value.is_dist_dense_tensor_type()
                ), f"param [{value.name}] is not dist tensor type"
                dist_attr = {
                    "dims_mapping": value.dist_attr().dims_mapping,
                    "process_shape": value.dist_attr().process_mesh.shape,
                    "process_group": value.dist_attr().process_mesh.process_ids,
                }
                # slice param_value with dist_attr
                # share sliced_param_value with param_tensor in global_scope
                pir_scope_param = global_scope().var(value_name).get_tensor()
                sliced_param = Converter.slice_with_dist_attr(
                    param.numpy(), dist_attr
                )
                pir_scope_param.set(sliced_param, place)
                param.get_tensor()._clear()

            elif param.is_dist():
                value_name = dy_param_name_to_pir_param_name[param.name]
                value = value_name_to_value[value_name]
                # assert value.is_dist_dense_tensor_type(), "param [{}] is not dist tensor type".format(value.name)
                pir_scope_param = global_scope().var(value_name).get_tensor()
                pir_scope_param._share_data_with(
                    param.get_tensor().get_tensor()
                )
                param.get_tensor()._clear()

        world_group = get_world_process_group()
        if (
            is_comm
            and world_group.nranks > 1
            and paddle.distributed.get_world_size() > 1
        ):
            paddle.disable_static()
            barrier_tensor = paddle.full([1], 1, dtype="int32")
            paddle._legacy_C_ops.barrier(
                barrier_tensor, barrier_tensor, 'ring_id', 0
            )
            paddle.enable_static()

    def init(self, main_program, place, dist_context):
        if self.lazy_init:
            return

        amp_strategy = dist_context.strategy.amp
        amp_config = copy.deepcopy(amp_strategy.to_dict())
        need_cast_parameter = amp_strategy.enable and amp_config["level"] in [
            "o2",
            "o3",
        ]
        is_comm = False
        for param in self.concrete_program.parameters:
            if param.is_dist():
                serial_main_program = self.concrete_program.main_program
                var = serial_main_program.global_block().vars[param.name]
                var_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                    var
                )
                is_comm = True
                # No need to construct backward.
                with paddle.no_grad():
                    tmp = paddle.base.core.reshard(param, var_dist_attr)
                if tmp._is_initialized():
                    param.get_tensor()._share_data_with(tmp.get_tensor())
                else:
                    # Only setting the "param" to "None" can't release the memory
                    param.get_tensor()._clear()
                    param = None
            paddle.device.synchronize()

            # create var in scope and share parameters to scope
            if param is None:
                continue
            if param.name not in main_program.global_block().vars:
                # Release the redundant params
                param.get_tensor()._clear()
                continue
            if not param._is_initialized():
                continue
            if param.is_dense():
                # get param_var's dist_attr
                var = main_program.global_block().vars[param.name]
                var_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                    var
                )
                dist_attr = {
                    "dims_mapping": var_dist_attr.dims_mapping,
                    "process_shape": var_dist_attr.process_mesh.shape,
                    "process_group": var_dist_attr.process_mesh.process_ids,
                }
                # slice param_value with dist_attr
                # share sliced_param_value with param_tensor in global_scope
                param_tensor = global_scope().var(param.name).get_tensor()
                sliced_param = Converter.slice_with_dist_attr(
                    param.numpy(), dist_attr
                )
                param_tensor.set(sliced_param, place)
                if not need_cast_parameter:
                    param.get_tensor()._clear()
            elif param.is_dist():
                dense_tensor = global_scope().var(param.name).get_tensor()
                dense_tensor._share_data_with(param.get_tensor().get_tensor())

        # transform the parameter in eager mode for amp.
        if need_cast_parameter:
            for param in self.concrete_program.parameters:
                amp_dtype = amp_config["dtype"]
                scope_var = global_scope().find_var(param.name)
                # The parameter is not in this rank.
                if not scope_var:
                    continue
                # The parameter do not need to transform
                if param.dtype in [paddle.float16, paddle.bfloat16]:
                    continue
                scope_tensor = global_scope().var(param.name).get_tensor()
                assert (
                    scope_var and scope_tensor._is_initialized()
                ), f"Parameter: {param.name} is not put into global_scope or not initialized."
                param_used = param
                # For the params without dist_attr.
                # NOTE(lizhiyu): In principle, each param should have dist_attr.
                if param.is_dense():
                    # get param_var's dist_attr
                    var = main_program.global_block().vars[param.name]
                    var_dist_attr = (
                        dist_context.get_tensor_dist_attr_for_program(var)
                    )
                    dist_attr = {
                        "dims_mapping": var_dist_attr.dims_mapping,
                        "process_shape": var_dist_attr.process_mesh.shape,
                        "process_group": var_dist_attr.process_mesh.process_ids,
                    }
                    # slice param_value with dist_attr
                    sliced_param = Converter.slice_with_dist_attr(
                        param.numpy(), dist_attr
                    )
                    with paddle.base.dygraph.guard():
                        param_used = paddle.to_tensor(
                            sliced_param, place=param.place
                        )
                    param.get_tensor()._clear()
                with paddle.base.dygraph.guard():
                    if amp_dtype == "float16":
                        with paddle.no_grad():
                            with paddle.base.framework._dygraph_place_guard(
                                place=place
                            ):
                                t_casted = param_used.cast(
                                    dtype=core.VarDesc.VarType.FP16
                                )
                    elif amp_dtype == "bfloat16":
                        with paddle.no_grad():
                            with paddle.base.framework._dygraph_place_guard(
                                place=place
                            ):
                                t_casted = param_used.cast(
                                    dtype=core.VarDesc.VarType.BF16
                                )
                    # NOTE(lizhiyu): Clear the origin param. Don't use `param_used.get_tensor().get_tensor()._clear()` to
                    #                clear the `DistTensor`, because it can't clear the `_holder`,
                    #                which `param_used.get_tensor().get_tensor()` will copy one `DenseTensor`.
                    param_used.get_tensor()._clear()
                    if t_casted.is_dist():
                        scope_tensor._share_data_with(
                            t_casted.get_tensor().get_tensor()
                        )
                    else:
                        scope_tensor._share_data_with(t_casted.get_tensor())

        world_group = get_world_process_group()
        if (
            is_comm
            and world_group.nranks > 1
            and paddle.distributed.get_world_size() > 1
        ):
            paddle.disable_static()
            barrier_tensor = paddle.full([1], 1, dtype="int32")
            paddle._legacy_C_ops.barrier(
                barrier_tensor, barrier_tensor, 'ring_id', 0
            )
            paddle.enable_static()

    def cache_whole_graph_dist_attr(self, all_params):
        for param_value in all_params:
            dist_attr = param_value.dist_attr()
            if dist_attr:
                process_mesh = dist_attr.process_mesh
                dims_mapping = dist_attr.dims_mapping
                self._all_params_dist_attr[param_value.name] = [
                    process_mesh,
                    dims_mapping,
                ]

    @property
    def concrete_program(self):
        return self.static_func().concrete_program

    @property
    def main_program(self):
        return self.concrete_program.main_program

    @property
    def startup_program(self):
        try:
            return self.proxy_layer.startup_program
        except Exception as err:
            self._logger.warning(
                "The startup_program is not built by `lazy init`."
            )
            if isinstance(err, AssertionError):
                return self.concrete_program.startup_program
            raise err

    @property
    def input_vars(self):
        return to_list(self.proxy_layer.input_vars)

    @property
    def output_vars(self):
        return to_list(self.proxy_layer.output_vars)

    @property
    def label_vars(self):
        return to_list(self.proxy_layer.label_vars)

    @property
    def loss_vars(self):
        return to_list(self.proxy_layer.loss_vars)

    @property
    def loss_names(self):
        return to_list(self.proxy_layer.loss_names)

    @property
    def metric_vars(self):
        return to_list(self.proxy_layer.metric_vars)

    def named_parameters(self):
        static_func = self.static_func()
        partial_program = static_func.get_concrete_program(
            self.inputs_spec, self.labels_spec
        )[-1]
        # TODO(xiongkun): support pir in the feature.
        return {param.name: param for param in partial_program._params}
