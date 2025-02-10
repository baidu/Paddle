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

import contextlib
import inspect
import os
import pickle
import socket
import time
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Union,
    overload,
)

import numpy as np
from typing_extensions import TypeAlias

import paddle
import paddle.distributed as dist
import paddle.optimizer
from paddle import base
from paddle.autograd import no_grad
from paddle.base import core
from paddle.base.executor import global_scope
from paddle.base.framework import (
    Variable,
    _current_expected_place_ as _get_device,
    _get_paddle_place,
)
from paddle.distributed import fleet
from paddle.distributed.collective import _init_parallel_env
from paddle.distributed.fleet.base import role_maker
from paddle.framework import in_dynamic_mode, in_pir_mode
from paddle.framework.io_utils import is_belong_to_optimizer
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.metric import Metric
from paddle.static import InputSpec as Input

from .callbacks import EarlyStopping, config_callbacks
from .model_summary import summary

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from paddle import Tensor
    from paddle._typing.dtype_like import _DTypeLiteral

    from .callbacks import Callback
    from .model_summary import ModelSummary

    _InputBatch: TypeAlias = Union[
        Tensor,
        npt.NDArray[Any],
        List[Tensor],
        List[npt.NDArray[Any]],
    ]


__all__ = []

_parallel_context_initialized = False


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def to_numpy(var):
    assert isinstance(
        var, (Variable, base.core.eager.Tensor, paddle.base.libpaddle.pir.Value)
    ), "not a variable"
    if isinstance(var, base.core.eager.Tensor):
        return np.array(var)
    t = global_scope().find_var(var.name).get_tensor()
    return np.array(t)


def flatten_list(l):
    assert isinstance(l, list), "not a list"
    outl = []
    splits = []
    for sl in l:
        assert isinstance(sl, list), "sub content not a list"
        splits.append(len(sl))
        outl += sl
    return outl, splits


def restore_flatten_list(l, splits):
    outl = []
    for split in splits:
        assert len(l) >= split, "list length invalid"
        sl, l = l[:split], l[split:]
        outl.append(sl)
    return outl


def extract_args(func):
    return inspect.getfullargspec(func).args


def _all_gather(x):
    output = []
    dist.all_gather(output, x)
    output = paddle.concat(output, axis=0)
    return output


def wait_server_ready(endpoints):
    assert not isinstance(endpoints, str)
    while True:
        all_ok = True
        not_ready_endpoints = []
        for ep in endpoints:
            ip_port = ep.split(":")
            with contextlib.closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                if result != 0:
                    all_ok = False
                    not_ready_endpoints.append(ep)
        if not all_ok:
            time.sleep(3)
        else:
            break


def init_communicator(
    program, rank, nranks, wait_port, current_endpoint, endpoints
):
    if nranks < 2:
        return
    endpoints_str = ",".join(endpoints)
    other_endpoints = endpoints[:]
    other_endpoints.remove(current_endpoint)
    block = program.global_block()
    if rank == 0 and wait_port:
        wait_server_ready(other_endpoints)
    if core.is_compiled_with_cuda():
        use_new_comm = paddle.get_flags("FLAGS_dynamic_static_unified_comm")[
            "FLAGS_dynamic_static_unified_comm"
        ]
        if use_new_comm:
            _init_parallel_env("nccl")
        else:
            nccl_id_var = block.create_var(
                name=base.unique_name.generate('nccl_id'),
                persistable=True,
                type=base.core.VarDesc.VarType.RAW,
            )

            block.append_op(
                type='c_gen_nccl_id',
                inputs={},
                outputs={'Out': nccl_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                },
            )

            block.append_op(
                type='c_comm_init',
                inputs={'X': nccl_id_var},
                outputs={},
                attrs={
                    'nranks': nranks,
                    'rank': rank,
                    'ring_id': 0,
                    'endpoints': endpoints_str,
                },
            )
    elif core.is_compiled_with_xpu():
        bkcl_id_var = block.create_var(
            name=base.unique_name.generate('bkcl_id'),
            persistable=True,
            type=base.core.VarDesc.VarType.RAW,
        )

        block.append_op(
            type='c_gen_bkcl_id',
            inputs={},
            outputs={'Out': bkcl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
            },
        )

        block.append_op(
            type='c_comm_init',
            inputs={'X': bkcl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': 0,
                'endpoints': endpoints_str,
            },
        )
    elif (
        paddle.distributed.ParallelEnv().device_type
        in paddle.device.get_all_custom_device_type()
    ):
        xccl_id_var = block.create_var(
            name=base.unique_name.generate('xccl_id'),
            persistable=True,
            type=base.core.VarDesc.VarType.RAW,
        )

        block.append_op(
            type='c_gen_xccl_id',
            inputs={},
            outputs={'Out': xccl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
            },
        )

        block.append_op(
            type='c_comm_init',
            inputs={'X': xccl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': 0,
                'endpoints': endpoints_str,
            },
        )


def prepare_distributed_context(place=None):
    if place is None:
        place = (
            base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
            if paddle.distributed.ParallelEnv().nranks > 1
            else base.CUDAPlace(0)
        )

    place = _get_paddle_place(place)
    strategy = paddle.distributed.parallel.ParallelStrategy()
    strategy.nranks = paddle.distributed.ParallelEnv().nranks
    strategy.local_rank = paddle.distributed.ParallelEnv().local_rank
    strategy.trainer_endpoints = (
        paddle.distributed.ParallelEnv().trainer_endpoints
    )
    strategy.current_endpoint = (
        paddle.distributed.ParallelEnv().current_endpoint
    )

    if strategy.nranks < 2:
        return

    global _parallel_context_initialized

    if not _parallel_context_initialized and isinstance(place, base.CUDAPlace):

        def _init_context():
            communicator_prog = base.Program()
            init_communicator(
                communicator_prog,
                strategy.local_rank,
                strategy.nranks,
                True,
                strategy.current_endpoint,
                strategy.trainer_endpoints,
            )
            exe = base.Executor(place)
            exe.run(communicator_prog)

        if in_dynamic_mode():
            base.disable_dygraph()
            _init_context()
            base.enable_dygraph(place)

    else:
        assert "Only support CUDAPlace for now."

    _parallel_context_initialized = True
    return strategy


def _update_input_info(inputs):
    "Get input shape list by given inputs in Model initialization."
    shapes = None
    dtypes = None
    if isinstance(inputs, Input):
        shapes = [list(inputs.shape)]
        dtypes = [inputs.dtype]
    elif isinstance(inputs, (list, tuple)):
        shapes = [list(input.shape) for input in inputs]
        dtypes = [input.dtype for input in inputs]
    elif isinstance(inputs, dict):
        shapes = [list(inputs[name].shape) for name in inputs]
        dtypes = [inputs[name].dtype for name in inputs]
    else:
        return None
    return shapes, dtypes


class StaticPIRGraphAdapter:
    """

    Model training/inference with PIR.

    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # with `_build_once` gone, parameters are now created in `__init__`
        # so we need to keep track of the parameters already created
        self._startup_prog = paddle.pir.core.default_startup_program()
        self._orig_prog = paddle.pir.core.default_main_program()

        self._label_vars = {}
        self._input_vars = {}
        self._endpoints = {}
        self._loss_endpoint = None
        self._executor = None
        self._progs = {}
        self._compiled_progs = {}

        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0,
        }

        self._nranks = paddle.distributed.ParallelEnv().nranks
        self._local_rank = paddle.distributed.ParallelEnv().local_rank

        self._amp_level = "O0"
        self._amp_configs = {}
        self._amp_custom_lists = {}
        self._use_fp16_guard = None

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    def train_batch(self, inputs, labels=None, update=True):
        assert (
            self.model._optimizer
        ), "model not ready, please call `model.prepare()` first"
        self.mode = 'train'
        assert (
            update is True
        ), "Does not support `update == False` in static graph mode by now."
        return self._run(inputs, labels)

    def eval_batch(self, inputs, labels=None):
        self.mode = 'eval'
        return self._run(inputs, labels)

    def predict_batch(self, inputs):
        self.mode = 'test'
        return self._run(inputs, None)

    def parameters(self, *args, **kwargs):
        return self.model.network.parameters(*args, **kwargs)

    def save(self, path):
        def _save(state, path):
            if not state:
                return
            state = {
                k: (
                    to_numpy(v)
                    if isinstance(v, paddle.base.libpaddle.pir.Value)
                    else v
                )
                for k, v in state.items()
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)

        def get_tensor(var):
            t = global_scope().find_var(var.name).get_tensor()
            return np.array(t)

        base = os.path.basename(path)
        assert base != "", "path should be of 'dirname/filename' format"
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        param_path = path + ".pdparams"
        _save(self.model.network.state_dict(), param_path)

        prog = self._progs.get('train', None)

        if prog is None or self.model._optimizer is None:
            return
        # XXX `optimizer.state_dict()` only work in dygraph mode
        optim_path = path + ".pdopt"

        opts = []
        for var in prog.list_vars():
            if var.persistable and var.get_defining_op().name() == "pd_op.data":
                opts.append(var)

        opt_dict = {
            var.name: get_tensor(var) for var in opts if var.persistable
        }
        if not opt_dict:
            return

        _save(opt_dict, optim_path)

    def _set_var(self, name, ndarray):
        t = global_scope().find_var(name).get_tensor()
        p = t._place()
        if p.is_cpu_place():
            place = paddle.base.CPUPlace()
        elif p.is_cuda_pinned_place():
            place = paddle.base.CUDAPinnedPlace()
        elif p.is_xpu_place():
            p = paddle.base.core.Place()
            p.set_place(t._place())
            place = paddle.base.XPUPlace(p.xpu_device_id())
        elif p.is_custom_place():
            p = paddle.base.core.Place()
            p.set_place(t._place())
            place = paddle.base.CustomPlace(
                paddle.device.get_device().split(':')[0], p.custom_device_id()
            )
        else:
            p = paddle.base.core.Place()
            p.set_place(t._place())
            place = paddle.base.CUDAPlace(p.gpu_device_id())

        t.set(ndarray, place)

    def load(self, param_state_pairs, optim_state):
        if self._executor is None:
            executor = base.Executor(base.CPUPlace())._default_executor
        else:
            executor = self._executor._default_executor

        paddle.base.libpaddle.pir.create_loaded_parameter(
            [param for param, state in param_state_pairs],
            global_scope(),
            executor,
        )
        for param, state in param_state_pairs:
            self._set_var(param.name, state)

        # restore optimizer states
        # FIXME what if a different optimizer is used?
        if not self.model._optimizer or not optim_state:
            return
        self._load_optimizer(optim_state, executor)

    def _load_optimizer(self, state, executor):
        prog = self._progs.get('train', None)
        optim = []
        for var in prog.list_vars():
            if not var.is_parameter:
                if var.persistable:
                    optim.append(var)

        if not optim:
            return

        base.core._create_loaded_parameter(optim, global_scope(), executor)

        converted_state = dict(state)
        for var in optim:
            if var.name.startswith("learning_rate_"):
                # When using static learning rate, static-graph would make it
                # a persistable var named 'unique_name.generate("learning_rate")',
                # However, dygraph wouldn't save it.
                if var.name not in state:
                    continue
            assert (
                var.name in converted_state
            ), f"variable [{var.name}] is not in optimizer state file"
            self._set_var(var.name, converted_state[var.name])

    def _run(self, inputs, labels=None):
        compiled_prog = self._compiled_progs.get(self.mode, None)
        assert (
            compiled_prog
        ), "Model is not ready, please call `model.prepare()` first"

        inputs = to_list(inputs)
        if labels is not None:
            labels = to_list(labels)
        assert len(inputs) == len(self._input_vars[self.mode]), (
            "number of inputs"
            + " does not match number of arguments of `forward` method"
        )

        feed = {}
        input_names = [v.name for v in self._input_vars[self.mode]]
        input_dtypes = [v.dtype for v in self._input_vars[self.mode]]

        for idx, n in enumerate(input_names):
            # train and test may take different arguments
            if inputs[idx] is not None:
                feed[n] = inputs[idx]
            if self._amp_level == 'O2' and input_dtypes[idx] == paddle.float16:
                if isinstance(feed[n], core.DenseTensor):
                    feed[n] = feed[n]._as_type(paddle.pir.core.DataType.FLOAT16)
                elif isinstance(feed[n], np.ndarray):
                    feed[n] = feed[n].astype('float16')

        if labels is not None:
            for idx, v in enumerate(self._label_vars[self.mode]):
                feed[v.name] = labels[idx]

        endpoints = self._endpoints[self.mode]
        if self.mode == 'test':
            fetch_list = endpoints['output']
        else:
            metric_list, metric_splits = flatten_list(endpoints['metric'])
            fetch_list = endpoints['loss'] + metric_list
            num_loss = len(endpoints['loss'])

        # if fetch Variable is same as input Variable, do not fetch
        # from program, get it from input directly
        pruned_fetch_list = []
        pruned_fetch_idx_name_map = [""] * len(fetch_list)
        for i, fetch_var in enumerate(fetch_list):
            if fetch_var in feed.keys():
                pruned_fetch_idx_name_map[i] = fetch_var
            else:
                pruned_fetch_list.append(fetch_var)

        rets = self._executor.run(
            compiled_prog,
            feed=feed,
            fetch_list=pruned_fetch_list,
            return_numpy=False,
        )

        # restore pruned fetch_list Variable from feeds
        for i, name in enumerate(pruned_fetch_idx_name_map):
            if len(name) > 0:
                rets.insert(i, feed[name])

        # DenseTensor cannot be fetch as numpy directly
        rets = [np.array(v) for v in rets]
        if self.mode == 'test':
            return rets[:]

        metric_states = restore_flatten_list(rets[num_loss:], metric_splits)
        metrics = []
        for metric, state in zip(self.model._metrics, metric_states):
            metrics.append(metric.update(*state))

        if num_loss and len(metrics):
            return rets[:num_loss], metrics
        else:
            return rets[:num_loss] if num_loss else metrics

    def prepare(self):
        modes = ['train', 'test', 'eval']
        for mode in modes:
            self._make_program(mode)
            self._initialize(self._progs[mode], mode)

    def _make_program(self, mode):
        prog = self._progs.get(mode, None)
        if prog is not None:
            return

        prog = self._orig_prog.clone()

        losses = []
        metrics = []
        prog_param = prog.get_all_parameter_values()

        named_sublayers = self.model.network.named_sublayers(
            prefix='',
            include_self=True,
            remove_duplicate=True,
        )
        for layer_prefix, sublayer in named_sublayers:
            params = sublayer._parameters.items()
            for key, param in params:
                sublayer._parameters[key] = prog_param[param.name]

        with base.program_guard(prog, self._startup_prog):
            inputs = self.model._inputs
            labels = self.model._labels if self.model._labels else []
            inputs = [k._create_feed_layer() for k in to_list(inputs)]
            labels = [k._create_feed_layer() for k in to_list(labels)]
            self._label_vars[mode] = labels

            if mode == 'train' and self.model._optimizer:
                opt_param = []
                for key, value in prog_param.items():
                    opt_param.append(value)

                self.model._optimizer._parameter_list = list(opt_param)

                if self._amp_level != "O0" and core.is_compiled_with_cuda:
                    self.model.network, self.model._optimizer = (
                        paddle.amp.decorate(
                            models=self.model.network,
                            optimizers=self.model._optimizer,
                            level=self._amp_level,
                        )
                    )

                    with paddle.amp.auto_cast(
                        level=self._amp_level, dtype='float16', use_promote=True
                    ):
                        outputs = to_list(self.model.network.forward(*inputs))

                        if mode != 'test' and self.model._loss:
                            losses = self.model._loss(*(outputs + labels))

                        if mode != 'test':
                            for metric in self.model._metrics:
                                metrics.append(
                                    to_list(metric.compute(*(outputs + labels)))
                                )
                else:
                    outputs = to_list(self.model.network.forward(*inputs))

                    if mode != 'test' and self.model._loss:
                        losses = self.model._loss(*(outputs + labels))

                    if mode != 'test':
                        for metric in self.model._metrics:
                            metrics.append(
                                to_list(metric.compute(*(outputs + labels)))
                            )

                self._loss_endpoint = paddle.add_n(losses)

                self.model._optimizer.minimize(self._loss_endpoint)
            else:
                outputs = to_list(self.model.network.forward(*inputs))

                if mode != 'test' and self.model._loss:
                    losses = self.model._loss(*(outputs + labels))

                if mode != 'test':
                    for metric in self.model._metrics:
                        metrics.append(
                            to_list(metric.compute(*(outputs + labels)))
                        )

        if mode != 'train':
            # Some operators, e.g., :ref:`api_paddle_base_layers_batch_norm` , behave differently between
            # training and testing. They have an attribute, :code:`is_test`, to
            # control this behaviour. This method will change the :code:`is_test`
            # attribute of them to :code:`True` when :code:`for_test=True`.
            prog.set_is_test_attr()

        self._input_vars[mode] = inputs

        self._progs[mode] = prog
        self._endpoints[mode] = {
            "output": outputs,
            "loss": to_list(losses),
            "metric": metrics,
        }

    def _initialize(self, prog, mode):
        assert (
            self.model._place is not None
        ), "device is not set, please call `model.prepare()` first"

        place = self.model._place

        if self._executor is None:
            self._executor = base.Executor(place)
            self._executor.run(self._startup_prog)

        if (
            self._amp_level == "O2"
            and mode == 'train'
            and core.is_compiled_with_cuda()
        ):
            self.model._optimizer.amp_init(place)

        self._compiled_progs[mode] = prog


class StaticGraphAdapter:
    """

    Model training/inference with a static graph.

    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # with `_build_once` gone, parameters are now created in `__init__`
        # so we need to keep track of the parameters already created
        self._startup_prog = base.default_startup_program()
        self._orig_prog = base.default_main_program()

        self._label_vars = {}  # label variables
        self._input_vars = {}  # label variables
        self._endpoints = {}
        self._loss_endpoint = None
        self._executor = None
        self._progs = {}
        self._compiled_progs = {}

        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0,
        }

        self._nranks = paddle.distributed.ParallelEnv().nranks
        self._local_rank = paddle.distributed.ParallelEnv().local_rank

        self._amp_level = "O0"
        self._amp_configs = {}
        self._amp_custom_lists = {}
        self._use_fp16_guard = None

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    def train_batch(self, inputs, labels=None, update=True):
        assert (
            self.model._optimizer
        ), "model not ready, please call `model.prepare()` first"
        self.mode = 'train'
        assert (
            update is True
        ), "Does not support `update == False` in static graph mode by now."
        return self._run(inputs, labels)

    def eval_batch(self, inputs, labels=None):
        self.mode = 'eval'
        return self._run(inputs, labels)

    def predict_batch(self, inputs):
        self.mode = 'test'
        return self._run(inputs, None)

    def parameters(self, *args, **kwargs):
        return self.model.network.parameters(*args, **kwargs)

    def save(self, path):
        def _save(state, path):
            if not state:
                return
            state = {
                k: to_numpy(v) if isinstance(v, Variable) else v
                for k, v in state.items()
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)

        base = os.path.basename(path)
        assert base != "", "path should be of 'dirname/filename' format"
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        param_path = path + ".pdparams"
        _save(self.model.network.state_dict(), param_path)
        prog = self._progs.get('train', None)
        if prog is None or self.model._optimizer is None:
            return
        # XXX `optimizer.state_dict()` only work in dygraph mode
        optim_path = path + ".pdopt"
        optim = {
            p.name: p for p in filter(is_belong_to_optimizer, prog.list_vars())
        }
        if not optim:
            return

        _save(optim, optim_path)

    # TODO: Support save/load scaler state in static graph
    def load(self, param_state_pairs, optim_state):
        if self._executor is None:
            executor = base.Executor(base.CPUPlace())._default_executor
        else:
            executor = self._executor._default_executor

        # restore parameter states
        base.core._create_loaded_parameter(
            [param for param, state in param_state_pairs],
            global_scope(),
            executor,
        )
        for param, state in param_state_pairs:
            self._set_var(param, state)

        # restore optimizer states
        # FIXME what if a different optimizer is used?
        if not self.model._optimizer or not optim_state:
            return
        self._load_optimizer(optim_state, executor)

    def _load_optimizer(self, state, executor):
        prog = self._progs.get('train', None)
        optim = list(filter(is_belong_to_optimizer, prog.list_vars()))
        if not optim:
            return

        base.core._create_loaded_parameter(optim, global_scope(), executor)

        converted_state = dict(state)
        for var in optim:
            if var.name in ["@LR_DECAY_COUNTER@", "global_step"]:
                # When using learning rate scheduler, dygraph would name the
                # global step var as "global_step" to save, while static-graph
                # would has a state var named as "@LR_DECAY_COUNTER@".
                # NOTE: dygraph saved global_step is 1 larger than that in
                # static-graph, since the time of global_step to increase is
                # different.
                state_val = (
                    (np.array(converted_state.pop("global_step")) - 1)
                    if "global_step" in converted_state
                    else converted_state.pop("@LR_DECAY_COUNTER@", None)
                )
                if state_val is not None:
                    converted_state[var.name] = state_val
            elif var.name.startswith("learning_rate_"):
                # When using static learning rate, static-graph would make it
                # a persistable var named 'unique_name.generate("learning_rate")',
                # However, dygraph wouldn't save it.
                if var.name not in state:
                    continue
            else:
                # moment and other accumulators
                if var.name not in converted_state:
                    # try to convert from dygraph name
                    opt_name = self.model._optimizer._name
                    opt_cls_name = self.model._optimizer.__class__.__name__
                    opt_unq_name = None
                    for name in self.model._optimizer._accumulators.keys():
                        accum_name = (
                            name
                            if opt_name is None
                            else name[len(opt_name) + 1 :]
                        )
                        for (
                            param_name,
                            state_var,
                        ) in self.model._optimizer._accumulators[name].items():
                            if opt_unq_name is None:
                                # can not infer out the exact unique(opt_name),
                                # thus try to extract rather than generate
                                for state_key in sorted(
                                    state.keys(),
                                    key=lambda x: len(x),
                                    reverse=True,
                                ):
                                    prefix = (
                                        param_name
                                        + "_"
                                        + (
                                            opt_cls_name
                                            if opt_name is None
                                            else opt_name
                                        )
                                        + "_"
                                    )
                                    if state_key.startswith(prefix):
                                        prefix_offset = state_key[
                                            len(prefix) :
                                        ].find("_") + len(prefix)
                                        opt_unq_name = state_key[
                                            len(
                                                param_name + "_"
                                            ) : prefix_offset
                                        ]
                                        # TODO: assert
                                        # assert opt_unq_name is None
                                    # gen(param.name + "_" + gen(opt_name) + "_" + accum_name)
                                    # always end with "_0" since the unique optimizer._name
                            dy_state_name = (
                                param_name
                                + "_"
                                + opt_unq_name
                                + "_"
                                + accum_name
                                + "_0"
                            )
                            converted_state[state_var.name] = (
                                converted_state.pop(dy_state_name)
                            )

            assert (
                var.name in converted_state
            ), f"variable [{var.name}] is not in optimizer state file"
            self._set_var(var, converted_state[var.name])

    def _set_var(self, var, ndarray):
        t = global_scope().find_var(var.name).get_tensor()
        p = t._place()
        if p.is_cpu_place():
            place = base.CPUPlace()
        elif p.is_cuda_pinned_place():
            place = base.CUDAPinnedPlace()
        else:
            p = base.core.Place()
            p.set_place(t._place())
            place = base.CUDAPlace(p.gpu_device_id())

        t.set(ndarray, place)

    def _run(self, inputs, labels=None):
        compiled_prog = self._compiled_progs.get(self.mode, None)
        assert (
            compiled_prog
        ), "Model is not ready, please call `model.prepare()` first"

        inputs = to_list(inputs)
        if labels is not None:
            labels = to_list(labels)
        assert len(inputs) == len(self._input_vars[self.mode]), (
            "number of inputs"
            + " does not match number of arguments of `forward` method"
        )

        feed = {}
        input_names = [v.name for v in self._input_vars[self.mode]]
        input_dtypes = [v.dtype for v in self._input_vars[self.mode]]

        for idx, n in enumerate(input_names):
            # train and test may take different arguments
            if inputs[idx] is not None:
                feed[n] = inputs[idx]
            if self._amp_level == 'O2' and input_dtypes[idx] == paddle.float16:
                if isinstance(feed[n], core.DenseTensor):
                    feed[n] = feed[n]._as_type(core.VarDesc.VarType.FP16)
                elif isinstance(feed[n], np.array):
                    feed[n] = feed[n].astype('float16')

        if labels is not None:
            for idx, v in enumerate(self._label_vars[self.mode]):
                feed[v.name] = labels[idx]

        endpoints = self._endpoints[self.mode]
        if self.mode == 'test':
            fetch_list = endpoints['output']
        else:
            metric_list, metric_splits = flatten_list(endpoints['metric'])
            fetch_list = endpoints['loss'] + metric_list
            num_loss = len(endpoints['loss'])

        # if fetch Variable is same as input Variable, do not fetch
        # from program, get it from input directly
        pruned_fetch_list = []
        pruned_fetch_idx_name_map = [""] * len(fetch_list)
        for i, fetch_var in enumerate(fetch_list):
            if fetch_var.name in feed.keys():
                pruned_fetch_idx_name_map[i] = fetch_var.name
            else:
                pruned_fetch_list.append(fetch_var)

        rets = self._executor.run(
            compiled_prog,
            feed=feed,
            fetch_list=pruned_fetch_list,
            return_numpy=False,
        )

        # restore pruned fetch_list Variable from feeds
        for i, name in enumerate(pruned_fetch_idx_name_map):
            if len(name) > 0:
                rets.insert(i, feed[name])

        # DenseTensor cannot be fetch as numpy directly
        rets = [np.array(v) for v in rets]
        if self.mode == 'test':
            return rets[:]

        metric_states = restore_flatten_list(rets[num_loss:], metric_splits)
        metrics = []
        for metric, state in zip(self.model._metrics, metric_states):
            # cut off padding size
            if (
                self.mode != 'train'
                and self.model._test_dataloader is not None
                and isinstance(self.model._test_dataloader, DataLoader)
                and self._nranks > 1
            ):
                total_size = len(self.model._test_dataloader.dataset)
                # TODO: fixme if have better way to get batch size
                samples = state[0].shape[0]
                current_count = self._merge_count.get(self.mode + '_total', 0)
                if current_count + samples >= total_size:
                    state = [
                        s[: int(total_size - current_count), ...] for s in state
                    ]
                    self._merge_count[self.mode + '_total'] = 0
                    self._merge_count[self.mode + '_batch'] = int(
                        total_size - current_count
                    )
                else:
                    self._merge_count[self.mode + '_total'] += samples
                    self._merge_count[self.mode + '_batch'] = samples

            metrics.append(metric.update(*state))

        if num_loss and len(metrics):
            return rets[:num_loss], metrics
        else:
            return rets[:num_loss] if num_loss else metrics

    def prepare(self):
        modes = ['train', 'eval', 'test']
        for mode in modes:
            self._make_program(mode)
            self._compile_and_initialize(self._progs[mode], mode)

    def _make_program(self, mode):
        prog = self._progs.get(mode, None)
        if prog is not None:
            return

        prog = self._orig_prog.clone()
        # NOTE: When defining learning rate scheduling in static-graph, ops to
        # increase the global step var and calculate learning rate would be
        # prepended into _orig_prog. test program marked by `_orig_prog.clone`
        # also would include these ops. Thus must prune these ops in test
        # program, otherwise the global step would be changed in test.
        if mode != 'train':
            for op in list(prog.global_block().ops):
                prog.global_block()._remove_op(0)
        if (
            mode == 'train'
            and self.model._optimizer
            and self.model._optimizer._learning_rate_map
        ):
            # HACK workaround learning rate map issue
            lr_var = self.model._optimizer._learning_rate_map[self._orig_prog]
            new_lr_var = prog.global_block().vars[lr_var.name]
            self.model._optimizer._learning_rate_map[prog] = new_lr_var

        losses = []
        metrics = []
        with base.program_guard(prog, self._startup_prog):
            inputs = self.model._inputs
            labels = self.model._labels if self.model._labels else []
            inputs = [k._create_feed_layer() for k in to_list(inputs)]
            labels = [k._create_feed_layer() for k in to_list(labels)]
            self._label_vars[mode] = labels
            outputs = to_list(self.model.network.forward(*inputs))

            if mode != 'test' and self.model._loss:
                losses = self.model._loss(*(outputs + labels))

            if self._nranks > 1 and mode != 'train':
                outputs = [_all_gather(o) for o in outputs]
                if mode != 'test':
                    labels = [_all_gather(l) for l in labels]

            if mode != 'test':
                for metric in self.model._metrics:
                    metrics.append(to_list(metric.compute(*(outputs + labels))))

            if mode == 'train' and self.model._optimizer:
                self._loss_endpoint = paddle.add_n(losses)
                if self._nranks > 1:
                    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                    fleet.init(role)
                    dist_strategy = fleet.DistributedStrategy()
                    if self._amp_level != 'O0':
                        dist_strategy.amp = True
                        dist_strategy.amp_configs = self._amp_configs.copy()
                        dist_strategy.amp_configs.update(self._amp_custom_lists)
                        dist_strategy.amp_configs['use_pure_fp16'] = (
                            self._amp_level == 'O2'
                        )
                    self.model._optimizer = fleet.distributed_optimizer(
                        self.model._optimizer, strategy=dist_strategy
                    )
                elif self._amp_level != "O0" and core.is_compiled_with_cuda:
                    amp_lists = (
                        paddle.static.amp.AutoMixedPrecisionLists(
                            **self._amp_custom_lists
                        )
                        if self._amp_custom_lists
                        else None
                    )
                    self.model._optimizer = paddle.static.amp.decorate(
                        self.model._optimizer,
                        amp_lists=amp_lists,
                        use_pure_fp16=self._amp_level == "O2",
                        use_fp16_guard=self._use_fp16_guard,
                        **self._amp_configs,
                    )

                self.model._optimizer.minimize(self._loss_endpoint)

        if mode != 'train':  # clone again to put it in test mode
            prog = prog.clone(for_test=True)

        self._input_vars[mode] = inputs

        self._progs[mode] = prog
        self._endpoints[mode] = {
            "output": outputs,
            "loss": to_list(losses),
            "metric": metrics,
        }

    def _compile_and_initialize(self, prog, mode):
        compiled_prog = self._compiled_progs.get(mode, None)
        if compiled_prog is not None:
            return compiled_prog

        assert (
            self.model._place is not None
        ), "device is not set, please call `model.prepare()` first"

        place = self.model._place

        # XXX *ALL WEIGHTS* should be initialized upon model construction
        # even if `forward()` may run different code path for different mode
        # therefore startup program only needs to run once
        if self._executor is None:
            self._executor = base.Executor(place)
            # XXX incremental initialization
            uninitialized = []
            for var_py in self._startup_prog.list_vars():
                var = base.global_scope().find_var(var_py.name)
                if (
                    not var_py.name.startswith('nccl_id')
                    and var
                    and var.get_tensor()._is_initialized()
                ):
                    continue

                uninitialized.append(var_py)

            # for RawProgramOptimizer, it will insert OP with no outputs like:
            #       c_comm_init(inputs={X=['comm_id_0']}
            # but we cannot prune this op.
            block = self._startup_prog.global_block()
            for op in block.ops:
                if op.type == "c_comm_init":
                    uninitialized.append(op)

            if uninitialized:
                startup_prog = self._startup_prog._prune(uninitialized)
                self._executor.run(startup_prog)

        if (
            self._amp_level == "O2"
            and mode == 'train'
            and core.is_compiled_with_cuda()
        ):
            self.model._optimizer.amp_init(place)

        if self._nranks < 2:
            compiled_prog = base.CompiledProgram(prog)
        else:
            compiled_prog = prog

        self._compiled_progs[mode] = compiled_prog


class DynamicGraphAdapter:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._nranks = paddle.distributed.ParallelEnv().nranks
        self._local_rank = paddle.distributed.ParallelEnv().local_rank
        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0,
        }

        self._input_info = None
        self._amp_level = "O0"
        self._amp_configs = {}
        self._amp_custom_lists = {}
        self._use_fp16_guard = True

        if self._nranks > 1:
            dist.init_parallel_env()
            strategy = paddle.distributed.parallel.ParallelStrategy()
            strategy.nranks = paddle.distributed.ParallelEnv().nranks
            strategy.local_rank = paddle.distributed.ParallelEnv().local_rank
            strategy.trainer_endpoints = (
                paddle.distributed.ParallelEnv().trainer_endpoints
            )
            strategy.current_endpoint = (
                paddle.distributed.ParallelEnv().current_endpoint
            )
            self.ddp_model = paddle.DataParallel(self.model.network, strategy)

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    # TODO multi device in dygraph mode not implemented at present time
    def train_batch(self, inputs, labels=None, update=True):
        assert (
            self.model._optimizer
        ), "model not ready, please call `model.prepare()` first"
        self.model.network.train()
        self.mode = 'train'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        labels = [paddle.to_tensor(l) for l in to_list(labels)]

        # scaler should be initialized only once
        if self._amp_level != "O0" and self.model._scaler is None:
            self.model._scaler = paddle.amp.GradScaler(**self._amp_configs)

        with paddle.amp.auto_cast(
            enable=self._amp_level != 'O0',
            **self._amp_custom_lists,
            level=self._amp_level,
        ):
            if self._nranks > 1:
                outputs = self.ddp_model(*[paddle.to_tensor(x) for x in inputs])
            else:
                outputs = self.model.network(
                    *[paddle.to_tensor(x) for x in inputs]
                )

        losses = self.model._loss(*(to_list(outputs) + labels))
        losses = to_list(losses)
        final_loss = paddle.add_n(losses)

        if self._amp_level != "O0":
            scaled = self.model._scaler.scale(final_loss)
            scaled.backward()
            if update:
                self.model._scaler.minimize(self.model._optimizer, scaled)
                self.model.network.clear_gradients()
        else:
            final_loss.backward()
            if update:
                self.model._optimizer.minimize(final_loss)
                self.model.network.clear_gradients()

        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.compute(*(to_list(outputs) + labels))
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return (
            ([to_numpy(l) for l in losses], metrics)
            if len(metrics) > 0
            else [to_numpy(l) for l in losses]
        )

    def eval_batch(self, inputs, labels=None):
        self.model.network.eval()
        self.mode = 'eval'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        labels = [paddle.to_tensor(l) for l in to_list(labels)]

        outputs = self.model.network(*[paddle.to_tensor(x) for x in inputs])

        # Transform data to expected device
        expected_device = paddle.device.get_device()
        for o in to_list(outputs):
            o._to(device=expected_device)

        for l in labels:
            l._to(device=expected_device)

        if self.model._loss:
            losses = self.model._loss(*(to_list(outputs) + labels))
            losses = to_list(losses)

        if self._nranks > 1:
            outputs = [_all_gather(o) for o in to_list(outputs)]
            labels = [_all_gather(l) for l in labels]

            if self.model._test_dataloader is not None and isinstance(
                self.model._test_dataloader, DataLoader
            ):
                total_size = len(self.model._test_dataloader.dataset)
                samples = outputs[0].shape[0]
                current_count = self._merge_count.get(self.mode + '_total', 0)

                if current_count + samples >= total_size:
                    outputs = [
                        o[: int(total_size - current_count)] for o in outputs
                    ]
                    labels = [
                        l[: int(total_size - current_count)] for l in labels
                    ]
                    self._merge_count[self.mode + '_total'] = 0
                    self._merge_count[self.mode + '_batch'] = int(
                        total_size - current_count
                    )
                else:
                    self._merge_count[self.mode + '_total'] += samples
                    self._merge_count[self.mode + '_batch'] = samples

        metrics = []
        for metric in self.model._metrics:
            # cut off padding value.
            metric_outs = metric.compute(*(to_list(outputs) + labels))
            m = metric.update(*[to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        if self.model._loss and len(metrics):
            return [to_numpy(l) for l in losses], metrics
        elif self.model._loss:
            return [to_numpy(l) for l in losses]
        else:
            return metrics

    def predict_batch(self, inputs):
        self.model.network.eval()
        self.mode = 'test'
        inputs = [paddle.to_tensor(x) for x in to_list(inputs)]
        self._input_info = _update_input_info(inputs)
        outputs = self.model.network(*inputs)
        if self._nranks > 1 and isinstance(self.model._place, base.CUDAPlace):
            outputs = [_all_gather(o) for o in to_list(outputs)]

        return [to_numpy(o) for o in to_list(outputs)]

    def parameters(self, *args, **kwargs):
        return self.model.network.parameters(*args, **kwargs)

    def save(self, path):
        params = self.model.network.state_dict()
        paddle.save(params, path + '.pdparams')
        if self.model._optimizer is not None:
            if self.model._optimizer.state_dict():
                optim = self.model._optimizer.state_dict()
                paddle.save(optim, path + '.pdopt')
        if hasattr(self.model, '_scaler') and self.model._scaler is not None:
            if self.model._scaler.state_dict():
                scaler = self.model._scaler.state_dict()
                paddle.save(scaler, path + '.pdscaler')

    def load(self, param_state_pairs, optim_state, scaler_state=None):
        # restore parameter states
        for param, state in param_state_pairs:
            param.set_value(state)

        if hasattr(self.model, '_scaler') and self.model._scaler is not None:
            if scaler_state:
                self.model._scaler.load_state_dict(scaler_state)

        # restore optimizer states
        if not self.model._optimizer or not optim_state:
            return

        # If optimizer performs set_state_dict when state vars haven't been created,
        # which would happen when set_state_dict before minimize, the state would be
        # stored in optimizer._accumulators_holder and loaded lazily.
        # To contrive this when loading from static-graph saved states, extend
        # state dict to include keys named according to dygraph naming rules.
        # TODO: if len(self.model._optimizer._accumulators) > 0
        converted_state = dict(optim_state)
        opt_unq_name = self.model._optimizer._name
        if opt_unq_name is None:
            opt_unq_name = ''

        opt_cls_name = self.model._optimizer.__class__.__name__
        opt_name = opt_unq_name[: opt_unq_name.rfind("_")]  # remove suffix idx
        param_names = [param.name for param in self.model.network.parameters()]
        for var_name, state_var in sorted(
            optim_state.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if var_name in ["@LR_DECAY_COUNTER@", "global_step"]:
                # NOTE: dygraph saved global_step is 1 larger than that in
                # static-graph, since the time of global_step to increase is
                # different.
                if var_name == "@LR_DECAY_COUNTER@":
                    converted_state["global_step"] = (
                        np.array(converted_state.pop("@LR_DECAY_COUNTER@")) + 1
                    )
            else:
                # moment and other accumulators
                # extend state dict to include promising dygraph names
                for param_name in param_names:
                    if var_name.startswith(param_name + "_" + opt_name):
                        # when init optimizer with name
                        accum_name = var_name[
                            len(param_name + "_" + opt_name + "_") :
                        ]
                    elif (
                        var_name.startswith(param_name + "_")
                        and opt_name == opt_cls_name
                    ):
                        # when init optimizer without name
                        accum_name = var_name[len(param_name + "_") :]
                    else:
                        continue
                    # remove suffix idx
                    accum_name = accum_name[: accum_name.rfind("_")]
                    # state names always end with "_0" in dygraph because of the
                    # unique optimizer._name
                    dy_state_name = (
                        param_name
                        + "_"
                        + opt_unq_name
                        + "_"
                        + accum_name
                        + "_0"
                    )
                    converted_state[dy_state_name] = state_var

        if not hasattr(self.model._optimizer, 'set_state_dict'):
            warnings.warn(
                "paddle.base.optimizer is deprecated in API 2.0, please use paddle.optimizer instead."
            )
            self.model._optimizer.set_dict(converted_state)
        else:
            self.model._optimizer.set_state_dict(converted_state)

    def prepare(self):
        if (
            self._amp_level == "O2"
            and self.model.mode == 'train'
            and core.is_compiled_with_cuda()
        ):
            self.model.network, self.model._optimizer = paddle.amp.decorate(
                models=self.model.network,
                optimizers=self.model._optimizer,
                level='O2',
            )
        if self._amp_level != "O0":
            self.model._scaler = None


class Model:
    """

    An Model object is network with training and inference features.
    Dynamic graph and static graph are supported at the same time,
    switched by `paddle.enable_static()`. The usage is as follows.
    But note, the switching between dynamic and static should be before
    instantiating a Model. The input description, i.e, paddle.static.InputSpec,
    must be required for static graph.

    When training on GPU, auto mixed precision (AMP O1) and pure float16
    (AMP O2) training are both supported in static graph mode and dynamic mode.
    In static graph mode, before training with pure float16 (AMP O2),
    `multi_precision` could be set to True when creating optimizer, which can
    avoid poor accuracy or slow convergence in a way, and inputs of dtype float
    should be cast to float16 by users. `paddle.static.amp.fp16_guard` API
    should be also used to limit the range of pure float16 training, otherwise,
    'use_fp16_guard' should be set to False by users. However, limiting the
    range of is not supported during training using AMP.

    Args:
        network (paddle.nn.Layer): The network is an instance of
            paddle.nn.Layer.
        inputs (InputSpec|list|tuple|dict|None, optional): `inputs`, entry points of network,
            could be a InputSpec instance, or list/tuple of InputSpec instances,
            or dict ({name: InputSpec}), and it couldn't be None in static
            graph. Default: None.
        labels (InputSpec|list|tuple|None, optional): `labels`, entry points of network,
            could be a InputSpec instance or list/tuple of InputSpec instances,
            or None. For static graph, if labels is required in loss,
            labels must be set. Otherwise, it could be None. Default: None.


    Examples:
        1. A common example

        .. code-block:: python
            :name: code-example1

            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.vision.transforms as T
            >>> from paddle.static import InputSpec

            >>> device = paddle.set_device('cpu') # or 'gpu'

            >>> net = nn.Sequential(
            ...     nn.Flatten(1),
            ...     nn.Linear(784, 200),
            ...     nn.Tanh(),
            ...     nn.Linear(200, 10))
            ...
            >>> # inputs and labels are not required for dynamic graph.
            >>> input = InputSpec([None, 784], 'float32', 'x')
            >>> label = InputSpec([None, 1], 'int64', 'label')

            >>> model = paddle.Model(net, input, label)
            >>> optim = paddle.optimizer.SGD(learning_rate=1e-3,
            ...     parameters=model.parameters())
            ...
            >>> model.prepare(optim,
            ...             paddle.nn.CrossEntropyLoss(),
            ...             paddle.metric.Accuracy())
            ...
            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            >>> ])
            >>> data = paddle.vision.datasets.MNIST(mode='train', transform=transform)
            >>> model.fit(data, epochs=2, batch_size=32, verbose=1)


        2. An example using mixed precision training.

        .. code-block:: python
            :name: code-example2

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> import paddle.nn as nn
            >>> import paddle.vision.transforms as T

            >>> def run_example_code():
            ...     device = paddle.set_device('gpu')
            ...
            ...     net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(),
            ...                         nn.Linear(200, 10))
            ...
            ...     model = paddle.Model(net)
            ...     optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
            ...
            ...     amp_configs = {
            ...         "level": "O1",
            ...         "custom_white_list": {'conv2d'},
            ...         "use_dynamic_loss_scaling": True
            ...     }
            ...     model.prepare(optim,
            ...         paddle.nn.CrossEntropyLoss(),
            ...         paddle.metric.Accuracy(),
            ...         amp_configs=amp_configs)
            ...
            ...     transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
            ...     data = paddle.vision.datasets.MNIST(mode='train', transform=transform)
            ...     model.fit(data, epochs=2, batch_size=32, verbose=1)
            ...
            >>> # mixed precision training is only supported on GPU now.
            >>> if paddle.is_compiled_with_cuda():
            ...     run_example_code()
            ...
    """

    mode: Literal["train", "eval", "test"]
    network: paddle.nn.Layer
    stop_training: bool

    def __init__(
        self,
        network: paddle.nn.Layer,
        inputs: Input | Sequence[Input] | dict[str, Input] | None = None,
        labels: Input | Sequence[Input] | None = None,
    ) -> None:
        self.mode = 'train'
        self.network = network
        self._inputs = None
        self._labels = None
        self._loss = None
        self._loss_weights = None
        self._optimizer = None
        self._input_info = None
        self._is_shape_inferred = False
        self._test_dataloader = None
        self.stop_training = False

        if not in_dynamic_mode():
            if not isinstance(inputs, (list, tuple, dict, Input)):
                raise TypeError(
                    "'inputs' must be list or tuple or dict, and couldn't be None."
                )
        elif inputs:
            self._input_info = _update_input_info(inputs)

        self._inputs = self._verify_spec(inputs, is_input=True)
        self._labels = self._verify_spec(labels)

        # init backend
        if in_dynamic_mode():
            self._adapter = DynamicGraphAdapter(self)
        elif in_pir_mode():
            self._adapter = StaticPIRGraphAdapter(self)
        else:
            self._adapter = StaticGraphAdapter(self)

    def train_batch(
        self,
        inputs: _InputBatch,
        labels: _InputBatch | None = None,
        update: bool = True,
    ) -> list[float] | tuple[list[npt.NDArray[Any]], list[float]]:
        """

        Run one training step on one batch of data. And using `update` indicates
        whether optimizer update gradients computing by this batch.

        Args:
            inputs (numpy.ndarray|Tensor|list): Batch of input data. It could
                be a numpy array or paddle.Tensor, or a list of arrays or
                tensors (in case the model has multiple inputs).
            labels (numpy.ndarray|Tensor|list, optional): Batch of labels. It could be
                a numpy array or paddle.Tensor, or a list of arrays or tensors
                (in case the model has multiple labels). If has no labels,
                set None. Default: None.
            update (bool, optional): Whether update parameters after loss.backward() computing.
                Set it to False to accumulate gradients. Default: True.

        Returns:
            A list of scalar training loss if the model has no metrics,
            or a tuple (list of scalar loss, list of metrics) if the model
            set metrics.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> from paddle.static import InputSpec
                >>> paddle.seed(2023)

                >>> device = paddle.set_device('cpu') # or 'gpu'

                >>> net = nn.Sequential(
                ...     nn.Linear(784, 200),
                ...     nn.Tanh(),
                ...     nn.Linear(200, 10))
                ...
                >>> input = InputSpec([None, 784], 'float32', 'x')
                >>> label = InputSpec([None, 1], 'int64', 'label')
                >>> model = paddle.Model(net, input, label)
                >>> optim = paddle.optimizer.SGD(learning_rate=1e-3,
                ...     parameters=model.parameters())
                >>> model.prepare(optim, paddle.nn.CrossEntropyLoss())
                >>> data = paddle.rand((4, 784), dtype="float32")
                >>> label = paddle.randint(0, 10, (4, 1), dtype="int64")
                >>> loss = model.train_batch([data], [label])
                >>> print(loss)
                [array(3.0039132, dtype=float32)]

        """
        loss = self._adapter.train_batch(inputs, labels, update)
        if in_dynamic_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    @no_grad()
    def eval_batch(
        self, inputs: _InputBatch, labels: _InputBatch | None = None
    ) -> list[float] | tuple[list[npt.NDArray[Any]], list[float]]:
        """

        Run one evaluating step on a batch of data.

        Args:
            inputs (numpy.ndarray|Tensor|list): Batch of input data. It could
                be a numpy array or paddle.Tensor, or a list of arrays or
                tensors (in case the model has multiple inputs).
            labels (numpy.ndarray|Tensor|list, optional): Batch of labels. It could be
                a numpy array or paddle.Tensor, or a list of arrays or tensors
                (in case the model has multiple labels). If has no labels,
                set None. Default: None.

        Returns:
            A list of scalar testing loss if the model has no metrics,
            or a tuple (list of scalar loss, list of metrics) if the model
            set metrics.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> from paddle.static import InputSpec
                >>> paddle.seed(2023)

                >>> device = paddle.set_device('cpu') # or 'gpu'

                >>> net = nn.Sequential(
                ...     nn.Linear(784, 200),
                ...     nn.Tanh(),
                ...     nn.Linear(200, 10))
                ...
                >>> input = InputSpec([None, 784], 'float32', 'x')
                >>> label = InputSpec([None, 1], 'int64', 'label')
                >>> model = paddle.Model(net, input, label)
                >>> optim = paddle.optimizer.SGD(learning_rate=1e-3,
                ...     parameters=model.parameters())
                >>> model.prepare(optim,
                ...               paddle.nn.CrossEntropyLoss(),
                ...               metrics=paddle.metric.Accuracy())
                >>> data = paddle.rand((4, 784), dtype="float32")
                >>> label = paddle.randint(0, 10, (4, 1), dtype="int64")
                >>> loss, acc = model.eval_batch([data], [label])
                >>> print(loss, acc)
                [array(3.0039132, dtype=float32)] [0.0]

        """
        loss = self._adapter.eval_batch(inputs, labels)
        if in_dynamic_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    @no_grad()
    def predict_batch(self, inputs: _InputBatch) -> list[npt.NDArray[Any]]:
        """

        Run one predicting step on a batch of data.

        Args:
            inputs (numpy.ndarray|Tensor|list): Batch of input data. It could
                be a numpy array or paddle.Tensor, or a list of arrays or
                tensors (in case the model has multiple inputs).

        Returns:
            A list of numpy.ndarray of predictions, that is the outputs
            of Model forward.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> from paddle.static import InputSpec
                >>> paddle.seed(2023)

                >>> device = paddle.set_device('cpu') # or 'gpu'

                >>> input = InputSpec([None, 784], 'float32', 'x')
                >>> label = InputSpec([None, 1], 'int64', 'label')

                >>> net = nn.Sequential(
                ...     nn.Linear(784, 200),
                ...     nn.Tanh(),
                ...     nn.Linear(200, 10),
                ...     nn.Softmax())
                ...
                >>> model = paddle.Model(net, input, label)
                >>> model.prepare()
                >>> data = paddle.rand((1, 784), dtype="float32")
                >>> out = model.predict_batch([data])
                >>> print(out)
                [array([[0.10844935, 0.04650883, 0.11790176, 0.04962315, 0.10899059,
                         0.08197589, 0.03125402, 0.03232312, 0.3786293 , 0.04434395]],
                      dtype=float32)]

        """
        loss = self._adapter.predict_batch(inputs)
        if in_dynamic_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    def save(self, path: str, training: bool = True) -> None:
        """

        This function saves parameters, optimizer information or model and
        parameters only for inference to path. It depends on the parameter
        `training`.

        If `training` is set to True, the parameters saved contain all
        the trainable Variable, will save to a file with suffix ".pdparams".
        The optimizer information contains all the variable used by optimizer.
        For Adam optimizer, contains beta1, beta2, momentum etc. All the
        information will save to a file with suffix ".pdopt". (If the optimizer
        have no variable need to save (like SGD), the fill will not generated).
        This function will silently overwrite existing file at the target location.

        If `training` is set to False, only inference model will be saved.

        Args:
            path (str): The file prefix to save model. The format
                is 'dirname/file_prefix' or 'file_prefix'. if empty str.
                A exception will be raised.
            training (bool, optional): Whether to save for training. If not, save
                for inference only. Default: True.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> # doctest: +TIMEOUT(80)
                >>> import paddle
                >>> import paddle.nn as nn
                >>> import paddle.vision.transforms as T
                >>> from paddle.static import InputSpec
                >>> from paddle.vision.datasets import MNIST

                >>> dynamic = True  # False
                >>> # If use static graph, do not set
                >>> if not dynamic:
                ...     paddle.enable_static()

                >>> transform = T.Compose([T.Transpose(),
                ...                        T.Normalize([127.5], [127.5])])
                >>> train_dataset = MNIST(mode='train', transform=transform)
                >>> train_loader = paddle.io.DataLoader(train_dataset, batch_size=64)
                >>> val_dataset = MNIST(mode='test', transform=transform)
                >>> val_loader = paddle.io.DataLoader(val_dataset, batch_size=64)

                >>> input = InputSpec([None, 1, 28, 28], 'float32', 'image')
                >>> label = InputSpec([None, 1], 'int64', 'label')

                >>> model = paddle.Model(paddle.vision.models.LeNet(), input, label)
                >>> optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
                >>> model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy(topk=(1, 2)))
                >>> model.fit(train_loader, val_loader, epochs=2, verbose=0)
                >>> model.save('checkpoint/test')  # save for training
                >>> model.save('inference_model', False)  # save for inference

        """

        if paddle.distributed.ParallelEnv().local_rank == 0:
            if not training:
                self._save_inference_model(path)
            else:
                self._adapter.save(path)

    def load(
        self,
        path: str,
        skip_mismatch: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """

        Load from files storing the model states and optimizer states. The file
        for optimizer states is not necessary if no need to restore the optimizer.

        NOTE: parameters are retrieved out from the file storing model states
        according to their structured names.

        For fine-tuning or transfer-learning models where some of the layers have
        changed, keep parameters needed to restore have same structured names in
        the pre-trained model and fine-tuning model.

        Args:
            path (str): The prefix of files storing the model states and
                optimizer states. The files would be `path.pdparams` and
                `path.pdopt` separately, and the latter is not necessary
                when no need to restore.
            skip_mismatch (bool, optional): Whether to skip the loading of mismatch
                parameter or raise an error when mismatch happens (not found
                the parameter in file storing model states of or receives a
                mismatch shape). Default: False.
            reset_optimizer (bool, optional): If True, ignore the providing file storing
                optimizer states and initialize optimizer states from scratch.
                Otherwise, restore optimizer states from `path.pdopt` if
                a optimizer has been set to the model. Default: False.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> from paddle.static import InputSpec

                >>> device = paddle.set_device('cpu')

                >>> input = InputSpec([None, 784], 'float32', 'x')

                >>> model = paddle.Model(nn.Sequential(
                ...     nn.Linear(784, 200),
                ...     nn.Tanh(),
                ...     nn.Linear(200, 10),
                ...     nn.Softmax()), input)
                ...
                >>> model.save('checkpoint/test')
                >>> model.load('checkpoint/test')

        """

        def _load_state_from_path(path):
            if not os.path.exists(path):
                return
            with open(path, 'rb') as f:
                return pickle.load(f, encoding='latin1')

        def _check_match(key, param):
            state = param_state.get(key, None)
            if state is None:
                raise ValueError(f"{key} is not found in the providing file.")
            if list(state.shape) != list(param.shape):
                raise ValueError(
                    f"{key} receives a shape {list(state.shape)}, but the expected shape is {list(param.shape)}."
                )
            return param, state

        def _strip_postfix(path):
            path, ext = os.path.splitext(path)
            assert ext in [
                '',
                '.pdparams',
                '.pdopt',
                '.pdmodel',
            ], f"Unknown postfix {ext} from weights"
            return path

        path = _strip_postfix(path)
        param_state = _load_state_from_path(path + ".pdparams")
        assert param_state, "Failed to load parameters, please check path."

        matched_param_state = []
        for key, param in self.network.state_dict().items():
            try:
                match_res = _check_match(key, param)
            except ValueError as err:
                if skip_mismatch:
                    warnings.warn(f"Skip loading for {key}. " + str(err))
                    # reset optimizer when mismatch happens
                    reset_optimizer = True
                else:
                    raise err
            matched_param_state.append(match_res)

        optim_state = (
            None if reset_optimizer else _load_state_from_path(path + ".pdopt")
        )

        # TODO: support save/load scaler state in static graph
        if in_dynamic_mode():
            scaler_state = None
            if hasattr(self, '_scaler') and self._scaler is not None:
                if os.path.exists(path + '.pdscaler'):
                    scaler_state = paddle.load(path + '.pdscaler')

            return self._adapter.load(
                matched_param_state, optim_state, scaler_state
            )
        else:
            return self._adapter.load(matched_param_state, optim_state)

    def parameters(self, *args: Any, **kwargs: Any) -> list[Tensor]:
        """

        Returns a list of parameters of the model.

        Returns:
            A list of Parameter in static graph.
            A list of ParamBase in dynamic graph.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn
                >>> from paddle.static import InputSpec
                >>> paddle.seed(2023)
                >>> input = InputSpec([None, 784], 'float32', 'x')

                >>> model = paddle.Model(nn.Sequential(
                ...     nn.Linear(784, 200),
                ...     nn.Tanh(),
                ...     nn.Linear(200, 10)), input)
                ...
                >>> params = model.parameters()
                >>> print(params)
                [Parameter containing:
                Tensor(shape=[784, 200], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[ 0.05713400,  0.00314646, -0.03754271, ..., -0.02529256,
                   0.04872842, -0.06670858],
                 ...,
                 [ 0.06268418,  0.06550254, -0.02103353, ...,  0.06395906,
                   0.05509177, -0.06355451]]), Parameter containing:
                Tensor(shape=[200], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0., 0., 0., ..., 0., 0.]), Parameter containing:
                Tensor(shape=[200, 10], dtype=float32, place=Place(cpu), stop_gradient=False,
                [[ 0.12933084,  0.07726504,  0.05336720, ...,  0.10865459,
                   0.06605886,  0.13684085],
                 ...,
                 [-0.10171061, -0.01649965, -0.13420501, ...,  0.11190581,
                  -0.12700224,  0.02916957]]), Parameter containing:
                Tensor(shape=[10], dtype=float32, place=Place(cpu), stop_gradient=False,
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]

        """
        return self._adapter.parameters()

    def _prepare_amp(self, amp_configs):
        def _check_pure_fp16_configs():
            # pure float16 training has some restricts now
            if self._adapter._amp_level == "O2" and self._optimizer._grad_clip:
                # clip by value is not supported
                assert isinstance(
                    self._optimizer._grad_clip,
                    (paddle.nn.ClipGradByGlobalNorm, paddle.nn.ClipGradByNorm),
                ), "Only ClipGradByNorm and ClipGradByGlobalNorm are supported in amp training with level=O2 currently."

        self._adapter._amp_custom_lists = {}
        self._adapter._amp_configs = {}

        # check and get level of mixed precision training
        if not amp_configs:
            self._adapter._amp_level = 'O0'
            return
        elif isinstance(amp_configs, str):
            if amp_configs not in ('O0', 'O1', 'O2'):
                raise ValueError(
                    "The level of amp_configs should be 'O0', 'O1' or 'O2'."
                )
            self._adapter._amp_level = amp_configs
            _check_pure_fp16_configs()
            return
        else:
            if 'level' not in amp_configs:
                self._adapter._amp_level = 'O1'
            elif amp_configs['level'] not in ('O0', 'O1', 'O2'):
                raise ValueError(
                    "amp_configs['level'] should be 'O0', 'O1' or 'O2'."
                )
            else:
                self._adapter._amp_level = amp_configs['level']
        amp_config_key_set = set(amp_configs.keys()) - {'level'}
        if not amp_config_key_set or self._adapter._amp_level == 'O0':
            return

        if 'use_pure_fp16' in amp_configs:
            raise ValueError(
                "'use_pure_fp16' is an invalid parameter, the level of mixed precision training only depends on 'O1' or 'O2'."
            )

        _check_pure_fp16_configs()

        # construct amp_custom_lists
        if self._adapter._amp_level != 'O0' and amp_config_key_set:
            for param_name in [
                'custom_white_list',
                'custom_black_list',
                'custom_black_varnames',
            ]:
                if param_name in amp_config_key_set:
                    self._adapter._amp_custom_lists[param_name] = amp_configs[
                        param_name
                    ]
                    amp_config_key_set -= {param_name}

        def _check_amp_configs(amp_config_key_set):
            accepted_param_set = {
                'init_loss_scaling',
                'incr_ratio',
                'decr_ratio',
                'incr_every_n_steps',
                'decr_every_n_nan_or_inf',
                'use_dynamic_loss_scaling',
                'use_fp16_guard',
            }
            if amp_config_key_set - accepted_param_set:
                raise ValueError(
                    f"Except for 'level', the keys of 'amp_configs' must be accepted by mixed precision APIs, but {tuple(amp_config_key_set - accepted_param_set)} could not be recognized."
                )

            if 'use_fp16_guard' in amp_config_key_set:
                if in_dynamic_mode():
                    raise ValueError(
                        "'use_fp16_guard' is supported in static graph mode only."
                    )
                self._adapter._use_fp16_guard = amp_configs['use_fp16_guard']
                amp_config_key_set.remove('use_fp16_guard')

            return amp_config_key_set

        amp_configs_set = _check_amp_configs(amp_config_key_set)
        for key in amp_configs_set:
            self._adapter._amp_configs[key] = amp_configs[key]

    def prepare(
        self,
        optimizer: paddle.optimizer.Optimizer | None = None,
        loss: (
            paddle.nn.Layer | Callable[[Tensor, Tensor], Tensor] | None
        ) = None,
        metrics: Metric | list[Metric] | None = None,
        amp_configs: str | dict[str, Any] | None = None,
    ) -> None:
        """

        Configures the model before running.

        Args:
            optimizer (Optimizer|None, optional): Optimizer must be set in training
                and should be a Optimizer instance. It can be None in eval
                and test mode. Default: None.
            loss (Loss|Callable|None, optional): Loss function can
                be a `paddle.nn.Layer` instance or any callable function
                taken the predicted values and ground truth values as input.
                It can be None when there is no loss. Default: None.
            metrics (Metric|list[Metric]|None, optional): If metrics is set, all
                metrics will be calculated and output in train/eval mode. Default: None.
            amp_configs (str|dict|None, optional): AMP configurations. If AMP or pure
                float16 training is used, the key 'level' of 'amp_configs'
                should be set to 'O1' or 'O2' respectively. Otherwise, the
                value of 'level' defaults to 'O0', which means float32
                training. In addition to 'level', parameters consistent with
                mixed precision API could also be passed in. The supported
                keys are: 'init_loss_scaling', 'incr_ratio', 'decr_ratio',
                'incr_every_n_steps', 'decr_every_n_nan_or_inf',
                'use_dynamic_loss_scaling', 'custom_white_list',
                'custom_black_list', and 'custom_black_varnames'or
                'use_fp16_guard' is only supported in static graph mode. Mixed
                precision API documentations  :ref:`api_paddle_amp_auto_cast`
                and  :ref:`api_paddle_amp_GradScaler` could be referenced
                for details. For convenience, 'amp_configs' could be set to
                'O1' or 'O2' if no more parameters are needed. 'amp_configs'
                could be None in float32 training. Default: None.

        Returns:
            None

        """
        self._place = _get_device()
        if isinstance(self._place, base.CUDAPlace):
            global _parallel_context_initialized
            if (
                paddle.distributed.ParallelEnv().nranks > 1
                and not _parallel_context_initialized
            ):
                if in_dynamic_mode():
                    main_prog_seed = base.default_main_program().random_seed
                    startup_prog_seed = (
                        base.default_startup_program().random_seed
                    )
                    base.disable_dygraph()
                    paddle.disable_static(self._place)
                    # enable_dygraph would create and switch to a new program,
                    # thus also copy seed to the new program
                    base.default_main_program().random_seed = main_prog_seed
                    base.default_startup_program().random_seed = (
                        startup_prog_seed
                    )
                else:
                    prepare_distributed_context(self._place)
                _parallel_context_initialized = True

        self._optimizer = optimizer
        if loss is not None:
            if not isinstance(loss, paddle.nn.Layer) and not callable(loss):
                raise TypeError(
                    "'loss' must be sub classes of `paddle.nn.Layer` or any callable function."
                )
        self._loss = loss

        metrics = metrics or []
        for metric in to_list(metrics):
            assert isinstance(
                metric, Metric
            ), f"{metric.__class__.__name__} is not sub class of Metric"
        self._metrics = to_list(metrics)
        self._prepare_amp(amp_configs)

        self._adapter.prepare()

    def fit(
        self,
        train_data: Dataset | DataLoader | None = None,
        eval_data: Dataset | DataLoader | None = None,
        batch_size: int | list[int] = 1,
        epochs: int = 1,
        eval_freq: int = 1,
        log_freq: int = 10,
        save_dir: str | None = None,
        save_freq: int = 1,
        verbose: int = 2,
        drop_last: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        callbacks: Sequence[Callback] | Callback | None = None,
        accumulate_grad_batches: int = 1,
        num_iters: int | None = None,
    ) -> None:
        """

        Trains the model for a fixed number of epochs. If `eval_data` is set,
        evaluation will be done at the end of each epoch.

        Args:
            train_data (Dataset|DataLoader, optional): An iterable data loader is used for
                train. An instance of paddle paddle.io.Dataset or
                paddle.io.Dataloader is recommended. Default: None.
            eval_data (Dataset|DataLoader, optional): An iterable data loader is used for
                evaluation at the end of epoch. If None, will not do evaluation.
                An instance of paddle.io.Dataset or paddle.io.Dataloader
                is recommended. Default: None.
            batch_size (int|list, optional): The batch size of train_data and eval_data. When
                train_data and eval_data are both the instance of Dataloader, this
                parameter will be ignored. Default: 1.
            epochs (int, optional): The number of epochs to train the model. Default: 1.
            eval_freq (int, optional): The frequency, in number of epochs, an evaluation
                is performed. Default: 1.
            log_freq (int, optional): The frequency, in number of steps, the training logs
                are printed. Default: 10.
            save_dir(str|None, optional): The directory to save checkpoint during training.
                If None, will not save checkpoint. Default: None.
            save_freq (int, optional): The frequency, in number of epochs, to save
                checkpoint. Default: 1.
            verbose (int, optional): The verbosity mode, should be 0, 1, or 2. 0 = silent,
                1 = progress bar, 2 = one line per epoch. Default: 2.
            drop_last (bool, optional): Whether drop the last incomplete batch of
                train_data when dataset size is not divisible by the batch size.
                When train_data is an instance of Dataloader, this parameter
                will be ignored. Default: False.
            shuffle (bool, optional): Whether to shuffle train_data. When train_data is
                an instance of Dataloader, this parameter will be ignored.
                Default: True.
            num_workers (int, optional): The number of subprocess to load data, 0 for no
                subprocess used and loading data in main process.
                When train_data and eval_data are both the instance of
                Dataloader, this parameter will be ignored. Default: 0.
            callbacks (Sequence[Callback]|Callback|None, optional): A list of `Callback` instances to apply
                during training. If None, :ref:`api_paddle_callbacks_ProgBarLogger` and
                :ref:`api_paddle_callbacks_ModelCheckpoint` are automatically inserted. Default: None.
            accumulate_grad_batches (int, optional): The number of batches to accumulate gradient
                during training process before optimizer updates. It can mimic large batch
                size. Default: 1.
            num_iters (int|None, optional): The number of iterations to evaluate the model.
                If None, evaluate on whole input dataset, otherwise, evaluate `num_iters` times.
                Default: None.

        Returns:
            None

        Examples:
            1. An example use Dataset and set batch size, shuffle in fit.
               How to make a batch is done internally.

            .. code-block:: python
                :name: code-example3

                >>> # doctest: +TIMEOUT(80)
                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.vision.datasets import MNIST
                >>> from paddle.static import InputSpec

                >>> dynamic = True
                >>> if not dynamic:
                ...     paddle.enable_static()
                ...
                >>> transform = T.Compose([T.Transpose(),
                ...                        T.Normalize([127.5], [127.5])])
                >>> train_dataset = MNIST(mode='train', transform=transform)
                >>> val_dataset = MNIST(mode='test', transform=transform)

                >>> input = InputSpec([None, 1, 28, 28], 'float32', 'image')
                >>> label = InputSpec([None, 1], 'int64', 'label')

                >>> model = paddle.Model(
                ...     paddle.vision.models.LeNet(),
                ...     input, label)
                >>> optim = paddle.optimizer.Adam(
                ...     learning_rate=0.001, parameters=model.parameters())
                >>> model.prepare(
                ...     optim,
                ...     paddle.nn.CrossEntropyLoss(),
                ...     paddle.metric.Accuracy(topk=(1, 2)))
                >>> model.fit(train_dataset,
                ...             val_dataset,
                ...             epochs=2,
                ...             batch_size=64,
                ...             save_dir='mnist_checkpoint')
                ...

            2. An example use DataLoader, batch size and shuffle is set in
               DataLoader.

            .. code-block:: python
                :name: code-example4

                >>> # doctest: +TIMEOUT(80)
                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.vision.datasets import MNIST
                >>> from paddle.static import InputSpec

                >>> dynamic = True
                >>> if not dynamic:
                ...     paddle.enable_static()
                ...
                >>> transform = T.Compose([T.Transpose(),
                ...                        T.Normalize([127.5], [127.5])])
                >>> train_dataset = MNIST(mode='train', transform=transform)
                >>> train_loader = paddle.io.DataLoader(train_dataset,
                ...     batch_size=64)
                >>> val_dataset = MNIST(mode='test', transform=transform)
                >>> val_loader = paddle.io.DataLoader(val_dataset,
                ...     batch_size=64)
                ...
                >>> input = InputSpec([None, 1, 28, 28], 'float32', 'image')
                >>> label = InputSpec([None, 1], 'int64', 'label')

                >>> model = paddle.Model(
                ...     paddle.vision.models.LeNet(), input, label)
                >>> optim = paddle.optimizer.Adam(
                ...     learning_rate=0.001, parameters=model.parameters())
                >>> model.prepare(
                ...     optim,
                ...     paddle.nn.CrossEntropyLoss(),
                ...     paddle.metric.Accuracy(topk=(1, 2)))
                >>> model.fit(train_loader,
                ...             val_loader,
                ...             epochs=2,
                ...             save_dir='mnist_checkpoint')
                ...
        """
        assert train_data is not None, "train_data must be given!"

        if isinstance(batch_size, (tuple, list)) and all(
            isinstance(x, int) for x in batch_size
        ):
            assert (
                len(batch_size) == 2
            ), "batch_size length error, expected train_batch_size and eval_batch_size."
            train_batch_size, eval_batch_size = batch_size
        elif isinstance(batch_size, int):
            train_batch_size, eval_batch_size = batch_size, batch_size

        if isinstance(train_data, Dataset):
            train_sampler = DistributedBatchSampler(
                train_data,
                batch_size=train_batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True,
            )
        else:
            train_loader = train_data

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=eval_batch_size
            )
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True,
            )
        elif eval_data is not None:
            eval_loader = eval_data
        else:
            eval_loader = None

        do_eval = eval_loader is not None
        self._test_dataloader = eval_loader

        self._accumulate = accumulate_grad_batches

        steps = self._len_data_loader(train_loader)
        self.num_iters = num_iters
        if (
            num_iters is not None
            and isinstance(num_iters, int)
            and isinstance(steps, int)
        ):
            assert num_iters > 0, "num_iters must be greater than 0!"
            epochs = (num_iters // steps) + 1
            steps = min(num_iters, steps)
        cbks = config_callbacks(
            callbacks,
            model=self,
            epochs=epochs,
            steps=steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(),
        )

        if any(isinstance(k, EarlyStopping) for k in cbks) and not do_eval:
            warnings.warn("EarlyStopping needs validation data.")

        cbks.on_begin('train')
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch)
            logs = self._run_one_epoch(train_loader, cbks, 'train')
            cbks.on_epoch_end(epoch, logs)

            if do_eval and epoch % eval_freq == 0:
                eval_steps = self._len_data_loader(eval_loader)
                cbks.on_begin(
                    'eval',
                    {'steps': eval_steps, 'metrics': self._metrics_name()},
                )

                eval_logs = self._run_one_epoch(eval_loader, cbks, 'eval')

                cbks.on_end('eval', eval_logs)
            if self.stop_training:
                break

        cbks.on_end('train', logs)
        self._test_dataloader = None

    def evaluate(
        self,
        eval_data: Dataset | DataLoader,
        batch_size: int = 1,
        log_freq: int = 10,
        verbose: int = 2,
        num_workers: int = 0,
        callbacks: Sequence[Callback] | Callback | None = None,
        num_iters: int | None = None,
    ) -> dict[str, float | npt.NDArray[Any]]:
        """
        Evaluate the loss and metrics of the model on input dataset.

        Args:
            eval_data (Dataset|DataLoader): An iterable data loader is used for
                evaluation. An instance of paddle.io.Dataset or
                paddle.io.Dataloader is recommended.
            batch_size (int, optional): The batch size of train_data and eval_data.
                When eval_data is the instance of Dataloader, this argument will be
                ignored. Default: 1.
            log_freq (int, optional): The frequency, in number of steps, the eval logs
                are printed. Default: 10.
            verbose (int, optional): The verbosity mode, should be 0, 1, or 2. 0 = silent,
                1 = progress bar, 2 = one line per epoch. Default: 2.
            num_workers (int, optional): The number of subprocess to load data,
                0 for no subprocess used and loading data in main process. When
                train_data and eval_data are both the instance of Dataloader,
                this parameter will be ignored. Default: 0.
            callbacks (Sequence[Callback]|Callback|None, optional): A list of `Callback` instances to apply
                during training. If None, `ProgBarLogger` and `ModelCheckpoint`
                are automatically inserted. Default: None.
            num_iters (int|None, optional): The number of iterations to evaluate the model.
                If None, evaluate on whole input dataset, otherwise, evaluate `num_iters` times.
                Default: None.
        Returns:
            dict: Result of metric. The key is the names of Metric,
                value is a scalar or numpy.array.

        Examples:

            .. code-block:: python

                >>> # doctest: +SKIP('Cause each step's acc and using time are not same when repeat running')
                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.static import InputSpec

                >>> # declarative mode
                >>> transform = T.Compose([T.Transpose(),
                ...                        T.Normalize([127.5], [127.5])])
                >>> val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

                >>> input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
                >>> label = InputSpec([None, 1], 'int64', 'label')
                >>> model = paddle.Model(paddle.vision.models.LeNet(), input, label)
                >>> model.prepare(metrics=paddle.metric.Accuracy())
                >>> result = model.evaluate(val_dataset, batch_size=64)
                >>> print(result)
                {'acc': 0.0699}
        """

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=batch_size
            )
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True,
            )
        else:
            eval_loader = eval_data

        self._test_dataloader = eval_loader

        cbks = config_callbacks(
            callbacks,
            model=self,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(),
        )

        eval_steps = self._len_data_loader(eval_loader)
        self.num_iters = num_iters
        if (
            num_iters is not None
            and isinstance(num_iters, int)
            and isinstance(eval_steps, int)
        ):
            assert num_iters > 0, "num_iters must be greater than 0!"
            eval_steps = min(num_iters, eval_steps)
            self.num_iters = eval_steps
        cbks.on_begin(
            'eval', {'steps': eval_steps, 'metrics': self._metrics_name()}
        )

        logs = self._run_one_epoch(eval_loader, cbks, 'eval')

        cbks.on_end('eval', logs)

        self._test_dataloader = None

        eval_result = {}
        for k in self._metrics_name():
            eval_result[k] = logs[k]

        return eval_result

    @overload
    def predict(
        self,
        test_data: Dataset | DataLoader,
        batch_size: int = ...,
        num_workers: int = ...,
        stack_outputs: Literal[True] = ...,
        verbose: int = ...,
        callbacks: Sequence[Callback] | Callback | None = ...,
    ) -> list[npt.NDArray[Any]]: ...

    @overload
    def predict(
        self,
        test_data: Dataset | DataLoader,
        batch_size: int = ...,
        num_workers: int = ...,
        stack_outputs: Literal[False] = ...,
        verbose: int = ...,
        callbacks: Sequence[Callback] | Callback | None = ...,
    ) -> list[tuple[npt.NDArray[Any], ...]]: ...

    @overload
    def predict(
        self,
        test_data: Dataset | DataLoader,
        batch_size: int = ...,
        num_workers: int = ...,
        stack_outputs: bool = ...,
        verbose: int = ...,
        callbacks: Sequence[Callback] | Callback | None = ...,
    ) -> list[npt.NDArray[Any] | tuple[npt.NDArray[Any], ...]]: ...

    def predict(
        self,
        test_data,
        batch_size=1,
        num_workers=0,
        stack_outputs=False,
        verbose=1,
        callbacks=None,
    ):
        """
        Compute the output predictions on testing data.

        Args:
            test_data (Dataset|DataLoader): An iterable data loader is used for
                predict. An instance of paddle.io.Dataset or paddle.io.Dataloader
                is recommended.
            batch_size (int, optional): The batch size of test_data. When test_data is the
                instance of Dataloader, this argument will be ignored. Default: 1.
            num_workers (int, optional): The number of subprocess to load data, 0 for no subprocess
                used and loading data in main process. When test_data is the instance of Dataloader,
                this argument will be ignored. Default: 0.
            stack_outputs (bool, optional): Whether stack output field like a batch, as for an output
                field of a sample is in shape [X, Y], test_data contains N samples, predict
                output field will be in shape [N, X, Y] if stack_output is True, and will
                be a length N list in shape [[X, Y], [X, Y], ..., [X, Y]] if stack_outputs
                is False. stack_outputs as False is used for DenseTensor output situation,
                it is recommended set as True if outputs contains no DenseTensor. Default: False.
            verbose (int, optional): The verbosity mode, should be 0, 1, or 2. 0 = silent,
                1 = progress bar, 2 = one line per batch. Default: 1.
            callbacks(Sequence[Callback]|Callback|None, optional): A Callback instance, Default: None.

        Returns:
            list: output of models.

        Examples:

            .. code-block:: python

                >>> import numpy as np
                >>> import paddle
                >>> from paddle.static import InputSpec

                >>> class MnistDataset(paddle.vision.datasets.MNIST):
                ...     def __init__(self, mode, return_label=True):
                ...         super().__init__(mode=mode)
                ...         self.return_label = return_label
                ...
                ...     def __getitem__(self, idx):
                ...         img = np.reshape(self.images[idx], [1, 28, 28])
                ...         if self.return_label:
                ...             return img, np.array(self.labels[idx]).astype('int64')
                ...         return img
                ...
                ...     def __len__(self):
                ...         return len(self.images)
                ...
                >>> test_dataset = MnistDataset(mode='test', return_label=False)

                >>> # imperative mode
                >>> input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
                >>> model = paddle.Model(paddle.vision.models.LeNet(), input)
                >>> model.prepare()
                >>> result = model.predict(test_dataset, batch_size=64)
                >>> print(len(result[0]), result[0][0].shape)
                157 (64, 10)
                >>> # declarative mode
                >>> device = paddle.set_device('cpu')
                >>> paddle.enable_static()
                >>> input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
                >>> model = paddle.Model(paddle.vision.models.LeNet(), input)
                >>> model.prepare()
                >>> result = model.predict(test_dataset, batch_size=64)
                >>> print(len(result[0]), result[0][0].shape)
                157 (64, 10)
        """

        if test_data is not None and isinstance(test_data, Dataset):
            test_sampler = DistributedBatchSampler(
                test_data, batch_size=batch_size
            )
            test_loader = DataLoader(
                test_data,
                batch_sampler=test_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True,
            )
        else:
            test_loader = test_data

        self._test_dataloader = test_loader

        cbks = config_callbacks(callbacks, model=self, verbose=verbose)

        test_steps = self._len_data_loader(test_loader)
        logs = {'steps': test_steps}

        cbks.on_begin('predict', logs)

        outputs = []

        logs, outputs = self._run_one_epoch(test_loader, cbks, 'predict')

        outputs = list(zip(*outputs))

        # NOTE: for lod tensor output, we should not stack outputs
        # for stacking may lose its detail info
        if stack_outputs:
            outputs = [np.vstack(outs) for outs in outputs]

        self._test_dataloader = None

        cbks.on_end('predict', logs)
        return outputs

    def _save_inference_model(self, path: str) -> None:
        """
        Save inference model can be used in static or dynamic mode.

        Args:
            path (str): The path prefix to save model. The format is
                ``dirname/file_prefix`` or ``file_prefix``.
        Returns:
            None
        """

        if in_dynamic_mode():
            with base.framework._dygraph_guard(None):
                layer = self.network
                if self._input_info is None:  # No provided or inferred
                    raise RuntimeError(
                        "Saving inference model needs 'inputs' or running before saving. Please specify 'inputs' in Model initialization or input training data and perform a training for shape derivation."
                    )
                if self._is_shape_inferred:
                    warnings.warn(
                        f"'inputs' was not specified when Model initialization, so the input shape to be saved will be the shape derived from the user's actual inputs. The input shape to be saved is {self._input_info[0]}. For saving correct input shapes, please provide 'inputs' for Model initialization."
                    )

                paddle.jit.save(layer, path, input_spec=self._inputs)

        else:
            # Path check
            file_prefix = os.path.basename(path)
            if file_prefix == "":
                raise ValueError(
                    "The input path MUST be format of dirname/file_prefix "
                    "[dirname\\file_prefix in Windows system], but received "
                    "file_prefix is empty string."
                )

            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            model_path = dirname
            model_filename = file_prefix + INFER_MODEL_SUFFIX
            params_filename = file_prefix + INFER_PARAMS_SUFFIX

            prog = self._adapter._progs.get('test', None)
            assert (
                prog
            ), "Model is not ready, please call `model.prepare()` first"

            if in_pir_mode():
                infer_prog = prog
            else:
                infer_prog = prog.clone(for_test=True)

            inputs = list(self._adapter._input_vars['test'])
            endpoints = self._adapter._endpoints['test']['output']

            paddle.static.save_inference_model(
                model_path,
                inputs,
                endpoints,
                self._adapter._executor,
                program=infer_prog,
            )

    def _run_one_epoch(
        self,
        data_loader,
        callbacks,
        mode,
        logs={},
    ):
        outputs = []
        for step, data in enumerate(data_loader):
            # Data might come from different types of data_loader and have
            # different format, as following:
            # 1. DataLoader in static graph:
            #    [[input1, input2, ..., label1, label2, ...]]
            # 2. DataLoader in dygraph
            #    [input1, input2, ..., label1, label2, ...]
            # 3. custumed iterator yield concated inputs and labels:
            #   [input1, input2, ..., label1, label2, ...]
            # 4. custumed iterator yield separated inputs and labels:
            #   ([input1, input2, ...], [label1, label2, ...])
            # To handle all of these, flatten (nested) list to list.
            data = paddle.utils.flatten(data)
            # DenseTensor.shape is callable, where DenseTensor comes from
            # DataLoader in static graph

            batch_size = (
                data[0].shape()[0]
                if callable(data[0].shape)
                else data[0].shape[0]
            )

            callbacks.on_batch_begin(mode, step, logs)

            if mode != 'predict':
                _inputs = [data[: len(self._inputs)], data[len(self._inputs) :]]
                if mode == 'train':
                    _inputs.append(
                        (step + 1) % self._accumulate == 0
                        or step + 1 == len(data_loader)
                    )

                outs = getattr(self, mode + '_batch')(*_inputs)

                if self._metrics and self._loss:
                    metrics = [[float(l) for l in outs[0]]]
                elif self._loss:
                    metrics = [[float(l) for l in outs]]
                else:
                    metrics = []

                # metrics
                for metric in self._metrics:
                    res = metric.accumulate()
                    metrics.extend(to_list(res))

                assert len(self._metrics_name()) == len(metrics)
                for k, v in zip(self._metrics_name(), metrics):
                    logs[k] = v
            else:
                if self._inputs is not None:
                    outs = self.predict_batch(data[: len(self._inputs)])
                else:
                    outs = self.predict_batch(data)

                outputs.append(outs)

            logs['step'] = step
            if (
                mode == 'train'
                or self._adapter._merge_count.get(mode + '_batch', 0) <= 0
            ):
                logs['batch_size'] = (
                    batch_size * paddle.distributed.ParallelEnv().nranks
                )
            else:
                logs['batch_size'] = self._adapter._merge_count[mode + '_batch']

            callbacks.on_batch_end(mode, step, logs)
            if hasattr(self, 'num_iters') and self.num_iters is not None:
                self.num_iters -= 1
                if self.num_iters <= 0:
                    self.stop_training = True
                    del self.num_iters
                    break
        self._reset_metrics()

        if mode == 'predict':
            return logs, outputs
        return logs

    def summary(
        self,
        input_size: (
            tuple[int, ...] | Input | list[tuple[int, ...] | Input] | None
        ) = None,
        dtype: _DTypeLiteral | None = None,
    ) -> ModelSummary:
        """Prints a string summary of the network.

        Args:
            input_size (tuple|InputSpec|list[tuple|InputSpec], optional): Size of input tensor.
                if not set, input_size will get from ``self._inputs`` if network only have
                one input, input_size can be tuple or InputSpec. if model have multiple
                input, input_size must be a list which contain every input's shape. Default: None.
            dtype (str, optional): If dtype is None, 'float32' will be used, Default: None.

        Returns:
            Dict: A summary of the network including total params and total trainable params.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.static import InputSpec

                >>> input = InputSpec([None, 1, 28, 28], 'float32', 'image')
                >>> label = InputSpec([None, 1], 'int64', 'label')
                >>> model = paddle.Model(paddle.vision.models.LeNet(), input, label)
                >>> optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
                >>> model.prepare(optim, paddle.nn.CrossEntropyLoss())
                >>> params_info = model.summary()
                >>> print(params_info)
                ---------------------------------------------------------------------------
                Layer (type)       Input Shape          Output Shape         Param #
                ===========================================================================
                  Conv2D-1       [[1, 1, 28, 28]]      [1, 6, 28, 28]          60
                    ReLU-1        [[1, 6, 28, 28]]      [1, 6, 28, 28]           0
                  MaxPool2D-1     [[1, 6, 28, 28]]      [1, 6, 14, 14]           0
                  Conv2D-2       [[1, 6, 14, 14]]     [1, 16, 10, 10]         2,416
                    ReLU-2       [[1, 16, 10, 10]]     [1, 16, 10, 10]           0
                  MaxPool2D-2    [[1, 16, 10, 10]]      [1, 16, 5, 5]            0
                  Linear-1          [[1, 400]]            [1, 120]           48,120
                  Linear-2          [[1, 120]]            [1, 84]            10,164
                  Linear-3          [[1, 84]]             [1, 10]              850
                ===========================================================================
                Total params: 61,610
                Trainable params: 61,610
                Non-trainable params: 0
                ---------------------------------------------------------------------------
                Input size (MB): 0.00
                Forward/backward pass size (MB): 0.11
                Params size (MB): 0.24
                Estimated Total Size (MB): 0.35
                ---------------------------------------------------------------------------
                {'total_params': 61610, 'trainable_params': 61610}

        """
        assert (
            input_size is not None or self._inputs is not None
        ), "'input_size' or 'self._input' must be set"
        if input_size is not None:
            _input_size = input_size
        else:
            _input_size = self._inputs
        return summary(self.network, _input_size, dtypes=dtype)

    def _verify_spec(self, specs, shapes=None, dtypes=None, is_input=False):
        out_specs = []

        if specs is None:
            # Note(Aurelius84): If not specific specs of `Input`, using argument names of `forward` function
            # to generate `Input`. But how can we know the actual shape of each input tensor?

            if is_input:
                arg_names = extract_args(self.network.forward)[1:]
                # While Saving inference model in dygraph, and providing inputs only in running.
                if (
                    shapes is not None
                    and dtypes is not None
                    and in_dynamic_mode()
                ):
                    out_specs = [
                        Input(name=n, dtype=dtypes[i], shape=shapes[i])
                        for i, n in enumerate(arg_names)
                    ]
                else:
                    out_specs = [Input(name=n, shape=[None]) for n in arg_names]
            else:
                out_specs = to_list(specs)
        elif isinstance(specs, dict):
            assert is_input is False
            out_specs = [
                specs[n]
                for n in extract_args(self.network.forward)
                if n != 'self'
            ]
        else:
            out_specs = to_list(specs)
        # Note: checks each element has specified `name`.
        if out_specs is not None:
            for i, spec in enumerate(out_specs):
                assert isinstance(spec, Input)
                if spec.name is None:
                    raise ValueError(
                        f"Requires Input[{i}].name != None, but receive `None` with {spec}."
                    )

        return out_specs

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _metrics_name(self):
        metrics_name = ['loss'] if self._loss else []
        for m in self._metrics:
            metrics_name.extend(to_list(m.name()))
        return metrics_name

    def _len_data_loader(self, data_loader):
        try:
            steps = len(data_loader)
        except Exception:
            steps = None
        return steps

    def _update_inputs(self):
        "Update self._inputs according to given inputs."
        self._input_info = self._adapter._input_info
        if self._input_info is not None and len(self._input_info) == 2:
            self._inputs = self._verify_spec(
                None, self._input_info[0], self._input_info[1], True
            )
            self._is_shape_inferred = True
