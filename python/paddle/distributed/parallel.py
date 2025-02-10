# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
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

import itertools
import os
import sys
import time
import warnings
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from multiprocessing import Manager, Process
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

import paddle
from paddle import _legacy_C_ops, framework
from paddle.distributed.collective import (
    Group,
    _default_group_name,
    _get_group_map_by_name,
    _new_process_group_impl,
    _set_default_backend,
    _set_default_store,
    _set_group_map,
    _set_group_map_backend,
    _set_group_map_by_name,
    _valid_backend_list,
)
from paddle.distributed.communication.group import (
    _add_new_group,
    _get_global_group,
    is_initialized,
)
from paddle.distributed.fleet.base.private_helper_function import (
    wait_server_ready,
)
from paddle.distributed.fleet.launch_utils import check_backend

# (TODO: GhostScreaming) It will be removed later.
from paddle.framework import (
    _set_expected_place,
    base as imperative_base,
    core,
    in_dynamic_mode,
)
from paddle.nn.layer import Layer
from paddle.utils import deprecated

from . import parallel_helper
from .backup_env import getenv_or_backup

if TYPE_CHECKING:
    from collections.abc import Generator

    from paddle import Tensor
    from paddle.nn.layer.layers import _StateDict
__all__ = []

ParallelStrategy = core.ParallelStrategy


def _build_default_parallel_strategy():
    strategy = ParallelStrategy()
    strategy.nranks = paddle.distributed.ParallelEnv().nranks
    strategy.local_rank = paddle.distributed.ParallelEnv().local_rank
    strategy.trainer_endpoints = (
        paddle.distributed.ParallelEnv().trainer_endpoints
    )
    strategy.current_endpoint = (
        paddle.distributed.ParallelEnv().current_endpoint
    )
    return strategy


def _coalesce_tensors(var_groups):
    coalesced_grads_and_grad_vars = []
    for group_id, grad_vars in var_groups.items():
        flattened_vars = []
        g_var_shapes = []
        for g_var in grad_vars:
            g_var_shapes.append(g_var.shape)
            flattened_vars.append(
                paddle.reshape(x=g_var, shape=[np.prod(g_var.shape)])
            )
        coalesced_grad = paddle.concat(flattened_vars)
        coalesced_grads_and_grad_vars.append(
            [coalesced_grad, grad_vars, g_var_shapes]
        )
    return coalesced_grads_and_grad_vars


@framework.dygraph_only
def _reshape_inplace(x, shape):
    x_shape = framework._create_tensor(dtype=x.dtype)
    framework._dygraph_tracer().trace_op(
        type="reshape2",
        inputs={'X': x},
        outputs={'Out': x, 'XShape': x_shape},
        attrs={'shape': shape},
    )


@framework.dygraph_only
def _split_tensors(coalesced_grads_and_grad_vars):
    if in_dynamic_mode():
        for (
            coalesced_grad,
            origin_grad_vars,
            grad_shapes,
        ) in coalesced_grads_and_grad_vars:
            grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
            attrs = ()
            attrs += ('sections', grad_var_len)
            attrs += ('axis', 0)
            _legacy_C_ops.split(coalesced_grad, origin_grad_vars, *attrs)
            for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
                g_var.reshape_(shape=g_shape)
                assert g_var.shape == g_shape


@imperative_base.no_grad
@framework.dygraph_only
def build_groups(
    vars: list[Tensor], group_size: int
) -> list[list[Tensor | list[Tensor] | list[int]]]:
    group_idx = 0
    memory_counter = 0
    var_groups = OrderedDict()
    dtype = vars[0].dtype

    for var in vars:
        var_dtype = var.dtype
        if isinstance(var_dtype, core.DataType):
            var_dtype = paddle.pir.core.datatype_to_vartype[var_dtype]
        bytes = np.prod(var.shape) * core.size_of_dtype(var_dtype)
        if memory_counter < group_size and dtype == var.dtype:
            memory_counter += bytes
        else:
            memory_counter = bytes
            dtype = var.dtype
            group_idx += 1
        var_groups.setdefault(group_idx, []).append(var)
    return _coalesce_tensors(var_groups)


@imperative_base.no_grad
@framework.dygraph_only
def sync_params_buffers(
    model: Layer,
    comm_group: Group | None = None,
    src_rank: int = 0,
    is_model_parallel: bool = False,
    fuse_params: bool = True,
) -> None:
    model_vars = []
    for _, param in model._obtain_parameters_buffers().items():
        if not isinstance(param, core.eager.Tensor):
            raise TypeError(
                f"The data type of '{param.name}' must be core.eager.Tensor"
            )

        if is_model_parallel:
            if hasattr(param, "is_distributed") and param.is_distributed:
                continue

        # NOTE(shenliang03): Support situations that do not require synchronization parameters,
        # such as moe's expert parameters
        if getattr(param, "no_sync", False):
            continue

        if param.type == core.VarDesc.VarType.VOCAB:
            continue

        model_vars.append(param.detach())
    if len(model_vars) == 0:
        return

    if fuse_params:
        # group size is 128M
        coalesced_vars = build_groups(model_vars, 128 * 1024 * 1024)

        for coalesced_var, _, _ in coalesced_vars:
            paddle.distributed.broadcast(
                coalesced_var, src=src_rank, group=comm_group, sync_op=True
            )
        for coalesced_var, origin_vars, var_shapes in coalesced_vars:
            var_len = [np.prod(v_shape) for v_shape in var_shapes]
            paddle.base.framework._dygraph_tracer().trace_op(
                type='split',
                inputs={'X': coalesced_var},
                outputs={'Out': origin_vars},
                attrs={'sections': var_len, 'axis': 0},
            )
    else:
        for var in model_vars:
            # NOTE(shenliang03): Now, we dont support contiguous tensor in dp
            var = var.contiguous()
            paddle.distributed.broadcast(
                var, src=src_rank, group=comm_group, sync_op=True
            )


class DataParallel(Layer):
    """
    Run the dygraph module with data parallelism.

    Currently, DataParallel class only supports to run the dynamic graph
    with multi-process.

    Now supports two ways to start training:

    1. start by ``paddle.distributed.spawn`` method, for example:

        ``python demo.py`` (spawn need to be called in ``__main__`` method)

    2. start by ``paddle.distributed.launch`` module, for example:

        ``python -m paddle.distributed.launch --gpus=0,1 demo.py`` .

    And the content of `demo.py` is the code of examples.

    Args:
        layers(Layer): The module that should be executed by data parallel.
        strategy(ParallelStrategy, optional): (deprecated) The strategy of data parallelism,
            contains environment configuration related to parallel execution. Default: None.
        comm_buffer_size(int, optional):  It limits the memory size(MB) of one buffer
                                          parameters' gradient which is the input of communication
                                          calling(e.g NCCLAllReduce). Default: 25.
        last_comm_buffer_size(float, optional): It limits memory size(MB) of last buffer in communication
                                         calling. Making the last communication buffer size small is useful to
                                         improve performance. Default: 1.
        find_unused_parameters(bool, optional): Whether to traverse the entire backward graph from the
                                                all tensors in the return value of the wrapped model's
                                                forward function. For parameters not involved in loss
                                                calculation, their gradients will be marked as ready in
                                                advance to prepare reduce. Please note that all forward
                                                outputs derived from the wrapped model parameters must
                                                participate in the calculation of loss and subsequent
                                                gradient calculations. If not, serious error will occur.
                                                Note that setting the find_unused_parameters to True
                                                will affect computing performance. Therefore, if all parameters
                                                are sure to participate in the loss calculation and the
                                                autograd graph construction, please set it False. Default: False.

    Returns:
        Layer: The data paralleled module.

    Examples:

        .. code-block:: python
            :name: dp-example

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt
            >>> import paddle.distributed as dist

            >>> class LinearNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear1 = nn.Linear(10, 10)
            ...         self._linear2 = nn.Linear(10, 1)
            ...     def forward(self, x):
            ...         return self._linear2(self._linear1(x))

            >>> def train():
            ...     # 1. initialize parallel environment
            ...     dist.init_parallel_env()
            ...     # 2. create data parallel layer & optimizer
            ...     layer = LinearNet()
            ...     dp_layer = paddle.DataParallel(layer)
            ...     loss_fn = nn.MSELoss()
            ...     adam = opt.Adam(
            ...         learning_rate=0.001, parameters=dp_layer.parameters())
            ...     # 3. run layer
            ...     inputs = paddle.randn([10, 10], 'float32')
            ...     outputs = dp_layer(inputs)
            ...     labels = paddle.randn([10, 1], 'float32')
            ...     loss = loss_fn(outputs, labels)
            ...     loss.backward()
            ...     adam.step()
            ...     adam.clear_grad()

            >>> if __name__ == '__main__':
            ...     # 1. start by ``paddle.distributed.spawn`` (default)
            ...     dist.spawn(train, nprocs=2)
            ...     # 2. start by ``paddle.distributed.launch``
            ...     # train()

    .. note::
        ``PyLayer`` is not supported in DataParallel. To solve problems of this kind,
        it's recommended to skip gradient synchronization among multiple cards by 'no_sync',
        and manually implement 'all_reduce' before model optimization. There is an example
        showing specific implementation processing.

    Examples:

        .. code-block:: python
            :name: dp-pylayer-example

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import numpy
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle.autograd import PyLayer
            >>> from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

            >>> class cus_tanh(PyLayer):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         y = paddle.tanh(x)
            ...         ctx.save_for_backward(y)
            ...         return y
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         y, = ctx.saved_tensor()
            ...         grad = dy * (1 - paddle.square(y))
            ...         return grad

            >>> class SimpleNet(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = paddle.nn.Linear(2, 2)
            ...     def forward(self, inputs):
            ...         inputs = cus_tanh.apply(inputs)
            ...         return self.linear(inputs)

            >>> if __name__ == '__main__':
            ...     dist.init_parallel_env()
            ...     model = SimpleNet()
            ...     model = paddle.DataParallel(model)
            ...     opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
            ...     for step in range(10):
            ...         x_data = numpy.random.randn(2, 2).astype(numpy.float32) # type: ignore[var-annotated]
            ...         x = paddle.to_tensor(x_data)
            ...         x.stop_gradient = False
            ...         # step 1 : skip gradient synchronization by 'no_sync'
            ...         with model.no_sync():
            ...             y_pred = model(x)
            ...             loss = y_pred.mean()
            ...             loss.backward()
            ...         # step 2 : fuse + allreduce manually before optimization
            ...         fused_allreduce_gradients(list(model.parameters()), None)
            ...         opt.step()
            ...         opt.clear_grad()

    """

    find_unused_parameters: bool
    grad_need_sync: bool
    group: Group | None
    var_dtype: Tensor
    comm_buffer_size: int
    last_comm_buffer_size: int

    def __init__(
        self,
        layers: Layer,
        strategy: ParallelStrategy | None = None,
        comm_buffer_size: int = 25,
        last_comm_buffer_size: float = 1,
        find_unused_parameters: bool = False,
        group: Group | None = None,
    ) -> None:
        super().__init__(layers.full_name() + "_data_parallel")

        assert (
            in_dynamic_mode()
        ), "It's not supported to construct DataParallel in static graph mode."

        self._layers = layers
        self.find_unused_parameters = find_unused_parameters
        self.grad_need_sync = True
        self.group = group
        self.var_dtype = core.eager.Tensor

        # NOTE(chenweihang): The ParallelStrategy here is not strictly a strategy.
        # It just stores some environment variables, which can be constructed by
        # ParallelEnv. Here it is set as an optional argument.
        # This parameter is not removed because of compatibility with 1.x writing.
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = _build_default_parallel_strategy()

        if self._strategy.nranks > 1:
            # check the environment
            assert parallel_helper.__parallel_ctx__clz__ is not None, (
                "ParallelContext must be initialized before. You should use init_parallel_env() before"
                "constructing the DataParallel."
            )

            if in_dynamic_mode():
                self.group = (
                    paddle.distributed.collective._get_default_group()
                    if self.group is None
                    else self.group
                )

                assert isinstance(
                    self.group, paddle.distributed.collective.Group
                ), "ProcessGroup must be an instance of Group in DataParallel."

                [
                    warnings.warn(
                        f"param [{name}] is not contiguous, please check it and make it contiguous."
                    )
                    for name, param in self._layers.named_parameters()
                    if not param.is_contiguous()
                ]
            # sync buffer and params
            sync_params_buffers(self._layers, fuse_params=False)

            self.comm_buffer_size = int(comm_buffer_size * 1024 * 1024)
            # NOTE(shenliang03): We can set environment variables to control
            # the size of the group, Default: 1MB. The role of this small group is:
            # when the last group allreduce, the overlap cannot work. Making the
            # the last group small is useful to improve performance.
            self.last_comm_buffer_size = int(
                last_comm_buffer_size * 1024 * 1024
            )
            self.init_reducer()
        else:
            warnings.warn(
                "The program will return to single-card operation. "
                "Please check 1, whether you use spawn or fleetrun "
                "to start the program. 2, Whether it is a multi-card "
                "program. 3, Is the current environment multi-card."
            )

    def init_reducer(self) -> None:
        layers_param = []
        params_set = set()
        for sublayer in self.sublayers():
            for _, param in sublayer.named_parameters(include_sublayers=False):
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                if not isinstance(param, self.var_dtype):
                    raise TypeError(
                        f"The data type of '{param.name}' must be '{self.var_dtype}'"
                    )
                if param.trainable:
                    layers_param.append((sublayer, param))

        trainable_parameters = list(
            filter(
                lambda x: not getattr(x, "no_sync", False),
                [param for _, param in layers_param],
            )
        )

        assert len(trainable_parameters) > 0, (
            "This model does not have any parameters to train, and "
            "does not need to use DataParallel"
        )

        # NOTE(shenliang03): Here we can only use the attributes to judge whether
        # parameter is sparse(or SelectedRows). The reason is that the sparse message
        # can't be obtained when bp hasn't happened yet. So if layer supports sparse parameter,
        # we should add the layer here like "paddle.nn.layer.common.Embedding".
        def check_layer_sparse(sublayer):
            if isinstance(sublayer, paddle.nn.layer.common.Embedding):
                return sublayer._sparse
            return False

        is_sparse_gradient = [
            check_layer_sparse(sublayer)
            for sublayer, param in layers_param
            if not getattr(param, "no_sync", False)
        ]

        if in_dynamic_mode():
            self.group_indices = core.eager_assign_group_by_size(
                trainable_parameters,
                is_sparse_gradient,
                [self.last_comm_buffer_size, self.comm_buffer_size],
            )

            self._reducer = core.EagerReducer(
                trainable_parameters,
                list(reversed(self.group_indices)),
                is_sparse_gradient,
                self.group.process_group,
                [self.last_comm_buffer_size, self.comm_buffer_size],
                self.find_unused_parameters,
            )

    def _find_tensor(self, obj):
        var_type = core.eager.Tensor
        if isinstance(obj, var_type):
            return [obj]
        if isinstance(obj, (list, tuple)):
            return itertools.chain(*map(self._find_tensor, obj))
        if isinstance(obj, dict):
            return itertools.chain(*map(self._find_tensor, obj.values()))
        return []

    @contextmanager
    def no_sync(self) -> Generator[None, None, None]:
        """
        A context manager to stop gradient synchronization. Within no_sync(),
        gradients of parameters will only be accumulated on model and not
        synchronized util the first forward-backward out of this context.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> import paddle
                >>> import paddle.nn as nn
                >>> import paddle.distributed as dist

                >>> class SimpleNet(nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = nn.Linear(10, 1)
                ...     def forward(self, x):
                ...         return self._linear(x)

                >>> dist.init_parallel_env()
                >>> model = SimpleNet()
                >>> dp_model = paddle.DataParallel(model)

                >>> inputs_1 = paddle.randn([10, 10], 'float32')
                >>> inputs_2 = paddle.ones([10, 10], 'float32')

                >>> with dp_model.no_sync():
                ...     # gradients will not be synchronized
                ...     dp_model(inputs_1).backward()

                >>> # synchronization happens here
                >>> dp_model(inputs_2).backward()

        """
        tmp_grad_need_sync = self.grad_need_sync
        self.grad_need_sync = False
        try:
            yield
        finally:
            self.grad_need_sync = tmp_grad_need_sync

    def forward(self, *inputs: Any, **kwargs: Any) -> Tensor:
        outputs = self._layers(*inputs, **kwargs)
        if (
            self._strategy.nranks > 1
            and framework._dygraph_tracer()._has_grad
            and self.grad_need_sync
        ):
            self._reducer.prepare_for_backward(list(self._find_tensor(outputs)))
        return outputs

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore."
    )
    def scale_loss(self, loss):
        """
        Deprecated method, now ``scale_loss`` is an empty method,
        keep this method just for compatibility.
        """
        return loss

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore."
    )
    def apply_collective_grads(self):
        """
        Deprecated method, now ``apply_collective_grads`` is an empty method,
        keep this method just for compatibility.
        """
        return

    def state_dict(
        self,
        destination: _StateDict | None = None,
        include_sublayers: bool = True,
        structured_name_prefix: str = "",
    ) -> _StateDict:
        '''
        Get all parameters and persistable buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True

        Returns:
            dict: a dict contains all the parameters and persistable buffers.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> import paddle
                >>> import paddle.distributed as dist

                >>> dist.init_parallel_env()

                >>> emb = paddle.nn.Embedding(10, 10)
                >>> emb = paddle.DataParallel(emb)

                >>> state_dict = emb.state_dict()
                >>> paddle.save(state_dict, "paddle_dy.pdparams")

        '''

        return self._layers.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
        )

    @framework.deprecate_stat_dict
    def set_state_dict(
        self, state_dict: _StateDict, use_structured_name: bool = True
    ) -> None:
        '''
        Set parameters and persistable buffers from state_dict. All the parameters and buffers will be reset by the tensor in the state_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters and persistable buffers.
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter or buffer name as key.
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> import paddle
                >>> import paddle.distributed as dist

                >>> dist.init_parallel_env()

                >>> emb = paddle.nn.Embedding(10, 10)
                >>> emb = paddle.DataParallel(emb)

                >>> state_dict = emb.state_dict()
                >>> paddle.save(state_dict, "paddle_dy.pdparams")

                >>> para_state_dict = paddle.load("paddle_dy.pdparams")
                >>> emb.set_state_dict(para_state_dict)

        '''

        self._layers.set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )

    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict


# NOTE(chenweihang): Maintain a global parallel env to avoid
# initializing ParallelEnv every time and improve performance
_global_parallel_env = None


class ParallelEnv:
    """
    .. note::
        This API is not recommended, if you need to get rank and world_size,
        it is recommended to use ``paddle.distributed.get_rank()`` and
        ``paddle.distributed.get_world_size()`` .

    This class is used to obtain the environment variables required for
    the parallel execution of ``paddle.nn.Layer`` in dynamic mode.

    The parallel execution in dynamic mode needs to be started using ``paddle.distributed.launch``
    or ``paddle.distributed.spawn`` .

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> def train():
            ...     # 1. initialize parallel environment
            ...     dist.init_parallel_env()
            ...     # 2. get current ParallelEnv
            ...     parallel_env = dist.ParallelEnv()
            ...     print("rank: ", parallel_env.rank)
            ...     print("world_size: ", parallel_env.world_size)

            >>> if __name__ == '__main__':
            ...     # 1. start by ``paddle.distributed.spawn`` (default)
            ...     dist.spawn(train, nprocs=2)
            ...     # 2. start by ``paddle.distributed.launch``
            ...     train()

            # Print result in process 1:
            rank: 1
            world_size: 2

            # Print result in process 2:
            rank: 2
            world_size: 2

    """

    def __init__(self):
        self._rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._world_size = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self._device_type = str(os.getenv("PADDLE_XCCL_BACKEND", ""))
        self._pg_timeout = int(os.getenv("PADDLE_PG_TIMEOUT", "1800000"))

        # imperative only support one gpu or xpu
        if self._device_type != "":
            FLAGS_selected_custom_devices = (
                f'FLAGS_selected_{self._device_type}s'
            )
            selected_custom_devices = os.getenv(
                FLAGS_selected_custom_devices, "0"
            ).split(",")
            self._device_id = int(selected_custom_devices[0])
        else:
            if core.is_compiled_with_cuda():
                selected_gpus = os.getenv("FLAGS_selected_gpus", "0").split(",")
                self._device_id = int(selected_gpus[0])
            elif core.is_compiled_with_xpu():
                selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
                self._device_id = int(selected_xpus[0])

        self._trainer_endpoints = getenv_or_backup(
            "PADDLE_TRAINER_ENDPOINTS", ""
        ).split(",")
        self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "")
        self._nrings = int(os.getenv("FLAGS_nccl_nrings", "1"))
        assert (
            self._nrings > 0
        ), "nccl_nrings must be an integer greater than 0."
        assert (
            self._nrings < 9
        ), "nccl_nrings should be less than 9, which is enough in most scenarios."

    @property
    def rank(self) -> int:
        """
        Rank of current trainer.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ID`` . The default value is 0.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # execute this command in terminal: export PADDLE_TRAINER_ID=0
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> print("The rank is %d" % env.rank)
                The rank is 0

        """
        return self._rank

    @property
    def world_size(self) -> int:
        """
        The number of trainers (number of processes participating in current job).

        Its value is equal to the value of the environment variable ``PADDLE_TRAINERS_NUM`` . The default value is 1.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> print("The world_size is %d" % env.world_size)
                The world_size is 4

        """
        return self._world_size

    @property
    def device_id(self) -> int:
        """
        The ID of selected GPU card for parallel training.

        Its value is equal to the value of the environment variable ``FLAGS_selected_gpus`` . The default value is 0.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # execute this command in terminal: export FLAGS_selected_gpus=1
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> print("The device id are %d" % env.device_id)
                The device id are 1
        """
        return self._device_id

    @property
    def device_type(self) -> str:
        """
        The type of custom device for parallel training.

        Its value is equal to the value of the environment variable ``PADDLE_XCCL_BACKEND`` . The default value is None.

        """
        return self._device_type

    @property
    def current_endpoint(self) -> str:
        """
        The endpoint of current trainer, it is in the form of (node IP + port).

        Its value is equal to the value of the environment variable ``PADDLE_CURRENT_ENDPOINT`` . The default value is "".

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> print("The current endpoint are %s" % env.current_endpoint)
                The current endpoint are 127.0.0.1:6170
        """
        return self._current_endpoint

    @property
    def trainer_endpoints(self) -> list[str]:
        """
        The endpoints of all trainer nodes in the task,
        which are used to broadcast the NCCL ID when NCCL2 is initialized.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ENDPOINTS`` . The default value is "".

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> print("The trainer endpoints are %s" % env.trainer_endpoints)
                The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']

        """
        return self._trainer_endpoints

    @property
    def nrings(self) -> int:
        """
        Nrings of current trainer.

        Its value is equal to the value of the environment variable ``FLAGS_nccl_nrings`` . The default value is 1.

        Examples:
            .. code-block:: python

                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # execute this command in terminal: export FLAGS_nccl_nrings=1
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> print("The nrings is %d" % env.nrings)
                The nrings is 1
        """
        return self._nrings

    @property
    def pg_timeout(self) -> int:
        """
        timeout of process group.

        Its value is equal to the value of the environment variable ``PADDLE_PG_TIMEOUT`` . The default value is 30 minutes.

        Examples:
            .. code-block:: python

                >>> # execute this command in terminal: export PADDLE_PG_TIMEOUT=1800000
                >>> import paddle.distributed as dist

                >>> env = dist.ParallelEnv()
                >>> # the pg_timeout of process group 1800000
        """
        return self._pg_timeout

    # [aliases] Compatible with old method names
    local_rank = rank
    nranks = world_size
    dev_id = device_id


def _get_global_parallel_env():
    global _global_parallel_env
    if _global_parallel_env is None:
        _global_parallel_env = ParallelEnv()
    return _global_parallel_env


def _start_kv_server(port, http_server_d, size):
    from paddle.distributed.fleet.utils.http_server import KVServer

    http_server = KVServer(int(port), size=size)
    http_server.start()
    wait_seconds = 3
    while http_server_d.get("running", False) or not http_server.should_stop():
        time.sleep(wait_seconds)
    http_server.stop()


def _is_cpuonly(backend):
    check_backend(backend)
    if (
        backend in ['auto', 'nccl', 'bkcl', 'heter', 'ucc']
        and (core.is_compiled_with_cuda() or core.is_compiled_with_xpu())
    ) or backend == 'xccl':
        # passes 'auto' and can use cuda or xpu, use the default logics. so return False
        return False
    else:
        return True


def _check_var_exists(var_name):
    var = getenv_or_backup(var_name, None)
    if var is None:
        raise ValueError(
            "paddle.distributed initialize error, "
            f"environment variable {var_name} is needed, but not set."
        )


def _get_modified_flags():
    ret = []
    FLAGS = namedtuple('FLAGS', ['name', 'current_value', 'default_value'])
    global_flags = core.globals()
    for key in global_flags.keys():
        value = global_flags.get(key)
        default_value = global_flags.get_default(key)
        if not value == default_value:
            ret.append(FLAGS(key, value, default_value))
    return ret


def _print_modified_flags(modified_flags):
    if len(modified_flags) > 0:
        sys.stderr.write(
            "======================= Modified FLAGS detected =======================\n"
        )
        for flag in modified_flags:
            sys.stderr.write(str(flag))
            sys.stderr.write("\n")
        sys.stderr.write(
            "=======================================================================\n"
        )


def init_parallel_env() -> Group:
    """

    Initialize parallel training environment in dynamic graph mode.

    Note:
        Now initialize both `NCCL` and `GLOO` contexts for communication.

    Args:
        backend (string): A string represents the backend used by DataParallel,
            should be one of 'gloo'(for cpu), 'nccl'(for cuda), 'bkcl'(for xpu), 'auto'(auto detect).
            The auto detection prefer 'nccl', 'bkcl' than 'gloo'.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU, env:DISTRIBUTED)
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt
            >>> import paddle.distributed as dist

            >>> class LinearNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear1 = nn.Linear(10, 10)
            ...         self._linear2 = nn.Linear(10, 1)
            ...     def forward(self, x):
            ...         return self._linear2(self._linear1(x))

            >>> def train():
            ...     # 1. initialize parallel environment
            ...     dist.init_parallel_env()
            ...     # 2. create data parallel layer & optimizer
            ...     layer = LinearNet()
            ...     dp_layer = paddle.DataParallel(layer)
            ...     loss_fn = nn.MSELoss()
            ...     adam = opt.Adam(
            ...         learning_rate=0.001, parameters=dp_layer.parameters())
            ...     # 3. run layer
            ...     inputs = paddle.randn([10, 10], 'float32')
            ...     outputs = dp_layer(inputs)
            ...     labels = paddle.randn([10, 1], 'float32')
            ...     loss = loss_fn(outputs, labels)
            ...     loss.backward()
            ...     adam.step()
            ...     adam.clear_grad()

            >>> if __name__ == '__main__':
            ...     dist.spawn(train)

    """

    modified_flags = _get_modified_flags()
    _print_modified_flags(modified_flags)

    # 0. get env & check world size
    global _global_parallel_env
    # when call init_parallel_env, need update `_global_parallel_env`
    _global_parallel_env = ParallelEnv()
    parallel_env = _global_parallel_env
    # if not parallel, `init_parallel_env` do nothing
    if parallel_env.world_size < 2:
        warnings.warn(
            "Currently not a parallel execution environment, `paddle.distributed.init_parallel_env` will not do anything."
        )
        return
    # NOTE(xiongkun): support cpu gloo only, add this environment variable to
    #                 enable cpu only gloo parallel training)
    backend = os.environ.get('PADDLE_DISTRI_BACKEND', 'auto')
    is_cpu_only = _is_cpuonly(backend)
    # 1. gpu xpu check, must be gpu or xpu,
    if not (
        is_cpu_only
        or core.is_compiled_with_cuda()
        or core.is_compiled_with_xpu()
        or backend == "xccl"
    ):
        raise NotImplementedError(
            "If you want to use CPU-only version, please use 'gloo' as backend"
        )

    if backend == "xccl":
        FLAGS_selected_custom_devices = (
            f'FLAGS_selected_{parallel_env.device_type}s'
        )
        _check_var_exists(FLAGS_selected_custom_devices)
    else:
        if not is_cpu_only and core.is_compiled_with_cuda():
            _check_var_exists("FLAGS_selected_gpus")
            backend = "nccl" if backend == "auto" else backend
        elif not is_cpu_only and core.is_compiled_with_xpu():
            _check_var_exists('FLAGS_selected_xpus')
            backend = "bkcl" if backend == "auto" else backend

    _check_var_exists("PADDLE_TRAINER_ID")
    _check_var_exists("PADDLE_CURRENT_ENDPOINT")
    _check_var_exists("PADDLE_TRAINERS_NUM")

    # NOTE(chenweihang): [ why config global place here? ]
    # the dygraph mode will be set to default mode,
    # users will not call `dygraph.guard` or `enable_dygraph`
    # directly, if they want to switch default place,
    # they need to call a function to change default place,
    # here just set correctly place to users
    if backend == "xccl":
        place = core.CustomPlace(
            parallel_env.device_type, parallel_env.device_id
        )
    elif is_cpu_only:
        place = core.CPUPlace()
    elif core.is_compiled_with_cuda():
        place = core.CUDAPlace(parallel_env.device_id)
    elif core.is_compiled_with_xpu():
        place = core.XPUPlace(parallel_env.device_id)
    _set_expected_place(place)

    group = None

    if backend in _valid_backend_list and in_dynamic_mode():
        if _default_group_name in _get_group_map_by_name():
            return _get_group_map_by_name()[_default_group_name]
        _set_default_backend(backend)
        rank = int(os.getenv("PADDLE_TRAINER_ID"))
        world_size = int(os.getenv("PADDLE_TRAINERS_NUM"))
        assert rank >= 0 and world_size > rank and world_size > 1, (
            "rank must be non-negative and world_size must be the "
            "maximum rank plus one. Moreover, at least two processes are "
            "required to create a process group."
        )
        master_addr = os.getenv("MASTER_ADDR", None)
        master_port = os.getenv("MASTER_PORT", None)
        endpoints = (
            ":".join([master_addr, master_port])
            if master_addr and master_port
            else None
        )
        if endpoints is None:
            endpoints = os.getenv("PADDLE_MASTER", None)
        if endpoints is None:
            endpoints = getenv_or_backup("PADDLE_TRAINER_ENDPOINTS").split(',')[
                0
            ]
        assert endpoints, (
            "The environment variable 'MASTER_ADDR' and 'MASTER_PORT' "
            "must be specified, for example 'export MASTER_ADDR=127.0.0.1' "
            "and 'export MASTER_ADDR=54612'. Or you can start your training"
            "with paddle.distributed.run module."
        )
        master_addr, master_port = endpoints.split(":")
        master_port = int(master_port)
        is_master = rank == 0
        stop_check_timeout = int(os.getenv("FLAGS_stop_check_timeout", "900"))
        default_store = core.create_or_get_global_tcp_store()
        _set_default_store(default_store)

        if backend in ["nccl", 'xccl', 'bkcl', 'ucc']:
            core.CommContextManager.set_device_id(parallel_env.device_id)

        pg = _new_process_group_impl(
            backend,
            default_store,
            rank,
            world_size,
            _default_group_name,
            pg_options=None,
        )
        ranks = list(range(world_size))
        group = Group(rank, 0, ranks, pg=pg, name=_default_group_name)
        _set_group_map_by_name(_default_group_name, group)
        _set_group_map(0, group)
        _set_group_map_backend(group, backend)
        _add_new_group(group)
        parallel_helper._set_parallel_ctx(True)
        return group

    node_num = {i.split(":")[0] for i in parallel_env.trainer_endpoints}
    # 3: init gloo context (step 1: httpserver start)
    init_gloo = int(os.getenv("PADDLE_WITH_GLOO", "0"))
    if is_cpu_only or init_gloo or backend == "heter":
        ep_rank_0 = parallel_env.trainer_endpoints[0].split(":")
        manager = Manager()
        # global dict to store status
        http_server_d = manager.dict()
        http_server_d["running"] = False
        if parallel_env.rank == 0:
            # The scope for worker used by http server is '_worker'
            size = {'_worker': parallel_env.world_size}
            if backend == "heter":
                size = {'_worker': len(node_num)}
            http_server = Process(
                target=_start_kv_server,
                args=(int(ep_rank_0[1]), http_server_d, size),
            )
            http_server.daemon = True
            http_server_d["running"] = True
            http_server.start()

    # 4. init NCCL ParallelStrategy
    strategy = ParallelStrategy()
    if parallel_helper._is_parallel_ctx_initialized():
        warnings.warn("The parallel environment has been initialized.")
    strategy.nranks = parallel_env.world_size
    strategy.local_rank = parallel_env.rank
    strategy.trainer_endpoints = parallel_env.trainer_endpoints
    strategy.current_endpoint = parallel_env.current_endpoint
    strategy.nrings = parallel_env.nrings

    # init nccl or bkcl or heter context
    if is_cpu_only:
        parallel_helper._set_parallel_ctx(
            core.GLOOParallelContext(strategy, place)
        )
    elif backend == "heter":
        parallel_helper._set_parallel_ctx(
            core.HeterParallelContext(strategy, parallel_env.device_id)
        )
    elif core.is_compiled_with_cuda():
        parallel_helper._set_parallel_ctx(
            core.NCCLParallelContext(strategy, place)
        )
    elif core.is_compiled_with_xpu():
        parallel_helper._set_parallel_ctx(
            core.BKCLParallelContext(strategy, place)
        )

    if backend != "heter":
        other_endpoints = strategy.trainer_endpoints[:]
        other_endpoints.remove(strategy.current_endpoint)
        if not is_cpu_only and strategy.local_rank == 0:
            wait_server_ready(other_endpoints)

    parallel_helper._init_parallel_ctx()

    # 5: init gloo context (step 2: gloo init)
    # dividing init_gloo into two part because nccl and gloo
    # are separately looking for free ports which sometimes
    # leads to port-conflict.
    if (is_cpu_only or backend == "heter") and parallel_env.rank == 0:
        # compare to init_gloo, we don't need to
        # init gloo, because we do this in _init_parallel_ctx;
        http_server_d["running"] = False
        http_server.join()

    elif init_gloo:
        wait_server_ready([parallel_env.trainer_endpoints[0]])
        gloo_strategy = core.GlooParallelStrategy()
        gloo_strategy.rank = parallel_env.rank
        gloo_strategy.rank_num = parallel_env.world_size
        gloo_strategy.ip_address = ep_rank_0[0]
        gloo_strategy.ip_port = int(ep_rank_0[1])
        default_init_timeout_seconds = 3600
        default_run_timeout_seconds = 9999999
        gloo_strategy.init_seconds = default_init_timeout_seconds
        gloo_strategy.run_seconds = default_run_timeout_seconds
        gloo = core.GlooParallelContext(gloo_strategy)
        gloo.init()
        if parallel_env.rank == 0:
            http_server_d["running"] = False
            http_server.join()
    return group


def get_rank(group: Group | None = None) -> int:
    """
    Returns the rank of current trainer in the given group, ranks are consecutive integers in [0, ``world_size``).
    If none of the group is given, the global group will be used as default.

    Args:
        group (Group, optional): The communication group you want to get rank of current trainer, use global group as default if group is None.

    Returns:
        (int) The rank of current trainer in the given group. Return -1 if the process is not part of the given group.

    Warning:
        Argument ``group`` only supports in dygraph mode.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # Execute this script using distributed launch with one card configs.
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> print("The rank is %d" % dist.get_rank())
            The rank is 0

    """
    if in_dynamic_mode() and group:
        return group.rank

    assert group is None, "Only support group argument in eager mode."
    return _get_global_parallel_env().rank


def get_world_size(group: Group | None = None) -> int:
    """
    Returns the number of trainers (number of processes participating in current job) in the given group.
    If none of the group is given, the global group will be used as default.

    Args:
        group (Group, optional): The communication group you want to check world size, use global group as default if group is None.

    Returns:
        (int) The number of trainers in the given group. Return -1 if the process if not part of the given group.

    Warning:
        Argument ``group`` only supports in dygraph mode.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # Execute this script using distributed launch with one card configs.
            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> print("The world_size is %d" % dist.get_world_size())
            The world_size is 1

    """
    if in_dynamic_mode() and (group is None):
        if is_initialized():
            group = _get_global_group()

    if in_dynamic_mode() and group:
        return group.world_size

    assert group is None, "Only support group argument in eager mode."
    return _get_global_parallel_env().world_size
