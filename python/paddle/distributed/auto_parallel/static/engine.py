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

from __future__ import annotations

import copy
import json
import logging
import numbers
import os
import random
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import paddle
import paddle.distributed.auto_parallel.static.utils as auto_utils
from paddle import pir, static, utils
from paddle.base.executor import _to_name_str
from paddle.base.framework import auto_complete_op_role
from paddle.decomposition import decomp
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.distributed.passes.pass_base import new_pass
from paddle.distributed.passes.pass_utils import (
    _split_program_into_forward_backward_optimize,
    set_skip_gc_vars,
)
from paddle.framework import (
    IrGraph,
    _current_expected_place_ as _get_device,
    core,
    in_dynamic_mode,
)
from paddle.metric import Metric
from paddle.static import InputSpec, Operator, Variable, global_scope
from paddle.static.amp.fp16_utils import _convert_float_to_bfloat16

from ...utils.log_utils import get_logger
from ..interface import CollectionNames, fetch, get_collection
from ..static.dist_tensor import DistributedTensor
from ..strategy import Strategy
from .callbacks import config_callbacks
from .cluster import Cluster, get_default_cluster
from .converter import Converter
from .cost.estimate_cost import get_cost_from_engine
from .dist_context import DistributedContext, get_default_distributed_context
from .dist_input_spec import DistributedInputSpec
from .dist_loader import (
    DistributedDataLoader,
)
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .helper import ProgramHelper
from .mix_to_dist_pass import apply_mix2dist_pass
from .parallelizer_v2 import Parallelizer
from .pir_pass import (
    RemovePasses,
    ReshardPasses,
    apply_partition_pass,
    check_chunk_id,
    complete_chunk_id,
    fuse_attention_ffn_qkv_pass,
    pipeline_pass,
    remove_unuseful_comm_op_pass,
)
from .planner_v2 import Planner
from .process_group import get_all_process_groups, new_process_group
from .utils import set_all_ops_op_role

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import PlaceLike
    from paddle.hapi.callbacks import Callback
    from paddle.io import Dataset
    from paddle.io.reader import _CollateFn
    from paddle.nn import Layer
    from paddle.optimizer import Optimizer
    from paddle.pir import Value
    from paddle.static import Program

    _Mode: TypeAlias = Literal["train", "eval", "predict"]


class Engine:
    """
    An High-Level API for auto parallel, which could be used for distributed Training (engine.fit) and Inference (engine.predict).
    Static graph mode is supported natively, Dynamic graph mode is also supported under `@to_static <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/to_static_cn.html#to-static>`_ .

    Args:
        model (paddle.nn.Layer, optional): The model is an instance of
            paddle.nn.Layer.
        loss (Loss|Callable|None, optional): The loss can be a `paddle.nn.Layer`
            instance or any callable function taken the predicted values and
            ground truth values as input. It can be None when there is no loss.
            Default: None.
        optimizer (Optimizer|None, optional): The optimizer need to be set in training
            and should be None in eval and predict mode. Default: None.
        metrics (Metric|list[Metric]|None, optional): If metrics is set, all
            metrics will be calculated and output in train/eval mode. Default: None.
        cluster (Cluster|None, optional): The cluster represents the topology information
            about the used physical devices. Default: None. (Unused for now)
        strategy (Strategy|None, optional): The strategy is used to configure the
        parallelization and optimization behaviors. Default: None.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.vision.transforms as T
            >>> from paddle.distributed.fleet import auto
            >>> from paddle.vision.datasets import MNIST

            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            >>> ])
            >>> train_dataset = MNIST(mode='train', transform=transform)
            >>> valid_dataset = MNIST(mode='test', transform=transform)

            >>> model = paddle.vision.models.LeNet()
            >>> loss = paddle.nn.CrossEntropyLoss()
            >>> optimizer = paddle.optimizer.Adam(
            ...     learning_rate=0.001, parameters=model.parameters())
            >>> metrics = paddle.metric.Accuracy(topk=(1, 2))

            >>> engine = auto.Engine(model, loss, optimizer, metrics)
            >>> # fit
            >>> engine.fit(train_dataset,
            ...            epochs=2,
            ...            batch_size=64)
            >>> # evaluate
            >>> engine.evaluate(valid_dataset,
            ...                 batch_size=64)
            >>> # predict
            >>> engine.predict(valid_dataset,
            ...                batch_size=64)
            >>> # save
            >>> engine.save("./my_model")
            >>> # load
            >>> engine.load("./my_model")

    """

    def __init__(
        self,
        model: Layer | Callable[..., Any] | None = None,
        loss: Layer | Callable[..., Any] | Tensor | None = None,
        optimizer: Optimizer | None = None,
        metrics: Metric | Sequence[Metric] | None = None,
        cluster: Cluster | None = None,
        strategy: Strategy | None = None,
    ) -> None:
        if (
            model
            and not isinstance(model, paddle.nn.Layer)
            and not callable(model)
        ):
            raise TypeError(
                "'model must be sub classes of `paddle.nn.Layer` or any callable function."
            )
        self._model = model
        self._parameter_list = (
            None if not model else [p.name for p in model.parameters()]
        )

        if (
            loss
            and not isinstance(loss, (paddle.nn.Layer, Variable))
            and not callable(loss)
        ):
            raise TypeError(
                "'loss' must be sub classes of `paddle.nn.Layer` or any callable function or a Variable."
            )
        self._loss = loss

        if optimizer and not isinstance(
            optimizer,
            (paddle.optimizer.Optimizer),
        ):
            raise TypeError(
                "'optimizer' must be object of class `paddle.optimizer.Optimizer`"
            )
        # NOTE(ljz) Not support parameter groups
        param_list = []
        if optimizer is not None and (
            optimizer._parameter_list is not None
            and len(optimizer._parameter_list) > 0
            and not isinstance(optimizer._parameter_list[0], dict)
        ):
            for p in optimizer._parameter_list:
                if not p.stop_gradient:
                    param_list.append(p)
        self._parameter_name_list = [p.name for p in param_list]
        self._optimizer = auto_utils.validate_opt(optimizer)

        metrics = metrics or []
        for metric in auto_utils.to_list(metrics):
            if metric and not isinstance(metric, Metric):
                raise TypeError(
                    f"{metric.__class__.__name__} is not sub class of Metric"
                )
        self._metrics = auto_utils.to_list(metrics)

        if cluster and not isinstance(cluster, Cluster):
            raise TypeError(
                "'cluster' must be the object or class `paddle.distributed.auto_parallel.Cluster`"
            )

        if strategy and not isinstance(strategy, Strategy):
            raise TypeError(
                "'strategy' must be object of class `paddle.distributed.auto_parallel.Strategy`"
            )
        self._strategy = strategy or Strategy()

        self._logger = get_logger(logging.INFO)

        self._json_config = None
        if cluster:
            self._cluster = cluster
        else:
            auto_config = None
            if os.getenv("PADDLE_AUTO_PARALLEL_CONFIG"):
                try:
                    path = os.getenv("PADDLE_AUTO_PARALLEL_CONFIG")
                    with open(path, "r") as f:
                        self._json_config = json.load(f)
                except Exception as e:
                    self._logger.info(
                        "Load json failed, please check json file, engine will run default config."
                    )
                    self._json_config = None
            else:
                if os.getenv("PADDLE_AUTO_CLUSTER"):
                    auto_config = int(os.getenv("PADDLE_AUTO_CLUSTER"))
            self._cluster = get_default_cluster(self._json_config, auto_config)

        if self._cluster is None:
            raise TypeError(
                "'cluster' must be the object or class `paddle.distributed.auto_parallel.Cluster`"
            )

        # for compute cost
        # TODO: remove _fwd_main_progs and _orig_optimizer and _pir_main_progs
        self._fwd_dist_contexts = {}
        self._fwd_main_progs = {}
        self._startup_progs = {}
        self._pir_dist_main_progs = {}
        self._pir_dist_startup_progs = {}
        self._pir_dense_main_progs = {}
        self._pir_fetch_values = []
        self._pir_user_defined_fetch_names = []
        self._orig_optimizer = copy.deepcopy(self._optimizer)

        self._executor = None
        self._cur_rank = paddle.distributed.get_rank()
        self._nranks = paddle.distributed.get_world_size()
        self._saver = DistributedSaver()

        self._orig_main_prog = static.default_main_program()
        self._orig_startup_prog = static.default_startup_program()
        self._orig_dist_context = get_default_distributed_context()
        self._dist_contexts = {}
        self._planners = {}
        self._has_prepared = {"train": False, "eval": False, "predict": False}
        self._has_prepared_reader = {
            "train": False,
            "eval": False,
            "predict": False,
        }
        self._inputs_spec = []
        self._labels_spec = []
        self._inputs = []
        self._labels = []
        self._losses = []

        self._mode = None
        self._skip_build = False
        self._outside_dataloader = False
        self._planned_mode = None
        self._dygraph_mode = False
        self._tuning = self._strategy.tuning
        self._acc_steps = 1
        self._job_plan = None
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]
        if self._strategy.gradient_merge.enable:
            self._acc_steps = self._strategy.gradient_merge.k_steps
        elif self._strategy.pipeline.enable:
            self._acc_steps = self._strategy.pipeline.accumulate_steps

        if (
            self._strategy.pipeline.enable
            and self._strategy.pipeline.schedule_mode == "1F1B"
        ):
            assert (
                os.getenv("CUDA_MODULE_LOADING") != "LAZY"
            ), "EXP_CUDA_MODULE_LOADING_LAZY not supported in 1F1B pipeline."

        self.history = None

        paddle.framework.set_flags({'FLAGS_new_executor_sequential_run': 1})
        paddle.framework.set_flags({'FLAGS_new_executor_static_build': 1})

        is_pir_mode = os.environ.get("FLAGS_enable_pir_in_executor", None)
        if is_pir_mode is None:
            paddle.framework.set_flags({'FLAGS_enable_pir_in_executor': 1})

        self.enable_job_schedule_profiler = False

        self.fused_ffn_qkv = None

    # get dist input spec from shard dataloader
    def _prepare_data_spec_from_dataloader(self, dataloader):
        inputs_spec = []
        labels_spec = []
        data = next(dataloader())
        if hasattr(dataloader, "batch_sampler"):
            batch_sampler = dataloader.batch_sampler
        else:
            batch_sampler = dataloader._dataloader.batch_sampler
        if hasattr(batch_sampler, "set_epoch"):
            # Get data from DataLoader iterator directly may affect data generation randomness
            # of BatchSampler when `Shuffle=True`. It may cause difference of data feeding
            # between dynamic and to_static mode.
            batch_sampler.set_epoch(0)
        if isinstance(data, dict):
            data = list(data.values())
            if len(data) >= 2:
                labels = data.pop()
                inputs = data
            else:
                raise ValueError(
                    f"Data should be a dict at least two keys, but received {len(data)}."
                )
        elif isinstance(data, (list, tuple)):
            if len(data) >= 2:
                labels = data.pop()
                inputs = data
            else:
                raise ValueError(
                    f"Data should be a dict or list at list two element, but received {len(data)}."
                )
        else:
            raise TypeError(
                f"Data should be a dict or list, but received {type(data)}."
            )
        if not isinstance(inputs, (list, tuple)):
            inputs = auto_utils.to_list(inputs)
        labels = auto_utils.to_list(labels)

        def flatten_list(nested_list):
            flat_list = []
            for item in nested_list:
                if isinstance(item, (list, tuple)):
                    flat_list.extend(flatten_list(item))
                else:
                    flat_list.append(item)
            return flat_list

        # flatten [[1,2],3] - > [1,2,3]
        inputs = flatten_list(inputs)
        if inputs is not None:
            for i, item in enumerate(inputs):
                assert item is not None, "Receive None input."
                name = "input" + str(i)
                inputs_spec.append(
                    DistributedInputSpec.from_dtensor(item, name)
                )
        if labels is not None:
            for i, item in enumerate(labels):
                assert item is not None, "Receive None input."
                name = "label" + str(i)
                labels_spec.append(
                    DistributedInputSpec.from_dtensor(item, name)
                )

        inputs_spec = self._validate_spec(inputs_spec)
        labels_spec = self._validate_spec(labels_spec)
        return inputs_spec, labels_spec

    def _prepare_data_spec(self, data, split, batch_size):
        inputs_spec = []
        labels_spec = []
        if isinstance(data, paddle.io.IterableDataset):
            if split is None:
                inputs, labels = next(iter(data))
            else:
                sample = next(iter(data))
                inputs = sample[:split]
                labels = sample[split:]
        elif isinstance(data, paddle.io.Dataset):
            if split is None:
                inputs, labels = data[0]
            else:
                sample = data[0]
                inputs = sample[:split]
                labels = sample[split:]
        else:
            raise TypeError(
                f"Data should be a Dataset or IterableDataset, but received {type(data).__name__}."
            )
        inputs = auto_utils.to_list(inputs)
        labels = auto_utils.to_list(labels)

        num_shards = self._strategy.dataset.num_shards

        def _adjust_item_spec(num_shards, spec):
            if num_shards > 1 and len(spec.shape) > 1:
                spec.shape[0] = spec.shape[0] * num_shards

        def _infer_item_spec(item, name, batch_size, specs):
            if isinstance(item, np.ndarray):
                spec = InputSpec.from_numpy(item, name)
                if batch_size is None:
                    _adjust_item_spec(num_shards, spec)
                    specs.append(spec)
                else:
                    specs.append(spec.batch(batch_size))
            elif isinstance(item, (Variable, core.eager.Tensor)):
                spec = InputSpec.from_tensor(item, name)
                _adjust_item_spec(num_shards, spec)
                if batch_size is None:
                    specs.append(spec)
                else:
                    specs.append(spec.batch(batch_size))
            elif isinstance(item, numbers.Number):
                specs.append(InputSpec([batch_size], type(item), name))
            else:
                raise TypeError(
                    f"The sample's dtype returned of dataset should be number, np.ndarray or Tensor, but got {type(item).__name__}"
                )

        if inputs is not None:
            for i, item in enumerate(inputs):
                assert item is not None, "Receive None input."
                name = "input" + str(i)
                _infer_item_spec(item, name, batch_size, inputs_spec)
        if labels is not None:
            for i, item in enumerate(labels):
                assert item is not None, "Receive None input."
                name = "label" + str(i)
                _infer_item_spec(item, name, batch_size, labels_spec)

        inputs_spec = self._validate_spec(inputs_spec)
        labels_spec = self._validate_spec(labels_spec)
        return inputs_spec, labels_spec

    def _prepare_data_tensor(self, inputs_spec, labels_spec, inputs, labels):
        if in_dynamic_mode() or self._dygraph_mode:
            raise ValueError("Only support static graph mode.")

        if inputs_spec:
            assert isinstance(
                inputs_spec, list
            ), f"inputs should be list, but received {type(inputs_spec)}"
            assert isinstance(
                inputs, list
            ), f"inputs should be list, but received {type(inputs)}"
            assert len(inputs_spec) == len(
                inputs
            ), "the number of `inputs_spec` should be equal to `inputs`'s."
            for input_spec, input in zip(inputs_spec, inputs):
                if input_spec.shape != input.shape:
                    input.desc.set_shape(input_spec.shape)
        if labels_spec:
            assert isinstance(
                labels_spec, list
            ), f"labels should be list, but received {type(labels_spec)}"
            assert isinstance(
                labels, list
            ), f"labels should be list, but received {type(labels)}"
            assert len(labels_spec) == len(
                labels
            ), "the number of `labels_spec` should be equal to `labels`'s."
            for label_spec, label in zip(labels_spec, labels):
                if label_spec.shape != label.shape:
                    label.desc.set_shape(label_spec.shape)

        return inputs, labels

    def _prepare_reader(self, feed_list=[]):
        dist_context = self._dist_contexts[self._mode]
        dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: this list may be changed if Paddle changes the existing rules.
        related_reader_ops = [
            "create_py_reader",
            "create_double_buffer_reader",
            "read",
        ]
        # remove the first three ops if multiple run fit/evaluate/predict
        if dist_main_block.ops[0].type == 'create_py_reader':
            for i in range(len(related_reader_ops)):
                if dist_main_block.ops[0].type in related_reader_ops:
                    dist_main_block._remove_op(0, sync=False)
        dist_main_block._sync_with_cpp()
        # Step 1: find the reader ops
        reader_op_indices = []
        for idx, op in enumerate(dist_main_block.ops):
            if op.type in related_reader_ops:
                reader_op_indices.append(idx)
        # Step 2: insert the new reader ops to cpp
        # record the read ops' desc to insert to program of forward task_node
        read_ops_desc = []
        new_reader_ops = []
        for idx in reversed(reader_op_indices):
            new_op_desc = dist_main_block.desc._prepend_op()
            new_op_desc.copy_from(dist_main_block.ops[idx].desc)
            read_ops_desc.append(new_op_desc)
            new_op = Operator(
                dist_main_block, new_op_desc, type=new_op_desc.type()
            )
            new_reader_ops.append(new_op)
            dist_op = DistributedOperator(new_op)
            dist_context.add_dist_op_for_program(dist_op)
        # Step 3: insert the new reader ops to python
        for new_op in new_reader_ops:
            dist_main_block.ops.insert(0, new_op)
        for i in range(len(reader_op_indices)):
            reader_op_indices[i] += len(reader_op_indices)
        # Step 4: remove the old reader ops from python and cpp
        for idx in reversed(reader_op_indices):
            op = dist_main_block.ops.pop(idx)
            dist_main_block.desc._remove_op(idx, idx + 1)
        dist_main_block._sync_with_cpp()
        self._has_prepared_reader[self._mode] = True

    def _prepare_feed(self, data, user_feeds, mode):
        feeds = {}
        if data is not None:
            if isinstance(data, (list, tuple)):
                if len(data) == 1 and isinstance(data[0], dict):
                    for name, value in data[0].items():
                        feeds[name] = value
                else:
                    raise ValueError(f"Unsupported data {data}")
            elif isinstance(data, dict):
                for name, value in data.items():
                    feeds[name] = value
            else:
                raise ValueError(f"Unsupported data {data}")
        if user_feeds is not None:
            assert isinstance(
                user_feeds, dict
            ), f"user_feeds must be a dict, but receive {type(user_feeds).__name__}"
            for name, data in user_feeds.items():
                feeds[name] = data
        return feeds

    def _prepare_fetch(self, user_fetches, mode):
        if user_fetches is not None:
            assert isinstance(
                user_fetches, list
            ), f"user_fetches must be a list, but receive {type(user_fetches).__name__}"
        fetch_names = []
        fetch_indices = []

        # TODO(2024-Q2)
        if self._in_pir_mode:
            return fetch_names, fetch_indices

        def _process_fetch_group(group_name, var_list):
            group_indices = []
            for var in var_list:
                # Remove duplicate var_names
                if self._is_local_var(var):
                    var_name = _to_name_str(var)
                    if var_name not in fetch_names:
                        fetch_names.append(var_name)
                    group_indices.append(fetch_names.index(var_name))
            fetch_indices.append(group_indices)

        dist_context = self._dist_contexts[mode]
        fetch_vars = dist_context.serial_fetch_vars
        if mode != "predict":
            _process_fetch_group("loss", fetch_vars["loss"])
        if mode != "predict":
            metrics = fetch_vars["metrics"]
            for i, var_list in enumerate(metrics):
                _process_fetch_group("metrics_" + str(i), var_list)
        if mode == "predict":
            _process_fetch_group("outputs", fetch_vars["outputs"])
        for usr_fetch in user_fetches or []:
            var_name = _to_name_str(usr_fetch)
            fetch(var_name)
        user_fetches_collection = [
            item[1] for item in get_collection(CollectionNames.FETCHES)
        ]
        var_list = user_fetches_collection or []
        _process_fetch_group("fetches", var_list)
        return fetch_names, fetch_indices

    def _prepare_logger(
        self,
        outs,
        epoch=None,
        step=None,
        lr=None,
        fetch_names=None,
        fetch_indices=None,
        mode=None,
    ):
        logs = {}
        if epoch is not None:
            logs["epoch"] = epoch
        if step is not None:
            logs["step"] = step + 1
        if lr is not None:
            logs["lr"] = lr
        group_idx = 0
        if mode != "predict":
            # logging loss
            loss_indices = fetch_indices[group_idx]
            assert len(loss_indices) <= 1
            for idx in loss_indices:
                logs["loss"] = outs[idx]
            group_idx += 1
            # logging metrics
            dist_context = self._dist_contexts[mode]
            metric_vars = dist_context.serial_fetch_vars["metrics"]
            if metric_vars:
                for metric in self._metrics:
                    metrics_indices = fetch_indices[group_idx]
                    metric_out = []
                    for idx in metrics_indices:
                        metric_out.append(outs[idx])
                    if metric_out:
                        metric.update(*metric_out)
                        results = metric.accumulate()
                        for i, res in enumerate(auto_utils.to_list(results)):
                            logs[metric.name()[i]] = res
                    group_idx += 1
        # logging outputs
        elif mode == "predict":
            outputs_indices = fetch_indices[group_idx]
            logs_out = {}
            for idx in outputs_indices:
                logs_out[f"out{idx}"] = outs[idx]
            logs["outputs"] = logs_out
            group_idx += 1
        # logging user fetches
        collect_fetches = get_collection(CollectionNames.FETCHES)
        logs_fetch = {}
        for name, var_name in collect_fetches:
            if var_name in fetch_names:
                idx = fetch_names.index(var_name)
                logs_fetch[name or var_name] = outs[idx]
        logs["fetches"] = logs_fetch
        return logs

    def _parallel_pir(self, mode):
        """A concise and light weight parallel transform for auto parallel in pir mode.
        Its logic consist of Four parts:
            1. Complete program: build a completion program with forward-backward-optimizer from a forward program. (if in train mode, maybe re-placed.)
            2. Parallelism completion: rule-based entire-graph sharding propagation(Semi-Auto) Or algorithm/random-based parallel search(Fully-Auto).
            3. Graph partition: Partition(Pipeline-like parallel) and Reshard Pass(SPMD parallel).
            4. Parallel related Optimization Pass. (maybe re-placed.)

        It is experimental and subject to change.
        """
        mix_fw_program = self._fwd_main_progs[mode]
        startup_program = self._startup_progs[mode]

        # TODO(zhangbo) Open fused_ffn/fused_attention_qkv pass
        if os.getenv("FLAGS_enable_fused_ffn_qkv_pass") in [
            'True',
            'true',
            '1',
        ]:
            self.fused_ffn_qkv = fuse_attention_ffn_qkv_pass(
                startup_program,
                mix_fw_program,
                self.concrete_program,
                mode="all",
            )

            # update self._parameter_name_list after fused_ffn_qkv, otherwise opt stage will not update fused params
            for k in self.fused_ffn_qkv.keys():
                for fusion in self.fused_ffn_qkv[k]:
                    for after_fuse_name, before_fuse_params in fusion.items():
                        index = self._parameter_name_list.index(
                            before_fuse_params[0].name
                        )
                        self._parameter_name_list.insert(index, after_fuse_name)
                        for before_fuse_param in before_fuse_params:
                            self._parameter_name_list.remove(
                                before_fuse_param.name
                            )

        forward_op_start_idx = 0
        backward_op_start_idx = -1
        opt_op_start_idx = -1
        # Part 1: Complete program
        # Step 1.1: Mix2Dist Pass
        # TODO(JZ-LIANG) regulization pass with pass management.
        dist_program = mix_fw_program.clone()
        apply_mix2dist_pass(dist_program)
        if self._strategy.mp_optimization.replace_with_parallel_cross_entropy:
            auto_parallel_replace_with_parallel_cross_entropy_pass = new_pass(
                "replace_with_parallel_cross_entropy", {}
            )
            auto_parallel_replace_with_parallel_cross_entropy_pass.apply(
                [dist_program], [startup_program]
            )

        set_all_ops_op_role(dist_program.global_block(), OpRole.Forward)
        if (
            self._strategy.pipeline.enable
            and self._strategy.pipeline.schedule_mode == "VPP"
        ):
            complete_chunk_id(
                dist_program, startup_program, self._strategy.pipeline
            )

        if self._strategy.mp_optimization.replace_with_c_embedding:
            config = {}
            config["concrete_program"] = self.concrete_program
            auto_parallel_c_embedding_pass = new_pass(
                "auto_parallel_c_embedding_pass", config
            )
            auto_parallel_c_embedding_pass.apply(
                [dist_program], [startup_program]
            )

        # Step 1.2: pir backward
        if mode == "train" and self._loss and self._optimizer:
            loss = dist_program.get_output_value_by_name(self._loss_names[0])
            if loss.initialized():
                with static.program_guard(dist_program, startup_program):
                    if self._strategy.amp.enable:
                        self._strategy.amp.level = (
                            self._strategy.amp.level.upper()
                        )
                        amp_lists = paddle.static.amp.decorator.AutoMixedPrecisionLists(
                            custom_white_list=self._strategy.amp.custom_white_list,
                            custom_black_list=self._strategy.amp.custom_black_list,
                            dtype=self._strategy.amp.dtype,
                        )
                        self._optimizer._sorted = False
                        parameter_value_list = [
                            dist_program.get_parameter_value_by_name(pname)
                            for pname in self._parameter_name_list
                        ]

                        self._optimizer = paddle.static.amp.decorator.OptimizerWithMixedPrecision(
                            optimizer=self._optimizer,
                            amp_lists=amp_lists,
                            level=self._strategy.amp.level,
                            dtype=self._strategy.amp.dtype,
                            init_loss_scaling=self._strategy.amp.init_loss_scaling,
                            incr_every_n_steps=self._strategy.amp.incr_every_n_steps,
                            decr_every_n_nan_or_inf=self._strategy.amp.decr_every_n_nan_or_inf,
                            incr_ratio=self._strategy.amp.incr_ratio,
                            decr_ratio=self._strategy.amp.decr_ratio,
                            use_dynamic_loss_scaling=self._strategy.amp.use_dynamic_loss_scaling,
                            use_amp_guard=self._strategy.amp.use_fp16_guard,
                            use_master_grad=self._strategy.amp.use_master_grad,
                            use_promote=self._strategy.amp.use_promote,
                        )
                        with auto_complete_op_role(
                            dist_program, OpRole.Forward
                        ):
                            # bfloat16 needs no scaler
                            scaler = paddle.amp.GradScaler(
                                init_loss_scaling=self._strategy.amp.init_loss_scaling,
                                incr_ratio=self._strategy.amp.incr_ratio,
                                decr_ratio=self._strategy.amp.decr_ratio,
                                incr_every_n_steps=self._strategy.amp.incr_every_n_steps,
                                decr_every_n_nan_or_inf=self._strategy.amp.decr_every_n_nan_or_inf,
                                use_dynamic_loss_scaling=self._strategy.amp.use_dynamic_loss_scaling,
                                enable=self._strategy.amp.enable
                                and self._strategy.amp.dtype != 'bfloat16',
                            )
                            scaled = scaler.scale(loss)
                        optimizer_ops, params_grads = scaler.minimize(
                            self._optimizer,
                            scaled,
                            parameter_list=parameter_value_list,
                        )
                    else:
                        with auto_complete_op_role(
                            dist_program, OpRole.Backward
                        ):
                            params_grads = (
                                paddle.autograd.ir_backward.append_backward(
                                    loss
                                )
                            )
                        with auto_complete_op_role(
                            dist_program, OpRole.Optimize
                        ):
                            self._optimizer._apply_optimize(
                                loss, startup_program, params_grads=params_grads
                            )
            else:
                self._logger.info(
                    "loss value is not found, skip append backward."
                )

        # re-run apply_mix2dist_pass to dist accumulator.
        apply_mix2dist_pass(dist_program)
        if mode == "train" and self._strategy.recompute.enable:
            config = copy.deepcopy(self._strategy.recompute.to_dict())
            auto_parallel_recompute_pir_pass = new_pass(
                "auto_parallel_recompute_pir", config
            )
            auto_parallel_recompute_pir_pass.apply(
                [dist_program], [startup_program]
            )

        # Part 2: Parallelism search (for full auto-parallel)
        # NOTE make all parallelis search logic work as Pass,
        # and all the Pass in this Part should be optional to allow consistence in dynamic and static mode.
        if self._strategy.auto_mode == "semi-auto":
            # TODO(xxxx) Step 2.1 Entire Graph Completion in Pir.
            # dist_program = apply_completion_pass(dist_program)
            pass
        elif self._strategy.auto_mode == "random" or "full_random":
            # TODO(caozhou) Step 2.3 Basic Random / MCMC Algorithm for Fully Auto Parallel Search.
            # dist_program = apply_mcmc_parallel_search_pass(dist_program)
            pass
        elif self._strategy.auto_mode == "pattern-based":
            # TODO(caozhou) Step 2.3 pattern based Algorithm for Fully Auto Parallel Search.
            # dist_program = apply_pattern_based_parallel_search_pass(dist_program)
            pass
        else:
            raise ValueError("auto_mode [] is not supported yet.".format())

        # Part 3: Graph partition
        # TODO(JZ-LIANG) Step 3.1: Partition Pass
        #   insert reshard op if operand tensor's placements is different from what the cumsumer op need.
        #   Partition the computation graph into different pipeline stage if need.
        apply_partition_pass(dist_program)

        if mode == "train" and self._loss and self._optimizer:
            global_params_grads = params_grads
        else:
            global_params_grads = []
            params_grads = []

        # TODO(hitywt) Step 3.2: Reshard Pass
        #   resolute the reshard op into special collective operation.
        #   collect the communicator created during resolution.
        ReshardPasses.apply_reshard_pass(dist_program, global_params_grads)

        # Note(luchang): When using VPP pipeline pass, we need to split the whole graph into
        # multiple chunks and adjust the process mesh accordingly. Here, we need to store the
        # distributed information of the entire graph for later resharding of the dynamic graph parameters.
        all_params = dist_program.global_block().all_parameters()
        self.program_helper.cache_whole_graph_dist_attr(all_params)

        RemovePasses.apply_all(dist_program, startup_program, params_grads)

        # Part 4: Optimization Pass
        # NOTE Only those Optimization Pass that related to Parallelism (need dist attr) should be placed here and all the Pass should be Optional.

        # TODO(xxxx) Step 4.1 DP Optimization Pass
        if self._strategy.dp_optimization.enable:
            # dist_program = apply_dp_optimization_pass(dist_program)
            pass

        # TODO(xxxx) Step 4.2 SP Optimization Pass
        if self._strategy.sp_optimization.enable:
            # dist_program = apply_sp_optimization_pass(dist_program)
            pass

            # TODO(xxxx) Step 4.3 Sharding Optimization Pass
            # if self._strategy.sharding_optimization.enable:
            # dist_program = apply_sharding_optimization_pass(dist_program)
            pass

        if mode == "train" and self._strategy.pipeline.enable:
            self._strategy.gradient_merge.enable = True
            self._strategy.gradient_merge.k_steps = (
                self._strategy.pipeline.accumulate_steps
            )
            self._strategy.gradient_merge.avg = True

        if mode == "train" and self._strategy.gradient_merge.enable:
            config = copy.deepcopy(self._strategy.gradient_merge.to_dict())
            config["gradient_sync_after_accumulate"] = True
            config["params_grads"] = global_params_grads

            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config
            )
            auto_parallel_gradient_merge_pass.apply(
                [dist_program], [startup_program]
            )

        if (
            self._strategy.pipeline.enable
            and self._strategy.pipeline.schedule_mode == "VPP"
        ):
            check_chunk_id(dist_program)

        # TODO(JZ-LIANG) Step 4.4 Dist2Dense Pass
        # NOTE All optimization pass that need dist_attr info should be called before Dist2Dense Pass.
        dense_program = dist_program.clone()
        paddle.base.libpaddle.pir.apply_dist2dense_pass(dense_program)
        remove_unuseful_comm_op_pass(dense_program)

        if core._enable_dist_prim_all():
            logging.info("apply decompose in auto parallel")
            with decomp.prim_guard():
                decomp.decompose_dist_program(dense_program)

        if core._enable_auto_recompute():
            logging.info("apply auto_recompute in auto parallel")
            dense_program = decomp.auto_recompute_pir_program(
                dense_program,
                lambda op: bool(op.has_attr('op_role') and op.op_role == 0),
            )

        if (
            self._strategy.fused_passes.fused_passes_list is not None
            and "fused_gemm_epilogue_pass"
            in self._strategy.fused_passes.fused_passes_list
        ):
            pm = pir.PassManager()
            pm.add_pass("fused_gemm_epilogue_pass", {})
            pm.run(dense_program)
            self._strategy.fused_passes.fused_passes_list.remove(
                "fused_gemm_epilogue_pass"
            )

        if self._strategy.pipeline.enable:
            self._job_plan = pipeline_pass(
                [dense_program], [dense_program], self._strategy.pipeline
            )
        elif mode == "train" and self._strategy.gradient_merge.enable:
            sub_programs = _split_program_into_forward_backward_optimize(
                dense_program
            )
            job_types = ["forward", "backward", "optimize"]

            # If gradient_merge is enabled, we need to multiply the job list by k_steps.
            # When k_steps is 2, the jobs will be [forward, backward, forward, backward, optimize].
            jobs = []
            for i in range(self._strategy.gradient_merge.k_steps):
                forward_job = core.Job("forward")
                forward_job.set_micro_batch_id(i)
                jobs.append(forward_job)

                backward_job = core.Job("backward")
                backward_job.set_micro_batch_id(i)
                jobs.append(backward_job)

            opt_job = core.Job("optimize")
            opt_job.set_micro_batch_id(0)
            jobs.append(opt_job)

            type_to_program = set_skip_gc_vars(
                self._strategy.gradient_merge.k_steps,
                job_types,
                sub_programs,
                jobs,
            )
            self._job_plan = core.Plan(jobs, type_to_program)

        if (
            self._strategy.fused_passes.fused_passes_list is not None
            and self._strategy.fused_passes.fused_passes_list
        ):
            pm = pir.PassManager()
            for p in self._strategy.fused_passes.fused_passes_list:
                # Temporary implementation, it will be refined when auto_parallel refactored
                if p == 'eliminate_transpose':
                    from paddle.distributed.auto_parallel.static.pir_pass import (
                        eliminate_transpose_by_reshape,
                    )

                    if self._job_plan is None:
                        eliminate_transpose_by_reshape(dense_program)
                    else:
                        for job_type in self._job_plan.job_types():
                            ir_program = self._job_plan.ir_program(job_type)
                            eliminate_transpose_by_reshape(ir_program)

                else:
                    pm.add_pass(p, {})

            if self._job_plan is None:
                pm.run(dense_program)
            else:
                for job_type in self._job_plan.job_types():
                    ir_program = self._job_plan.ir_program(job_type)
                    pm.run(ir_program)

        remove_unuseful_comm_op_pass(dense_program)
        self._pir_dense_main_progs[mode] = dense_program
        self._pir_dist_main_progs[mode] = dist_program
        self._pir_dist_startup_progs[mode] = startup_program

    def _prepare_program(self, mode, init_parameters=True):
        if self._in_pir_mode:
            with paddle.amp.auto_cast(
                enable=self._strategy.amp.enable,
                custom_white_list=self._strategy.amp.custom_white_list,
                custom_black_list=self._strategy.amp.custom_black_list,
                level=self._strategy.amp.level,
                dtype=self._strategy.amp.dtype,
                use_promote=self._strategy.amp.use_promote,
            ):
                self._build(mode)
            self._parallel_pir(mode)
            # Init comm
            self._init_comm()
            # startup program
            self._initialize(mode, init_parameters)
            self._has_prepared[mode] = True
            return

        # legacy program
        # Do the build process
        self._build(mode)
        # Do the planning process
        self._plan(mode)
        # Do the parallel process
        self._parallel(mode)
        # Init comm
        self._init_comm()
        # startup program
        self._initialize(mode, init_parameters)
        # mark main program for further decompose
        self._mark_prim(mode)
        self._has_prepared[mode] = True

    def _process_dist_input_specs(self):
        if isinstance(self._inputs_spec[0], DistributedInputSpec):

            def _create_dist_input_var(input_var, input_spec):
                dist_tensor = DistributedTensor(input_var)

                dist_tensor.dist_attr.process_mesh = input_spec.mesh
                dist_tensor.dist_attr.dims_mapping = input_spec.dims_mapping
                dist_tensor.dist_attr.mark_annotated("process_mesh")
                dist_tensor.dist_attr.mark_annotated("dims_mapping")
                default_dist_ctx = get_default_distributed_context()
                default_dist_ctx.add_dist_tensor_for_program(dist_tensor)

            for index in range(len(self._inputs)):
                input_var = self._inputs[index]
                input_spec = self._inputs_spec[index]
                _create_dist_input_var(input_var, input_spec)

            for index in range(len(self._labels)):
                input_var = self._labels[index]
                input_spec = self._labels_spec[index]
                _create_dist_input_var(input_var, input_spec)

    def _build(self, mode):
        if in_dynamic_mode() or self._dygraph_mode:
            paddle.disable_static()
            self._dygraph_mode = True
            self._logger.info("Building model with 'to_static' method.")

            self.program_helper = ProgramHelper(
                self._model,
                self._loss,
                self._metrics,
                self._inputs_spec,
                self._labels_spec,
            )
            # build forward main program
            with utils.unique_name.guard():
                self.program_helper.build_program(mode)

            self.concrete_program = self.program_helper.concrete_program
            serial_main_prog = self.program_helper.main_program
            serial_startup_prog = self.program_helper.startup_program

            self._inputs = self.program_helper.input_vars
            self._labels = self.program_helper.label_vars
            # self._process_dist_input_specs()
            outputs = self.program_helper.output_vars
            self._losses = self.program_helper.loss_vars
            self._loss_names = self.program_helper.loss_names
            metrics = self.program_helper.metric_vars

            paddle.enable_static()
        else:
            # build program in static mode
            dist_context = self._dist_contexts.get(mode, None)
            if dist_context is not None:
                return

            outputs = []
            metrics = []
            self._losses = []
            serial_main_prog = self._orig_main_prog.clone()
            serial_startup_prog = self._orig_startup_prog.clone()
            if not self._skip_build:
                with static.program_guard(
                    serial_main_prog, serial_startup_prog
                ), utils.unique_name.guard():
                    self._inputs = [
                        s._create_feed_layer() for s in self._inputs_spec
                    ]
                    self._labels = [
                        s._create_feed_layer() for s in self._labels_spec
                    ]

                    outputs = auto_utils.to_list(self._model(*self._inputs))

                    if mode != "predict" and self._loss:
                        assert isinstance(
                            self._loss, paddle.nn.Layer
                        ) or callable(
                            self._loss
                        ), "the type of `loss` of the Engine arguments should be sub classes of `paddle.nn.Layer` or any callable function."
                        self._losses = auto_utils.to_list(
                            self._loss(*(outputs + self._labels))
                        )

                    if mode != "predict" and (outputs or self._labels):
                        for metric in self._metrics:
                            metrics.append(
                                auto_utils.to_list(
                                    metric.compute(*(outputs + self._labels))
                                )
                            )
            elif mode == "train":
                assert isinstance(
                    self._loss, Variable
                ), "the type of `loss` of the Engine arguments should be Variable."
                self._losses = auto_utils.to_list(self._loss)

        # TODO(zhiqiu): distributed_context is no longer used in pir_program
        # so, just return here and need to reimplement the logics below
        if self._in_pir_mode:
            # TODO(ljz): pir not support clone_for_test,
            # so we need to update the method to create eval/test program in engine.

            # if mode != "train":
            #     self._fwd_main_progs[mode] = serial_main_prog.clone(
            #         for_test=True
            #     )
            # else:

            # concrete_program: <class 'paddle.jit.dy2static.program_translator.ConcreteProgram'>
            # serial_main_prog:  <class 'paddle.base.libpaddle.pir.Program'>
            self._fwd_main_progs[mode] = serial_main_prog
            self._startup_progs[mode] = serial_startup_prog
            return

        default_ctx = get_default_distributed_context()
        if not default_ctx.has_annotation:
            # We build the world process group because the data parallel
            # needs all ranks by default.
            new_process_group(list(range(self._nranks)))
            default_ctx.data_parallel = True
            self._inputs = [
                auto_utils.set_data_parallel(var) for var in self._inputs
            ]
            self._labels = [
                auto_utils.set_data_parallel(var) for var in self._labels
            ]

        feed_vars = {"inputs": self._inputs, "labels": self._labels}

        fetch_vars = {
            "outputs": paddle.utils.flatten(outputs),
            "loss": self._losses,
            "metrics": metrics,
        }

        if mode != "train":
            serial_main_prog = serial_main_prog.clone(for_test=True)

        auto_utils.set_recompute_segments(
            self._model, self._losses, self._strategy, serial_main_prog
        )
        self._dist_contexts[mode] = DistributedContext(
            serial_main_prog,
            serial_startup_prog,
            self._optimizer,
            self._losses,
            feed_vars,
            fetch_vars,
            self._cluster,
            self._strategy,
            self._json_config,
        )
        self._fwd_dist_contexts[mode] = DistributedContext(
            serial_main_prog,
            serial_startup_prog,
            self._optimizer,
            self._losses,
            feed_vars,
            fetch_vars,
            self._cluster,
            self._strategy,
            self._json_config,
        )
        self._dist_contexts[mode].gradient_scale = self._strategy.gradient_scale
        self._dist_contexts[mode].gradient_scale_using_allreduce_avg = (
            self._strategy.gradient_scale_using_allreduce_avg
        )
        self._fwd_main_progs[mode] = serial_main_prog.clone()

    def _optimization_tuning(self, mode, dataset, batch_size):
        if not self._tuning.enable:
            raise ValueError("Please set `tuning.enable=True`.")

        assert mode == "train"
        # Do the build process
        self._build(mode)
        # Do the planning process
        self._plan(mode)

        dataset.dp_world_size = self._dp_world_sizes
        dataset.dp_rank = self._dp_ranks

        from .tuner.optimization_tuner import OptimizationTuner

        self._optimization_tuner = OptimizationTuner(
            self._dist_contexts[mode],
            dataset,
            self._inputs_spec,
            self._labels_spec,
            batch_size=batch_size,
            rank=self._cur_rank,
        )

        self._optimization_tuner.tune()

        if self._tuning.run_after_tuning:
            # update the strategy
            self._dist_contexts[mode]._strategy = (
                self._optimization_tuner.get_best_config()
            )

    def _plan(self, mode):
        if self._planned_mode is None:
            self._planned_mode = mode
        elif self._strategy.auto_mode != "semi":
            self._init_dist_context(mode)

        self._planners[mode] = Planner(mode, self._dist_contexts[mode])
        self._planners[mode].plan()

        # infer data parallel info
        inputs_var = self._dist_contexts[mode].serial_feed_vars["inputs"]
        labels_var = self._dist_contexts[mode].serial_feed_vars["labels"]
        block = self._dist_contexts[mode].serial_main_program.global_block()
        # TODO: check this feed_list
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in block.vars:
                feed_list.append(block.vars[var.name])

        self._dp_world_sizes = getattr(self, "_dp_world_sizes", [])
        self._dp_ranks = getattr(self, "_dp_ranks", [])
        if mode in ['eval', 'predice'] or (
            not self._dp_world_sizes and not self._dp_ranks
        ):
            self._dp_world_sizes = []
            self._dp_ranks = []
            for feed_var in feed_list:
                dp_world_size, dp_rank = auto_utils.get_input_split_info(
                    self._cur_rank, feed_var, self._dist_contexts[mode]
                )
                self._dp_world_sizes.append(dp_world_size)
                self._dp_ranks.append(dp_rank)

    def _parallel(self, mode, all_ranks=False):
        # Parallelize program based on the planner's results
        # For now, the completer has to be passed to the Parallelizer,
        # because we may use it to complete the annotation of the backward and update.
        parallelizer = Parallelizer(
            mode,
            self._planners[mode].completer,
            self._dist_contexts[mode],
        )
        if not all_ranks:
            parallelizer.parallel(self._cur_rank, self._parameter_list)
        else:
            parallelizer.parallel_all(self._parameter_list)

    def _init_dist_context(self, mode):
        # Init dist_context['mode'] with the first planned dist_context
        # to guarantee that train/eval/predict mode have same parallel strategy
        dist_context = self._dist_contexts[mode]
        origin_main_prog = dist_context._original_serial_main_program
        ref_mode = self._planned_mode
        ref_dist_context = self._dist_contexts[ref_mode]
        ref_origin_main_prog = ref_dist_context._original_serial_main_program
        ref_blocks = ref_origin_main_prog.blocks
        for ib, block in enumerate(origin_main_prog.blocks):
            for iop, op in enumerate(block.ops):
                ref_op = ref_blocks[ib].ops[iop]
                assert (
                    op.type == ref_op.type
                ), f"'{mode}' mode op '{op.type}' is different with '{ref_mode}' op '{ref_op.type}'. "
                ref_op_dist_attr = (
                    ref_dist_context.get_op_dist_attr_for_program(ref_op)
                )
                dist_context.set_op_dist_attr_for_program(op, ref_op_dist_attr)

    def _init_comm(self):
        if self._nranks > 1:
            if self._in_pir_mode:
                # TODO(hitywt) Initialize the communicator collected in Reshard Pass.
                # pir_init_comms()
                all_process_groups = get_all_process_groups()
                for process_group in all_process_groups:
                    process_group.instantiate()
                return

            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()

            if self._strategy.auto_mode == "full_random":
                auto_utils.initialize_pg_in_full_mode(
                    all_process_groups, self._cur_rank
                )
            else:
                for process_group in all_process_groups:
                    process_group.instantiate()

    def _init_lr(self, main_program):
        # hack to find learning_rate op
        lr_name = None
        for op in main_program.global_block().ops:
            if (
                op.name() == "pd_op.data"
                and 'learning_rate' in op.attrs()["name"]
            ):
                lr_name = op.attrs()["name"]
                break
        if lr_name is not None:
            buffer_tensor = global_scope().var(lr_name).get_tensor()
            if isinstance(self._optimizer._learning_rate, float):
                buffer_tensor.set(
                    np.float32(self._optimizer._learning_rate), self._place
                )

    def _initialize(self, mode, init_parameters=True):
        self._place = _get_device()
        if isinstance(self._place, paddle.framework.CUDAPlace):
            self._place = paddle.framework.CUDAPlace(
                paddle.distributed.ParallelEnv().dev_id
            )

        if self._in_pir_mode:
            # FIXME(ljz) avoid shared same tensor more than once in different mode
            if mode != "train":
                return
            # TODO(2024-Q2)
            # 1. unify random control
            # 2. initialization of non-parameter buffer
            # 3. run startup program for pir
            # 4. lazy init adaption
            # 5. amp init adaption
            # 6. vpp init adaption

            # self._init_lr(self._pir_dense_main_progs[mode])
            self.program_helper.init_pir(
                self._pir_dist_main_progs[mode], self._place
            )
            changed_output_op_list = []
            if self._executor is None:
                self._executor = paddle.static.Executor(self._place)
                startup_prog = self._startup_progs[mode].clone()
                dist_main_prog = self._pir_dist_main_progs[mode]
                name_map_value = {}
                for op in dist_main_prog.global_block().ops:
                    if op.name() == "pd_op.data":
                        var_name = op.str_attr("name")
                        assert (
                            var_name not in name_map_value
                        ), f"The value {var_name} in {op} is already exist"
                        name_map_value[var_name] = op.result(0)
                del_ops = []
                block = startup_prog.global_block()
                for op in block.ops:
                    if op.name() == "builtin.set_parameter":
                        var_name = op.str_attr("parameter_name")
                    elif op.name() == "builtin.shadow_output":
                        var_name = op.str_attr("output_name")
                    else:
                        continue
                    scope_var = global_scope().find_var(var_name)
                    if scope_var and scope_var.get_tensor()._is_initialized():
                        param = op.operand_source(0)
                        initial_op = param.get_defining_op()
                        new_param = block.add_kwarg(var_name, param.type())
                        new_param.persistable = True
                        new_param.place_attr = scope_var.get_tensor()._place()
                        param.replace_all_uses_with(new_param)
                        del_ops.append(op)
                        del_ops.append(initial_op)
                    elif var_name in name_map_value:
                        local_shape = name_map_value[var_name]._local_shape
                        global_shape = name_map_value[var_name].shape
                        if local_shape != global_shape:
                            src_value = op.operand_source(0)
                            assert src_value.shape == global_shape
                            dst_dist_attr = name_map_value[var_name].dist_attr()
                            if not src_value.is_dist():
                                src_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                    dst_dist_attr.process_mesh,
                                    [-1] * len(src_value.shape),
                                    {},
                                )
                                src_value.set_type(
                                    paddle.base.libpaddle.pir.cvt_to_dist_type(
                                        src_value.type(), src_dist_attr
                                    )
                                )
                            pir.set_insertion_point_after(
                                src_value.get_defining_op()
                            )
                            reshard_var = paddle._C_ops.reshard_v2(
                                src_value, dst_dist_attr
                            )
                            if src_value.persistable:
                                src_value.persistable = False
                                changed_output_op_list.append(op)
                            op.operand(0).set_source(reshard_var)
                for del_op in del_ops:
                    del_op.erase()

                set_all_ops_op_role(startup_prog.global_block(), OpRole.Forward)
                ReshardPasses.apply_reshard_pass(startup_prog)
                paddle.base.libpaddle.pir.apply_dist2dense_pass(startup_prog)
                remove_unuseful_comm_op_pass(startup_prog)

                for op in changed_output_op_list:
                    op.operand_source(0).persistable = True
                self._executor.run(startup_prog)
                if self._job_plan is not None:
                    # pipeline scheduling should be enabled after running
                    # startup program, otherwise the startup program cannot
                    # run correctly.
                    self._executor._set_plan(self._job_plan)
            return

        if self._strategy.seed:
            paddle.seed(self._strategy.seed + self._dp_ranks[0])
            np.random.seed(self._strategy.seed + self._dp_ranks[0])
            random.seed(self._strategy.seed + self._dp_ranks[0])

        dist_context = self._dist_contexts[mode]
        dist_main_program = dist_context.dist_main_programs[self._cur_rank]
        if self._dygraph_mode:
            self.program_helper.init(
                dist_main_program, self._place, dist_context
            )
            # The model's instance variables (not parameters), used in forward function,
            # have been initialized when initialize model in dynamic mode.
            if self._model and len(self._model.buffers()) > 0:
                for buffer in self._model.buffers():
                    if dist_main_program.global_block().has_var(buffer.name):
                        dest_type = (
                            dist_main_program.global_block()
                            .var(buffer.name)
                            .dtype
                        )
                        scope_var = global_scope().find_var(buffer.name)
                        buffer_tensor = (
                            global_scope().var(buffer.name).get_tensor()
                        )
                        if scope_var and buffer_tensor._is_initialized():
                            continue
                        # for amp
                        if dest_type == paddle.bfloat16:
                            buffer_tensor.set(
                                _convert_float_to_bfloat16(
                                    self._place, buffer.numpy()
                                ),
                                self._place,
                            )
                        elif dest_type == paddle.float16:
                            buffer_tensor.set(
                                np.float16(buffer.numpy()), self._place
                            )
                        else:
                            buffer_tensor.set(buffer.numpy(), self._place)

        if self._executor is None:
            self._executor = paddle.static.Executor(self._place)
            uninitialized = []
            dist_startup_prog = dist_context.dist_startup_programs[
                self._cur_rank
            ]
            for var in dist_startup_prog.list_vars():
                scope_var = global_scope().find_var(var.name)
                if scope_var and scope_var.get_tensor()._is_initialized():
                    continue
                uninitialized.append(var)
            # Make sure the number of communication operators is consistent
            commu_ops = []
            if self._nranks > 1:
                for op in dist_startup_prog.global_block().ops:
                    if auto_utils.is_comm_op(op):
                        commu_ops.append(op)
            reserved_vars_and_ops = uninitialized + commu_ops
            if reserved_vars_and_ops:
                prune_startup_prog = dist_startup_prog._prune(
                    reserved_vars_and_ops
                )
                self._executor.run(prune_startup_prog)

            if hasattr(self, "_state_dict") and hasattr(self, "_dist_attr"):
                self._set_state_dict(
                    mode, self._strict, self._state_dict, self._dist_attr
                )

        if self._strategy.reinit:
            self._logger.info("NOTE: parameters will be re-initialized.")
            dist_startup_prog = dist_context.dist_startup_programs[
                self._cur_rank
            ]
            self._executor.run(dist_startup_prog)

    # distributed training combined with prim mechanism (prim is behind of distributed)
    # for local main subprogram after distributed partition,
    # mark _need_decomp=True to tag this program needs to be decomposed
    # get _grad_var_to_var from distributed context and set it to main program for further decomposing in static executor
    def _mark_prim(self, mode):
        if os.getenv("FLAGS_enable_prim_after_distribute") in [
            'True',
            'true',
            '1',
        ]:
            dist_context = self._dist_contexts[mode]
            dist_main_program = dist_context.dist_main_programs[self._cur_rank]
            dist_main_program._need_decomp = True
            grad_var_to_var = auto_utils.get_grad_var_to_var(
                dist_context,
            )
            auto_utils.update_grad_var_to_var(
                dist_main_program, self._strategy, grad_var_to_var
            )
            dist_main_program._grad_var_to_var = grad_var_to_var

    def fit(
        self,
        train_data: Dataset,
        train_sample_split: int | None = None,
        batch_size: int = 1,
        epochs: int = 1,
        steps_per_epoch: int | None = None,
        log_freq: int = 10,
        save_dir: str | None = None,
        save_freq: int = 1,
        valid_data: Dataset | None = None,
        valid_sample_split: int | None = None,
        valid_freq: int = 1,
        valid_steps: int | None = None,
        collate_fn: _CollateFn | None = None,
        callbacks: Sequence[Callback] | None = None,
        verbose: int = 2,
        nvprof_range: list[int] | tuple[int, int] = [-1, -1],
    ) -> None:
        """
        Trains the model for a fixed number of epochs. If `valid_data` is set,
        evaluation will be done at the end of each epoch.

        Args:
            train_data (Dataset): An instance of paddle paddle.io.Dataset. Default: None.
            train_sample_split (int|None, optional): Each sample of the train dataset is assumed
                to be a (input, label) pair by default and has two items. If each sample has
                more than two items, train_sample_split specifies how to split these items into
                input and label. The items before it are input and the left are label. Default: None.
            batch_size (int, optional): The batch size of train_data and valid_data if provided.
                The user's data will be used directly without batching if set to None. Default: 1.
            epochs (int, optional): The number of epochs to train the model. Default: 1.
            steps_per_epoch (int|None, optional): The total number of steps (batches of samples)
                is executed in one epoch before stating the next one. If None, it is equal to
                the number samples in your dataset divided by the batch size. Default: None.
            valid_data (Dataset|None, optional): An instance of paddle paddle.io.Dataset used for
                evaluation at the end of epoch. No evaluation will be done if set to None.
                Default: None. (Unsupported for now)
            valid_freq (int, optional): Only relevant if valid_data is provided. This specifies
                how many training epochs before a new evaluation is performed. Default: 1.
            valid_sample_split (int|None, optional): Only relevant if valid_data is provided.
                Each sample of the valid dataset is assumed to be a (input, label) pair
                by default and has two items. If each sample has more than two items,
                valid_sample_split specifies how to split these items into input and label.
                The items before it are input and the left are label. Default: None.
            valid_steps (int|None, optional): Only relevant if valid_data is provided.
                It is the total number of steps (batches of samples) to draw before
                stopping validation at the end of every epoch. If None, validation will run until the
                `valid_data` dataset is exhausted. The validation will start from the
                beginning of the dataset at each epoch. Default: None.
            collate_fn(callable|None, optional): function to generate mini-batch data by merging
                the sample list, None for only stack each fields of sample in axis
                0. Default None.
            callbacks (Callback|None, optional): A list of `Callback` instances to apply
                during training. Default: None. (Unused for now)
            nvprof_range(list, optional): A list of integers indicating nvprof ranges in form of [start_step, end_step]. Note that if start_step >= end_step, the nvprof will not apply.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.distributed.fleet import auto
                >>> from paddle.vision.datasets import MNIST

                >>> transform = T.Compose([
                ...     T.Transpose(),
                ...     T.Normalize([127.5], [127.5])
                >>> ])
                >>> train_dataset = MNIST(mode='train', transform=transform)

                >>> model = paddle.vision.models.LeNet()
                >>> loss = paddle.nn.CrossEntropyLoss()
                >>> optimizer = paddle.optimizer.Adam(
                ...     learning_rate=0.001, parameters=model.parameters())
                >>> metrics = paddle.metric.Accuracy(topk=(1, 2))

                >>> engine = auto.Engine(model, loss, optimizer, metrics)
                >>> engine.fit(train_dataset,
                ...             epochs=2,
                ...             batch_size=64)
        """
        self._mode = 'train'

        if not self._has_prepared[self._mode]:
            self._inputs_spec, self._labels_spec = self._prepare_data_spec(
                train_data, train_sample_split, batch_size
            )
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        local_batch_size = self._validate_batch_size(batch_size)
        train_dataloader = self._prepare_dataloader(
            train_data,
            return_list=False,
            batch_size=local_batch_size,
            epochs=epochs,
            collate_fn=collate_fn,
        )
        steps_per_epoch = (
            len(train_dataloader)
            if steps_per_epoch is None
            else steps_per_epoch
        )

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        cbks = config_callbacks(
            callbacks,
            engine=self,
            batch_size=local_batch_size,
            epochs=epochs,
            steps=steps_per_epoch,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(),
            acc_step=(
                1 if self._strategy.pipeline.enable else self._acc_steps
            ),  # lr update once every local batch
        )

        cbks.on_begin('train')
        for epoch in range(epochs):
            logs = {}
            cbks.on_epoch_begin(epoch)

            for step, batch in enumerate(train_dataloader):
                batches = self._validate_batch(batch)

                for micro_batch in batches:
                    with paddle.profiler.utils._nvprof_range(
                        iter_id=step,
                        start=nvprof_range[0],
                        end=nvprof_range[1],
                    ):
                        cbks.on_batch_begin('train', step, logs)
                        outs = self._executor.run(
                            self.main_program,
                            feed=micro_batch,
                            fetch_list=fetch_names,
                            use_program_cache=self._strategy.use_cache,
                            return_numpy=self._strategy.return_numpy,
                        )

                        lr = auto_utils.get_lr(self.optimizer)
                        logs = self._prepare_logger(
                            outs,
                            epoch,
                            step,
                            lr,
                            fetch_names,
                            fetch_indices,
                            self._mode,
                        )
                        cbks.on_batch_end('train', step, logs)

                if steps_per_epoch and step >= steps_per_epoch:
                    break

            if valid_data and (epoch + 1) % valid_freq == 0:
                val_logs = self.evaluate(
                    valid_data,
                    valid_sample_split,
                    batch_size,
                    valid_steps,
                    log_freq,
                    collate_fn,
                    callbacks,
                    verbose,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                logs.update(val_logs)
                self._switch_mode("train")
            else:
                self._reset_metrics()

            cbks.on_epoch_end(epoch, logs)

        cbks.on_end('train', logs)
        return self.history

    def evaluate(
        self,
        valid_data: Dataset,
        valid_sample_split: int | None = None,
        batch_size: int = 1,
        steps: int | None = None,
        log_freq: int = 10,
        collate_fn: _CollateFn | None = None,
        callbacks: Sequence[Callback] | None = None,
        verbose: int = 2,
    ) -> dict[str, Any]:
        """
        Evaluate the loss and metrics of the model on evaluation data.

        Args:
            valid_data (Dataset): An instance of paddle paddle.io.Dataset. Default: None.
            valid_sample_split (int|None, optional): Each sample of the eval dataset is assumed
                to be a (input, label) pair by default and has two items. If each sample has
                more than two items, valid_sample_split specifies how to split these items into
                input and label. The items before it are input and the left are label. Default: None.
            batch_size (int, optional): The batch size of valid_data. The user's data will
                be used directly without batching if set to None. Default: 1.
            steps (int|None, optional): It is the total number of steps (batches of samples) to draw before
                stopping evaluation. If None, evaluation will run until the `valid_data` dataset is exhausted.
                The evaluation will start from the beginning of the dataset in each run. Default: None.
            collate_fn(callable|None, optional): function to generate mini-batch data by merging
                the sample list, None for only stack each fields of sample in axis
                0. Default None.
            callbacks (Callback|None, optional): A list of `Callback` instances to apply
                during evaluating. Default: None. (Unused for now)

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.distributed.fleet import auto
                >>> from paddle.vision.datasets import MNIST

                >>> transform = T.Compose([
                ...     T.Transpose(),
                ...     T.Normalize([127.5], [127.5])
                >>> ])
                >>> valid_dataset = MNIST(mode='test', transform=transform)

                >>> model = paddle.vision.models.LeNet()
                >>> loss = paddle.nn.CrossEntropyLoss()
                >>> metrics = paddle.metric.Accuracy(topk=(1, 2))

                >>> engine = auto.Engine(model, loss, metrics=metrics)
                >>> engine.evaluate(valid_dataset, batch_size=64)

        """
        self._mode = 'eval'

        if not self._has_prepared[self._mode]:
            self._inputs_spec, self._labels_spec = self._prepare_data_spec(
                valid_data, valid_sample_split, batch_size
            )
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        local_batch_size = self._validate_batch_size(batch_size)
        valid_dataloader = self._prepare_dataloader(
            valid_data,
            return_list=False,
            batch_size=local_batch_size,
            collate_fn=collate_fn,
        )
        steps_per_epoch = len(valid_dataloader) if steps is None else steps

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        cbks = config_callbacks(
            callbacks,
            engine=self,
            batch_size=local_batch_size,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(),
        )

        eval_steps = steps_per_epoch
        cbks.on_begin(
            'eval', {'steps': eval_steps, 'metrics': self._metrics_name()}
        )
        logs = {}
        for step, batch in enumerate(valid_dataloader):
            batches = self._validate_batch(batch)
            for micro_batch in batches:
                cbks.on_batch_begin('eval', step, logs)
                outs = self._executor.run(
                    self.main_program,
                    feed=micro_batch,
                    fetch_list=fetch_names,
                    use_program_cache=self._strategy.use_cache,
                    return_numpy=self._strategy.return_numpy,
                )

            if steps_per_epoch and step >= steps_per_epoch:
                break
            logs = self._prepare_logger(
                outs, None, step, None, fetch_names, fetch_indices, self._mode
            )
            cbks.on_batch_end('eval', step, logs)
        cbks.on_end('eval', logs)
        self._reset_metrics()
        return logs

    def predict(
        self,
        test_data: Dataset,
        test_sample_split: int | None = None,
        batch_size: int = 1,
        steps: int | None = None,
        collate_fn: _CollateFn | None = None,
        callbacks: Sequence[Callback] | None = None,
        verbose: int = 2,
    ) -> list[Any]:
        """
        Compute the output predictions on testing data.

        Args:
            test_data (Dataset): An instance of paddle paddle.io.Dataset. Default: None.
            test_sample_split (int, optional): Each sample of the test dataset is assumed
                to be a (input, label) pair by default and has two items. If each sample has
                more than two items, test_sample_split specifies how to split these items into
                input and label. The items before it are input and the left are label. Default: None.
            batch_size (int, optional): The batch size of test_data. The user's data will
                be used directly without batching if set to None. Default: 1.
            steps (int, optional): It is the total number of steps (batches of samples) to draw before
                stopping predict. If None, predict will run until the `test_data` dataset is exhausted.
                The predict will start from the beginning of the dataset in each run. Default: None.
            collate_fn(callable, optional): function to generate mini-batch data by merging
                the sample list, None for only stack each fields of sample in axis
                0. Default None.
            callbacks (Callback|None, optional): A list of `Callback` instances to apply
                during testing. Default: None. (Unused for now)

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.distributed.fleet import auto
                >>> from paddle.vision.datasets import MNIST

                >>> transform = T.Compose([
                ...     T.Transpose(),
                ...     T.Normalize([127.5], [127.5])
                >>> ])
                >>> valid_dataset = MNIST(mode='test', transform=transform)

                >>> model = paddle.vision.models.LeNet()

                >>> engine = auto.Engine(model)
                >>> engine.predict(valid_dataset, batch_size=64)
        """
        self._mode = 'predict'

        if not self._has_prepared[self._mode]:
            self._inputs_spec, self._labels_spec = self._prepare_data_spec(
                test_data, test_sample_split, batch_size
            )
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        local_batch_size = self._validate_batch_size(batch_size)
        test_dataloader = self._prepare_dataloader(
            test_data,
            return_list=False,
            batch_size=local_batch_size,
            collate_fn=collate_fn,
        )
        steps_per_epoch = len(test_dataloader) if steps is None else steps

        fetch_names, fetch_indices = self._prepare_fetch(None, mode=self._mode)

        outputs = []
        cbks = config_callbacks(callbacks, engine=self, verbose=verbose)
        test_steps = steps_per_epoch
        cbks.on_begin('predict', {'steps': test_steps})
        logs = {}
        for step, batch in enumerate(test_dataloader):
            batches = self._validate_batch(batch)
            for micro_batch in batches:
                cbks.on_batch_begin('predict', step, logs)
                outs = self._executor.run(
                    self.main_program,
                    feed=micro_batch,
                    fetch_list=fetch_names,
                    use_program_cache=self._strategy.use_cache,
                    return_numpy=self._strategy.return_numpy,
                )

            if steps_per_epoch and step >= steps_per_epoch:
                break
            logs = self._prepare_logger(
                outs, None, step, None, fetch_names, fetch_indices, self._mode
            )
            cbks.on_batch_end('predict', step, logs)
            outputs.append(list(logs["outputs"].values()))
        cbks.on_end('predict', logs)
        return outputs

    def dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
        collate_fn: _CollateFn | None = None,
        num_workers: int = 0,
        use_buffer_reader: bool = True,
        use_shared_memory: bool = True,
        timeout: int = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        epochs: int = 1,
        steps_per_epoch: int | None = None,
        sample_split: int = 1,
        mode: _Mode | None = None,
        places: PlaceLike | Sequence[PlaceLike] | None = None,
    ) -> DistributedDataLoader:
        if mode is not None:
            self.to_mode(mode)

        if not self._has_prepared[self._mode]:
            self._inputs_spec, self._labels_spec = self._prepare_data_spec(
                dataset, sample_split, batch_size
            )
            self._prepare_program(self._mode)
        else:
            self._switch_mode(self._mode)

        batch_size = self._validate_batch_size(batch_size)
        dataloader = self._prepare_dataloader(
            dataset,
            return_list=False,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=use_buffer_reader,
            use_shared_memory=use_shared_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            places=places,
        )
        return dataloader

    def prepare(
        self,
        inputs_spec: InputSpec | None = None,
        labels_spec: InputSpec | None = None,
        inputs: Sequence[Tensor] | None = None,
        labels: Sequence[Tensor] | None = None,
        main_program: Program | None = None,
        startup_program: Program | None = None,
        mode: _Mode | None = None,
        init_parameters: bool = True,
    ) -> None:
        if mode is not None:
            self.to_mode(mode)

        if not self._mode:
            raise ValueError(
                "Please set mode to be prepared with `prepare(mode=...)`"
            )

        if self._has_prepared[self._mode]:
            return

        inputs_spec = self._validate_spec(inputs_spec)
        labels_spec = self._validate_spec(labels_spec)
        inputs = self._validate_vars(inputs)
        labels = self._validate_vars(labels)

        self._orig_main_prog = main_program
        self._orig_startup_prog = startup_program
        if inputs or labels:
            self._skip_build = True
            inputs, labels = self._prepare_data_tensor(
                inputs_spec, labels_spec, inputs, labels
            )
            if self._orig_main_prog is None:
                self._orig_main_prog = static.default_main_program()
            if self._orig_startup_prog is None:
                self._orig_startup_prog = static.default_startup_program()
        elif inputs_spec or labels_spec:
            self._outside_dataloader = True
            if self._orig_main_prog is None:
                self._orig_main_prog = static.default_main_program()
            if self._orig_startup_prog is None:
                self._orig_startup_prog = static.default_startup_program()
        else:
            assert (
                self._inputs_spec and self._labels_spec
            ), "Please call the dataloader(...) before calling prepare(...)"

        self._inputs_spec, self._labels_spec = inputs_spec, labels_spec
        self._inputs, self._labels = inputs, labels
        if not self._has_prepared[self._mode]:
            self._prepare_program(self._mode, init_parameters)
        else:
            self._switch_mode(self._mode)

    def run(
        self,
        data: (
            list[dict[str, Any]] | tuple[dict[str, Any]] | dict[str, Any] | None
        ) = None,
        feed: dict[str, Any] | None = None,
        fetch_list: list[Tensor | str | Operator | Value] | None = None,
        mode: _Mode | None = None,
    ) -> dict[str, Any]:
        if mode is not None:
            self.to_mode(mode)
        feed_dict = self._prepare_feed(data, feed, self._mode)
        fetch_names, fetch_indices = self._prepare_fetch(fetch_list, self._mode)
        if (
            self._outside_dataloader
            and not self._has_prepared_reader[self._mode]
        ):
            self._prepare_reader()

        self._executor.enable_job_schedule_profiler = (
            self.enable_job_schedule_profiler
        )

        # TODO(2024-Q2)
        use_cache = self._strategy.use_cache
        if self._in_pir_mode:
            use_cache = False
            no_fetch = False  # not last rank should not fetch loss in pipeline parallel
            if self._job_plan is None:
                program_for_executor = self.main_program
            else:
                # NOTE: If pipeline scheduling is enabled, The program_for_executor
                # is used to tell the executor where to feed data and add fetch op,
                # not the program to be executed. The ``plan`` object is already
                # constructed, and the programs to be executed are  stored in the
                # ``plan`` object.
                loss_job_type = "forward"
                if self._strategy.pipeline.schedule_mode == "VPP":
                    vpp_degree = self._strategy.pipeline.vpp_degree
                    loss_job_type = f"forward{vpp_degree - 1}"

                program_for_executor = self._job_plan.ir_program(loss_job_type)

            loss_value = program_for_executor.get_output_value_by_name(
                self._loss_names[0]
            )
            if pir.is_fake_value(loss_value):
                no_fetch = True
                fetch_names = []
            else:
                fetch_names = [loss_value]
            fetch_names += self._pir_fetch_values

        outs = self._executor.run(
            self.main_program,
            feed=feed_dict,
            fetch_list=fetch_names,
            use_program_cache=use_cache,
            return_numpy=self._strategy.return_numpy,
        )

        if self._in_pir_mode:
            if no_fetch:
                logs = {"outputs": None, "loss": None}
                start_idx = 0
            else:
                logs = {"outputs": outs[0], "loss": outs[0]}
                start_idx = 1
            for i, name in enumerate(self._pir_user_defined_fetch_names):
                logs[name] = outs[start_idx + i]
            return logs

        logs = self._prepare_logger(
            outs, None, None, None, fetch_names, fetch_indices, self._mode
        )
        return logs

    def get_feed_list(self) -> list[Tensor]:
        dist_context = self._dist_contexts[self._mode]
        dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
        dist_startup_prog = dist_context.dist_startup_programs[self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list, then insert dataloader op with sharded var shape.
        # Cause predict_program does not contain labels var,
        # then we will add labels var from serial_program to dist_program,
        # that maintains the length of feed_list equal to the length of dataset's values.
        inputs_var = dist_context.serial_feed_vars["inputs"]
        labels_var = dist_context.serial_feed_vars["labels"]
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in dist_main_block.vars:
                feed_list.append(dist_main_block.vars[var.name])
            else:
                copy_var = dist_main_block._clone_variable(var, var.persistable)
                copy_var.desc.set_original_id(var.desc.original_id())
                feed_list.append(copy_var)

        return feed_list

    def get_feed_name_list(self) -> list[str]:
        return [spec.name for spec in self._inputs_spec + self._labels_spec]

    def _prepare_dataloader(
        self,
        dataset,
        return_list=True,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        collate_fn=None,
        num_workers=0,
        use_buffer_reader=True,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        steps_per_epoch=None,
        places=None,
    ):
        dist_context = self._dist_contexts[self._mode]
        dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
        dist_startup_prog = dist_context.dist_startup_programs[self._cur_rank]
        dist_main_block = dist_main_prog.global_block()

        # NOTE: Get feed_list, then insert dataloader op with sharded var shape.
        # Cause predict_program does not contain labels var,
        # then we will add labels var from serial_program to dist_program,
        # that maintains the length of feed_list equal to the length of dataset's values.
        inputs_var = dist_context.serial_feed_vars["inputs"]
        labels_var = dist_context.serial_feed_vars["labels"]
        feed_list = []
        for var in inputs_var + labels_var:
            if var.name in dist_main_block.vars:
                feed_list.append(dist_main_block.vars[var.name])
            else:
                copy_var = dist_main_block._clone_variable(var, var.persistable)
                copy_var.desc.set_original_id(var.desc.original_id())
                feed_list.append(copy_var)

        # insert read op at the end of program
        with static.program_guard(dist_main_prog, dist_startup_prog):
            dataloader = DistributedDataLoader(
                dataset,
                feed_list=feed_list,
                places=places,
                return_list=return_list,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                num_workers=num_workers,
                use_buffer_reader=use_buffer_reader,
                use_shared_memory=use_shared_memory,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                split_data=self._strategy.split_data,
                data_parallel_world_size=self._dp_world_sizes,
                data_parallel_rank=self._dp_ranks,
            )

        return dataloader

    def _tune(self, tune_data, tune_sample_split=None, batch_size=1):
        self._mode = 'train'
        self._inputs_spec, self._labels_spec = self._prepare_data_spec(
            tune_data, tune_sample_split, batch_size
        )
        self._optimization_tuning(self._mode, tune_data, batch_size)

    def _validate_batch_size(self, batch_size):
        if batch_size is None:
            return None

        assert (
            len(set(self._dp_world_sizes)) == 1
        ), f"DistributedBatchSampler only support one data parallel group, but got [{len(set(self._dp_world_sizes))}] different data parallel groups"
        assert (
            batch_size % self._dp_world_sizes[0] == 0
        ), f"batch_size [{batch_size}] is not divisible by dp_world_size [{self._dp_world_sizes[0]}]"
        return batch_size // self._dp_world_sizes[0]

    def _validate_batch(self, batch):
        if batch is None:
            return [None]

        if self._strategy.pipeline.enable or self._acc_steps == 1:
            # pp with schedule or naive-pp
            return batch
        else:
            # split feed data with gradient_merge k_steps
            feed_names = []
            split_batches = []
            for feed_name, cur_feed in batch[0].items():
                feed_names.append(feed_name)
                split_batches.append(
                    np.split(np.array(cur_feed), self._acc_steps, 0)
                )
            baches = []
            for i in range(self._acc_steps):
                micro_batch = [split_batch[i] for split_batch in split_batches]
                baches.append(dict(zip(feed_names, micro_batch)))
            return baches

    def _validate_spec(self, specs):
        specs = auto_utils.to_list(specs)
        if specs is not None:
            for i, spec in enumerate(specs):
                if not isinstance(spec, InputSpec) and not isinstance(
                    spec, DistributedInputSpec
                ):
                    raise TypeError(
                        "'spec' must be object of class `paddle.static.InputSpec` or `DistributedInputSpec`."
                    )
                if spec.name is None:
                    raise ValueError(
                        f"Requires Input[{i}].name != None, but receive `None` with {spec}."
                    )
                if self._acc_steps > 1:
                    shape = list(spec.shape)
                    assert (
                        shape[0] % self._acc_steps == 0
                    ), f"Requires batch_size[{spec.shape[0]}] to be divisible by k_steps[{self._acc_steps}]."
                    shape[0] //= self._acc_steps
                    spec.shape = shape
        return specs or []

    def _validate_vars(self, vars):
        vars = auto_utils.to_list(vars)
        if vars is not None:
            for i, var in enumerate(vars):
                if not isinstance(var, Variable):
                    raise TypeError("'var' must be a `Variable`.")
        return vars or []

    def _is_local_var(self, var):
        var_name = _to_name_str(var)
        return var_name in self.main_program.global_block().vars

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _metrics_name(self):
        metrics_name = ['loss'] if self._loss else []
        for m in self._metrics:
            metrics_name.extend(auto_utils.to_list(m.name()))
        return metrics_name

    def _switch_mode(self, mode):
        assert (
            mode in self._dist_contexts
        ), f"{mode} model is not ready, please call `prepare()` first."
        self.to_mode(mode)

    def to_mode(self, mode: _Mode) -> None:
        assert mode in [
            "train",
            "eval",
            "predict",
        ], f"mode {mode} should be one of ['train', 'eval', 'predict']"
        self._mode = mode

    def _set_state_dict(self, mode, strict, state_dict, dist_attr):
        dist_context = self._dist_contexts[mode]
        program = dist_context.dist_main_programs[self._cur_rank]
        cur_dist_attr = auto_utils.get_dist_attr(program, dist_context)
        converter = Converter(state_dict, dist_attr, cur_dist_attr)
        state_dict = converter.convert(strict=strict)
        for name, param in program.state_dict().items():
            param_array = np.array(param)
            if name not in state_dict:
                continue
            if param_array.dtype != state_dict[name].dtype:
                self._logger.info(
                    f"cast {name}'s dtype from '{state_dict[name].dtype}' to '{param_array.dtype}'"
                )
                state_dict[name] = state_dict[name].astype(param_array.dtype)
        program.set_state_dict(state_dict)

    def save(self, path: str, training: bool = True) -> None:
        """
        Saves the model, parameters, optimizer state to path.
        If `training` is set to False, only inference model will be saved.

        Args:
            path (str): The file prefix to save model. The format
                is 'dirname/file_prefix' or 'file_prefix'. if empty str.
                A exception will be raised.
            training (bool, optional): Whether to save for training. If not, save
                for inference only. If `training` is set to True, the optimizer state
                will be saved. Otherwise, only the model and parameters are saved.
                This function will silently overwrite existing file at the target
                location. Default: True.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.distributed.fleet import auto
                >>> from paddle.vision.datasets import MNIST

                >>> transform = T.Compose([
                ...     T.Transpose(),
                ...     T.Normalize([127.5], [127.5])
                >>> ])
                >>> train_dataset = MNIST(mode='train', transform=transform)

                >>> model = paddle.vision.models.LeNet()
                >>> loss = paddle.nn.CrossEntropyLoss()
                >>> optimizer = paddle.optimizer.Adam(
                ...     learning_rate=0.001, parameters=model.parameters())
                >>> metrics = paddle.metric.Accuracy(topk=(1, 2))

                >>> engine = auto.Engine(model, loss, optimizer, metrics)
                >>> engine.fit(train_dataset,
                ...             epochs=1,
                ...             batch_size=64)
                >>> engine.save("./my_model")

        """
        if training:
            assert self._mode in self._dist_contexts
            dist_context = self._dist_contexts[self._mode]
            serial_program = dist_context.serial_main_program
            dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
            self._saver.save(
                path,
                serial_program=serial_program,
                dist_main_program=dist_main_prog,
                dist_context=dist_context,
            )
        else:
            assert "predict" in self._dist_contexts
            dist_context = self._dist_contexts["predict"]
            feed_vars = dist_context.serial_feed_vars['inputs']
            fetch_vars = dist_context.serial_fetch_vars['outputs']
            dist_main_prog = dist_context.dist_main_programs[self._cur_rank]
            if self._strategy.qat.enable and self._strategy.qat.onnx_format:
                from paddle.static.quantization import QuantWeightPass

                self._logger.info("export quantized model.")
                self._logger.info(
                    f"convert config {self._strategy.qat.to_dict()}"
                )
                test_graph = IrGraph(
                    core.Graph(dist_main_prog.desc), for_test=True
                )
                quant_weight_pass = QuantWeightPass(global_scope(), self._place)
                for sub_graph in test_graph.all_sub_graphs():
                    quant_weight_pass.apply(sub_graph)
                dist_main_prog = test_graph.to_program()
            self._saver.save_inference_model(
                path,
                feed_vars,
                fetch_vars,
                self._executor,
                program=dist_main_prog,
            )

    def load(
        self, path: str, strict: bool = True, load_optimizer: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Load the stored model, parameters and optimizer states.

        Args:
            path (str): The prefix of files storing the model states and
                optimizer states.
            strict (bool, optional): Whether to skip the loading of mismatch
                parameter or raise an error when mismatch happens (not found
                the parameter in file storing model states of or receives a
                mismatch shape). Default: True.
            load_optimizer (bool, optional): If True, the stored optimizer
                states is restored. Otherwise, the optimizer states is initialized
                from scratch. Default: True.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.vision.transforms as T
                >>> from paddle.distributed.fleet import auto
                >>> from paddle.vision.datasets import MNIST

                >>> transform = T.Compose([
                ...     T.Transpose(),
                ...     T.Normalize([127.5], [127.5])
                >>> ])
                >>> train_dataset = MNIST(mode='train', transform=transform)

                >>> model = paddle.vision.models.LeNet()
                >>> loss = paddle.nn.CrossEntropyLoss()
                >>> optimizer = paddle.optimizer.Adam(
                ...     learning_rate=0.001, parameters=model.parameters())
                >>> metrics = paddle.metric.Accuracy(topk=(1, 2))

                >>> engine = auto.Engine(model, loss, optimizer, metrics)
                >>> engine.fit(train_dataset,
                ...             epochs=1,
                ...             batch_size=64)
                >>> engine.save("./my_model")
                >>> engine.load("./my_model")

        """
        self._strict = strict
        self._state_dict, self._dist_attr = self._saver.load(
            path, load_optimizer
        )
        return self._state_dict, self._dist_attr

    def cost(
        self,
        inputs_spec: InputSpec | None = None,
        labels_spec: InputSpec | None = None,
        mode: _Mode | None = None,
    ) -> tuple[int, int] | None:
        """
        Get and Print cost, including memory of every rank,
        max memory among all ranks, and the global cost of one step based on
        communication cost(computation cost is 0 by default).
        In the future, the flops information of every rank and global cost including
        computation cost will be added.

        Args:
            inputs_spec(InputSpec): The specification of inputs. Default: None.
            labels_spec(InputSpec): The specification of labels. Default: None.
            mode (str): The engine mode must be in ["train", "predict", "eval"]. Default: None.

        Returns:
            Return the global execution time (ms) and max memory (B).

        """
        # Check parallel mode
        if self._strategy.auto_mode == "full":
            self._logger.info(
                "The cost will be calculated in the search process when the auto mode is full."
            )
            return

        # Check mode
        mode = mode if mode is not None else self._mode
        assert mode is not None, "Please set mode."
        if mode not in self._has_prepared:
            raise ValueError(
                f"The mode {mode} is not in accepted modes {list(self._has_prepared.keys())}"
            )
        self.to_mode(mode)

        if inputs_spec is not None and not self._has_prepared[mode]:
            self._inputs_spec = self._validate_spec(inputs_spec)
            self._labels_spec = self._validate_spec(labels_spec)
            self._build(mode)
            self._plan(mode)
        else:
            if in_dynamic_mode() or self._dygraph_mode:
                raise ValueError(
                    "Please call `prepare()` or `fit()` or  `evaluate()` or  `predict()` before calling `cost()`."
                )
            else:
                self._logger.info(
                    "The program whose cost to be estimated must be static default program. Otherwise, please call `prepare()`before calling `cost()`."
                )
                program = paddle.static.default_main_program()
                if (
                    not program.global_block().ops
                    or not program.global_block().ops
                ) and not self._has_prepared[mode]:
                    raise ValueError(
                        "Please call `prepare()` or `fit()` or  `evaluate()` or  `predict()` before calling `cost()`."
                    )

        # Estimate the exec cost and max memory
        global_cost, max_memory = get_cost_from_engine(self, mode)

        return global_cost.time, max_memory

    def get_dist_main_program(self, mode: _Mode) -> Program:
        if self._in_pir_mode:
            return self._pir_dist_main_progs[self._mode]
        return self._dist_contexts[mode].dist_main_programs[self._cur_rank]

    def get_dist_startup_program(self, mode: _Mode) -> Program:
        if self._in_pir_mode:
            return self._pir_dist_startup_progs[self._mode]
        return self._dist_contexts[mode].dist_startup_programs[self._cur_rank]

    def get_serial_main_program(self, mode: _Mode) -> Program:
        if self._in_pir_mode:
            return self._fwd_main_progs[mode]
        return self._dist_contexts[mode].serial_main_program

    def get_serial_startup_program(self, mode: _Mode) -> Program:
        if self._in_pir_mode:
            return self._startup_progs[mode]
        return self._dist_contexts[mode].serial_startup_program

    @property
    def main_program(self) -> Program:
        if self._in_pir_mode:
            return self._pir_dense_main_progs[self._mode]
        dist_context = self._dist_contexts[self._mode]
        return dist_context.dist_main_programs[self._cur_rank]

    @property
    def startup_program(self) -> Program:
        dist_context = self._dist_contexts[self._mode]
        return dist_context.dist_startup_programs[self._cur_rank]

    @property
    def dist_context(self) -> DistributedContext:
        return self._dist_contexts[self._mode]

    @property
    def serial_main_program(self) -> Program:
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_main_program

    @property
    def serial_startup_program(self) -> Program:
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_startup_program

    @property
    def feed_vars(self) -> dict[str, list[Tensor]]:
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_feed_vars

    @property
    def fetch_vars(self) -> dict[str, list[Tensor]]:
        dist_context = self._dist_contexts[self._mode]
        return dist_context.serial_fetch_vars

    @property
    def optimizer(self) -> Optimizer:
        dist_context = self._dist_contexts[self._mode]
        if dist_context._serial_optimizer:
            return dist_context._serial_optimizer
        return self._optimizer

    @property
    def inputs(self) -> list[Tensor]:
        return self._inputs

    @property
    def labels(self) -> list[Tensor]:
        return self._labels
