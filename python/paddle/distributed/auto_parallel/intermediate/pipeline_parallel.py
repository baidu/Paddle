#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import logging
import re
from collections import OrderedDict
from enum import Enum

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.utils.log_utils import get_logger

from .parallel_base import ParallelModel, ParallelOptimizer, is_tensor

logger = get_logger("INFO", __name__)


class SplitPoint(Enum):
    """
    Marking the position of the split.
    BEGINNING: will split the model before the specified layer.
    END: will split the model after the specified layer.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> pp_config = {
            ...     'fc1': dist.SplitPoint.END
            ... }
    """

    BEGINNING = 0
    END = 1


class PipelineParallel(ParallelModel):
    def __init__(self, model, split_spec, global_spec, pipeline_layers):
        super().__init__(model)
        self.split_spec = split_spec
        self.global_spec = global_spec
        self.pipeline_layers = pipeline_layers
        self.pp_parallelizer = self.pipeline_parallel_fn
        self.name_to_layer = {}
        for layer_name, layer in model.named_sublayers():
            self.name_to_layer[layer_name] = layer

    def get_layer_by_name(self, name):
        assert (
            name in self.name_to_layer
        ), f"layer name:{name} not in the model, please check the split_spec"
        return self.name_to_layer[name]

    def pipeline_parallel_fn(self, model):
        mesh = fleet.auto.get_mesh()
        pipeline_stage_num = mesh.get_dim_size("pp")
        assert len(self.split_spec) == pipeline_stage_num - 1

        name_to_layer = {}
        for layer_name, layer in model.named_sublayers():
            name_to_layer[layer_name] = layer

        def forward_post_hook(layer, input, output):
            pipeline_stage_index = layer.pipeline_stage_index
            split_point = layer.split_point
            assert split_point == SplitPoint.END
            # reshard to next pipeline stage
            if isinstance(output, (dict, OrderedDict)):
                for key, tensor in output.items():
                    assert is_tensor(tensor)
                    output[key] = dist.reshard(
                        tensor,
                        self.get_mesh(pipeline_stage_index + 1),
                        tensor.placements,
                    )
            elif isinstance(output, (list, tuple)):
                for i in range(len(output)):
                    assert is_tensor(output[i])
                    output[i] = dist.reshard(
                        output[i],
                        self.get_mesh(pipeline_stage_index + 1),
                        output[i].placements,
                    )
            elif is_tensor(output):
                output = dist.reshard(
                    output,
                    self.get_mesh(pipeline_stage_index + 1),
                    output.placements,
                )
            else:
                raise ValueError(
                    f"output should be a dict of tensors or list of tensors or tensor, but {type(output)}"
                )
            return output

        def forward_pre_hook(layer, input):
            split_point = layer.split_point
            assert split_point == SplitPoint.BEGINNING
            # TODO(deepllz): support in the future
            return input

        # step1: set every layer's own pipeline_stage_index
        split_layer_names = list(self.split_spec.keys())
        sublayer_names = [name for name, _ in model.named_sublayers()]
        # Mark which layer is the next pipeline stage
        pipeline_layer_mark = [0 for _ in range(len(sublayer_names))]
        for split_layer_name in split_layer_names:
            split_point = self.split_spec[split_layer_name]
            index = sublayer_names.index(split_layer_name)
            if split_point == SplitPoint.END:
                is_valid = False
                for i in range(index + 1, len(sublayer_names)):
                    if not sublayer_names[i].startswith(split_layer_name):
                        pipeline_layer_mark[i] = 1
                        is_valid = True
                        break
                assert (
                    is_valid
                ), f"the last layer:{split_layer_name} must not be SplitPoint.END, please check the split_spec"
            else:
                raise NotImplementedError(
                    "SplitPoint.BEGINNING is not supported currently"
                )
                pipeline_layer_mark[index] = 1
        # the inclusiveSum of pipeline_layer_mark is the pipeline stage index
        pipeline_stage_index = list(itertools.accumulate(pipeline_layer_mark))
        for index, (name, layer) in enumerate(model.named_sublayers()):
            layer.pipeline_stage_index = pipeline_stage_index[index]

        # step2: insert reshard
        for name in split_layer_names:
            layer = self.get_layer_by_name(name)
            split_point = self.split_spec[name]
            layer.split_point = split_point
            if split_point == SplitPoint.END:
                layer.register_forward_post_hook(forward_post_hook)
            else:
                raise NotImplementedError(
                    "SplitPoint.BEGINNING is not supported currently"
                )
                layer.register_forward_pre_hook(forward_pre_hook)
        if self.global_spec:
            self.process_global_mesh_layers()
        return model

    def process_global_mesh_layers(self):
        g_mesh = fleet.auto.get_mesh()
        g_mesh = g_mesh.get_mesh_with_dim("pp")

        def forward_post_hook(layer, input, output):
            if isinstance(output, (list, tuple)):
                global_output = list(output)
                for ind in range(len(global_output)):
                    if is_tensor(global_output[ind]):
                        global_output[ind] = dist.shard_tensor(
                            global_output[ind],
                            g_mesh,
                            [
                                dist.Replicate()
                                for _ in range(len(g_mesh._shape))
                            ],
                        )
                if isinstance(output, tuple):
                    global_output = tuple(global_output)
                return global_output
            elif is_tensor(output):
                return dist.shard_tensor(
                    output,
                    g_mesh,
                    [dist.Replicate() for _ in range(len(g_mesh._shape))],
                )
            else:
                raise TypeError(
                    "layer output can only be tensor or list/tuple of tensor"
                )

        def forward_pre_hook(layer, input):
            pp_idx = getattr(layer, "pipeline_stage_index", 0)
            new_input = []
            for t in input:
                if is_tensor(t) and t.is_dist() and t.process_mesh == g_mesh:
                    new_input.append(
                        dist.reshard(
                            t,
                            self.get_mesh(pp_idx),
                            [dist.Replicate(), dist.Replicate()],
                        )
                    )
                else:
                    new_input.append(t)
            return tuple(new_input)

        for layer_name in self.global_spec:
            layer = self.get_layer_by_name(layer_name)
            layer.register_forward_post_hook(forward_post_hook)

        for layer_name in self.pipeline_layers:
            layer = self.get_layer_by_name(layer_name)
            layer.register_forward_pre_hook(forward_pre_hook)


def pipeline_parallel(model, optimizer=None, config=None):
    """
    pipeline_parallel converts model and optimizer to pipelined distributed model

    Args:
        model (paddle.nn.Layer): A single card model to be distributed
        optimizer (paddle.optimizer.Optimizer): An optimizer to be distributed
        config (dict): {
            "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                if split_spec is a string or list, such as "llama.layer" or ["llama.layerA", "llama.layerB"], Then the layer with same prefix a will be divided equally according to the size of pipeline degree.
                if split_spec is a OrderedDict|dict, key is the layer name, and the value is the split position that can be SplitPoint.BEGINNING or SplitPoint.END, the order of the keys is the order of the pipeline stage.
                NOTE: dict is also ordered after python3.7, so use dict at this time.
            "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
        }

    Returns:
        PipelineParallel: a distributed model
        ParallelOptimizer: a distributed optimizer
    """

    split_spec = config.get("split_spec")
    if split_spec is None:
        logging.warning("No split_spec, pipeline parallel won't do anything.")
        return model, optimizer

    mesh = fleet.auto.get_mesh()
    assert (
        mesh is not None
    ), "global mesh must not be None, please call fleet.auto.set_mesh(global_mesh) firstly"
    assert (
        "pp" in mesh.dim_names
    ), "pp must in the mesh dim_names when use pipeline_parallel"

    global_spec = config.get("global_spec")
    if isinstance(split_spec, str):
        split_spec = [split_spec]

    matched_layer_name = None
    if isinstance(split_spec, (list, tuple)):
        # match layer_name with split_spec following by a dot and numbers and no other characters
        # such as split_spec = ["llama.layer"], then llama.layer.0 is matched, llama.layer.0.mlp is not matched
        patterns = [rf"{prefix}\.\d+$" for prefix in split_spec]

        def is_match(layer_name):
            for pattern in patterns:
                if re.match(pattern, layer_name):
                    return True
            return False

        matched_layer_name = [
            name for name, _ in model.named_sublayers() if is_match(name)
        ]

        pp_size = mesh.get_dim_size("pp")
        layer_num = len(matched_layer_name)
        assert (
            layer_num > 0
        ), "No layer match the split_spec, please check its correctness"
        assert (
            layer_num % pp_size == 0
        ), f"The number of layers must be divisible by the pp size, but got {layer_num} and {pp_size}"
        layers_per_rank = layer_num // pp_size
        split_spec_dict = OrderedDict(
            [
                (
                    matched_layer_name[i * layers_per_rank - 1],
                    SplitPoint.END,
                )
                for i in range(1, pp_size)
            ]
        )
    else:
        split_spec_dict = split_spec
        if global_spec:
            raise NotImplementedError(
                "global_spec should be None if split_spec is a dict"
            )
    if global_spec:
        if isinstance(global_spec, str):
            global_spec = [global_spec]
        else:
            assert isinstance(
                global_spec, (list, tuple)
            ), f"global_spec can only be list or list(str), but got:{type(global_spec)}"

    logger.info(
        f"split_spec_dict: {split_spec_dict}, global_spec: {global_spec}"
    )

    model = PipelineParallel(
        model, split_spec_dict, global_spec, matched_layer_name
    )
    if optimizer is not None:
        optimizer = ParallelOptimizer(optimizer)

    return model, optimizer
