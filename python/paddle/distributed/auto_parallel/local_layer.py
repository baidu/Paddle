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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import paddle
import paddle.distributed as dist
from paddle.nn import Layer

if TYPE_CHECKING:
    from paddle.distributed import Placement, ProcessMesh


class LocalLayer(Layer):
    """
    The `LocalLayer` class is a specialized `Layer` for managing distributed tensors during
    forward and backward passes in a parallelized training environment. It converts distributed tensors
    to local tensors for computation and then back to distributed tensors as output, ensuring seamless
    integration with distributed parallelism frameworks.

    Args:
        out_dist_attrs (list[tuple[ProcessMesh, list[Placement]]]):
            A list where each entry is a tuple containing the `ProcessMesh` and the list of `Placement`
            attributes for the corresponding output tensors. These attributes define the distribution
            strategy for the outputs.

    Examples:
        .. code-block:: python

            >>> from __future__ import annotations

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle import Tensor
            >>> from paddle.distributed import ProcessMesh

            >>> class CustomLayer(dist.LocalLayer):
            ...     def __init__(self, out_dist_attrs):
            ...         super().__init__(out_dist_attrs)
            ...         self.local_result = paddle.to_tensor(0.0)

            ...     def forward(self, x):
            ...         mask = paddle.zeros_like(x)
            ...         if dist.get_rank() == 0:
            ...             mask[1:3] = 1
            ...         else:
            ...             mask[4:7] = 1

            ...         x = x * mask
            ...         mask_sum = paddle.sum(x)
            ...         mask_sum = mask_sum / mask.sum()
            ...         self.local_result = mask_sum
            ...         return mask_sum

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> dist.init_parallel_env()
            >>> mesh = ProcessMesh([0, 1], dim_names=["x"])
            >>> out_dist_attrs = [
            ...     (mesh, [dist.Partial(dist.ReduceType.kRedSum)]),
            ... ]
            >>> local_input = paddle.arange(0, 10, dtype="float32")
            >>> local_input = local_input + dist.get_rank()
            >>> input_dist = dist.auto_parallel.api.dtensor_from_local(
            ...     local_input, mesh, [dist.Shard(0)]
            ... )
            >>> custom_layer = CustomLayer(out_dist_attrs)
            >>> output_dist = custom_layer(input_dist)

            >>> local_value = custom_layer.local_result
            >>> gathered_values: list[Tensor] = []
            >>> dist.all_gather(gathered_values, local_value)

            >>> print(f"[Rank 0] local_loss={gathered_values[0]}")
            [Rank 0] local_loss=1.5
            >>> print(f"[Rank 1] local_loss={gathered_values[1]}")
            [Rank 1] local_loss=6.0
            >>> print(f"global_loss (distributed)={output_dist}")
            global_loss (distributed)=7.5

            >>> # This case needs to be executed in a multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1
            >>> # python -m paddle.distributed.launch {test_case}.py
    """

    def __init__(
        self, out_dist_attrs: list[tuple[ProcessMesh, list[Placement]]]
    ) -> None:
        super().__init__()
        self.out_dist_attrs = out_dist_attrs

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Overrides the base `Layer`'s `__call__` method. Transforms distributed tensors to local tensors
        before computation, invokes the parent class's `__call__` method, and then transforms the
        outputs back to distributed tensors based on the specified distribution attributes.
        """
        inputs = list(inputs)
        for idx in range(len(inputs)):
            if inputs[idx].is_dist():
                inputs[idx] = dist.auto_parallel.api.dtensor_to_local(
                    inputs[idx]
                )

        outputs = Layer.__call__(self, *inputs, **kwargs)
        list_outs = paddle.utils.flatten(outputs)
        if len(list_outs) != len(self.out_dist_attrs):
            raise ValueError(
                f"The number of outputs ({len(list_outs)}) does not match "
                f"the number of distribution attributes ({len(self.out_dist_attrs)})."
            )

        dist_outs = []
        for idx in range(len(list_outs)):
            dist_outs.append(
                dist.auto_parallel.api.dtensor_from_local(
                    list_outs[idx],
                    self.out_dist_attrs[idx][0],
                    self.out_dist_attrs[idx][1],
                )
            )
        return paddle.utils.pack_sequence_as(outputs, dist_outs)
