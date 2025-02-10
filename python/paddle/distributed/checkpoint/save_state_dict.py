# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import os
import time
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorIndex, LocalTensorMetadata, Metadata
from .utils import (
    check_unique_id,
    compute_local_shape_and_global_offset,
    flatten_state_dict,
    get_max_id,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.distributed.collective import Group

async_save_queue = []


def check_exitcode(task):
    exitcode = task.exitcode
    if exitcode != 0:
        logger.error(
            f"Error: save ckpt process failed with exitcode {exitcode}!!!"
        )


def clear_async_save_task_queue():
    """
    wait until all async save task to be done.
    """
    while len(async_save_queue) > 0:
        task = async_save_queue.pop()
        if task and task.is_alive():
            task.join(timeout=60)
            if task.is_alive():
                logger.error("Error: save ckpt process timeout!!!")
                async_save_queue.append(task)
            else:
                check_exitcode(task)
        else:
            check_exitcode(task)


def copy_dict_to_cpu(nested_dict):
    """
    Copy the paddle.Tensor objects in the nested dictionary to the CPU and return a new dict.
    """
    new_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, paddle.Tensor):
            new_dict[key] = value.cpu()
            paddle.device.synchronize()
        elif isinstance(value, dict):
            new_dict[key] = copy_dict_to_cpu(value)
        else:
            new_dict[key] = value
    return new_dict


def merge_state_dict_metadata(global_state_dict_metadata):
    assert isinstance(
        global_state_dict_metadata, list
    ), "The global_state_dict should be a list."
    out = {}
    for state_dict in global_state_dict_metadata:
        for key, val in state_dict.items():
            if key in out:
                if val in out[key]:
                    continue
                out[key].append(val)
            else:
                out[key] = [val]
    return out


def dedup_key_in_dict(global_storage_metadata):
    out = {}
    for storage_metadata in global_storage_metadata:
        for key, val in storage_metadata.items():
            if key in out:
                continue
            out[key] = val
    return out


def dedup_tensor(
    local_state_dict, local_storage_metadata, global_storage_metadata
):
    """
    Dedup the replicated tensor in local state_dict.

    Args:
        local_state_dict(Dict[str, paddle.Tensor]): The state_dict of current rank.
        local_storage_metadata(Dict[LocalTensorIndex, str]): The storage metadata of current rank.
        global_storage_metadata(Dict[LocalTensorIndex, str]): The final storage metadata of all ranks.

    Examples:
        In rank0, local_state_dict:{"w1": t1_0, "w2": t2}, local_storage_metadata:{LocalTensorIndex("w1", (0,0)): "0_0.distcp", LocalTensorIndex("w2", (0,0)): "0_0.distcp"},
        in rank1, local_state_dict:{"w1": t1_1, "w2": t2}, local_storage_metadata:{LocalTensorIndex("w1", (1,0)): "1_0.distcp", LocalTensorIndex("w2", (0,0)): "1_0.distcp"},
        global_storage_metadata:{LocalTensorIndex("w1", (0,0)): "0_0.distcp", LocalTensorIndex("w1", (1,0)): "1_0.distcp", LocalTensorIndex("w2", (0, 0)): "0_0.distcp"}.
        w2 is replicated in rank0 and rank1. We save it in rank0 as default thus need to remove it in other ranks.
        Finally, the local_state_dict:{"w1": t1_1, "w2": t2} in rank1 update to {"w1": t1_1}.
    """

    for tensor_index, file_name in global_storage_metadata.items():
        rank = int(file_name.split(".")[0].split("_")[0])
        if (
            tensor_index in local_storage_metadata
            and rank != paddle.distributed.get_rank()
        ):
            local_state_dict.pop(tensor_index.tensor_key)


def save_state_dict(
    state_dict: dict[str, Tensor],
    path: str,
    process_group: Group | None = None,
    coordinator_rank: int = 0,
    unique_id: int | None = None,
    async_save: bool = False,
) -> None:
    """
    Save the state_dict of model to path.

    Args:
        state_dict(Dict[str, paddle.Tensor]): The state_dict to save.
        path(str): The directory to save state_dict.
        process_group(paddle.distributed.collective.Group): ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank(int): The rank used to save non distributed values. Rank 0 is used by default.
        unique_id(int): The unique id of ckeckpoint, used to distinguish between different checkpoint versions. Default is None, in which case the id 0 when save for the first time and increased by 1 each time when calling save_state_dict in the same path. If unique_id is given and there is already checkpoint with the same unique_id, it will be overrited.
        async_save(bool): Async save the state_dict, default is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed mode')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> mesh = dist.ProcessMesh([0, 1])
            >>> sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0), dist.Replicate()])
            >>> state_dict = {"w1": sharded_w1}
            >>> dist.save_state_dict(state_dict, "./checkpoint")
            >>> # doctest: -SKIP

    """
    with paddle.base.dygraph.guard():
        assert isinstance(
            state_dict, dict
        ), "The state_dict should be a dictionary."
        flat_state_dict, mapping = flatten_state_dict(state_dict)
        if len(flat_state_dict) > 0:
            for val in flat_state_dict.values():
                assert isinstance(
                    val, paddle.Tensor
                ), f"The value of state_dict should be a paddle.Tensor, but got: {val}."

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        if use_dist and process_group is None and not is_initialized():
            # Init the default global process group
            paddle.distributed.init_parallel_env()

        if unique_id is None:
            max_unique_id = get_max_id(path)
            logger.debug(f"Max unique id: {max_unique_id}")
            if max_unique_id is None:
                unique_id = 0
            else:
                unique_id = max_unique_id
        else:
            assert unique_id >= 0, f'{unique_id} should be >= 0'
        if use_dist:
            check_unique_id(unique_id, process_group)

        file_name = f"{paddle.distributed.get_rank()}_{unique_id}.distcp"
        logger.debug(f"The checkpoint is saved to file_name:{file_name}")

        metadata = Metadata()
        local_state_dict = {}
        local_state_dict_metadata = {}
        local_storage_metadata = {}
        for key, val in flat_state_dict.items():
            if isinstance(val, paddle.Tensor):
                # Case1: not initialized means this tensor is placed in another mesh which do not contain this rank
                if not val._is_initialized():
                    continue
                if val.is_dist():
                    local_tensor = val._local_value()
                    # Note: The local_tensor must keep the same name with the original tensor. Otherwise, the StructuredToParameterName@@ mapping will be wrong.
                    local_tensor.name = val.name
                    # when val is scalar, the shape is []
                    (
                        local_shape,
                        global_offset,
                    ) = (
                        compute_local_shape_and_global_offset(
                            val.shape,
                            val.process_mesh,
                            val.placements,
                        )
                        if len(val.shape) > 0
                        else ((), ())
                    )
                    if local_shape is None or global_offset is None:
                        continue
                else:
                    local_shape = tuple(val.shape)
                    global_offset = (
                        tuple([0] * len(val.shape))
                        if len(val.shape) > 0
                        else ()
                    )
                    local_tensor = val
                local_state_dict[key] = local_tensor
                local_tensor_dtype = str(local_tensor.dtype).split('.')[1]
                local_state_dict_metadata[key] = LocalTensorMetadata(
                    global_offset, local_shape, local_tensor_dtype
                )
                local_storage_metadata[
                    LocalTensorIndex(key, tuple(global_offset))
                ] = file_name

        global_state_dict_metadata = []
        global_storage_metadata = []
        global_flatten_mapping = []
        if use_dist:
            paddle.distributed.all_gather_object(
                global_state_dict_metadata,
                local_state_dict_metadata,
                process_group,
            )
            paddle.distributed.all_gather_object(
                global_storage_metadata, local_storage_metadata, process_group
            )
            paddle.distributed.all_gather_object(
                global_flatten_mapping, mapping, process_group
            )
        else:
            global_state_dict_metadata.append(local_state_dict_metadata)
            global_storage_metadata.append(local_storage_metadata)
            global_flatten_mapping.append(mapping)

        metadata.state_dict_metadata = merge_state_dict_metadata(
            global_state_dict_metadata
        )
        metadata.storage_metadata = dedup_key_in_dict(global_storage_metadata)
        metadata.flat_mapping = dedup_key_in_dict(global_flatten_mapping)
        if coordinator_rank == paddle.distributed.get_rank():
            logger.debug(f"metadata:{metadata}")
            paddle.save(metadata, os.path.join(path, f"{unique_id}.metadata"))

        dedup_tensor(
            local_state_dict, local_storage_metadata, metadata.storage_metadata
        )

        if async_save:
            cpu_state_dict = copy_dict_to_cpu(local_state_dict)
            clear_async_save_task_queue()

            attempt = 0
            ctx = multiprocessing.get_context("spawn")

            def start_process():
                nonlocal attempt
                try:
                    p = ctx.Process(
                        target=paddle.save,
                        args=(cpu_state_dict, os.path.join(path, file_name)),
                    )
                    p.start()
                    return p
                except Exception as e:
                    logger.error(
                        f"Attempt {attempt + 1} failed with error: {e}"
                    )
                    attempt += 1
                    time.sleep(1)
                    return start_process()

            p = start_process()
            async_save_queue.append(p)
        else:
            paddle.save(local_state_dict, os.path.join(path, file_name))
