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

import copy
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle.base.framework import (
    _current_expected_place,
)
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorIndex, LocalTensorMetadata
from .utils import (
    check_unique_id,
    compute_local_shape_and_global_offset,
    flatten_state_dict,
    get_max_id,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.distributed.collective import Group


@dataclass(frozen=True)
class ReadItem:
    local_tensor_index: LocalTensorIndex
    rank: int
    dtype: str
    cur_offset: tuple[int]
    storage_offset: tuple[int]
    lengths: tuple[int]


PATH_TO_CHECKPOINT_FILES: dict[str, tuple[list, list]] = {}


def get_checkpoint_files(path, use_cache=True, unique_id=None):
    # if unique_id is None, all file ends with .metadata and .distcp is returned
    if unique_id is None:
        unique_id = ''
    global PATH_TO_CHECKPOINT_FILES
    if use_cache and path in PATH_TO_CHECKPOINT_FILES:
        return PATH_TO_CHECKPOINT_FILES[path]
    accessible_files = os.listdir(path)
    metadata_files = [
        file
        for file in accessible_files
        if file.endswith(f"{unique_id}.metadata")
    ]
    assert (
        len(metadata_files) > 0
    ), f"No metadata file ends with '{unique_id}.metadata' found in the checkpoint directory: {path}."
    local_data_files = [
        file
        for file in accessible_files
        if file.endswith(f"{unique_id}.distcp")
    ]
    assert (
        len(local_data_files) > 0
    ), f"No data file ends with '{unique_id}.distcp' found in the checkpoint directory:{path}."
    if use_cache:
        PATH_TO_CHECKPOINT_FILES[path] = (metadata_files, local_data_files)
    return (metadata_files, local_data_files)


def get_rank_to_files(
    metadata_list,
    local_data_files,
    state_dict,
    process_group,
    use_dist,
    mw_name_compatibility=True,
):
    """
    Get the mapping of rank to its accessible files.
    """

    # The necessary files to be read
    tensor_key_list = []
    necessary_files = []
    mw_name_compatibility_mapping = {}

    for metadata in metadata_list:
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            assert (
                local_tensor_index not in tensor_key_list
            ), f"Duplicate tensor_key:{local_tensor_index} found. Check whether the metadata."
            tensor_key_list.append(local_tensor_index.tensor_key)
            if local_tensor_index.tensor_key in state_dict:
                necessary_files.append(file_name)

    all_necessary_files = []
    if use_dist:
        paddle.distributed.all_gather_object(
            all_necessary_files, necessary_files, process_group
        )
    else:
        all_necessary_files.append(necessary_files)

    global_necessary_files = [
        file for files in all_necessary_files for file in files
    ]

    global_necessary_files_set = set(global_necessary_files)
    if len(global_necessary_files_set) <= 0:
        logger.warning(
            "No necessary data files found in the checkpoint directory. Please check the metadata."
        )
        missing_keys = set(state_dict.keys())
        return {}, missing_keys, mw_name_compatibility_mapping

    # allgather all accessible files
    global_data_files = []
    if use_dist:
        paddle.distributed.all_gather_object(
            global_data_files, local_data_files, process_group
        )
    else:
        global_data_files.append(local_data_files)
    tmp = []
    for files in global_data_files:
        tmp += files
    global_data_files_set = set(tmp)
    logger.debug(
        f"necessary_data_files_set:{global_necessary_files_set}, global_data_files_set:{global_data_files_set}"
    )
    # check necessary files in global_data_files
    assert (
        global_data_files_set & global_necessary_files_set
        == global_necessary_files_set
    ), f"The checkpoint files are not complete. Please check the checkpoint directory. global_data_files_set:{global_data_files_set}, necessary_data_files_set:{global_necessary_files_set}"
    missing_keys = set(state_dict.keys()) - set(tensor_key_list)
    if len(missing_keys) > 0:
        if mw_name_compatibility:
            mw_name_compatibility_mapping = _modify_mw_name_for_compatibility(
                state_dict, missing_keys, tensor_key_list
            )
            if len(missing_keys) > 0:
                logger.warning(
                    f"Missing keys:{missing_keys}, check whether the checkpoint is complete."
                )
        else:
            logger.warning(
                f"Missing keys:{missing_keys}, check whether the checkpoint is complete."
            )

    rank_to_files = {}
    for rank, need_files in enumerate(all_necessary_files):
        seen = set()
        unique_need_files = [
            f for f in need_files if not (f in seen or seen.add(f))
        ]
        rank_to_files[rank] = unique_need_files
    logger.debug(f"mapping rank_to_files:{rank_to_files}")
    return rank_to_files, missing_keys, mw_name_compatibility_mapping


def _modify_mw_name_for_compatibility(
    state_dict, missing_keys, tensor_key_list
):
    """
    Adjust the master weight name within the optimizer's state_dict to ensure compatibility between semi-automatic parallel execution in both dynamic and static graph modes.
    Args:
        state_dict(Dict[str, paddle.Tensor]): The state_dict to load. It will be modified inplace after loading.
        missing_keys(Set[str]): A set of keys that are expected to be loaded but are missing.
        tensor_key_list(List[str]): A list of tensor keys from the source checkpoint (ckpt).
    """
    compatibility_set = set()
    mw_name_compatibility_mapping = {}
    compatibility_key = None
    for missing_key in missing_keys:
        parts = missing_key.split(".")
        # Determine compatibility key based on naming style
        if "master_weights" in parts:
            parts.remove("master_weights")
            compatibility_key = ".".join(parts) + "_fp32_master_0"
        elif parts[-1].endswith("_fp32_master_0"):
            parts[-1] = parts[-1].replace("_fp32_master_0", "")
            parts.insert(1, "master_weights")
            compatibility_key = ".".join(parts)
        if compatibility_key in tensor_key_list:
            logger.info(
                f"Modify master weights {missing_key} -> {compatibility_key}"
            )
            compatibility_set.add(missing_key)
            mw_name_compatibility_mapping[missing_key] = compatibility_key
            state_dict[compatibility_key] = state_dict.pop(missing_key)
    # update missing_keys
    missing_keys -= compatibility_set
    return mw_name_compatibility_mapping


def get_rank_to_read_files(rank_to_files, rank_to_local_data_files):
    cross_node_file_names = []
    rank_to_need_files = copy.deepcopy(rank_to_files)
    for rank, need_files in rank_to_need_files.items():
        local_data_files = rank_to_local_data_files[rank]
        file_need_to_remove = []
        for file in need_files:
            if file not in local_data_files:
                file_need_to_remove.append(file)
        for file in file_need_to_remove:
            need_files.remove(file)
        cross_node_file_names += file_need_to_remove

    not_read_file_ranks = []
    for rank, files in rank_to_need_files.items():
        if len(files) == 0:
            not_read_file_ranks.append(rank)
    for rank in not_read_file_ranks:
        rank_to_need_files.pop(rank)

    rank_load_files = _get_rank_to_read_files(rank_to_need_files)

    for rank in not_read_file_ranks:
        rank_load_files[rank] = []

    cur_load_files = []
    for rank, load_file in rank_load_files.items():
        cur_load_files += load_file

    unload_files = []
    for file in cross_node_file_names:
        if file not in cur_load_files:
            unload_files.append(file)

    file_to_ranks = {}
    for rank, files in rank_to_local_data_files.items():
        for file in files:
            if file not in file_to_ranks:
                file_to_ranks[file] = [rank]
            else:
                file_to_ranks[file].append(rank)

    seen = set()
    unload_files = [x for x in unload_files if not (x in seen or seen.add(x))]
    for file in unload_files:
        sub_rank_load_files = {}
        for rank in file_to_ranks[file]:
            sub_rank_load_files[rank] = rank_load_files[rank]
        min_rank = min(
            sub_rank_load_files,
            key=lambda rank: (len(sub_rank_load_files[rank]), rank),
        )
        rank_load_files[min_rank].append(file)

    cur_rank = paddle.distributed.get_rank()
    if cur_rank in rank_load_files:
        return rank_load_files[cur_rank]
    else:
        logger.warning(f"rank:{cur_rank} does not need to load checkpoint")
        return []


def _get_rank_to_read_files(rank_to_files):
    """
    Load files in a load-balanced manner.

    Args:
        rank_to_files (dict): mapping from rank to files.

    Example:
        Case1: all ranks access the same data files
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp, 2_0.distcp, 3_0.distcp], rank1:[0_0.distcp, 1_0.distcp, 2_0.distcp, 3_0.distcp]}
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
        Case2: all ranks access different data files but some overlapped
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp, 2_0.distcp], rank1:[2_0.distcp, 3_0.distcp]
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
        Case3: all ranks access different data files and no overlapped
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp], rank1:[2_0.distcp, 3_0.distcp]
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
    """
    file_to_ranks = {}
    for rank, files in rank_to_files.items():
        for file in files:
            if file not in file_to_ranks:
                file_to_ranks[file] = []
            file_to_ranks[file].append(rank)
    rank_to_not_read_files = copy.deepcopy(rank_to_files)
    rank_to_read_files = {rank: [] for rank in rank_to_not_read_files.keys()}
    for file, ranks in file_to_ranks.items():
        if len(ranks) == 1:
            rank = ranks[0]
            rank_to_read_files[rank].append(file)
            rank_to_not_read_files[rank].remove(file)
            if len(rank_to_not_read_files[rank]) == 0:
                rank_to_not_read_files.pop(rank)

    logger.debug(
        f"rank_to_read_files:{rank_to_read_files}, rank_to_not_read_files:{rank_to_not_read_files}"
    )

    def get_least_read_files_ranks(rank_to_read_files):
        nums = [
            (rank, len(files)) for rank, files in rank_to_read_files.items()
        ]
        nums = sorted(nums, key=lambda x: x[1])
        ranks = [rank for rank, num in nums if num == nums[0][1]]
        return ranks

    def get_read_rank_file(rank_to_not_read_files, ranks):
        if len(rank_to_not_read_files) == 0:
            return (None, None)
        nums = [
            (rank, len(files))
            for rank, files in rank_to_not_read_files.items()
            if rank in ranks
        ]
        # 'ranks' refer to the ranks that have read the fewest number of files so far. However, the files containing the weights required
        # . by these ranks may have already been completely read. In this case, they will not read any more files.
        if len(nums) == 0:
            nums = [
                (rank, len(files))
                for rank, files in rank_to_not_read_files.items()
            ]
        nums = sorted(nums, key=lambda x: x[1])
        rank = nums[0][0]
        return (rank, rank_to_not_read_files[rank][0])

    def update(rank_to_read_files, rank_to_not_read_files, rank_file):
        rank, file = rank_file
        if rank is None and file is None:
            return
        if rank not in rank_to_read_files:
            rank_to_read_files[rank] = []
        rank_to_read_files[rank].append(file)
        # update rank_to_not_read_files
        file_to_ranks = {}
        for r, files in rank_to_not_read_files.items():
            for f in files:
                if f not in file_to_ranks:
                    file_to_ranks[f] = []
                file_to_ranks[f].append(r)
        logger.debug(f"file_to_ranks:{file_to_ranks}")
        if file in file_to_ranks:
            for r in file_to_ranks[file]:
                rank_to_not_read_files[r].remove(file)
                if len(rank_to_not_read_files[r]) == 0:
                    rank_to_not_read_files.pop(r)

    while len(rank_to_not_read_files) > 0:
        ranks = get_least_read_files_ranks(rank_to_read_files)
        rank_file = get_read_rank_file(rank_to_not_read_files, ranks)
        update(rank_to_read_files, rank_to_not_read_files, rank_file)
        logger.debug(
            f"update rank_to_read_files:{rank_to_read_files}, rank_to_not_read_files:{rank_to_not_read_files}, ranks:{ranks}, rank_file:{rank_file}"
        )
    return rank_to_read_files


def get_load_infos(metadata_list, local_load_files, process_group, use_dist):
    load_info = {}
    for metadata in metadata_list:
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            if file_name in local_load_files:
                load_info[local_tensor_index] = (
                    paddle.distributed.get_rank(),
                    file_name,
                )

    load_info_list = []
    if use_dist:
        paddle.distributed.all_gather_object(
            load_info_list, load_info, process_group
        )
    else:
        load_info_list.append(load_info)
    load_infos = {}
    for load_info in load_info_list:
        for local_tensor_index, (rank, file_name) in load_info.items():
            assert local_tensor_index not in load_infos
            load_infos[local_tensor_index] = (rank, file_name)
    return load_infos


def compute_overlap(
    cur_chunk_metadata: LocalTensorMetadata,
    storage_local_tensor_metadata: LocalTensorMetadata,
):
    cur_offsets = []
    storage_offsets = []
    lengths = []
    for cur_len, cur_offset, storage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset,
    ):
        begin_offset = max(cur_offset, storage_offset)
        end_offset = min(cur_offset + cur_len, storage_offset + storage_len)
        if begin_offset == cur_offset:
            cur_offsets.append(0)
            storage_offsets.append(begin_offset - storage_offset)
        elif begin_offset == storage_offset:
            cur_offsets.append(begin_offset - cur_offset)
            storage_offsets.append(0)
        else:
            raise ValueError(
                f"Invalid begin_offset:{begin_offset}, cur_offset:{cur_offset}, storage_offset:{storage_offset}"
            )
        lengths.append(end_offset - begin_offset)
        assert (
            lengths[-1] >= 0
        ), f"Invalid length:{lengths[-1]}, end_offset:{end_offset}, begin_offset:{begin_offset}"
    return cur_offsets, storage_offsets, lengths


def not_overlap(
    cur_chunk_metadata: LocalTensorMetadata,
    storage_local_tensor_metadata: LocalTensorMetadata,
):
    for cur_len, cur_offset, storage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset,
    ):
        if (
            cur_offset >= (storage_offset + storage_len)
            or (cur_offset + cur_len) <= storage_offset
        ):
            return True
    return False


def get_read_items(metadata_list, state_dict, process_group, use_dist):
    storage_state_dict_metadata = {}
    for metadata in metadata_list:
        for (
            tensor_key,
            local_tensor_metadata,
        ) in metadata.state_dict_metadata.items():
            if tensor_key not in storage_state_dict_metadata:
                storage_state_dict_metadata[tensor_key] = []
            storage_state_dict_metadata[tensor_key] += local_tensor_metadata

    read_items = []
    logger.debug(f"storage_state_dict_metadata:{storage_state_dict_metadata}")
    for tensor_key, val in state_dict.items():
        if isinstance(val, paddle.Tensor):
            if val.is_dist():
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
                    tuple([0] * len(val.shape)) if len(val.shape) > 0 else ()
                )
            cur_chunk_metadata = LocalTensorMetadata(
                global_offset, local_shape, str(val.dtype).split(".")[1]
            )
            assert (
                tensor_key in storage_state_dict_metadata
            ), f"tensor_key:{tensor_key} not found in storage_state_dict_metadata:{storage_state_dict_metadata}."
            for storage_local_tensor_metadata in storage_state_dict_metadata[
                tensor_key
            ]:
                if not_overlap(
                    cur_chunk_metadata, storage_local_tensor_metadata
                ):
                    continue
                cur_offsets, storage_offsets, lengths = compute_overlap(
                    cur_chunk_metadata, storage_local_tensor_metadata
                )
                storage_local_tensor_index = LocalTensorIndex(
                    tensor_key,
                    tuple(storage_local_tensor_metadata.global_offset),
                )
                read_items.append(
                    ReadItem(
                        storage_local_tensor_index,
                        paddle.distributed.get_rank(),
                        storage_local_tensor_metadata.dtype,
                        tuple(cur_offsets),
                        tuple(storage_offsets),
                        tuple(lengths),
                    )
                )
        else:
            raise ValueError(
                f"Only support paddle.Tensor., val type:{type(val)}"
            )
    global_read_items = []
    tmp = []
    if use_dist:
        paddle.distributed.all_gather_object(tmp, read_items, process_group)
    else:
        tmp.append(read_items)
    for items in tmp:
        for item in items:
            global_read_items.append(item)
    return global_read_items


def load_state_dict(
    state_dict: dict[str, Tensor],
    path: str,
    process_group: Group | None = None,
    coordinator_rank: int = 0,
    unique_id: int | None = None,
    offload: bool = False,
    mw_name_compatibility: bool = True,
) -> None:
    """
    Load the state_dict inplace from a checkpoint path.

    Args:
        state_dict(Dict[str, paddle.Tensor]): The state_dict to load. It will be modified inplace after loading.
        path(str): The directory to load checkpoint files.
        process_group(paddle.distributed.collective.Group): ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank(int): The rank used to coordinate the checkpoint. Rank0 is used by default.
        unique_id(int): The unique id of ckeckpoint, used to distinguish between different checkpoint versions. Default is None, in which case the id the max id of given path, and the newest version checkpoint is loaded.
        offload(bool): Whether to offload the checkpoint data from GPU to CPU.
        mw_name_compatibility(bool): Enable name compatibility between dynamic and static graph semi-automatic parallel. Default is True.
    Example:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> mesh = dist.ProcessMesh([0, 1])
            >>> sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
            >>> state_dict = {"w1": sharded_w1}
            >>> dist.save_state_dict(state_dict, ckpt_path)
            >>> w1_to_load = paddle.zeros_like(w1)
            >>> sharded_w1_to_load = dist.shard_tensor(w1, mesh, [dist.Replicate()])
            >>> state_dict_to_load = {"w1": sharded_w1_to_load}
            >>> dist.load_state_dict(state_dict_to_load, ckpt_path)
            >>> print(f"state_dict_to_load:{state_dict_to_load}")
            state_dict_to_load:{'w1': Tensor(shape=[4, 8], dtype=int64, place=Place(gpu:0), stop_gradient=True, dist_attr={process_mesh: {shape: [2], process_ids: [0,1], dim_names: [d0]}, dims_mappings: [-1,-1], batch_dim: 0, dynamic_dims: [0,0], annotated: [dims_mapping: 1,process_mesh: 1], partial: [].}, GlobalDenseTensor=
            [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20, 21, 22, 23],
             [24, 25, 26, 27, 28, 29, 30, 31]])}
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

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        if use_dist and process_group is None and not is_initialized():
            # Init the default global process group
            paddle.distributed.init_parallel_env()

        if use_dist:
            # sync to avoid some ranks not write path yet
            paddle.distributed.barrier(process_group)
        if unique_id is None:
            unique_id = get_max_id(path)
        else:
            assert unique_id >= 0, f'{unique_id} should be >= 0'
        logger.info(f"The unique_id:{unique_id} is uesed.")

        if use_dist:
            check_unique_id(unique_id, process_group)

        metadata_files, local_data_files = get_checkpoint_files(
            path, unique_id=unique_id
        )

        metadata_list = []
        for file in metadata_files:
            metadata_list.append(paddle.load(os.path.join(path, file)))

        rank_to_files, missing_keys, mw_name_compatibility_mapping = (
            get_rank_to_files(
                metadata_list,
                local_data_files,
                flat_state_dict,
                process_group,
                use_dist,
                mw_name_compatibility,
            )
        )

        if len(missing_keys) > 0:
            logger.warning(
                f"The following keys:{missing_keys} are not found in checkpoint path: {path}."
            )
        if len(rank_to_files) <= 0:
            return

        cur_rank = paddle.distributed.get_rank()
        global_local_data_files = []
        if use_dist:
            paddle.distributed.all_gather_object(
                global_local_data_files,
                {cur_rank: local_data_files},
                process_group,
            )
        else:
            global_local_data_files = [{cur_rank: local_data_files}]

        rank_to_local_data_files = {}
        for d in global_local_data_files:
            rank_to_local_data_files.update(d)

        local_load_files = get_rank_to_read_files(
            rank_to_files, rank_to_local_data_files
        )

        source_state_dict = {}
        for file in local_load_files:
            if offload:
                state_dict_numpy = paddle.load(
                    os.path.join(path, file), return_numpy=True
                )
                source_state_dict[file] = {
                    key: paddle.to_tensor(value, place=paddle.CPUPlace())
                    for key, value in state_dict_numpy.items()
                }
            else:
                source_state_dict[file] = paddle.load(os.path.join(path, file))

        _load_state_dict(
            flat_state_dict,
            source_state_dict,
            metadata_list,
            process_group,
            coordinator_rank,
            offload,
        )

        for flat_key, keys in mapping.items():
            if (
                mw_name_compatibility
                and flat_key in mw_name_compatibility_mapping
            ):
                flat_key = mw_name_compatibility_mapping[flat_key]
            tmp = state_dict
            for key in keys[:-1]:
                tmp = tmp[key]
            tmp[keys[-1]] = flat_state_dict[flat_key]


def _load_state_dict(
    target_state_dict,
    source_state_dict,
    metadata_list,
    process_group=None,
    coordinator_rank=0,
    offload=False,
) -> None:
    with paddle.base.dygraph.guard():
        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        local_load_files = list(source_state_dict.keys())
        # load_infos: {LocalTensorIndex: (rank, file_name)}, which local tensor located in which file, and the file is load in which rank.
        load_infos = get_load_infos(
            metadata_list, local_load_files, process_group, use_dist
        )
        # read_items: [ReadItem(local_tensor_index, rank, cur_offsets, storage_offsets, lengths)],
        # slice the storage local tensor in (storage_offsets, lengths) to assign the current tensor in (cur_offsets, lengths) in rank.
        read_items = get_read_items(
            metadata_list, target_state_dict, process_group, use_dist
        )
        state_dict_in_cpu = []
        idx = 0
        for item in read_items:
            key = item.local_tensor_index.tensor_key
            if key in target_state_dict:
                if target_state_dict[key].place.is_cpu_place():
                    state_dict_in_cpu.append(key)
                    target_state_dict[key] = target_state_dict[key].cuda()
            assert (
                item.local_tensor_index in load_infos
            ), f"read item:{item}, load_infos:{load_infos}"

            logger.debug(f"read item: {item}")
            src_rank, file_name = load_infos[item.local_tensor_index]
            storage_chunk_tensor = None
            cur_chunk_tensor = None
            # The src rank need to load the state_dict.
            if src_rank == paddle.distributed.get_rank():
                assert file_name in source_state_dict
                storage_state_dict = source_state_dict[file_name]
                assert item.local_tensor_index.tensor_key in storage_state_dict
                storage_local_tensor = storage_state_dict[
                    item.local_tensor_index.tensor_key
                ]

                if offload:
                    storage_local_tensor = paddle.to_tensor(
                        storage_local_tensor, place=_current_expected_place()
                    )

                storage_offsets = item.storage_offset
                storage_lengths = item.lengths
                storage_ends = [
                    storage_offset + storage_length
                    for storage_offset, storage_length in zip(
                        storage_offsets, storage_lengths
                    )
                ]
                # The storage_chunk_tensor and storage_local_tensor share the same memory.
                if len(storage_lengths) > 0:
                    storage_chunk_tensor = paddle.slice(
                        storage_local_tensor,
                        list(range(len(storage_lengths))),
                        storage_offsets,
                        storage_ends,
                    )
                else:
                    storage_chunk_tensor = storage_local_tensor
            # The read item rank need to be assigned
            if item.rank == paddle.distributed.get_rank():
                assert (
                    item.local_tensor_index.tensor_key in target_state_dict
                ), f"item:{item}, state_dict:{target_state_dict}"

                cur_local_tensor = (
                    target_state_dict[
                        item.local_tensor_index.tensor_key
                    ]._local_value()
                    if use_dist
                    and target_state_dict[
                        item.local_tensor_index.tensor_key
                    ].is_dist()
                    else target_state_dict[item.local_tensor_index.tensor_key]
                )

                cur_offsets = item.cur_offset
                cur_lengths = item.lengths
                cur_ends = [
                    cur_offset + cur_length
                    for cur_offset, cur_length in zip(cur_offsets, cur_lengths)
                ]
                # The cur_chunk_tensor and cur_local_tensor share the same memory.
                if len(cur_lengths) > 0:
                    cur_chunk_tensor = paddle.slice(
                        cur_local_tensor,
                        list(range(len(cur_lengths))),
                        cur_offsets,
                        cur_ends,
                    )
                else:
                    cur_chunk_tensor = cur_local_tensor
            else:
                # Why we use item.dtype: In static mode, the state_dict maybe incomplete in pp, the dtype is stored in advance.
                cur_chunk_tensor = paddle.zeros(
                    item.lengths,
                    item.dtype,
                )

            # Src_rank represents the rank of data read from ckpt, item_rank is the rank of the parameter of the data to be loaded.
            if src_rank == item.rank:
                if src_rank == paddle.distributed.get_rank():
                    # Assign value locally: in the case of src_rank is cur_rank, it means that the ckpt and the parameters to be loaded are both in the current node.
                    paddle.assign(storage_chunk_tensor, cur_chunk_tensor)
            else:
                # Assign value remotely: src_rank broadcasts the ckpt, and the parameters to be loaded receive the data broadcast by src_rank.
                if src_rank == paddle.distributed.get_rank():
                    storage_chunk_tensor = storage_chunk_tensor.contiguous()
                    paddle.distributed.broadcast(
                        storage_chunk_tensor, src=src_rank, group=process_group
                    )
                else:
                    # The memory hold by cur_chunk_tensor may be non-contiguous, and the broadcast API does not support this type of tensor.
                    tmp_tensor = paddle.assign(cur_chunk_tensor)
                    paddle.distributed.broadcast(
                        tmp_tensor, src=src_rank, group=process_group
                    )
                    paddle.assign(tmp_tensor, cur_chunk_tensor)
            if (
                key in state_dict_in_cpu
                and idx + 1 < len(read_items)
                and read_items[idx + 1].local_tensor_index.tensor_key != key
            ):
                target_state_dict[key] = target_state_dict[key].cpu()
            idx = idx + 1

        if use_dist:
            paddle.distributed.barrier(process_group)


def compute_global_shape(local_tensor_indices):
    rank = len(local_tensor_indices[0].local_shape)
    global_shape = []
    for dim in range(rank):
        max_size = max(
            m.global_offset[dim] + m.local_shape[dim]
            for m in local_tensor_indices
        )
        global_shape.append(max_size)
    return global_shape


def load_merged_state_dict(
    path: str, prefix=None, unique_id=None, offload=False
):
    """
    Load the distributed checkpoint and merge it to unsharded state_dict.

    Args:
        path(str): The directory to load checkpoint files.
        prefix(str): The flat_mapping prefix of state_dict key. e.g., 'model', Default None.
        unique_id(int): The unique id of ckeckpoint, used to distinguish between different checkpoint versions. Default is None, in which case the id the max id of given path, and the newest version checkpoint is loaded.
        offload(bool): Whether to offload the checkpoint data from GPU to CPU, set to True if GPU memory is not enough.

    Returns:
        dict: Merged state_dict.

    Example:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> mesh = dist.ProcessMesh([0, 1])
            >>> sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
            >>> state_dict = {"w1": sharded_w1}
            >>> dist.save_state_dict(state_dict, ckpt_path) # save sharded checkpoint

            >>> # doctest: +SKIP('run in single-card mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> unsharded_state_dict = dist.checkpoint.utils.merge_state_dict(ckpt_path) # load unsharded checkpoint
            >>> print(f"unsharded_state_dict:{unsharded_state_dict}")
            unsharded_state_dict:{'w1':
            [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20, 21, 22, 23],
             [24, 25, 26, 27, 28, 29, 30, 31]])}
            >>> # doctest: -SKIP
    """
    if unique_id is None:
        unique_id = get_max_id(path)
    else:
        assert unique_id >= 0, f'{unique_id} should be >= 0'

    metadata_files, local_data_files = get_checkpoint_files(
        path, unique_id=unique_id
    )

    metadata_list = []
    for file in metadata_files:
        metadata_list.append(paddle.load(os.path.join(path, file)))

    # create target state_dict by local_tensor_meta
    state_dict_to_save = {}
    for metadata in metadata_list:
        for (
            tensor_key,
            local_tensor_meta,
        ) in metadata.state_dict_metadata.items():
            if prefix is None or tensor_key.startswith(prefix):
                global_shape = compute_global_shape(local_tensor_meta)
                t = paddle.zeros(global_shape, dtype=local_tensor_meta[0].dtype)
                if offload:
                    t = t.cpu()
                state_dict_to_save[tensor_key] = t.cpu()
            else:
                continue

    load_state_dict(state_dict_to_save, path, offload=offload)

    # Update dictionary keys in place
    for key in list(
        state_dict_to_save.keys()
    ):  # Use list(data.keys()) to avoid runtime error
        if prefix and key.startswith(prefix):
            new_key = key[len(prefix) + 1 :]  # Remove the "str" prefix
            state_dict_to_save[new_key] = state_dict_to_save.pop(
                key
            )  # Add new key and remove the old one
    return state_dict_to_save
