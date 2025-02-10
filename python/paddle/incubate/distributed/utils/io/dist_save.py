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
import re
import sys
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import paddle
import paddle.distributed as dist
from paddle.base.framework import dygraph_only
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger

from .save_for_auto import save_for_auto_inference

if TYPE_CHECKING:
    from collections.abc import Sequence
    from io import BytesIO

    from typing_extensions import Unpack

    from paddle import Tensor
    from paddle._typing import NestedStructure
    from paddle.nn.layer.layers import _StateDict
    from paddle.static import Program

    class _SaveConfig(TypedDict, total=False):
        use_binary_format: bool
        gather_to: int | Sequence[int] | None
        state_type: Literal['params', 'opt']
        max_grouped_size: str | int


__all__ = ["save", "save_for_auto_inference"]


@dygraph_only
def save(
    state_dict: dict[str, Any] | _StateDict | NestedStructure[Tensor] | Program,
    path: str | BytesIO,
    **configs: Unpack[_SaveConfig],
) -> None:
    '''
    Save a state dict to the specified path in both distributed and single-card environment.

    Note:
        Now supports saving ``state_dict`` of Layer/Optimizer, Tensor and nested structure containing Tensor, Program.

    Note:
        Different from ``paddle.jit.save``, since the save result of ``paddle.save`` is a single file,
        there is no need to distinguish multiple saved files by adding a suffix. The argument ``path``
        of ``paddle.save`` will be directly used as the saved file name instead of a prefix.
        In order to unify the saved file name format, we recommend using the paddle standard suffix:
        1. for ``Layer.state_dict`` , recommend to use ``.pdparams`` ;
        2. for ``Optimizer.state_dict`` , recommend to use ``.pdopt`` .
        For specific examples, please refer to API code examples.

    Args:
        obj(Object) : The object to be saved.
        path(str|BytesIO) : The path/buffer of the object to be saved.
            If saved in the current directory, the input path string will be used as the file name.
        protocol(int, optional): The protocol version of pickle module must be greater than 1 and less than 5.
            Default: 4.
        **configs(dict, optional): optional keyword arguments. The following options are currently supported:

            1. use_binary_format(bool):
                To be used in paddle.save. When the saved object is static graph variable, you can specify ``use_binary_for_var``.
                If True, save the file in the c++ binary format when saving a single static graph variable; otherwise, save it in pickle format.
                Default: False.
            2. gather_to(int|list|tuple|None):
                To specify which global rank to save in.Default is None.
                None value means distributed saving with no gathering to a single card.
            3. state_type(str):
                Value can be 'params' or 'opt', specifying to save parameters or optimizer state.
            4. max_grouped_size(str|int):
                To limit the max size(how many bits) a object group to be transferred a time.
                If str, the format must be as num+'G/M/K', for example, 3G, 2K, 10M, etc. Default is 3G.

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> # doctest: +SKIP('TODO: the error will be fixed in the future')
            >>> # type: ignore
            >>> import paddle
            >>> paddle.distributed.init_process_group(backend='nccl')
            >>> paddle.distributed.fleet.init(is_collective=True)

            >>> model = build_model()
            >>> optimizer = build_optimizer(model)

            >>> dist_optimizer = paddle.distributed_optimizer(optimizer)
            >>> dist_model = paddle.distributed_optimizer(model)

            >>> # gather params to rank 0 and then save
            >>> paddle.incubate.distributed.utils.io.save(model.state_dict(), path="path/to/save.pdparams", gather_to=[0], state_type="params")

            >>> # save whole params on all ranks
            >>> paddle.incubate.distributed.utils.io.save(model.state_dict(), path="path/to/save.pdparams", gather_to=[0,1], state_type="params")

            >>> # save optimizer state dict on rank 0
            >>> paddle.incubate.distributed.utils.io.save(optimizer.state_dict(), path="path/to/save.pdopt", gather=0, state_type="opt")

    '''

    gather_to = configs.get("gather_to", None)
    if dist.get_world_size() == 1 or gather_to is None:
        configs = _remove_not_supported_conf(configs)
        return paddle.save(state_dict, path, **configs)

    # gather_to is not None and world size > 1
    state_type = configs.get("state_type", None)
    assert isinstance(
        state_type, str
    ), "must pass an arg state_type='params' or state_type='opt' to specify whether to save model state_dict or optimizer state_dict"
    assert state_type in [
        "params",
        "opt",
    ], "must pass an arg state_type='params' or state_type='opt'"

    if re.search(f"{state_type}$", path) is None:
        logger.warning(
            f"You are saving {state_type}, while the path({path} does not end with {state_type})"
        )

    hcg = fleet.get_hybrid_communicate_group()
    assert (
        hcg.get_model_parallel_world_size() == 1
        and hcg.get_pipe_parallel_world_size() == 1
    ), f"Only DP and Sharding is supported now. However, current MP={hcg.get_model_parallel_world_size()} , PP={hcg.get_pipe_parallel_world_size()}"

    sharding_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()

    if state_type == "params":
        if dp_group.nranks > 1:
            assert _same_keys(
                state_dict, dp_group
            ), "only sharding stage 1/2 and DP are supported now"
        if sharding_group.nranks > 1:
            assert _same_keys(
                state_dict, sharding_group
            ), "only sharding stage 1/2 and DP are supported now"
        configs = _remove_not_supported_conf(configs)
        return paddle.save(state_dict, path, **configs)

    # state_type == "opt"
    if sharding_group.nranks == 1:
        configs = _remove_not_supported_conf(configs)
        return paddle.save(state_dict, path, **configs)
    if _same_keys(state_dict, sharding_group):
        return paddle.save(state_dict, path, **configs)
    assert isinstance(gather_to, (list, tuple, int))
    if isinstance(gather_to, int):
        gather_to = [gather_to]
    max_size = configs.get("max_grouped_size", "3G")
    try:
        logger.info("state_dict_keys:" + str(state_dict.keys()))
        gathered_state_dict = _gather_state_dict(
            state_dict, gather_to, sharding_group, max_size=max_size
        )
        logger.info("gathered_state_dict_keys:" + str(state_dict.keys()))
        if dist.get_rank() in gather_to:
            configs = _remove_not_supported_conf(configs)
            paddle.save(gathered_state_dict, path, **configs)
    except:
        raise RuntimeError(
            f'''Saving failed. Following are some suggestions:
    1) pass the param max_grouped_size to turn the grouped size smaller (current value of max_grouped_size is {max_size})
    2) if sharding stage is 1, use paddle.save rather than paddle.distributed.save
    3) Concat the developers
'''
        )


def _state_dict_groups(state_dict, max_size):
    """
    Description:
        Generator of state dict groups to transfer.the size of each group is less than max_size.
    """

    # find the max size of a whole tensor
    # now we only support to transfer  at least one whole tensor
    max_tensor_size = 0
    for k, v in state_dict.items():
        if max_tensor_size < sys.getsizeof(v) + sys.getsizeof(k):
            max_tensor_size = sys.getsizeof(v) + sys.getsizeof(k)

    max_size = max(max_size, max_tensor_size)
    logger.debug(f"max tensor size: {max_size}")

    state_group = {}
    k_list = list(state_dict.keys())
    index = 0
    bits = 0

    # generate groups utils the end
    while index < len(k_list):
        bsize = sys.getsizeof(state_dict[k_list[index]]) + sys.getsizeof(
            k_list[index]
        )
        if bits + bsize >= max_size:
            yield state_group
            state_group = {}
            bits = 0

        state_group[k_list[index]] = state_dict[k_list[index]]
        index += 1
        bits += bsize

        if index == len(k_list) and bits > 0:
            yield state_group


def all_empty(dict_list):
    """
    Check if all items are empty
    """
    for v in dict_list:
        if len(v) > 0:
            return False
    return True


def _parse_mem_size_to_bits(max_size):
    """
    Parse an integer or a mem size str to an integer
    convert xxxG to xxx * 1024^3
    convert xxxM to xxx * 1024^2
    convert xxxK to xxx * 1024^1
    """
    assert isinstance(max_size, (int, str))
    if isinstance(max_size, str):
        assert re.search(
            "^[0-9]*[GMK]$", max_size
        ), f"Wrong max_size 's format, the format ust be like 10K, 9M, 200G , etc, or an integer. However this is {max_size}"
        num = int(max_size[:-1])
        if max_size[-1] == "G":
            max_size = num * 1024**3
        elif max_size[-1] == "M":
            max_size = num * 1024**2
        else:
            max_size = num * 1024
    return max_size


def _gather_state_dict(state_dict, dst, group, max_size="3G"):
    """
    Description:
        Gather state dicts across all group ranks to dst, Depiring the same elements. including LR_Scheduler.
    Args:
        state_dict(dict):
            local state dict
        dst(int|list|tuple):
            ranks the state dicts are gathered to
        group(ProcessGroup):
            group across which the state dicts are gathered
        max_size(int|str):
            The max limitation of the gathered tensor group size transformed a time. Default is 3G bits.
            Each rank 's max tensor group before gathering is max_size // group.size
    Returns:
        Gathered state dict
    """
    assert isinstance(
        dst, (list, tuple, int)
    ), "dst' type must be one of int, list and tuple"
    if isinstance(dst, int):
        dst = [dst]

    max_size = _parse_mem_size_to_bits(max_size)
    max_size //= dist.get_world_size(group)

    logger.debug("len state_dict: len(state_dict)")

    state_dict_ = copy.copy(state_dict)
    mw = None
    has_mw = False
    has_lr = False

    # Remove master_weights and LR_Scheduler to ensure that all the elements of the state dict are str->Tensor
    if "master_weights" in state_dict_:
        mw = state_dict_.pop("master_weights", None)
        has_mw = True
    if "LR_Scheduler" in state_dict_:
        lr = state_dict_.pop("LR_Scheduler", None)
        has_lr = True

    # Gather optimizer state_dict
    output = _grouped_gather_data_dict(state_dict_, dst, group, max_size)

    # Gather master_weights if it exists
    if isinstance(mw, dict):
        masters = _grouped_gather_data_dict(mw, dst, group, max_size)
    else:
        assert mw is None, f"Wrong type of master weights . type: {type(mw)}"

    # assign master_weights and LR_Scheduler
    # Because LR_Schedulers are same across group, it just needs to be reset
    if has_mw:
        output["master_weights"] = masters
    if has_lr:
        output["LR_Scheduler"] = lr
    return output


def _grouped_gather_data_dict(state_data_dict, dst, group, max_size):
    """
    Description:
        Gather state data dict by groups.
    Args:
        state__data_dict(dict):
            local dict to transfer.The state_data_dict only contains the mapping: str->paddle.Tensor
        dst(int|list|tuple):
            ranks the state dicts are gathered to
        group(ProcessGroup):
            group across which the state dicts are gathered
        max_size(int|str):
            The max limitation of the gathered tensor group size transformed a time. Default is 3G bits.
            Each rank 's max tensor group before gathering is max_size // group.size
    Returns:
        Gathered state_data_dict

    """
    numpy_dict = {}
    logger.debug(f"len state_tict_ : {len(state_data_dict)}")

    for k, v in state_data_dict.items():
        try:
            numpy_dict[k] = v.numpy()
        except:
            raise TypeError(
                f"the object (type of {type(v)}) of '{k}' is neither tensor nor parameter"
            )

    total = 0
    output_state = {}

    logger.info("start all gather ...")
    # gather all state_dict by groups
    for state in _state_dict_groups(numpy_dict, max_size):
        s_list = []
        total += len(state)
        logger.info(f"gen to gather: {total} / {len(numpy_dict)}")
        dist.all_gather_object(s_list, state, group)
        if dist.get_rank() in dst:
            for s in s_list:
                for k, v in s.items():
                    logger.debug(f"gathered: {k}, {v.shape}")
                output_state.update(s)

        logger.debug(
            f"s list size: {sum(len(s) for s in s_list)} output: {len(output_state)}"
        )

    # Because each size of groups may be different, here we should wait all objects gathered.
    # The while block breaks until all objects from every rank are empty, which means all of the objects transforming is done.
    while True:
        s_list = []
        state = {}
        logger.debug("while True")
        dist.all_gather_object(s_list, state, group)
        if all_empty(s_list):
            break
        if dist.get_rank() in dst:
            for s in s_list:
                for k, v in s.items():
                    logger.debug(f"gathered: {k}, {v.shape}")
                output_state.update(s)
        logger.debug(
            f"s list size: {sum(len(s) for s in s_list)} output: {len(output_state)}"
        )

    logger.debug("all gathered ...")

    if dist.get_rank() in dst:
        # convert numpy.ndarray to Tensor in cpu place
        place = paddle.CPUPlace()
        for k in output_state.keys():
            output_state[k] = paddle.to_tensor(output_state[k], place=place)
            output_state[k].name = k
        return output_state
    return {}


def _same_keys(state_dict, group):
    """
    Check whether all keys in each dict in the group are the same.
    Used in sharding strategy to determine whether a dict needs to be gathered.
    """
    keys = list(state_dict.keys())
    key_list = []
    logger.info(keys)
    dist.all_gather_object(key_list, keys, group=group)
    for k in key_list:
        if not k == keys:
            return False
    return True


def _remove_not_supported_conf(configs):
    """
    Remove the config values not supported by paddle.save
    """
    __supported_by_save__ = ["use_binary_format"]
    configs_ = copy.copy(configs)
    for k in configs.keys():
        if k not in __supported_by_save__:
            configs_.pop(k, None)
    return configs_
