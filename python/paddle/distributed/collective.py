#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import datetime
import hashlib

import paddle

# (TODO: GhostScreaming) It will be removed later.
from paddle.base import core
from paddle.framework import in_dynamic_mode

from .communication.group import Group, _add_new_group, is_initialized
from .fleet.layers.mpu.mp_ops import (  # noqa: F401
    _c_concat,
    _c_identity,
    _c_lookup_table,
    _c_softmax_with_cross_entropy,
    _c_softmax_with_multi_label_cross_entropy,
    _c_split,
    _Linear,
    _linear,
    _mp_allreduce,
    _parallel_embedding,
    _parallel_linear,
    _set_var_distributed,
    split,
)

__all__ = []

_global_env = None


def _get_global_env():
    global _global_env
    if not _global_env:
        _global_env = paddle.distributed.ParallelEnv()
    return _global_env


# group map : the map of all group, 0 for GlobalGroup
# Dict[int, Group]
_group_map = {}
_global_env_gid = 0

# group map by name : the map of all groups from their names
# Dict[name, Group]
_group_map_by_name = {}

# backend map by group : the map of all backend from their groups
# Dict[group, backend]
_group_map_backend = {}

# Name of the default group for init_parallel_env
_default_group_name = "_default_pg"

_valid_backend_list = ['nccl', 'gloo', 'heter', 'xccl', 'bkcl']
_default_store = None  # the default tcp store
_default_backend = None
_default_timeout = datetime.timedelta(seconds=1800)
_start_ring_id = 0


def _set_default_backend(backend):
    global _default_backend
    _default_backend = backend


def _set_default_store(store):
    global _default_store
    _default_store = store


def _get_group_map():
    global _group_map
    if _global_env_gid not in _group_map:
        genv = _get_global_env()
        _group_map[_global_env_gid] = Group(
            genv.rank, 0, list(range(genv.world_size))
        )
    return _group_map


def _get_global_group():
    return _get_group_map()[_global_env_gid]


def _get_group_map_by_name():
    global _group_map_by_name
    return _group_map_by_name


def _get_default_group():
    global _group_map_by_name
    assert is_initialized(), (
        "Call paddle.distributed.init_parallel_env first "
        "to initialize the distributed environment."
    )
    return _get_group_map_by_name()[_default_group_name]


def _set_group_map(gid, group):
    global _group_map
    assert gid not in _group_map
    _group_map[gid] = group


def _set_group_map_by_name(name, group):
    global _group_map_by_name
    assert name not in _group_map_by_name
    _group_map_by_name[name] = group


def _set_group_map_backend(group, backend):
    global _group_map_backend
    assert group not in _group_map_backend
    _group_map_backend[group] = backend


def _new_ring_id():
    # NOTE(liyurui): For compatible reason, auto parallel and eager mode relay on previous syntax.
    if in_dynamic_mode():
        global _start_ring_id
        _start_ring_id += 1
        return _start_ring_id + max(_get_global_env().nrings, 9)
    else:
        return len(_get_group_map()) + max(_get_global_env().nrings, 9)


def _new_process_group_impl(
    backend,
    store,
    rank,
    world_size,
    group_name,
    pg_options,
    group_id=0,
    nccl_comm_init_option=0,
):
    pg = None
    genv = _get_global_env()
    assert backend in _valid_backend_list, "Unsupported backend: %s." % backend
    if backend == "gloo":
        pg = core.ProcessGroupGloo.create(store, rank, world_size, group_id)
    elif backend == "nccl":
        pg = core.ProcessGroupNCCL.create(
            store,
            rank,
            world_size,
            group_id,
            genv.pg_timeout,
            nccl_comm_init_option,
        )
    elif backend == "xccl":
        pg = core.ProcessGroupCustom.create(
            store, genv.device_type, rank, world_size, group_id
        )
    elif backend == "bkcl":
        pg = core.ProcessGroupBKCL.create(store, rank, world_size, group_id)
    return pg


# _custom_gid provides a way for users to
# set the group id, which is usually useful
# to be compatible with the static graph mode.
_custom_gid = None


def _set_custom_gid(gid):
    global _custom_gid
    _custom_gid = gid


def new_group(
    ranks=None,
    backend=None,
    timeout=_default_timeout,
    nccl_comm_init_option=0,
):
    """

    Creates a new distributed communication group.

    Args:
        ranks (list): The global ranks of group members.
        backend (str): The backend used to create group, only nccl is supported now.
        timeout (datetime.timedelta, optional): The waiting timeout for store relevant options, default is 30 minutes.

    Returns:
        Group: The group instance.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle

            >>> paddle.distributed.init_parallel_env()
            >>> tindata = paddle.randn(shape=[2, 3])
            >>> gp = paddle.distributed.new_group([2, 4, 6])
            >>> paddle.distributed.all_reduce(tindata, group=gp, sync_op=False)

    """
    global _custom_gid
    global _group_map
    if in_dynamic_mode():
        global _default_group_name
        gid = _custom_gid if _custom_gid else _new_ring_id()
        group_name = _default_group_name + str(gid)
        if backend != 'heter' and (ranks is None or len(ranks) > 1):
            global_group = _get_default_group()
            global_rank = global_group.rank
            global_ranks = global_group.ranks
            backend = _default_backend if backend is None else backend
            if ranks is None:
                ranks = global_ranks
            assert len(ranks) <= len(global_ranks), (
                "Size of new group must be less than or "
                "equal to that of the default global group."
            )
        size = len(ranks)
        ranks = sorted(ranks)
        if size > 1 and global_rank in ranks:
            rank = 0 if backend == 'heter' else ranks.index(global_rank)
            pg = _new_process_group_impl(
                backend,
                _default_store,
                rank,
                size,
                group_name,
                pg_options=None,
                group_id=gid,
                nccl_comm_init_option=nccl_comm_init_option,
            )
        else:
            rank = -1
            pg = None
        group = Group(rank, gid, ranks, pg=pg, name=group_name)
        _group_map_by_name[group_name] = group
        _group_map[gid] = group
        _group_map_backend[group] = backend
        # TODO: The method below is a new method for group management, will replace the previous
        # three in the future.
        _add_new_group(group)
        return group

    if not backend:
        backend = 'nccl'
    assert backend == 'nccl', "backend other than nccl is not supported yet"

    genv = _get_global_env()
    global_rank = genv.rank

    ring_id = _new_ring_id()

    if global_rank not in ranks:
        gp = Group(-1, ring_id, ranks)
        _group_map[ring_id] = gp
    else:
        ranks = sorted(ranks)
        group_rank = ranks.index(global_rank)
        group_size = len(ranks)
        gp = Group(group_rank, ring_id, ranks)
        _group_map[ring_id] = gp

        if group_size >= 2:
            strategy = core.ParallelStrategy()
            strategy.nranks = group_size
            strategy.local_rank = group_rank
            strategy.trainer_endpoints = [
                genv.trainer_endpoints[i] for i in ranks
            ]
            strategy.current_endpoint = genv.current_endpoint
            strategy.nrings = 1

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(genv.device_id)
                core.NCCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            elif core.is_compiled_with_xpu():
                place = core.XPUPlace(genv.device_id)
                core.BKCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            else:
                raise AssertionError("no cuda device found")
        else:
            return gp

    # TODO(shenliang03): This is a temporary solution to solve the problem of
    # hang caused by cross-creation of new_group
    tmp = (
        paddle.to_tensor([1], dtype="int32")
        if in_dynamic_mode()
        else paddle.full([0], 1, dtype="int32")
    )
    paddle.distributed.all_reduce(tmp, sync_op=True)
    paddle.distributed.wait(tmp)
    return gp


def is_available():
    """
    Check whether the distributed package is available.

    Returns:
        Returns True if the distributed package is available, otherwise False.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> print(paddle.distributed.is_available())

    """
    return core.is_compiled_with_dist()


def _init_parallel_env(backend):
    store = core.create_or_get_global_tcp_store()
    global_env = _get_global_env()
    rank = global_env.rank
    world_size = global_env.world_size
    dev_id = global_env.device_id

    if backend == "gloo":
        core.CommContextManager.create_gloo_comm_context(
            store, "0", rank, world_size
        )
    elif backend == "nccl":
        endpoints_str = ""
        for endpoint in global_env.trainer_endpoints:
            endpoints_str += endpoint
        endpoints_str += "ring_id:{}".format("0")
        endpoints_str_hash = hashlib.md5(
            endpoints_str.encode(encoding='UTF-8')
        ).hexdigest()
        core.CommContextManager.set_device_id(dev_id)
        core.CommContextManager.create_nccl_comm_context(
            store, "0", rank, world_size, endpoints_str_hash
        )
    elif backend == "xccl":
        dev_type = global_env.device_type
        paddle.device.set_device(f"{dev_type}:{dev_id}")
        core.CommContextManager.create_xccl_comm_context(
            store, "0", rank, world_size, dev_type
        )
    elif backend == "bkcl":
        endpoints_str = ""
        for endpoint in global_env.trainer_endpoints:
            endpoints_str += endpoint
        endpoints_str += "ring_id:{}".format("0")
        endpoints_str_hash = hashlib.md5(
            endpoints_str.encode(encoding='UTF-8')
        ).hexdigest()
        core.CommContextManager.set_device_id(dev_id)
        core.CommContextManager.create_bkcl_comm_context(
            store, "0", rank, world_size, endpoints_str_hash
        )
