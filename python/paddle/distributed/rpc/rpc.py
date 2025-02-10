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

import datetime
import os
import pickle
import time
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from paddle.base import core
from paddle.distributed.launch.context import Node
from paddle.distributed.rpc.internal import PythonFunc, _serialize
from paddle.distributed.utils.launch_utils import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    _RetT = TypeVar("_RetT", covariant=True)

    class _FutureWrapper(Protocol[_RetT]):
        def wait(self) -> _RetT: ...


WorkerInfo = namedtuple("WorkerInfo", ["name", "rank", "ip", "port"])

_DEFAULT_RPC_TIMEOUT = -1
_MAX_RPC_TIMEOUT_MS = 0x7FFFFFFF
_BARRIER_TIMEOUT_MAX_DAYS = 99999999
# tcp store for `_barrier_never_timeout`
_barrier_store = None
# count the number of `_barrier_never_timeout` is called and
# ensure that the barrier key is unique
_barrier_count = 0


def _set_barrier_store(store):
    global _barrier_store
    _barrier_store = store


def _del_barrier_store():
    global _barrier_store
    del _barrier_store


def _set_self_info(name, rank, ip, port):
    self_info = pickle.dumps(WorkerInfo(name, rank, ip, port))
    _barrier_store.set(str(rank), self_info)


def _exchange_all_service_infos(world_size):
    all_infos = []
    s = set()
    for rank in range(world_size):
        info = pickle.loads(_barrier_store.get(str(rank)))
        assert (
            info.name not in s
        ), "The Worker name must be unique, but name `{}` is repeated."
        s.add(info.name)
        all_infos.append(info)
    return all_infos


def _gen_endpoint():
    node = Node()
    ip = node.get_host_ip()
    free_port = node.get_free_port()
    return f"{ip}:{free_port}"


def init_rpc(
    name: str,
    rank: int | None = None,
    world_size: int | None = None,
    master_endpoint: str | None = None,
) -> None:
    """
    init rpc.

    Args:
        name (str): worker name.
        rank (int, optional): worker id, default is None.
        world_size (int, optional): number of workers, default is None.
        master_endpoint (str, optional): id address of master, other nodes communicate with the master to
            get the information of all worker nodes, default is None.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc

            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...             master_endpoint="127.0.0.1:8001")

            >>> rpc.shutdown()

    """
    rank = int(os.environ["PADDLE_TRAINER_ID"]) if rank is None else rank
    world_size = (
        int(os.environ["PADDLE_TRAINERS_NUM"])
        if world_size is None
        else world_size
    )
    worker_endpoint = os.getenv("PADDLE_WORKER_ENDPOINT", None)
    if worker_endpoint is None:
        worker_endpoint = _gen_endpoint()
    logger.info(f"Trainer {rank}: worker endpoint: {worker_endpoint}")
    master_endpoint = (
        master_endpoint
        if master_endpoint is not None
        else os.environ["PADDLE_MASTER_ENDPOINT"]
    )
    master_addr, master_port = master_endpoint.split(":")
    master_port = int(master_port)
    stop_check_timeout = int(os.getenv("FLAGS_stop_check_timeout", "900"))
    store = core.TCPStore(
        master_addr,
        master_port,
        rank == 0,
        world_size,
        timeout=stop_check_timeout,
    )
    _set_barrier_store(store)
    ip, port = worker_endpoint.split(":")
    port = int(port)
    _set_self_info(name, rank, ip, port)
    all_infos = _exchange_all_service_infos(world_size)
    c_infos = []
    for node_info in all_infos:
        info = core.WorkerInfo(
            node_info.name, node_info.rank, node_info.ip, node_info.port
        )
        c_infos.append(info)
    core.init_and_set_agent_instance(name, c_infos)
    core.rpc_start_worker()
    # ensure that all the workers are started
    _barrier_never_timeout(rank, world_size)
    core.rpc_start_client()
    logger.info(f"Trainer {rank}: Init RPC done!")


def rpc_sync(
    to: str,
    fn: Callable[..., _RetT],
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    timeout: int = _DEFAULT_RPC_TIMEOUT,
) -> _RetT:
    """
    Make a blocking RPC call to run function ``fn`` on worker ``to``. Attention: Users must use this API in a secure network environment.

    Args:
        to (str): name of the destination worker.
        fn (fn): a callable function, such as Python callables.
        args (tuple, optional): the argument tuple for the ``fn`` invocation, default is None.
        kwargs (dict, optional): is a dictionary of keyword arguments for the ``fn``
                       invocation, default is None.
        timeout (int, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value less than or equal to 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. The default value is -1.

    Returns:
        Returns the result of running ``fn`` with ``args`` and ``kwargs``.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc

            >>> def add(a, b):
            ...     return a + b

            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...         master_endpoint="127.0.0.1:8002")

            >>> ret = rpc.rpc_sync("worker0", add, args=(2, 3))
            >>> rpc.shutdown()

    """
    fut = _invoke_rpc(to, fn, args, kwargs, timeout)
    return fut.wait()


def rpc_async(
    to: str,
    fn: Callable[..., _RetT],
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    timeout: int = _DEFAULT_RPC_TIMEOUT,
) -> _FutureWrapper[_RetT]:
    """
    Make a non-blocking RPC call to run function ``fn`` on worker ``to``. Attention: Users must use this API in a secure network environment.

    Args:
        to (str): name of the destination worker.
        fn (fn): a callable function, such as Python callables.
        args (tuple, optional): the argument tuple for the ``fn`` invocation, default is None.
        kwargs (dict, optional): is a dictionary of keyword arguments for the ``fn``
                       invocation, default is None.
        timeout (int, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value less than or equal to 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. The default value is -1.

    Returns:
        Returns a :class:`FutureWrapper` object that can be waited
        on. When completed, the return value of ``fn`` on ``args`` and
        ``kwargs`` can be got by `fut.wait()`.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc

            >>> def add(a, b):
            ...     return a + b

            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...         master_endpoint="127.0.0.1:8003")

            >>> fut = rpc.rpc_async("worker0", add, args=(2, 3))
            >>> print(fut.wait())
            5

            >>> rpc.shutdown()

    """
    return _invoke_rpc(to, fn, args, kwargs, timeout)


def _invoke_rpc(to, fn, args, kwargs, timeout):
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    serial_obj = _serialize(PythonFunc(fn, args, kwargs))
    timeout_ms = timeout * 1000
    timeout_ms = _MAX_RPC_TIMEOUT_MS if timeout_ms <= 0 else timeout_ms
    future = core.invoke_rpc(to, serial_obj, timeout_ms)
    return future


def _barrier_never_timeout(global_rank, global_world_size):
    # max timeout
    timeout = datetime.timedelta(days=_BARRIER_TIMEOUT_MAX_DAYS)

    if global_world_size < 2:
        return

    global _barrier_count
    barrier_prefix = "Barrier/" + str(_barrier_count) + "/"
    _barrier_count += 1
    is_master = global_rank == 0

    def _check_keys_ready(wait_keys):
        start_time = time.time()
        while len(wait_keys) > 0:
            time.sleep(0.1)
            elapse_time = time.time() - start_time
            if datetime.timedelta(seconds=elapse_time) > timeout:
                raise RuntimeError(
                    f"Keys {wait_keys} are not ready since rank {global_rank} is waiting them."
                )
            wait_keys = list(
                filter(lambda key: int(_barrier_store.get(key)) != 1, wait_keys)
            )

    if is_master:
        # the master will add key, wait for all workers'exiting key and exit in the end.
        # Note: the master must exit in the end to ensure that the TcpServer is destroyed in the end.
        wait_keys = [
            barrier_prefix + str(rank) for rank in range(1, global_world_size)
        ]
        _barrier_store.add(barrier_prefix + str(0), 1)
        _check_keys_ready(wait_keys)
    else:
        wait_keys = [barrier_prefix + str(0)]
        _check_keys_ready(wait_keys)
        _barrier_store.add(barrier_prefix + str(global_rank), 1)


def shutdown() -> None:
    """
    Perform a shutdown of the RPC agent, stop the worker and destroy the agent.
    This will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc

            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...             master_endpoint="127.0.0.1:8004")

            >>> rpc.shutdown()

    """
    info = get_current_worker_info()
    rank = info.rank
    world_size = len(get_all_worker_infos())
    # master will exit in the end
    _barrier_never_timeout(rank, world_size)
    core.rpc_stop_worker()
    _del_barrier_store()
    logger.info(f"Trainer {rank}: rpc shutdown!")


def get_worker_info(name: str) -> WorkerInfo:
    """
    Get worker information by worker name.

    Args:
        name (str): name of the worker.

    Returns:
        class `WorkerInfo` with attribute `name`, `rank`, `ip` and `port`.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc
            >>> import os

            >>> os.environ["PADDLE_WORKER_ENDPOINT"] = "127.0.0.1:9002"
            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...             master_endpoint="127.0.0.1:8005")

            >>> print(rpc.get_worker_info("worker0"))
            {name: worker0, rank: 0, ip: 127.0.0.1, port: 9002}

            >>> rpc.shutdown()

    """
    return core.rpc_get_worker_info(name)


def get_all_worker_infos() -> list[WorkerInfo]:
    """
    Get all worker information.

    Returns:
        List[WorkerInfo].

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc
            >>> import os

            >>> os.environ["PADDLE_WORKER_ENDPOINT"] = "127.0.0.1:9003"
            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...         master_endpoint="127.0.0.1:8006")

            >>> print(rpc.get_all_worker_infos())
            [{name: worker0, rank: 0, ip: 127.0.0.1, port: 9003}]

            >>> rpc.shutdown()

    """
    return core.rpc_get_all_worker_infos()


def get_current_worker_info() -> WorkerInfo:
    """
    Get current worker information.

    Returns:
        class `WorkerInfo` with attribute `name`, `rank`, `ip` and `port`.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.distributed.rpc as rpc
            >>> import os

            >>> os.environ["PADDLE_WORKER_ENDPOINT"] = "127.0.0.1:9004"
            >>> rpc.init_rpc("worker0", rank=0, world_size=1,
            ...             master_endpoint="127.0.0.1:8007")

            >>> print(rpc.get_current_worker_info())
            {name: worker0, rank: 0, ip: 127.0.0.1, port: 9004}

            >>> rpc.shutdown()

    """
    return core.rpc_get_current_worker_info()
