#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
)

import paddle
from paddle.base import compiler
from paddle.base.wrapped_decorator import wrap_decorator
from paddle.framework import _global_flags, in_dynamic_mode
from paddle.framework.ir import apply_build_strategy

from .base import topology as tp
from .base.distributed_strategy import DistributedStrategy
from .base.meta_optimizer_factory import MetaOptimizerFactory
from .base.role_maker import PaddleCloudRoleMaker, RoleMakerBase
from .base.runtime_factory import RuntimeFactory
from .base.strategy_compiler import StrategyCompiler
from .meta_parallel import model_parallel_random_seed
from .utils.log_util import logger, set_log_level

if TYPE_CHECKING:

    from collections.abc import (
        Callable,
        Iterable,
        Sequence,
    )

    import numpy.typing as npt
    from typing_extensions import ParamSpec, Self, TypeVar, Unpack

    from paddle import Tensor
    from paddle._typing import PlaceLike
    from paddle.base.core import DistFleetWrapper, _Scope
    from paddle.distributed.collective import Group
    from paddle.optimizer import Optimizer
    from paddle.static import (
        BuildStrategy,
        Executor,
        Operator,
        Parameter,
        Program,
        Variable,
    )

    from .base.topology import CommunicateTopology, HybridCommunicateGroup

    class _SaveConfigs(TypedDict, total=False):
        mode: int

    class _SaveCacheConfigs(TypedDict, total=False):
        mode: int
        table_id: int

    _InputT = ParamSpec('_InputT')
    _RetT = TypeVar('_RetT')

__all__ = []

g_pre_alloc_memory = int(os.environ.get("FLAGS_pre_alloc_memory", 0))


def apply_ir_passes(
    main_program: Program,
    startup_program: Program,
    config: Fleet,
) -> BuildStrategy:
    build_strategy = config._user_defined_strategy.build_strategy._copy()
    if not _global_flags()['FLAGS_apply_pass_to_program']:
        return build_strategy

    pipeline_opt = getattr(main_program, "_pipeline_opt", {})
    if pipeline_opt:
        main_program = pipeline_opt["section_program"]
        startup_program = startup_program._pipeline_opt["startup_program"]

    pass_attrs = {"use_cuda": config._is_collective}
    fuse_all_reduce = config._user_defined_strategy.fuse_all_reduce_ops
    if fuse_all_reduce and build_strategy.fuse_all_optimizer_ops:
        # FIXME(zjl): currently, fuse_all_optimizer_ops
        # have conflict with fuse_all_reduce_ops because
        # RawProgramOptimizer also inserts coalesce_tensor
        # into program. These two procedures may conflict
        # in which vars are to be fused.
        logger.warning(
            'Currently, the fuse_all_optimizer_ops pass has conflict with fuse_all_reduce_ops pass. Disable the fuse_all_optimizer_ops pass temporarily.'
        )
        build_strategy.fuse_all_optimizer_ops = False

    return apply_build_strategy(
        main_program, startup_program, build_strategy, pass_attrs
    )


def _inited_runtime_handler_(
    func: Callable[_InputT, _RetT]
) -> Callable[_InputT, _RetT]:
    def __impl__(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        cls = args[0]

        if cls._runtime_handle is None:
            raise ValueError("Fleet can not find suitable runtime handler")

        return func(*args, **kwargs)

    return __impl__


def _is_non_distributed_check_(
    func: Callable[_InputT, _RetT]
) -> Callable[_InputT, _RetT]:
    def __impl__(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        cls = args[0]

        if (
            cls._role_maker is not None
            and cls._role_maker._is_non_distributed() is True
        ):
            logger.warning(
                f"{func.__name__}() function doesn't work when use non_distributed fleet."
            )
            return

        return func(*args, **kwargs)

    return __impl__


inited_runtime_handler = wrap_decorator(_inited_runtime_handler_)
is_non_distributed_check = wrap_decorator(_is_non_distributed_check_)


class Fleet:
    """
    Unified API for distributed training of PaddlePaddle.
    Please reference the https://github.com/PaddlePaddle/PaddleFleetX for details

    Returns:
        Fleet: A Fleet instance

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> # Example1: for collective training
            >>> import paddle
            >>> paddle.enable_static()
            >>> import paddle.distributed.fleet as fleet

            >>> fleet.init(is_collective=True)

            >>> strategy = fleet.DistributedStrategy()
            >>> linear = paddle.nn.Linear(10, 10)
            >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=linear.parameters())
            >>> optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

            >>> # do distributed training

        .. code-block:: python
            :name: code-example2

            >>> # Example2: for parameter server training
            >>> import paddle
            >>> paddle.enable_static()
            >>> import paddle.distributed.fleet as fleet
            >>> strategy = fleet.DistributedStrategy()
            >>> fleet.init(strategy=strategy)

            >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            >>> optimizer = fleet.distributed_optimizer(optimizer)

            >>> if fleet.is_first_worker():
            ...     print("this is first worker")

            >>> print("current node index: {}".format(fleet.worker_index()))
            >>> print("total number of worker num: {}".format(fleet.worker_num()))

            >>> if fleet.is_worker():
            ...     print("this is worker")
            >>> print("worker endpoints: {}".format(fleet.worker_endpoints(to_string=True)))

            >>> print("server num: {}".format(fleet.server_num()))
            >>> print("server endpoints: {}".format(fleet.server_endpoints(to_string=True)))

            >>> if fleet.is_server():
            ...     print("this is server")
            >>> fleet.stop_worker()

    """

    def __init__(self) -> None:
        self._role_maker = None
        self.strategy_compiler: StrategyCompiler | None = None
        self._is_collective = False
        self._runtime_handle = None
        self._util = None
        self._context = {}
        self.user_defined_optimizer: Optimizer = paddle.optimizer.Optimizer(0.0)

    def init(
        self,
        role_maker: RoleMakerBase | None = None,
        is_collective: bool = False,
        strategy: DistributedStrategy | None = None,
        log_level: int | str = "INFO",
    ) -> Self:
        """
        Initialize role_maker in Fleet.

        This function is responsible for the distributed architecture
        what you want to run your code behind.

        Args:
            role_maker (RoleMakerBase, optional): A ``RoleMakerBase`` containing the configuration
                of environment variables related to distributed training.If you did not initialize
                the rolemaker by yourself, it will be automatically initialized to PaddleRoleMaker.
                The default value is None.
            is_collective (Boolean, optional): A ``Boolean`` variable determines whether the program
                runs on Collective mode or ParameterServer mode. True means the program runs on
                Collective mode, and False means running on ParameterServer mode. The default value
                is False.
            strategy (DistributedStrategy): Extra properties for distributed training.
                For details, please refer to paddle.distributed.fleet.DistributedStrategy. Default: None.
            log_level (Integer, String, optional): A ``Integer`` or ``String`` Variable determining how height
                the logging level is. Default is "INFO".

        Returns:
            None

        Examples:
            .. code-block:: python
                :name: code-init-example1

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

            .. code-block:: python
                :name: code-init-example2

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init(is_collective=True)

            .. code-block:: python
                :name: code-init-example3

                >>> import paddle.distributed.fleet as fleet
                >>> role = fleet.PaddleCloudRoleMaker()
                >>> fleet.init(role)

            .. code-block:: python
                :name: code-init-example4

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> fleet.init(strategy=strategy)

            .. code-block:: python
                :name: code-init-example5

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> fleet.init(log_level = "DEBUG")

        """
        from paddle.distributed import parallel_helper

        set_log_level(log_level)

        if strategy is None:
            strategy = DistributedStrategy()
        self._user_defined_strategy = copy.deepcopy(strategy)

        if role_maker is None:
            if isinstance(is_collective, bool):
                self._is_collective = is_collective
                self._role_maker = PaddleCloudRoleMaker(
                    is_collective=self._is_collective
                )
            else:
                raise ValueError(
                    f"`is_collective` should be instance of `bool`, but got {type(is_collective)}"
                )
        else:
            if isinstance(role_maker, RoleMakerBase):
                self._role_maker = role_maker
                self._is_collective = role_maker._is_collective
            else:
                raise ValueError(
                    f"`role_maker` should be subclass of `RoleMakerBase`, but got {type(role_maker)}"
                )
        self._role_maker._generate_role()

        from paddle.distributed import fleet

        fleet.util._set_role_maker(self._role_maker)

        self.strategy_compiler = StrategyCompiler()

        if in_dynamic_mode():
            if parallel_helper._is_parallel_ctx_initialized():
                logger.warning(
                    "The dygraph parallel environment has been initialized."
                )
            else:
                # FLAGS_nccl_nrings is used for dynamic graph multi-stream communication
                if "FLAGS_nccl_nrings" in os.environ:
                    logger.warning(
                        "You have set the environment variable FLAGS_nccl_nrings "
                        "outside the program, so the nccl_comm_num in "
                        "DistributedStrategy will not take effect here."
                    )
                else:
                    os.environ["FLAGS_nccl_nrings"] = str(
                        self._user_defined_strategy.nccl_comm_num
                    )
                paddle.distributed.init_parallel_env()

            # hybrid parallel not support for npu/xpu
            if not self._user_defined_strategy.heter_ccl_mode:
                # init hybrid parallel environment in dygraph
                if tp._HYBRID_PARALLEL_GROUP is None:
                    self._init_hybrid_parallel_env()
                else:
                    logger.warning(
                        "The dygraph hybrid parallel environment has been initialized."
                    )
        elif self._is_collective:
            use_sharding = self._user_defined_strategy.sharding

            # global group
            global_rank = self.worker_index()
            global_world_size = self.worker_num()
            # NOTE(wangxi): see sharding_optimizer
            global_ring_id = 3 if use_sharding else 0
            global_ranks = list(range(global_world_size))

            if tp._HYBRID_PARALLEL_GROUP is None:
                tp._CommunicateGroup()
            cg = tp._HYBRID_PARALLEL_GROUP
            self._hcg = cg
            cg.set_comm_group(
                'global',
                global_rank,
                global_world_size,
                global_ring_id,
                global_ranks,
            )

            use_tensor_parallel = self._user_defined_strategy.tensor_parallel
            use_mp = use_sharding or use_tensor_parallel

            # hybrid group
            if use_mp is False:
                return

            mp_degree_sharding = 1
            mp_degree_tensor_parallel = 1
            if use_sharding:
                sharding_configs = self._user_defined_strategy.sharding_configs
                mp_degree_sharding = int(sharding_configs['mp_degree'])

            if use_tensor_parallel:
                tensor_parallel_configs = (
                    self._user_defined_strategy.tensor_parallel_configs
                )
                mp_degree_tensor_parallel = int(
                    tensor_parallel_configs['tensor_parallel_degree']
                )

            if use_sharding and use_tensor_parallel:
                assert mp_degree_sharding == mp_degree_tensor_parallel

            mp_degree = (
                mp_degree_sharding
                if use_sharding
                else mp_degree_tensor_parallel
            )

            if mp_degree > 1:
                assert global_world_size % mp_degree == 0
                # NOTE(wangxi): mp_ring_id sync with sharding_optimizer.py _build_groups
                mp_ring_id = 0
                mp_rank = global_rank % mp_degree
                mp_group_id = global_rank // mp_degree
                mp_group_ranks = [
                    idx
                    for idx in global_ranks
                    if idx // mp_degree == mp_group_id
                ]
                cg.set_comm_group(
                    'model', mp_rank, mp_degree, mp_ring_id, mp_group_ranks
                )
        self.maybe_pre_alloc_device_memory()
        return self

    def maybe_pre_alloc_device_memory(self):
        if g_pre_alloc_memory > 0:
            logger.warning(
                f'To avoid creating memory shards, pre-allocating a tensor whose memory capacity is {g_pre_alloc_memory} GB and then release it.'
            )
            memory_size = int((g_pre_alloc_memory * 1024 * 1024 * 1024) // 4)
            x = paddle.empty([memory_size], dtype=paddle.float32)
            del x

    # test allreduce perf
    def allreduce_perf(
        self,
        iteration: int,
        x: Tensor,
        group: Group,
        perf_size: int,
        perf_threshold_time: float,
        warmup: bool = False,
    ) -> None:
        if group is None or group.nranks <= 1:
            logger.warning("allreduce_perf is invalid, group invalid!")
            return
        paddle.distributed.barrier()
        paddle.device.cuda.synchronize()
        start_t = time.time()
        for _ in range(iteration):
            paddle.distributed.all_reduce(x, group=group)
        paddle.device.cuda.synchronize()
        end_t = time.time()
        ret = (end_t - start_t) / iteration
        if warmup:
            return
        logger.info(
            f"[AllReduceTest] nbytes {perf_size}B test result: {ret} s/iter"
        )
        if perf_threshold_time > -1 and ret > perf_threshold_time:
            logger.warning(
                f"[Perf Warning] AllReduce Test Timeout! {ret} > {perf_threshold_time}"
            )

    # test reduce perf
    def reduce_perf(
        self,
        iteration: int,
        x: Tensor,
        group: Group,
        perf_size: int,
        perf_threshold_time: float,
    ) -> None:
        if group is None or group.nranks <= 1:
            logger.warning("reduce_perf is invalid, group invalid!")
            return
        paddle.distributed.barrier()
        paddle.device.cuda.synchronize()
        start_t = time.time()
        for _ in range(iteration):
            paddle.distributed.reduce(x, dst=min(group.ranks), group=group)
        paddle.device.cuda.synchronize()
        end_t = time.time()
        ret = (end_t - start_t) / iteration
        logger.info(
            f"[ReduceTest] nbytes {perf_size}B test result: {ret} s/iter"
        )
        if perf_threshold_time > -1 and ret > perf_threshold_time:
            logger.warning(
                f"[Perf Warning] Reduce Test Timeout! {ret} > {perf_threshold_time}"
            )

    # test broadcast perf
    def broadcast_perf(
        self,
        iteration: int,
        x: Tensor,
        group: Group,
        perf_size: int,
        perf_threshold_time: float,
    ) -> None:
        if group is None or group.nranks <= 1:
            logger.warning("broadcast_perf is invalid, group invalid!")
            return
        paddle.distributed.barrier()
        paddle.device.cuda.synchronize()
        start_t = time.time()
        for _ in range(iteration):
            paddle.distributed.broadcast(x, src=min(group.ranks), group=group)
        paddle.device.cuda.synchronize()
        end_t = time.time()
        ret = (end_t - start_t) / iteration
        logger.info(
            f"[BroadcastTest] nbytes {perf_size}B test result: {ret} s/iter"
        )
        if perf_threshold_time > -1 and ret > perf_threshold_time:
            logger.warning(
                f"[Perf Warning] Broadcast Test Timeout! {ret} > {perf_threshold_time}"
            )

    # test allgather perf
    def allgather_perf(
        self,
        iteration: int,
        x: Tensor,
        group: Group,
        perf_size: int,
        perf_threshold_time: float,
    ) -> None:
        if group is None or group.nranks <= 1:
            logger.warning("allgather_perf is invalid, group invalid!")
            return
        paddle.distributed.barrier()
        paddle.device.cuda.synchronize()
        start_t = time.time()
        for _ in range(iteration):
            tmp = []
            paddle.distributed.all_gather(tmp, x, group=group)
        paddle.device.cuda.synchronize()
        end_t = time.time()
        ret = (end_t - start_t) / iteration
        logger.info(
            f"[AllgatherTest] nbytes {perf_size}B test result: {ret} s/iter"
        )
        if perf_threshold_time > -1 and ret > perf_threshold_time:
            logger.warning(
                f"[Perf Warning] Allgather Test Timeout! {ret} > {perf_threshold_time}"
            )

    # test reduce_scatter perf
    def reduce_scatter_perf(
        self,
        iteration: int,
        x: Tensor,
        group: Group,
        perf_size: int,
        perf_threshold_time: float,
    ) -> None:
        if group is None or group.nranks <= 1:
            logger.warning("reduce_scatter_perf is invalid, group invalid!")
            return
        paddle.distributed.barrier()
        paddle.device.cuda.synchronize()
        parallelism = group.nranks
        output_shape = x.shape
        if x.shape[0] % parallelism != 0:
            logger.warning(
                f"the shape of input[{x.shape[0]}] can't be divided exactly by reduce_scatter parallelism[{parallelism}], test stopped!"
            )
            return
        output_shape[0] = output_shape[0] // parallelism
        output = paddle.empty(shape=output_shape, dtype=x.dtype)
        start_t = time.time()
        for _ in range(iteration):
            paddle.distributed.stream.reduce_scatter(
                output,
                x,
                op=paddle.distributed.ReduceOp.SUM,
                group=group,
                sync_op=True,
            )
        paddle.device.cuda.synchronize()
        end_t = time.time()
        ret = (end_t - start_t) / iteration
        logger.info(
            f"[ReduceScatterTest] nbytes {perf_size}B test result: {ret} s/iter"
        )
        if perf_threshold_time > -1 and ret > perf_threshold_time:
            logger.warning(
                f"[Perf Warning] ReduceScatter Test Timeout! {ret} > {perf_threshold_time}"
            )

    def _collective_perf_impl(self, round=50, context={}, hcg=None):
        if hcg is None:
            hcg = self.get_hybrid_communicate_group()

        collective_perf_func_map = {
            "allreduce": self.allreduce_perf,
            "reduce": self.reduce_perf,
            "broadcast": self.broadcast_perf,
            "allgather": self.allgather_perf,
            "reduce_scatter": self.reduce_scatter_perf,
        }
        dp_group = hcg.get_data_parallel_group()
        sharding_group = hcg.get_sharding_parallel_group()
        mp_group = hcg.get_model_parallel_group()
        data_group = None
        if dp_group.nranks > 1:
            data_group = dp_group
        elif sharding_group.nranks > 1:
            data_group = sharding_group

        collective_perf_group_map = {
            "allreduce": data_group,
            "reduce": data_group,
            "broadcast": data_group,
            "allgather": mp_group,
            "reduce_scatter": mp_group,
        }

        for comm_type, size_and_time in context.items():
            # test 1M ~ 1G as default
            nbytes = 1 << 20  # 1048576(1MB)
            final_nbytes = 1 << 30  # 1073741824(1GB)
            dtype = paddle.float32
            time_threshold = 0

            if size_and_time is not None:
                nbytes = size_and_time[0]
                # Run only once when test specific message size.
                final_nbytes = nbytes
                time_threshold = size_and_time[1]
            if nbytes <= 0:
                logger.warning(
                    f"Size for collective performance check should be positive, but got {nbytes}"
                )
                return

            while nbytes <= final_nbytes:
                x = paddle.zeros([nbytes // 4], dtype=dtype)
                # warmup
                self.allreduce_perf(10, x, None, nbytes, 1, warmup=True)

                collective_perf_func_map[comm_type](
                    iteration=round,
                    x=x,
                    group=collective_perf_group_map[comm_type],
                    perf_size=nbytes,
                    perf_threshold_time=time_threshold,
                )
                nbytes = nbytes << 1

    def collective_perf(
        self,
        comm_type: Literal[
            'allreduce', 'broadcast', 'reduce', 'allgather', 'reduce_scatter'
        ],
        round: int = 50,
        size_and_time: dict[int, float] = {},
    ) -> None:
        """
        Run performance test for given communication type
        and compare the time cost with the threshold.

        Args:
            comm_type (str): Communication type for performance test. Currently support
                            "allreduce", "broadcast", "reduce", "allgather" and "reduce_scatter".
            round (int, optional): Loop times for performance test. More loops will cost more time
                            and provide more accurate result. Defaults to 50.
            size_and_time (dict, optional): Message sizes and time thresholds for performance test.
                            each pair will invoke a performance check. Defaults to {}, which indicates
                            acting performance check from 1MB to 1GB without threshold set.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init(is_collective=True)
                >>> # run two tests, one with 1MB (threshold 0.5s) and another with 1GB (threshold 1s)
                >>> size_and_time = {1<<20: 0.5, 1<<30: 1}
                >>> fleet.collective_perf("allreduce", round=50, size_and_time = size_and_time)
        """
        if not self._is_collective:
            logger.warning(
                "fleet.collective_perf is only for collective mode, will return with no test acted."
            )
            return
        for size, time_threshold in size_and_time.items():
            context = {comm_type: [size, time_threshold]}
            self._collective_perf_impl(round=round, context=context)

    def _init_hybrid_parallel_env(self):
        """initialize the hybrid environment."""
        self.hybrid_configs = self._user_defined_strategy.hybrid_configs
        self.dp_degree = self.hybrid_configs["dp_degree"]
        self.mp_degree = self.hybrid_configs["mp_degree"]
        self.pp_degree = self.hybrid_configs["pp_degree"]
        self.sep_degree = self.hybrid_configs["sep_degree"]
        self.sharding_degree = self.hybrid_configs["sharding_degree"]

        assert self.mp_degree >= 0, "mp_degree should be greater or equal to 0"
        assert self.pp_degree >= 0, "pp_degree should be greater or equal to 0"
        assert (
            self.sep_degree >= 0
        ), "sep_degree should be greater or equal to 0"
        assert (
            self.sharding_degree >= 0
        ), "sharding_degree should be greater or equal to 0"

        self.mp_degree = max(self.mp_degree, 1)
        self.pp_degree = max(self.pp_degree, 1)
        self.sep_degree = max(self.sep_degree, 1)

        if self.dp_degree < 0:
            nranks = paddle.distributed.get_world_size()
            self.dp_degree = nranks // (self.mp_degree * self.pp_degree)

        self.dp_degree = max(self.dp_degree, 1)

        d_hybrid_degree = {
            "dp": ["data", self.dp_degree],
            "pp": ['pipe', self.pp_degree],
            "sharding": ['sharding', self.sharding_degree],
            "mp": ['model', self.mp_degree],
            "sep": ["sep", self.sep_degree],
        }

        order = self._user_defined_strategy.hybrid_parallel_order
        if order[:].sort() != list(d_hybrid_degree.keys())[:].sort():
            raise AssertionError(
                'The order of hybrid_config setting is incorrect.'
            )

        hybrid_group_names = []
        dims = []
        for h_name in order:
            name, degree = d_hybrid_degree[h_name]
            hybrid_group_names.append(name)
            dims.append(degree)

        self._topology = tp.CommunicateTopology(
            hybrid_group_names=hybrid_group_names, dims=dims
        )

        self._hcg = tp.HybridCommunicateGroup(self._topology)

        if self.mp_degree > 1:
            tensor_parallel_configs = (
                self._user_defined_strategy.tensor_parallel_configs
            )
            tensor_init_seed = tensor_parallel_configs["tensor_init_seed"]
            if tensor_init_seed == -1:
                model_parallel_random_seed()
            else:
                model_parallel_random_seed(tensor_init_seed)

    def get_hybrid_communicate_group(self) -> HybridCommunicateGroup:
        assert self._hcg is not None
        return self._hcg

    def get_hybrid_parallel_topology(self) -> CommunicateTopology:
        assert self._topology is not None
        return self._topology

    def is_first_worker(self) -> bool:
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker, False if not.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.is_first_worker()

        """
        return self._role_maker._is_first_worker()

    def worker_index(self) -> int:
        """
        Get current worker index.

        Returns:
            int: node id

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.worker_index()

        """
        return self._role_maker._worker_index()

    def worker_num(self) -> int:
        """
        Get current total worker number.

        Returns:
            int: worker numbers

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.worker_num()

        """
        return self._role_maker._worker_num()

    def node_num(self) -> int:
        return self._role_maker._get_node_num()

    def local_rank(self) -> str | None:
        return self._role_maker._get_local_rank()

    def local_device_ids(self) -> str | None:
        return self._role_maker._get_local_device_ids()

    def world_device_ids(self) -> str | None:
        return self._role_maker._get_world_device_ids()

    def is_worker(self) -> bool:
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.is_worker()

        """
        return self._role_maker._is_worker()

    def is_coordinator(self) -> bool:
        return self._role_maker._is_coordinator()

    def worker_endpoints(self, to_string: bool = False) -> list[str] | str:
        """
        Get current worker endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.worker_endpoints()

        """
        if to_string:
            return ",".join(self._role_maker._get_trainer_endpoints())
        else:
            return self._role_maker._get_trainer_endpoints()

    def server_num(self) -> int:
        """
        Get current total worker number.

        Returns:
            int: server number

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.server_num()
        """
        return len(self._role_maker._get_pserver_endpoints())

    def server_index(self) -> int:
        """
        Get current server index.

        Returns:
            int: node id

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.server_index()

        """
        return self._role_maker._server_index()

    def server_endpoints(self, to_string: bool = False) -> list[str] | str:
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.server_endpoints()

        """

        if to_string:
            return ",".join(self._role_maker._get_pserver_endpoints())
        else:
            return self._role_maker._get_pserver_endpoints()

    def is_server(self) -> bool:
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.is_server()

        """
        return self._role_maker._is_server()

    def barrier_worker(self) -> None:
        """
        barrier all workers

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> fleet.barrier_worker()
        """
        self._role_maker._barrier("worker")

    def all_reduce(
        self,
        input: Any,
        mode: Literal['sum', 'mean', 'max'] = "sum",
    ) -> npt.NDArray[Any]:
        """
        all reduce input between all workers, mode can be sum, mean or max, default is sum

        Returns:
            list/int: all reduce result

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> res = fleet.all_reduce(5)

        """
        return self._role_maker._all_reduce(input, mode, "worker")

    @is_non_distributed_check
    @inited_runtime_handler
    def init_worker(self, scopes: Sequence[_Scope] | None = None) -> None:
        """
        initialize `Communicator` for parameter server training.


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.init_worker()

        """
        self._runtime_handle._init_worker(scopes)

    @is_non_distributed_check
    @inited_runtime_handler
    def init_coordinator(self, scopes: Sequence[_Scope] | None = None) -> None:
        """
        initialize coordinator node
        """
        self._runtime_handle._init_coordinator(scopes)

    def make_fl_strategy(self) -> None:
        self._runtime_handle._make_fl_strategy()

    @is_non_distributed_check
    @inited_runtime_handler
    def get_fl_client(self) -> DistFleetWrapper:
        """
        get worker(training node) ptr
        """
        return self._runtime_handle._worker

    @is_non_distributed_check
    @inited_runtime_handler
    def init_server(self, *args: Any, **kwargs: Any) -> None:
        """
        init_server executor to initialize startup program,
        if the `args` is not empty, it will run load_persistables for increment training.


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.init_server()

        """
        self._runtime_handle._init_server(*args, **kwargs)

    @is_non_distributed_check
    @inited_runtime_handler
    def load_model(self, path: str, mode: int) -> None:
        """
        load fleet model from path


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.load_model("path", mode=0)

        """
        self._runtime_handle._load_persistables(path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def load_one_table(self, table_id: int, path: str, mode: int) -> None:
        """
        load fleet one table from path


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.load_one_table(0, "path", mode=0)

        """
        self._runtime_handle._load_one_table(table_id, path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def load_inference_model(self, path: str, mode: int) -> None:
        """
        load fleet inference model from path


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.load_inference_model("path", mode=1)

        """
        self._runtime_handle._load_inference_model(path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def run_server(self) -> None:
        """
        run server will run pserver main program with executor.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> if fleet.is_server():
                ...     fleet.init_server()

        """
        self._runtime_handle._run_server()

    @is_non_distributed_check
    @inited_runtime_handler
    def stop_worker(self) -> None:
        """
        stop `Communicator` and give training complete notice to parameter server.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.init_server()

        """
        self._runtime_handle._stop_worker()

    @is_non_distributed_check
    @inited_runtime_handler
    def save(
        self,
        dirname: str,
        feed: list[str | Variable] = [],
        fetch: list[str | Variable] = [],
        **configs: Unpack[_SaveConfigs],
    ) -> None:
        inference = True

        if not feed and not fetch:
            inference = False

        place = paddle.CPUPlace()
        executor = paddle.static.Executor(place)

        if inference:
            feeded_var_names = []
            fetch_var_names = []

            for var in feed:
                if isinstance(var, str):
                    feeded_var_names.append(var)
                elif isinstance(var, paddle.static.Variable):
                    feeded_var_names.append(var.name)
                else:
                    raise ValueError("feed must be [str|Variable]")

            for var in fetch:
                if isinstance(var, str):
                    fetch_var_names.append(var)
                elif isinstance(var, paddle.static.Variable):
                    fetch_var_names.append(var.name)
                else:
                    raise ValueError("feed must be [str|Variable]")

            fetch_vars = [
                paddle.static.default_main_program().global_block().var(name)
                for name in fetch_var_names
            ]

            self._runtime_handle._save_inference_model(
                executor, dirname, feeded_var_names, fetch_vars, None, True, 0
            )
        else:
            increment_mode = 0
            if "mode" in configs:
                increment_mode = int(configs["mode"])
            self._runtime_handle._save_persistables(
                executor, dirname, main_program=None, mode=increment_mode
            )

    @is_non_distributed_check
    @inited_runtime_handler
    def save_inference_model(
        self,
        executor: Executor,
        dirname: str,
        feeded_var_names: list[str],
        target_vars: list[Variable],
        main_program: Program | None = None,
        export_for_deployment: bool = True,
        mode: int = 0,
    ) -> None:
        """
        save inference model for inference.

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.init_server()

        """

        self._runtime_handle._save_inference_model(
            executor,
            dirname,
            feeded_var_names,
            target_vars,
            main_program,
            export_for_deployment,
            mode,
        )

    @is_non_distributed_check
    @inited_runtime_handler
    def save_persistables(
        self,
        executor: Executor,
        dirname: str,
        main_program: Program | None = None,
        mode: int = 0,
    ) -> None:
        """

        saves all persistable tensors from :code:`main_program` to
        the folder :code:`dirname`. You can refer to

        The :code:`dirname` is used to specify the folder where persistable tensors
        are going to be saved. If you would like to save tensors in separate
        files, set :code:`filename` None.

        Args:
            executor(Executor): The executor to run for saving persistable tensors.
                                You can refer to :ref:`api_guide_executor_en` for
                                more details.

            dirname(str, optional): The saving directory path.
                                When you need to save the parameter to the memory, set it to None.
            main_program(Program, optional): The program whose persistable tensors will
                                             be saved. Default: None.


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet

                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> exe = paddle.static.Executor(paddle.CPUPlace())
                >>> fleet.save_persistables(exe, "dirname", paddle.static.default_main_program())

        """
        self._runtime_handle._save_persistables(
            executor, dirname, main_program, mode
        )

    @is_non_distributed_check
    @inited_runtime_handler
    def save_cache_model(
        self, dirname: str, **configs: Unpack[_SaveCacheConfigs]
    ) -> int:
        return self._runtime_handle._save_cache_model(dirname, **configs)

    @is_non_distributed_check
    @inited_runtime_handler
    def check_save_pre_patch_done(self) -> None:
        return self._runtime_handle._check_save_pre_patch_done()

    @is_non_distributed_check
    @inited_runtime_handler
    def save_cache_table(
        self,
        table_id: int,
        pass_id: int,
        mem_cache_key_threshold: int = 4000000000,
    ) -> None:
        return self._runtime_handle._save_cache_table(
            table_id, pass_id, mem_cache_key_threshold
        )

    @is_non_distributed_check
    @inited_runtime_handler
    def save_one_table(self, table_id: int, path: str, mode: int) -> None:
        """
        save fleet one table from path


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.save_one_table(0, "path", mode=0)

        """
        self._runtime_handle._save_one_table(table_id, path, mode)

    @is_non_distributed_check
    @inited_runtime_handler
    def save_dense_params(
        self,
        executor: Executor,
        dirname: str,
        scope: _Scope,
        program: Program,
        var_names: list[str] | None = None,
    ) -> None:
        """
        save fleet one table from path


        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()
                >>> import paddle
                >>> place = paddle.CPUPlace()
                >>> exe =  paddle.static.Executor(place)

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.save_dense_params(exe, "path", scope=paddle.static.global_scope(), program=paddle.static.default_main_program())

        """
        self._runtime_handle._save_dense_params(
            executor, dirname, scope, program, var_names
        )

    @is_non_distributed_check
    @inited_runtime_handler
    def set_date(self, table_id: int, day_id: str) -> None:
        """
        set_date for gpups table

        Returns:
            None

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init()

                >>> # build net
                >>> # fleet.distributed_optimizer(...)

                >>> fleet.set_date(0, "20250101")

        """
        self._runtime_handle._set_date(table_id, str(day_id))

    @is_non_distributed_check
    @inited_runtime_handler
    def print_table_stat(self, table_id: int, pass_id: int, threshold: float):
        """
        Print stat info of table_id for gpups table, format: tableid, feasign size, mf size.

        Args:

            table_id (int): The id of table.
            pass_id (int): The id of pass.
            threshold (float): The threshold of print.

        Examples:

            .. code-block:: text

                fleet.print_table_stat(0,6,600000)

        """
        self._runtime_handle._print_table_stat(table_id, pass_id, threshold)

    @is_non_distributed_check
    @inited_runtime_handler
    def shrink(self, threshold: int | None = None) -> None:
        self._runtime_handle._shrink(threshold)

    def distributed_optimizer(
        self,
        optimizer: Optimizer,
        strategy: DistributedStrategy | None = None,
    ) -> Self:
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributedStrategy): Extra properties for distributed optimizer.
                It is recommended to use DistributedStrategy in fleet.init(). The strategy
                here is for compatibility. If the strategy in fleet.distributed_optimizer()
                is not None, then it will overwrite the DistributedStrategy in fleet.init(),
                which will take effect in distributed training.

        Returns:
            Fleet: instance of fleet.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed.fleet as fleet
                >>> fleet.init(is_collective=True)
                >>> linear = paddle.nn.Linear(10, 10)
                >>> strategy = fleet.DistributedStrategy()
                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=linear.parameters())
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

        """
        self.user_defined_optimizer = optimizer

        if strategy is not None:
            if self._is_collective:
                logger.warning(
                    "It is recommended to use DistributedStrategy "
                    "in fleet.init(). The strategy here is only for compatibility. "
                    "If the strategy in fleet.distributed_optimizer() is "
                    "not None, then it will overwrite the DistributedStrategy in fleet.init(), "
                    "which will take effect in distributed training."
                )
            self._user_defined_strategy = copy.deepcopy(strategy)

        self._context = {}

        return self

    def _get_amp_optimizer(self):
        # imitate target optimizer retrieval
        amp_optimizer = None
        for optimizer in self.strategy_compiler._get_applied_meta_optimizer():
            if hasattr(optimizer, 'amp_init'):
                amp_optimizer = optimizer
                break

        if amp_optimizer is None:
            if hasattr(self.user_defined_optimizer, 'amp_init'):
                amp_optimizer = self.user_defined_optimizer

        assert (
            amp_optimizer is not None
        ), "amp_init can only be used when the amp(auto mixed precision) strategy is turned on."
        return amp_optimizer

    def get_loss_scaling(self) -> float:
        """Return the real-time loss scaling factor."""
        amp_optimizer = self._get_amp_optimizer()
        return amp_optimizer.get_loss_scaling()

    def amp_init(
        self,
        place: PlaceLike,
        scope: _Scope | None = None,
        test_program: Program | None = None,
        use_fp16_test: bool = False,
    ) -> None:
        """
        Init the amp training, such as cast fp32 parameters to fp16 type.

        Args:
            place(CUDAPlace): place is used to initialize
                fp16 parameters with fp32 values.
            scope(Scope): The scope is used to find fp32 parameters.
            test_program(Program): The program is used for testing.
            use_fp16_test(bool): Whether to use fp16 testing.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn.functional as F
                >>> paddle.enable_static()

                >>> def run_example_code():
                ...     place = paddle.CUDAPlace(0)
                ...     exe = paddle.static.Executor(place)
                ...     data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
                ...     conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
                ...     # 1) Use fp16_guard to control the range of fp16 kernels used.
                ...     with paddle.static.amp.fp16_guard():
                ...         bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                ...         pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                ...         hidden = paddle.static.nn.fc(pool, size=10)
                ...         loss = paddle.mean(hidden)
                ...     # 2) Create the optimizer and set `multi_precision` to True.
                ...     # Setting `multi_precision` to True can avoid the poor accuracy
                ...     # or the slow convergence in a way.
                ...     optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
                ...     # 3) These ops in `custom_black_list` will keep in the float32 computation type.
                ...     amp_list = paddle.static.amp.CustomOpLists(
                ...         custom_black_list=['pool2d'])
                ...     # 4) The entry of Paddle AMP.
                ...     # Enable pure fp16 training by setting `use_pure_fp16` to True.
                ...     optimizer = paddle.static.amp.decorate(
                ...         optimizer,
                ...         amp_list,
                ...         init_loss_scaling=128.0,
                ...         use_dynamic_loss_scaling=True,
                ...         use_pure_fp16=True)
                ...     # If you don't use the default_startup_program(), you should pass
                ...     # your defined `startup_program` into `minimize`.
                ...     optimizer.minimize(loss)
                ...     exe.run(paddle.static.default_startup_program())
                ...     # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
                ...     # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
                ...     optimizer.amp_init(place, scope=paddle.static.global_scope())

                >>> if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 0:
                ...     run_example_code()
        """
        amp_optimizer = self._get_amp_optimizer()
        return amp_optimizer.amp_init(place, scope, test_program, use_fp16_test)

    def _get_qat_optimizer(self):
        # imitate target optimizer retrieval
        qat_optimizer = None
        for optimizer in self.strategy_compiler._get_applied_meta_optimizer():
            if hasattr(optimizer, 'qat_init'):
                qat_optimizer = optimizer
                break

        if qat_optimizer is None:
            if hasattr(self.user_defined_optimizer, 'qat_init'):
                qat_optimizer = self.user_defined_optimizer

        assert (
            qat_optimizer is not None
        ), "qat_init can only be used when the qat(quantization aware training) strategy is turned on."
        return qat_optimizer

    def qat_init(
        self,
        place: PlaceLike,
        scope: _Scope | None = None,
        test_program: Program | None = None,
    ) -> None:
        """
        Init the qat training, such as insert qdq ops and scale variables.

        Args:
            place(CUDAPlace): place is used to initialize
                scale parameters.
            scope(Scope): The scope is used to find parameters and variables.
            test_program(Program): The program is used for testing.
        """
        qat_optimizer = self._get_qat_optimizer()
        return qat_optimizer.qat_init(
            place, scope=scope, test_program=test_program
        )

    def _final_strategy(self):
        if "valid_strategy" not in self._context:
            print(
                "WARNING: You may need to call minimize function before this function is called"
            )
            return {}
        else:
            return self._context["valid_strategy"]

    def _get_applied_meta_list(self):
        if "applied_meta_list" not in self._context:
            print(
                "WARNING: You may need to call minimize function before _get_applied_meta_list called"
            )
            return []
        else:
            return self._context["applied_meta_list"]

    def _get_applied_graph_list(self):
        if "applied_graph_list" not in self._context:
            print(
                "WARNING: You may need to call minimize function before _get_applied_graph_list called"
            )
            return []
        else:
            return self._context["applied_graph_list"]

    def minimize(
        self,
        loss: Tensor,
        startup_program: Program | None = None,
        parameter_list: Iterable[Tensor | str] | None = None,
        no_grad_set: set[Tensor | str] | None = None,
    ) -> tuple[
        list[Operator],
        list[tuple[Parameter, Tensor]],
    ]:
        """
        Add distributed operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_paddle_static_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_paddle_static_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) tensor pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to
            indicate program pruning. If so, the program will be pruned by ``feed`` and
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet
                >>> import paddle.nn.functional as F

                >>> hid_dim = 10
                >>> label_dim = 2
                >>> input_x = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
                >>> input_y = paddle.static.data(name='y', shape=[None, 1], dtype='int64')
                >>> fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim, activation='tanh')
                >>> fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim, activation='tanh')
                >>> prediction = paddle.static.nn.fc(x=[fc_2], size=label_dim, activation='softmax')
                >>> cost = F.cross_entropy(input=prediction, label=input_y)
                >>> avg_cost = paddle.mean(x=cost)

                >>> fleet.init(is_collective=True)
                >>> strategy = fleet.DistributedStrategy()
                >>> linear = paddle.nn.Linear(10, 10)
                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=linear.parameters())
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
                >>> optimizer.minimize(avg_cost)

                >>> # for more examples, please reference https://github.com/PaddlePaddle/PaddleFleetX

        """
        if not isinstance(loss, list):
            return self._minimize_impl(
                loss, startup_program, parameter_list, no_grad_set
            )
        else:
            if (
                in_dynamic_mode()
                or self._role_maker._is_non_distributed()
                or self._is_collective
            ):
                raise ValueError("loss can be list only in PS mode")
            return self._minimize_losses_impl(
                loss, startup_program, parameter_list, no_grad_set
            )

    def _minimize_impl(
        self,
        loss,
        startup_program=None,
        parameter_list=None,
        no_grad_set=None,
    ):
        context = {}
        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy
        )
        if in_dynamic_mode():
            # imitate target optimizer retrieval
            target_opt = self.user_defined_optimizer
            self._context = context
            return target_opt.minimize(loss)
        else:
            # cache original feed forward program
            self.origin_main_program = loss.block.program
            # add distributed attr
            if not hasattr(self.origin_main_program, "distributed_info_"):
                self.origin_main_program.distributed_info_ = {}
                self.origin_main_program.distributed_info_["dp_degree"] = (
                    self._user_defined_strategy.sharding_configs["dp_degree"]
                )
                self.origin_main_program.distributed_info_["mp_degree"] = (
                    self._user_defined_strategy.sharding_configs["mp_degree"]
                )
                self.origin_main_program.distributed_info_["pp_degree"] = (
                    self._user_defined_strategy.sharding_configs["pp_degree"]
                )
                self.origin_main_program.distributed_info_[
                    "sharding_degree"
                ] = self._user_defined_strategy.sharding_configs[
                    "sharding_degree"
                ]

            context["origin_main_program"] = self.origin_main_program
            context["origin_main_programs"] = [self.origin_main_program]
            context["loss"] = loss
            if startup_program is None:
                self.origin_startup_program = (
                    paddle.static.default_startup_program().clone(
                        for_test=False
                    )
                )
                startup_program = paddle.static.default_startup_program()
            else:
                self.origin_startup_program = startup_program.clone(
                    for_test=False
                )

            context["origin_startup_program"] = startup_program
            context["origin_startup_programs"] = [startup_program]
            context["role_maker"] = self._role_maker

            # Use the auto-parallel's routines instead
            if (
                self._user_defined_strategy.semi_auto
                or self._user_defined_strategy.auto_search
            ):
                from ..auto_parallel.static.parallelizer import AutoParallelizer

                auto_parallelizer = AutoParallelizer(self)
                (
                    optimize_ops,
                    params_grads,
                    dist_startup_prog,
                    dist_main_prog,
                ) = auto_parallelizer.parallelize(
                    loss, startup_program, parameter_list, no_grad_set
                )

                return (
                    optimize_ops,
                    params_grads,
                    dist_startup_prog,
                    dist_main_prog,
                )

            context["user_defined_strategy"] = copy.deepcopy(
                self._user_defined_strategy
            )
            copy_user_defined_strategy = copy.deepcopy(
                self._user_defined_strategy
            )

            can_not_apply_optimizer_list = []

            valid_optimizer_list = []
            valid_graph_optimizer_list = []
            skip_names = []
            if (
                self._is_collective
                and len(self._user_defined_strategy.sparse_table_configs) > 0
            ):
                skip_names.append("ShardingOptimizer")
            # compile time
            distributed_optimizer_list = (
                MetaOptimizerFactory()._get_valid_meta_optimizers(
                    self.user_defined_optimizer, skip_names
                )
            )
            # trigger the auto-parallel in very strict condition
            # strategy = DistributedStrategy()
            # strategy.auto = True
            # optimizer = paddle.optimizer.SGD(learning_rate=0.1)
            # optimizer = fleet.distributed_optimizer(optimizer, strategy)
            if copy_user_defined_strategy._is_strict_auto():
                # turn on all the strategy for each optimizer
                for opt in distributed_optimizer_list:
                    opt._enable_strategy(copy_user_defined_strategy, context)

            valid_optimizer_list = []
            valid_graph_optimizer_list = []
            # recall meta optimizers for ranking
            for opt in distributed_optimizer_list:
                opt._set_basic_info(
                    loss,
                    self._role_maker,
                    self.user_defined_optimizer,
                    copy_user_defined_strategy,
                )
                if opt._can_apply() and not opt._is_graph_out():
                    valid_optimizer_list.append(opt)
                elif opt._can_apply() and opt._is_graph_out():
                    valid_graph_optimizer_list.append(opt)
                else:
                    can_not_apply_optimizer_list.append(opt)
            # fix set collective and fleet ps gpu error
            if (
                self._is_collective
                and len(self._user_defined_strategy.sparse_table_configs) > 0
            ):
                context["use_fleet_ps"] = True

                from .meta_optimizers import ParameterServerOptimizer

                meta_optimizer = ParameterServerOptimizer(
                    self.user_defined_optimizer
                )
                meta_optimizer._set_basic_info(
                    loss,
                    self._role_maker,
                    self.user_defined_optimizer,
                    copy_user_defined_strategy,
                )
                valid_optimizer_list.clear()
                valid_optimizer_list.append(meta_optimizer)
                can_not_apply_optimizer_list.append(meta_optimizer)

                # meaningless, just for compatibility with other code
                graph_optimizer = None

                # valid_graph_optimizer_list.clear()
                # valid_graph_optimizer_list.append(graph_optimizer)
                # can_not_apply_optimizer_list.append(graph_optimizer)

            print("valid_optimizer_list=", valid_optimizer_list)
            # combine recalled meta optimizers to be a valid meta optimizer
            (
                meta_optimizer,
                graph_optimizer,
            ) = self.strategy_compiler.generate_optimizer(
                loss,
                self._role_maker,
                self.user_defined_optimizer,
                copy_user_defined_strategy,
                valid_optimizer_list,
                valid_graph_optimizer_list,
            )
            print("meta_optimizer=", meta_optimizer)
            print("graph_optimizer=", graph_optimizer)

            valid_strategy = self.strategy_compiler._get_valid_strategy(
                copy_user_defined_strategy, can_not_apply_optimizer_list
            )

            context["valid_strategy"] = copy.deepcopy(valid_strategy)
            logger.debug("valid_strategy: " + str(context["valid_strategy"]))
            logger.debug(
                "user_defined_strategy: "
                + str(context["user_defined_strategy"])
            )

            applied_meta_list = self.strategy_compiler._get_applied_meta_list()
            applied_graph_list = (
                self.strategy_compiler._get_applied_graph_list()
            )

            context['applied_meta_list'] = applied_meta_list
            context['applied_graph_list'] = applied_graph_list

            self._context = context

            self.valid_strategy = valid_strategy
            self.valid_strategy._enable_env()

            optimize_ops = []
            params_grads = []

            if (
                self._role_maker._is_non_distributed()
                and not self._is_collective
            ):
                if self._runtime_handle is None:
                    self._runtime_handle = RuntimeFactory()._create_runtime(
                        context
                    )

                compiled_program = compiler.CompiledProgram(
                    self.origin_main_program
                )
                loss.block.program._graph = compiled_program
                return self.user_defined_optimizer.minimize(
                    loss,
                    startup_program,
                    parameter_list,
                    no_grad_set=no_grad_set,
                )

            if meta_optimizer:
                logger.debug(
                    "before minimize program id: " + str(id(loss.block.program))
                )
                optimize_ops, params_grads = meta_optimizer.minimize(
                    loss,
                    startup_program,
                    parameter_list,
                    no_grad_set=no_grad_set,
                )
                logger.debug(
                    "after minimize program id: " + str(id(loss.block.program))
                )
                default_program = paddle.static.default_main_program()
                logger.debug("default program id: " + str(id(default_program)))

                if id(default_program) != id(loss.block.program):
                    paddle.framework.switch_main_program(loss.block.program)
                logger.debug(
                    "default program id after switch: "
                    + str(id(default_program))
                )

            else:
                (
                    optimize_ops,
                    params_grads,
                ) = self.user_defined_optimizer.minimize(
                    loss,
                    startup_program,
                    parameter_list,
                    no_grad_set=no_grad_set,
                )

            context["program_optimize_ops"] = optimize_ops
            context["program_params_grads"] = params_grads

            if graph_optimizer:
                logger.debug(
                    "before graph minimize program id: "
                    + str(id(loss.block.program))
                )
                optimize_ops, params_grads = graph_optimizer.minimize(
                    loss,
                    startup_program,
                    parameter_list,
                    no_grad_set=no_grad_set,
                )
                # since we do not encourage users to use graph operations
                # if a graph optimizer takes effect, mostly
                # optimizers_ops and params_grads are None
                # i.e. users can not modify current computation graph anymore
                context["graph_optimize_ops"] = optimize_ops
                context["graph_optimize_grads"] = params_grads
            elif loss.block.program._pass_applied is None:
                apply_ir_passes(loss.block.program, startup_program, self)

            if not self._role_maker._is_heter_parameter_server_mode:
                program = paddle.static.default_main_program()
                opt_info = (
                    {} if program._fleet_opt is None else program._fleet_opt
                )
                opt_info["mpi_size"] = self.worker_num()
                opt_info["mpi_rank"] = self.worker_index()
                for (
                    k,
                    v,
                ) in self._user_defined_strategy.trainer_desc_configs.items():
                    if v or k not in opt_info:
                        opt_info[k] = v
                program._fleet_opt = opt_info

            if self._runtime_handle is None:
                self._runtime_handle = RuntimeFactory()._create_runtime(context)

            from paddle.distributed import fleet

            fleet.util._set_strategy(context["valid_strategy"])

            return optimize_ops, params_grads

    def _minimize_losses_impl(
        self,
        losses,
        startup_programs=None,
        parameter_list=None,
        no_grad_set=None,
    ):
        context = {}

        # cache original feed forward program
        self.origin_main_program = losses[0].block.program
        context["origin_main_program"] = self.origin_main_program
        context["origin_main_programs"] = []
        for loss in losses:
            context["origin_main_programs"].append(loss.block.program)
        context["loss"] = losses

        if startup_programs is None:
            if len(losses) == 1:
                startup_programs = [paddle.static.default_startup_program()]
            else:
                raise ValueError(
                    "startup_program can't be None when loss is list."
                )
        ori_startup_programs = startup_programs.copy()
        self.origin_startup_program = startup_programs[0].clone(for_test=False)
        context["origin_startup_program"] = startup_programs[0]
        context["origin_startup_programs"] = []
        for program in startup_programs:
            context["origin_startup_programs"].append(program)

        context["role_maker"] = self._role_maker

        context["user_defined_strategy"] = copy.deepcopy(
            self._user_defined_strategy
        )

        context["valid_strategy"] = copy.deepcopy(self._user_defined_strategy)

        self._context = context

        self.valid_strategy = context["valid_strategy"]
        self.valid_strategy._enable_env()

        optimize_ops = []
        params_grads = []

        from .meta_optimizers import ParameterServerOptimizer

        ps_optimizer = ParameterServerOptimizer(self.user_defined_optimizer)
        ps_optimizer._set_basic_info(
            losses,
            self._role_maker,
            self.user_defined_optimizer,
            self._user_defined_strategy,
        )
        optimize_ops, params_grads = ps_optimizer.minimize_losses_impl(
            losses, startup_programs, parameter_list, no_grad_set=no_grad_set
        )

        # default_program = paddle.static.default_main_program()

        # if id(default_program) != id(losses[0].block.program):
        #     paddle.framework.switch_main_program(losses[0].block.program)
        # join phase program add communication ops from startup_programs. But python return original startup_program
        for i in range(len(ori_startup_programs)):
            ori_startup_programs[i]._rebuild_from_desc(startup_programs[i].desc)
        context["program_optimize_ops"] = optimize_ops
        context["program_params_grads"] = params_grads

        for loss in losses:
            program = loss.block.program
            opt_info = {} if program._fleet_opt is None else program._fleet_opt
            opt_info["mpi_size"] = self.worker_num()
            opt_info["mpi_rank"] = self.worker_index()
            for (
                k,
                v,
            ) in self._user_defined_strategy.trainer_desc_configs.items():
                if v or k not in opt_info:
                    opt_info[k] = v
            program._fleet_opt = opt_info
            logger.info(
                "fleet base opt info: "
                + str(id(program))
                + str(program._fleet_opt)
            )

        if self._runtime_handle is None:
            self._runtime_handle = RuntimeFactory()._create_runtime(context)

        from paddle.distributed import fleet

        fleet.util._set_strategy(context["valid_strategy"])

        return optimize_ops, params_grads
