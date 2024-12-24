#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import json
import logging
import os
import pathlib
import pickle
import shlex
import subprocess
import sys
import time

import paddle
from paddle.distributed.passes import PassContext, new_pass
from paddle.distributed.utils.log_utils import get_logger
from paddle.framework import core
from paddle.static import append_backward, program_guard

from .cluster import Cluster
from .completion import Completer
from .dist_context import DistributedContext, set_default_distributed_context
from .dist_op import DistributedOperator
from .dist_tensor import DistributedTensor
from .mapper import mapping
from .partitioner import Partitioner
from .planner import Planner
from .process_group import (
    ProcessGroup,
    _g_process_group_map,
    get_all_process_groups,
    get_process_group,
    get_world_process_group,
)
from .reshard import Resharder
from .utils import SerialProgramInfo, make_data_unshard

_logger = get_logger(logging.INFO)


class AutoParallelizer:
    """
    AutoParallelizer is the main controller class to do the auto parallel process.
    And the auto parallel process will be triggered in the wrapped parallelize function.
    To facilitate the auto parallelization, it will contain information about program, cluster and the
    related context. In this basic version, the program information will be retrieved from
    Fleet object, and the cluster information can be retrieved in the new created Cluster object,
    and the context information can be retrieved in the new created DistributedContext.
    """

    def __init__(self, fleet):
        self._fleet = fleet
        self._optimizer = self._fleet.user_defined_optimizer
        self._dist_strategy = self._fleet._user_defined_strategy
        self._dist_context = DistributedContext()
        self._cluster = None
        self._cluster_topo_path = os.getenv("PADDLE_CLUSTER_TOPO_PATH", None)
        if self._cluster_topo_path is not None:
            self._cluster = Cluster()
            self._cluster.build_from_file(self._cluster_topo_path)
        # Prepare information for auto mapping
        self._rank_mapping_path = os.getenv("PADDLE_RANK_MAPPING_PATH", None)
        enable_auto_mapping_env = os.getenv("PADDLE_ENABLE_AUTO_MAPPING", None)
        if enable_auto_mapping_env is None:
            self._enable_auto_mapping = False
        else:
            self._enable_auto_mapping = True
        self._pass_context = PassContext()

        self._need_rank_mapping = os.getenv("PADDLE_NEED_RANK_MAPPING")
        self._need_rank_mapping = (
            True
            if self._need_rank_mapping
            and self._need_rank_mapping.lower() == 'true'
            else False
        )
        # self._pass_context = None

    def _remove_distributed_attrs(self, main_program):
        suffix = core.kAutoParallelSuffix()
        # distributed attributes for variable have been removed
        # in previous process.
        for block in main_program.blocks:
            for op in block.ops:
                for attr_name in op.attr_names:
                    if suffix in attr_name:
                        op._remove_attr(attr_name)

    def _apply_pre_optimization_passes(
        self, main_program, startup_program, loss, params_grads, no_grad_set
    ):
        # apply amp pass
        if self._dist_strategy.amp:
            config = copy.deepcopy(self._dist_strategy.amp_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["loss"] = loss
            if config["use_pure_fp16"]:
                config["base_opt"] = self._optimizer
                auto_parallel_fp16_pass = new_pass("auto_parallel_fp16", config)
                auto_parallel_fp16_pass.apply(
                    [main_program], [startup_program], self._pass_context
                )
                loss = auto_parallel_fp16_pass.get_loss()
            else:
                auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
                auto_parallel_amp_pass.apply(
                    [main_program], [startup_program], self._pass_context
                )
                loss = auto_parallel_amp_pass.get_loss()

        # apply recompute pass
        if self._dist_strategy.recompute:
            config = copy.deepcopy(self._dist_strategy.recompute_configs)
            config["dist_context"] = self._dist_context
            config["no_grad_set"] = copy.deepcopy(no_grad_set)
            config["loss"] = loss
            auto_parallel_recompute_pass = new_pass(
                "auto_parallel_recompute", config
            )
            auto_parallel_recompute_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

    def _generate_backward(
        self,
        main_program,
        startup_program,
        loss,
        parameter_list,
        no_grad_set,
        callbacks,
    ):
        with program_guard(main_program, startup_program):
            params_grads = append_backward(
                loss,
                parameter_list,
                no_grad_set,
                callbacks,
                distop_context=self._dist_context.dist_op_context,
            )
        self._completer = Completer(self._dist_context)
        self._completer.complete_backward_annotation(main_program)
        self._dist_context.block_state.parse_backward_blocks(main_program)
        return params_grads

    def _apply_optimize(self, main_program, startup_program, params_grads):
        optimizer = copy.deepcopy(self._optimizer)
        with program_guard(main_program, startup_program):
            optimize_ops = optimizer.apply_gradients(params_grads)

        self._dist_context._serial_optimizer = optimizer
        # update completion
        self._completer = Completer(self._dist_context)
        self._completer.complete_update_annotation(main_program)

        return optimize_ops

    def _apply_post_optimization_passes(
        self, main_program, startup_program, rank, params_grads
    ):
        if self._dist_strategy.sharding:
            config = copy.deepcopy(self._dist_strategy.sharding_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["global_rank"] = rank
            auto_parallel_sharding_pass = new_pass(
                "auto_parallel_sharding", config
            )
            auto_parallel_sharding_pass.apply(
                [main_program], [startup_program], self._pass_context
            )
            params_grads = self._pass_context.get_attr("params_grads")

        config = copy.deepcopy(self._dist_strategy.sharding_configs)
        config["dist_context"] = self._dist_context
        config["params_grads"] = params_grads
        config["rank_id"] = rank
        auto_parallel_clip_pass = new_pass("auto_parallel_grad_clip", config)
        auto_parallel_clip_pass.apply(
            [main_program], [startup_program], self._pass_context
        )

        if self._dist_strategy.gradient_merge:
            config = copy.deepcopy(self._dist_strategy.gradient_merge_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config
            )
            auto_parallel_gradient_merge_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

    def _get_dist_program(self, rank, dist_context=None, relaunch_phase=False):
        completed_main_program = None
        serial_main_program = self._main_program.clone()
        serial_startup_program = self._startup_program.clone()
        serial_loss = serial_main_program.global_block().var(self._loss.name)

        # generating serial
        if dist_context is None:
            # Annotation completion
            self._dist_context = DistributedContext()
            _logger.info("Start annotation dist attr.")
            self._completer = Completer(self._dist_context)
            completed_main_program = (
                self._completer.complete_forward_annotation(serial_main_program)
            )
        else:
            completed_main_program = serial_main_program
            self._dist_context = copy.deepcopy(dist_context)

        # parse forward sub block
        self._dist_context.block_state.parse_forward_blocks(serial_main_program)

        # serial backward pass
        params_grads = self._generate_backward(
            completed_main_program,
            serial_startup_program,
            serial_loss,
            self._parameter_list,
            self._no_grad_set,
            self._callbacks,
        )

        # serial forward pass
        self._apply_pre_optimization_passes(
            completed_main_program,
            serial_startup_program,
            serial_loss,
            params_grads,
            self._no_grad_set,
        )
        # Logical partition
        partitioner = Partitioner(self._dist_context, rank)
        (
            dist_main_prog,
            dist_startup_prog,
            dist_params_grads,
        ) = partitioner.partition(
            completed_main_program, serial_startup_program, params_grads
        )

        # TODO refactor the placement of optimizer
        # generate optimize program
        dist_optimize_ops = self._apply_optimize(
            dist_main_prog, dist_startup_prog, dist_params_grads
        )

        make_data_unshard(dist_main_prog, dist_startup_prog, self._dist_context)

        resharder = Resharder(
            dist_main_prog,
            dist_startup_prog,
            rank,
            self._dist_context,
            dist_params_grads,
        )
        resharder.reshard()

        self._apply_post_optimization_passes(
            dist_main_prog, dist_startup_prog, rank, dist_params_grads
        )
        g_process_group_map = None
        if not relaunch_phase:
            g_process_group_map = copy.deepcopy(_g_process_group_map)
            _g_process_group_map.clear()
            _g_process_group_map[0] = ProcessGroup(0, [])
            for process_mesh in self._dist_context._process_meshes:
                _g_process_group_map[0].add_ranks(process_mesh.process_ids)
        return (
            dist_optimize_ops,
            dist_params_grads,
            dist_startup_prog,
            dist_main_prog,
            g_process_group_map,
        )

    def parallelize(
        self,
        loss,
        startup_program,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    ):
        assert startup_program is not None
        self._loss = loss
        self._startup_program = startup_program
        self._main_program = loss.block.program
        self._parameter_list = parameter_list
        self._no_grad_set = no_grad_set
        self._callbacks = callbacks

        if self._enable_auto_mapping and self._need_rank_mapping:
            # Do the mapping pass before parallelization
            assert (
                self._cluster is not None
            ), "The cluster must not be none when using auto mapping."
            dist_programs = {}
            world_process_group = get_world_process_group()
            dist_context = None
            # auto search
            if self._dist_strategy.auto_search:
                logging.info("Start searching dist attr.")
                serial_program_info = SerialProgramInfo(
                    self._main_program,
                    self._startup_program,
                    self._loss,
                    self._optimizer,
                    self._cluster,
                )
                planner = Planner(
                    serial_program_info,
                    self,
                    algorithm_config={"name": "mcmc", "max_search_times": 5},
                )
                dist_context, _ = planner.search()
                logging.info("End searching dist attr.")

            # serialize the dist context by planner
            if dist_context is not None:
                logging.info("Start serialize searched dist attr")
                cwd = pathlib.Path().cwd()
                searched_dist_context_path = os.path.join(
                    cwd, f"searched_dist_context_{time.time()}.pkl"
                )
                saved_dist_context = {}
                ops_dist_attr = {}
                tensors_dist_attr = {}
                for key, dist_op in dist_context._dist_ops_for_program.items():
                    ops_dist_attr[key] = dist_op.dist_attr
                for (
                    key,
                    dist_tensor,
                ) in dist_context._dist_tensors_for_program.items():
                    tensors_dist_attr[key] = dist_tensor.dist_attr
                saved_dist_context["ops_dist_attr"] = ops_dist_attr
                saved_dist_context["tensors_dist_attr"] = tensors_dist_attr
                saved_dist_context["process_meshes"] = (
                    dist_context._process_meshes
                )
                with open(
                    searched_dist_context_path, "wb"
                ) as dist_context_file:
                    pickle.dump(saved_dist_context, dist_context_file)
                    os.environ['PADDLE_SEARCHED_DIST_CONTEXT_PATH'] = (
                        searched_dist_context_path
                    )
                    logging.info(
                        f"End serialize searched dist attr to {searched_dist_context_path}"
                    )

            for rank in world_process_group.ranks:
                (
                    dist_optimize_ops,
                    dist_params_grads,
                    dist_startup_prog,
                    dist_main_prog,
                    g_process_group_map,
                ) = self._get_dist_program(rank, dist_context)
                dist_programs[rank] = [dist_main_prog, g_process_group_map]

            # Do the mapping between the distributed program graph and the cluster graph
            rank_mapping_dict = mapping(dist_programs, self._cluster)
            rank_mapping = list(rank_mapping_dict.values())

            # Relaunch the training by using the rank mapping file
            with open(self._rank_mapping_path, "w") as rank_mapping_file:
                json.dump(rank_mapping, rank_mapping_file)

            enable_elastic = os.getenv("PADDLE_ENABLE_ELASTIC")
            enable_elastic = (
                True
                if enable_elastic and enable_elastic.lower() == 'true'
                else False
            )
            if enable_elastic:
                print("Auto mapping finished, now do elastic re-launch")
                sys.exit(
                    paddle.distributed.fleet.elastic.manager.ELASTIC_AUTO_PARALLEL_EXIT_CODE
                )

            original_cmd_args = os.getenv("PADDLE_ORIGINAL_CMD_ARGS")
            rank_mapping_args = " ".join(
                ["--rank_mapping_path", self._rank_mapping_path]
            )
            if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
                coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
            else:
                coverage_args = []
            new_cmd_args = (
                "-m paddle.distributed.fleet.launch"
                + " "
                + rank_mapping_args
                + " "
                + original_cmd_args
            )
            new_cmd = [
                sys.executable,
                "-u",
                *coverage_args,
                *shlex.split(new_cmd_args),
            ]
            new_process = subprocess.Popen(new_cmd)
            new_process.wait()
            assert (
                new_process.returncode == 0
            ), "Launch failed with rank mapping"
            print("Successfully do the second launch for auto mapping!")
            sys.exit(0)
        else:
            # Parallelization after the mapping pass
            rank = paddle.distributed.get_rank()
            dist_context = None
            searched_dist_context_path = os.getenv(
                "PADDLE_SEARCHED_DIST_CONTEXT_PATH", None
            )
            if searched_dist_context_path is not None:
                with open(
                    searched_dist_context_path, "rb"
                ) as dist_context_file:
                    saved_dist_context = pickle.load(dist_context_file)
                    dist_context = DistributedContext()
                    for op in self._main_program.global_block().ops:
                        dist_attr = saved_dist_context["ops_dist_attr"][
                            op.desc.id()
                        ]
                        dist_op = DistributedOperator(op, dist_attr)
                        dist_context.add_dist_op_for_program(dist_op)

                    vars = self._main_program.global_block().vars
                    for var in vars.values():
                        dist_attr = saved_dist_context["tensors_dist_attr"][
                            var.desc.id()
                        ]
                        dist_tensor = DistributedTensor(var, dist_attr)
                        dist_context.add_dist_tensor_for_program(dist_tensor)

                    dist_context._process_meshes = saved_dist_context[
                        "process_meshes"
                    ]

            else:
                if self._dist_strategy.auto_search:
                    serial_program_info = SerialProgramInfo(
                        self._main_program,
                        self._startup_program,
                        self._loss,
                        self._optimizer,
                        cluster=self._cluster,
                    )
                    planner = Planner(
                        serial_program_info,
                        self,
                        algorithm_config={
                            "name": "mcmc",
                            "max_search_times": 5,
                        },
                    )
                    dist_context, _ = planner.search()

            # rebuild g_process_group
            if dist_context is not None:
                pg0 = get_process_group(0)
                for process_mesh in dist_context._process_meshes:
                    pg0.add_ranks(process_mesh.process_ids)
            (
                dist_optimize_ops,
                dist_params_grads,
                dist_startup_prog,
                dist_main_prog,
                _,
            ) = self._get_dist_program(rank, dist_context, relaunch_phase=True)

            # NOTE: This is a trick to fix hang in pipeline mode when dist context is searched by planner
            if self._dist_strategy.auto_search:
                is_pipeline = False
                for op in dist_main_prog.global_block().ops:
                    if op.type == "p_send" or op.type == "p_recv":
                        is_pipeline = True
                        break
                if is_pipeline:
                    with paddle.static.program_guard(dist_main_prog):
                        paddle.distributed.barrier()

            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()
            for process_group in all_process_groups:
                process_group.instantiate()

            # Copy distributed info to the default context
            set_default_distributed_context(self._dist_context)

            # The last step: remove all distributed attributes to be compatible
            # with inference.
            self._remove_distributed_attrs(dist_main_prog)

            return (
                dist_optimize_ops,
                dist_params_grads,
                dist_startup_prog,
                dist_main_prog,
            )

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (
                k == "_main_program"
                or k == "_startup_program"
                or k == "_dist_context"
                or k == "_fleet"
                or k == "_loss"
            ):
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result
