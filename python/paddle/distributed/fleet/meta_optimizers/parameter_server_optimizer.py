#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid
from .meta_optimizer_base import MetaOptimizerBase
from paddle.fluid import core
import subprocess
import re
import platform


class ParameterServerOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ParameterServerOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _is_graph_out(self):
        return False

    def _can_apply(self):
        if self.role_maker._is_collective:
            return False
        if self.user_defined_strategy.auto == True:
            return True

        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        return True if k_steps >= 0 else False

    def _get_distributed_strategy(self):
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        strategy = None

        if not self.user_defined_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_sync_strategy()

        if self.user_defined_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_async_strategy()

        if self.user_defined_strategy.a_sync and k_steps > 0:
            strategy = StrategyFactory.create_geo_strategy(k_steps)

        if not strategy:
            raise ValueError("k_steps must be invalid value, please check")

        return strategy

    def _build_trainer_programs(self, compiled_config):
        from paddle.fluid.incubate.fleet.parameter_server.ir import trainer_pass as worker

        _main = compiled_config.origin_main_program.clone()
        _startup = compiled_config.origin_startup_program.clone()

        if not compiled_config.is_geo_mode():
            # for main program
            _main = worker.delete_optimizer_pass(_main, compiled_config)
            _main = worker.distributed_ops_pass(_main, compiled_config)
            _main = worker.append_send_ops_pass(_main, compiled_config)

            # for startup program
            _startup = worker.fake_init_ops_pass(_startup, compiled_config)
            _startup = worker.init_from_server_pass(_startup, compiled_config)
            _startup = worker.delet_extra_optimizes_pass(_startup,
                                                         compiled_config)

            # for heter program
            if self.role_maker._is_heter_parameter_server_mode:
                from paddle.fluid.incubate.fleet.parameter_server.ir import heter_trainer_pass as heter_worker
                if self.role_maker._is_heter_worker():
                    # for heter worker
                    _main = heter_worker.split_heter_worker_ops_pass(
                        _main, compiled_config)
                else:
                    # for default worker
                    _main = heter_worker.split_trainer_ops_pass(_main,
                                                                compiled_config)
                # for startup change
                _startup = heter_worker.delete_startup_useless_ops_var_pass(
                    _startup, _main, compiled_config)
        else:
            _main = worker.append_send_ops_pass(_main, compiled_config)
            _startup = _startup

        return _main, _startup

    def _build_pserver_programs(self, compiled_config):
        from paddle.fluid.incubate.fleet.parameter_server.ir import pserver_pass as server

        _main = fluid.Program()
        _startup = fluid.Program()

        if not compiled_config.is_geo_mode():
            _main = server.add_listen_and_serv_pass(_main, compiled_config)
            _main = server.add_rpc_global_flags_pass(_main, compiled_config)
            _main = server.add_optimizer_pass(_main, compiled_config)
            _main = server.large_scale_sparse_pass(_main, _main,
                                                   compiled_config, False)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.large_scale_sparse_pass(_startup, _main,
                                                      compiled_config, True)

            if not compiled_config.is_sync_mode():
                _main = server.delete_unused_in_main_pass(_main,
                                                          compiled_config)

            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)
        else:
            _main = server.add_listen_and_serv_pass(_main, compiled_config)
            _main = server.add_rpc_global_flags_pass(_main, compiled_config)
            _main = server.add_geo_optimizer_pass(_main, compiled_config)
            _main = server.large_scale_sparse_pass(_main, _main,
                                                   compiled_config, False)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.large_scale_sparse_pass(_startup, _main,
                                                      compiled_config, True)
            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)

        return _main, _startup

    def _try_auto_apply_geo(self, program, compiled_config):
        def get_sys_free_mem():
            plat = platform.system()
            if platform.system() == "Darwin":
                vm = subprocess.Popen(
                    ['vm_stat'], stdout=subprocess.PIPE).communicate()[0]
                # Process vm_stat
                vmLines = vm.split('\n')
                sep = re.compile(':[\s]+')
                vmStats = {}
                for row in range(1, len(vmLines) - 2):
                    rowText = vmLines[row].strip()
                    rowElements = sep.split(rowText)
                    vmStats[(rowElements[0]
                             )] = int(rowElements[1].strip('\.')) * 4096
                return vmStats["Pages free"]
            elif platform.system() == "Linux":
                mems = {}
                with open('/proc/meminfo', 'rb') as f:
                    for line in f:
                        fields = line.split()
                        mems[fields[0]] = int(fields[1]) * 1024
                free = mems[b'MemFree:']
                return free
            else:
                raise ValueError(
                    "%s platform is unsupported is parameter server optimizer" %
                    (platform.system()))

        if self.user_defined_strategy.auto == False:
            return

        a_sync_configs = self.user_defined_strategy.a_sync_configs
        if a_sync_configs["k_steps"] >= 0:
            return

        self.user_defined_strategy.a_sync = True
        if not isinstance(self.inner_opt, fluid.optimizer.SGDOptimizer):
            # auto async
            a_sync_configs["k_steps"] = 0
            self.user_defined_strategy.a_sync_configs = a_sync_configs
            return

        from paddle.fluid.incubate.fleet.parameter_server.ir.vars_metatools import dtype_to_size
        free = get_sys_free_mem()

        param_grad_pairs = compiled_config.origin_sparse_pairs + compiled_config.origin_dense_pairs
        processed_var_names = set(["@EMPTY@"])

        param_memory_size = 0
        for param_grad_pair in param_grad_pairs:
            param, grad = param_grad_pair
            param_memory_size += param.m_size
            processed_var_names.add(param.name)

        upper_mem_use = param_memory_size * 5.0

        program_tmp_vars = dict()
        batch_size = 1024
        for op in program.global_block().ops:
            for var_name in op.output_arg_names:
                if var_name in processed_var_names:
                    continue
                processed_var_names.add(var_name)
                var = program.global_block().vars[var_name]

                if var.desc.type() != core.VarDesc.VarType.LOD_TENSOR:
                    continue

                data_count = 1
                neg_dim_count = 0
                for x in var.shape:
                    if x < 0:
                        if neg_dim_count >= 1:
                            raise ValueError(
                                "Var %s has more than one negative dim." %
                                (var_name))
                        neg_dim_count += 1
                        data_count *= (-x)
                    else:
                        data_count *= x
                program_tmp_vars[var_name] = (data_count, neg_dim_count,
                                              dtype_to_size[var.dtype])

        for varname in program_tmp_vars:
            data_count, neg_dim_count, type_size = program_tmp_vars[varname]
            if neg_dim_count == 1:
                data_count *= batch_size
            var_memory = data_count * type_size
            upper_mem_use += var_memory

        if upper_mem_use < free:
            # auto geo
            a_sync_configs["k_steps"] = 800
        else:
            # auto async
            a_sync_configs["k_steps"] = 0
        self.user_defined_strategy.a_sync_configs = a_sync_configs

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.inner_opt.minimize(loss, startup_program, parameter_list,
                                no_grad_set)

        _origin_main_program = loss.block.program
        _origin_startup_program = startup_program
        from paddle.fluid.incubate.fleet.parameter_server.ir import public as public

        compiled_config = public.CompileTimeStrategy(_origin_main_program,
                                                     _origin_startup_program,
                                                     None, self.role_maker)

        self._try_auto_apply_geo(_origin_main_program, compiled_config)

        strategy = self._get_distributed_strategy()
        compiled_config.strategy = strategy

        if self.role_maker.is_worker() or self.role_maker._is_heter_worker():
            main_program, startup_program = self._build_trainer_programs(
                compiled_config)
        elif self.role_maker.is_server():
            main_program, startup_program = self._build_pserver_programs(
                compiled_config)

        loss.block.program = main_program
        fluid.framework.switch_startup_program(startup_program)

        return None, None

    def _disable_strategy(self, dist_strategy):
        dist_strategy.a_sync_configs = {}
        self.user_defined_strategy.a_sync_configs = {}

    def _enable_strategy(self, dist_strategy):
        dist_strategy.a_sync = True
        dist_strategy.a_sync_configs = {}
