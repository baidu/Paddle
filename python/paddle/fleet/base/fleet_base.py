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

from __future__ import print_function
from paddle.fleet import RoleMakerBase
from . import obj_creator
from strategy_compiler import StrategyCompiler
from meta_optimizer import MetaOptimizerFactory

__all__ = ['Fleet']


class Fleet(object):
    """
    Unified API for distributed training of PaddlePaddle
    Fleet is initialized through init function
    """

    def __init__(self):
        pass

    def init(self, role_maker):
        self.role_maker = role_maker
        self.strategy_compiler = StrategyCompiler()

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        
        """
        return self._role_maker.is_first_worker()

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id
        """
        return self._role_maker.worker_index()

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers
        """
        return self._role_maker.worker_num()

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.
        """
        return self._role_maker.is_worker()

    def worker_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """

        if to_string:
            return ",".join(self._role_maker.get_trainer_endpoints())
        else:
            return self._role_maker.get_trainer_endpoints()

    def server_num(self):
        """
        Get current total worker number.

        Returns:
            int: server number
        """
        return len(self._role_maker.get_pserver_endpoints())

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id
        """
        return self._role_maker.server_index()

    def server_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """

        if to_string:
            return ",".join(self._role_maker.get_pserver_endpoints())
        else:
            return self._role_maker.get_pserver_endpoints()

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.
        """
        return self._role_maker.is_server()

    @property
    def util(self):
        """
        
        """
        return self._util

    def barrier_worker(self):
        """
        barrier between workers
        """
        self._role_maker.barrier_worker()

    @abc.abstractmethod
    def init_worker(self):
        pass

    @abc.abstractmethod
    def init_server(self, model_dir=None):
        pass

    @abc.abstractmethod
    def run_server(self):
        pass

    @abc.abstractmethod
    def stop_worker(self):
        pass

    def distributed_optimizer(self, optimizer, strategy):
        self.user_defined_optimizer = optimizer
        self.user_defined_strategy = strategy

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set):
        # cache original feed forward program
        self.origin_main_program = loss.block.program
        if startup_program == None:
            self.origin_startup_program = \
                paddle.default_startup_program().clone(for_test=False)
            startup_program = paddle.default_startup_program()
        else:
            self.origin_startup_program = \
                startup_program.clone(for_test=False)

        # compile time
        distributed_optimizer_list = \
            MetaOptimizerFactory()._get_valid_meta_optimizers()
        valid_optimizer_list = []
        # recall meta optimizers for ranking
        for opt in distributed_optimizer_list:
            if opt.can_apply(loss, self.role_maker, self.user_defined_optimizer,
                             self.user_defined_strategy):
                valid_optimizer_list.append(opt)
        # combine recalled meta optimizers to be a valid meta optimizer
        meta_optimizer, compiled_strategy = \
                self.strategy_compiler.generate_optimizer(
                    loss, self.role_maker, self.optimizer,
                    self.strategy, valid_optimizer_list)
        optimize_ops, params_grads = meta_optimizer.minimize(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        return optimize_ops, params_grads
