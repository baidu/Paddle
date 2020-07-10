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

from paddle.fleet.collective import RecomputeOptimizer

meta_optimizer_names = ["RecomputeOptimizer"]


class MetaOptimizerFactory(object):
    def __init__(self):
        pass

    def _get_valid_meta_optimizers(self):
        opt_list = []
        for opt_name in meta_optimizer_names:
            opt_list.append(globals()[opt_name]())


class MetaOptimizerBase(object):
    def __init__(self, optimizer):
        pass

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        self.loss = loss
        self.role_maker = role_maker
        self.user_defined_optimizer = user_defined_optimizer
        self.user_defined_strategy = user_defined_strategy

    def _update_inner_optimier(self, optimizer):
        self.inner_opt = optimizer

    def _can_update(self, optimizer):
        if str(optimizer.__class__.__name__) in self.meta_optimizers_white_list:
            return True

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        raise NotImplementedError("meta optimizer not implemented")

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        optimize_ops, params_grads = self.minimize_impl(
            loss, startup_program, parameter_list, no_grad_set)

        if isinstance(self.inner_opt, MetaOptimizerBase):
            self.inner_opt.optimize_ops = optimize_ops
            self.inner_opt.params_grads = params_grads
