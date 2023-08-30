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

import os
import random
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass():
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())
    paddle.utils.unique_name.switch()


class Test1F1BPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 2
        self.batch_num = 5
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, mode):
        reset_prog()

        strategy = apply_pass()
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model(mode)

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=self.rtol,
            atol=self.atol,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_pp_pass(self):
        os.environ['FLAGS_new_executor_micro_batching'] = 'True'
        os.environ['FLAGS_enable_new_ir_in_executor'] = 'True'

        # data parallel
        engine_dp = self.get_engine("dp")
        outs = engine_dp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        # navie pipeline parallel without schedule
        engine_pp = self.get_engine("pp")
        outs = engine_pp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        assert os.environ.get('FLAGS_new_executor_micro_batching') == "True"
        assert os.environ.get('FLAGS_enable_new_ir_in_executor') == "True"


if __name__ == "__main__":
    unittest.main()
