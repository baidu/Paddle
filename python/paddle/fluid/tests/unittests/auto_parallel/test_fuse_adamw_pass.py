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

import unittest

import numpy as np

import paddle
import paddle.nn as nn
from paddle.distributed.passes import PassManager, new_pass


def apply_passes(main_prog, startup_prog):
    pass_manager = PassManager([new_pass("fuse_adamw")])
    pass_manager.apply([main_prog], [startup_prog])


class MLPLayer(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size, n):
        super(MLPLayer, self).__init__()
        self.linear_first = nn.Linear(input_size, hidden_size)
        self.decoder_layers = nn.LayerList()
        for i in range(n):
            self.decoder_layers.append(nn.Linear(hidden_size, hidden_size))

        self.linear_last = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_first(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.linear_last(x)
        return x.mean()


class TestFuseAdamWPass(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.input_size = 30
        self.hidden_size = 50
        self.output_size = 20
        self.n = 2

    def get_loss_data(self, place, use_amp=False, use_apply_passes=False):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        if place == 'cpu':
            use_amp = False
        exe = paddle.static.Executor(place=place)
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.AdamW(multi_precision=use_amp)
        if use_amp:
            optimizer = paddle.static.amp.decorate(
                optimizer,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True,
                use_pure_fp16=True,
                use_fp16_guard=False,
            )
        with paddle.static.program_guard(train_program, startup_program):
            if use_amp:
                data = paddle.static.data(
                    shape=[10, 30], name='X', dtype='float16'
                )
            else:
                data = paddle.static.data(
                    shape=[10, 30], name='X', dtype='float32'
                )
            model = MLPLayer(30, 50, 20, 2)
            out = model(data)
            loss = paddle.mean(out)
            optimizer.minimize(loss)

        apply_passes(train_program, startup_program)

        exe.run(startup_program)
        if use_amp:
            optimizer.amp_init(place=place, scope=paddle.static.global_scope())
            x = np.random.random(size=(10, 30)).astype('float16')
        else:
            x = np.random.random(size=(10, 30)).astype('float32')
        for _ in range(5):
            loss_data = exe.run(
                train_program, feed={"X": x}, fetch_list=[loss.name]
            )
        return loss_data

    def test_fuse_adamw_pass(self):
        place = paddle.CUDAPlace(0)
        for use_amp in [True, False]:
            loss_without_passes = self.get_loss_data(place, use_amp, False)
            loss_with_passes = self.get_loss_data(place, use_amp, True)
            np.testing.assert_allclose(
                np.array(loss_without_passes),
                np.array(loss_with_passes),
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
