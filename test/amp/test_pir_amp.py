# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class TestAmpAttrs(unittest.TestCase):
    def test_pir_amp_attrs(self):
        with paddle.pir_utils.IrGuard():
            amp_attrs = core._get_amp_attrs()
            amp_attrs._use_promote = True
            amp_attrs._amp_level = core.AmpLevel.O2
            amp_attrs._amp_dtype = 'float16'
            np.testing.assert_equal(core._get_amp_attrs()._use_promote, True)
            np.testing.assert_equal(
                core._get_amp_attrs()._amp_level, core.AmpLevel.O2
            )
            np.testing.assert_equal(core._get_amp_attrs()._amp_dtype, 'float16')
            amp_attrs._use_promote = False
            amp_attrs._amp_level = core.AmpLevel.O0
            amp_attrs._amp_dtype = 'float32'


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "only support device's compute capability is at least 7.0",
)
class TestPirAMPProgram(unittest.TestCase):
    def test_linear_amp_o1(self):
        if not core.is_compiled_with_cuda():
            return
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [3, 4], 'float32')
                linear = paddle.nn.Linear(4, 5)
                with paddle.amp.auto_cast(
                    level='O1', dtype='float16', use_promote=True
                ):
                    out1 = linear(x)
                    out2 = paddle.mean(out1)

            cast_op_count = 0
            for op in main.global_block().ops:
                if op.name() == 'pd_op.cast':
                    cast_op_count += 1
            np.testing.assert_equal(out1.dtype, core.DataType.FLOAT32)
            np.testing.assert_equal(out2.dtype, core.DataType.FLOAT32)
            np.testing.assert_equal(cast_op_count, 3)
            _white_list, _black_list = core._get_amp_op_list()
            np.testing.assert_equal(len(_white_list), 0)
            np.testing.assert_equal(len(_black_list), 0)

    def test_linear_amp_bf16_o1(self):
        if not core.is_compiled_with_cuda():
            return
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [3, 4], 'float32')
                linear = paddle.nn.Linear(4, 5)
                with paddle.amp.auto_cast(
                    level='O1', dtype='bfloat16', use_promote=True
                ):
                    out1 = linear(x)
                    out2 = paddle.mean(out1)

            cast_op_count = 0
            for op in main.global_block().ops:
                if op.name() == 'pd_op.cast':
                    cast_op_count += 1
            np.testing.assert_equal(out1.dtype, core.DataType.FLOAT32)
            np.testing.assert_equal(out2.dtype, core.DataType.FLOAT32)
            np.testing.assert_equal(cast_op_count, 3)
            _white_list, _black_list = core._get_amp_op_list()
            np.testing.assert_equal(len(_white_list), 0)
            np.testing.assert_equal(len(_black_list), 0)

    def test_linear_amp_o2_without_scaler(self):
        if not core.is_compiled_with_cuda():
            return
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [3, 4], 'float32')
                linear = paddle.nn.Linear(4, 5)
                optimizer = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=linear.parameters()
                )
                linear, optimizer = paddle.amp.decorate(
                    models=linear,
                    optimizers=optimizer,
                    level='O2',
                    master_weight=True,
                    master_grad=True,
                )

                with paddle.amp.auto_cast(
                    level='O2', dtype='float16', use_promote=True
                ):
                    out = linear(x)
                    loss = paddle.mean(out)
                optimizer.minimize(loss)
            cast_op_count = 0
            for op in main.global_block().ops:
                if op.name() == 'pd_op.cast':
                    cast_op_count += 1
            np.testing.assert_equal(cast_op_count, 3)
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup)
            result = exe.run(
                main,
                feed={'x': np.random.rand(3, 4).astype('float32')},
                fetch_list=[loss],
            )

    def test_linear_amp_o2(self):
        if not core.is_compiled_with_cuda():
            return
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [3, 4], 'float32')
                linear = paddle.nn.Linear(4, 5)
                optimizer = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=linear.parameters()
                )
                linear, optimizer = paddle.amp.decorate(
                    models=linear,
                    optimizers=optimizer,
                    level='O2',
                    master_weight=True,
                    master_grad=True,
                )
                scaler = paddle.amp.GradScaler(
                    init_loss_scaling=2.0**16, use_dynamic_loss_scaling=True
                )

                with paddle.amp.auto_cast(
                    level='O2', dtype='float16', use_promote=True
                ):
                    out = linear(x)
                    loss = paddle.mean(out)
                scaled = scaler.scale(loss)
                opt_ops, _ = scaler.minimize(
                    optimizer, scaled, startup_program=startup
                )
            np.testing.assert_equal(len(opt_ops), 8)
            cast_op_count = 0
            for op in main.global_block().ops:
                if op.name() == 'pd_op.cast':
                    cast_op_count += 1
            np.testing.assert_equal(cast_op_count, 5)
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup)
            result = exe.run(
                main,
                feed={'x': np.random.rand(3, 4).astype('float32')},
                fetch_list=[loss],
            )

    def test_linear_amp_bf16_o2_without_scaler(self):
        if not core.is_compiled_with_cuda():
            return
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [3, 4], 'float32')
                linear = paddle.nn.Linear(4, 5)
                optimizer = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=linear.parameters()
                )
                linear, optimizer = paddle.amp.decorate(
                    models=linear,
                    optimizers=optimizer,
                    level='O2',
                    dtype='bfloat16',
                    master_weight=True,
                    master_grad=True,
                )

                with paddle.amp.auto_cast(
                    level='O2', dtype='bfloat16', use_promote=True
                ):
                    out = linear(x)
                    loss = paddle.mean(out)
                optimizer.minimize(loss)
            cast_op_count = 0
            for op in main.global_block().ops:
                if op.name() == 'pd_op.cast':
                    cast_op_count += 1
            np.testing.assert_equal(cast_op_count, 3)
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup)
            result = exe.run(
                main,
                feed={'x': np.random.rand(3, 4).astype('float32')},
                fetch_list=[loss],
            )


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(2, 2)

    def forward(self, x):
        out1 = self.linear(x)
        out2 = self.linear.weight + 1
        return out1 + out2


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "only support device's compute capability is at least 7.0",
)
class TestPirAMPMasterGrad(unittest.TestCase):
    def test_multi_param_grad(self):
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [2, 2])
                net = Net()
                opt = paddle.optimizer.Adam(
                    learning_rate=0.0001, parameters=net.parameters()
                )
                linear, opt = paddle.amp.decorate(
                    models=net,
                    optimizers=opt,
                    level='O2',
                    dtype='float16',
                    master_weight=False,
                    master_grad=True,
                )
                with paddle.amp.auto_cast(level='O2', dtype='float16'):
                    out = net(x)
                    loss = paddle.mean(out)
                opt.minimize(loss)

                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(startup)
                result = exe.run(
                    main,
                    feed={'x': np.random.rand(2, 2).astype('float32')},
                    fetch_list=[loss],
                )
                for op in main.global_block().ops:
                    if op.name() == 'builtin.combine':
                        for input in [
                            op.operand_source(0),
                            op.operand_source(1),
                        ]:
                            np.testing.assert_equal(input.dtype, paddle.float32)
                            np.testing.assert_equal(
                                input.get_defining_op().name(), 'pd_op.cast'
                            )


if __name__ == '__main__':
    unittest.main()
