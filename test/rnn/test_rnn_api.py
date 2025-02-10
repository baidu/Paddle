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


import paddle

paddle.set_default_dtype("float64")

import unittest

import numpy as np

from paddle import base

paddle.enable_static()

bidirectional_list = ["bidirectional", "bidirect"]


class TestSimpleRNN(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16
        self.seq_len = 12
        self.seed = 1234

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.

        place = paddle.set_device(self.place)
        paddle.disable_static(self.place)
        paddle.seed(self.seed)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
        else:
            paddle.framework.random._manual_program_seed(self.seed)
        cell_dy = paddle.nn.SimpleRNNCell(self.input_size, self.hidden_size)
        self.rnn_net = paddle.nn.RNN(cell_dy, time_major=self.time_major)

        paddle.enable_static()

        with paddle.base.unique_name.guard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(
                main_program=main_program, startup_program=startup_program
            ):
                paddle.seed(self.seed)
                paddle.framework.random._manual_program_seed(self.seed)

                self.exe = base.Executor(
                    base.CPUPlace()
                    if self.place == "cpu"
                    else base.CUDAPlace(0)
                )

                rnn_in_data = paddle.static.data(
                    "x",
                    [None, self.batch_size, self.hidden_size],
                    dtype="float64",
                )
                pre_h_data = paddle.static.data(
                    "pre_h",
                    [self.batch_size, self.hidden_size],
                    dtype="float64",
                )
                seq_len_data = paddle.static.data(
                    "seq_len", [self.batch_size], dtype="int64"
                )
                cell_st = paddle.nn.SimpleRNNCell(
                    self.input_size, self.hidden_size
                )
                self.rnn_st = paddle.nn.RNN(cell_st, time_major=self.time_major)
                st_out, st_last_h = self.rnn_st(
                    rnn_in_data, pre_h_data, sequence_length=seq_len_data
                )

                self.fetch_list = [st_out, st_last_h]

                self.exe.run(paddle.static.default_startup_program())

                self.main_program = paddle.static.default_main_program()

            paddle.disable_static(self.place)

    def test_base(self, test_seq_len=False):
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(4, 16)

        paddle.disable_static(self.place)
        if test_seq_len:
            seq_len = np.array([9, 10, 8, 12], "int64")
        else:
            seq_len = np.array([12, 12, 12, 12], "int64")

        y1, h1 = self.rnn_net(
            paddle.to_tensor(x),
            paddle.to_tensor(prev_h),
            sequence_length=paddle.to_tensor(seq_len),
        )

        paddle.enable_static()
        out = self.exe.run(
            self.main_program,
            feed={"x": x, "pre_h": prev_h, "seq_len": seq_len},
            fetch_list=[self.fetch_list],
        )

        y2, h2 = out

        np.testing.assert_allclose(y1.numpy(), y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1.numpy(), h2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_base()
        self.test_base(True)


class TestGRU(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16
        self.seq_len = 12
        self.seed = 1234

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.

        place = paddle.set_device(self.place)
        paddle.disable_static(self.place)
        paddle.seed(self.seed)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
        else:
            paddle.framework.random._manual_program_seed(self.seed)
        cell_dy = paddle.nn.GRUCell(self.input_size, self.hidden_size)
        self.rnn_net = paddle.nn.RNN(cell_dy, time_major=self.time_major)

        paddle.enable_static()

        with paddle.base.unique_name.guard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(
                main_program=main_program, startup_program=startup_program
            ):
                paddle.seed(self.seed)
                paddle.framework.random._manual_program_seed(self.seed)

                self.exe = base.Executor(
                    base.CPUPlace()
                    if self.place == "cpu"
                    else base.CUDAPlace(0)
                )

                rnn_in_data = paddle.static.data(
                    "x",
                    [None, self.batch_size, self.hidden_size],
                    dtype="float64",
                )
                pre_h_data = paddle.static.data(
                    "pre_h",
                    [self.batch_size, self.hidden_size],
                    dtype="float64",
                )
                seq_len_data = paddle.static.data(
                    "seq_len", [self.batch_size], dtype="int64"
                )
                cell_st = paddle.nn.GRUCell(self.input_size, self.hidden_size)
                self.rnn_st = paddle.nn.RNN(cell_st, time_major=self.time_major)
                st_out, st_last_h = self.rnn_st(
                    rnn_in_data, pre_h_data, sequence_length=seq_len_data
                )

                self.fetch_list = [st_out, st_last_h]

                self.exe.run(paddle.static.default_startup_program())

                self.main_program = paddle.static.default_main_program()

            paddle.disable_static(self.place)

    def test_base(self, test_seq_len=False):
        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(4, 16)

        paddle.disable_static(self.place)
        if test_seq_len:
            seq_len = np.array([9, 10, 8, 12], "int64")
        else:
            seq_len = np.array([12, 12, 12, 12], "int64")

        y1, h1 = self.rnn_net(
            paddle.to_tensor(x),
            paddle.to_tensor(prev_h),
            sequence_length=paddle.to_tensor(seq_len),
        )

        paddle.enable_static()
        out = self.exe.run(
            self.main_program,
            feed={"x": x, "pre_h": prev_h, "seq_len": seq_len},
            fetch_list=[self.fetch_list],
        )

        y2, h2 = out

        np.testing.assert_allclose(y1.numpy(), y2, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1.numpy(), h2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_base()
        self.test_base(True)


class TestGRUBackward(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super().__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.place = place
        self.batch_size = 4
        self.input_size = 4
        self.hidden_size = 4
        self.seq_len = 12
        self.seed = 1234

    def setUp(self):
        # Since `set_device` is global, set `set_device` in `setUp` rather than
        # `__init__` to avoid using an error device set by another test case.

        place = paddle.set_device(self.place)
        paddle.disable_static(self.place)
        paddle.seed(self.seed)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
        else:
            paddle.framework.random._manual_program_seed(self.seed)
        cell_dy = paddle.nn.SimpleRNNCell(self.input_size, self.hidden_size)
        self.rnn_net = paddle.nn.RNN(cell_dy, time_major=self.time_major)

        paddle.enable_static()

        with paddle.base.unique_name.guard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(
                main_program=main_program, startup_program=startup_program
            ):
                paddle.seed(self.seed)
                paddle.framework.random._manual_program_seed(self.seed)

                self.exe = paddle.base.Executor(
                    base.CPUPlace()
                    if self.place == "cpu"
                    else base.CUDAPlace(0)
                )

                rnn_in_data = paddle.static.data(
                    "x",
                    [None, self.batch_size, self.hidden_size],
                    dtype="float64",
                )
                pre_h_data = paddle.static.data(
                    "pre_h",
                    [self.batch_size, self.hidden_size],
                    dtype="float64",
                )
                seq_len_data = paddle.static.data(
                    "seq_len", [self.batch_size], dtype="int64"
                )
                pre_h_data.stop_gradient = False
                rnn_in_data.stop_gradient = False

                cell_st = paddle.nn.SimpleRNNCell(
                    self.input_size, self.hidden_size
                )
                self.rnn_st = paddle.nn.RNN(cell_st, time_major=self.time_major)

                st_out, st_last_h = self.rnn_st(
                    rnn_in_data, pre_h_data, sequence_length=seq_len_data
                )
                loss = paddle.sum(st_out)
                sgd = paddle.optimizer.SGD(0.0)

                if paddle.framework.in_pir_mode():
                    rnn_in_data.persistable = True
                    pre_h_data.persistable = True
                    params_grads = paddle.base.backward.append_backward(loss)
                    pre_h_data_grad = None
                    rnn_in_data_grad = None
                    for p, g in params_grads:
                        if p.is_same(rnn_in_data):
                            rnn_in_data_grad = g
                        elif p.is_same(pre_h_data):
                            pre_h_data_grad = g

                    self.fetch_list = [
                        st_out,
                        st_last_h,
                        pre_h_data_grad,
                        rnn_in_data_grad,
                    ]
                else:
                    sgd.minimize(loss)
                    self.fetch_list = [
                        st_out,
                        st_last_h,
                        "pre_h@GRAD",
                        "x@GRAD",
                    ]

                self.exe.run(paddle.static.default_startup_program())

                self.main_program = paddle.static.default_main_program()

            paddle.disable_static(self.place)

    def test_base(self, test_seq_len=False):
        x = np.random.randn(12, 4, self.hidden_size)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(4, self.hidden_size)

        paddle.disable_static(self.place)
        if test_seq_len:
            seq_len = np.array([9, 10, 8, 12], "int64")
        else:
            seq_len = np.array([12, 12, 12, 12], "int64")

        x_in = paddle.to_tensor(x)
        h_in = paddle.to_tensor(prev_h)
        x_in.stop_gradient = False
        h_in.stop_gradient = False
        y1, h1 = self.rnn_net(
            x_in,
            h_in,
            sequence_length=paddle.to_tensor(seq_len),
        )
        loss = y1.sum()
        loss.backward()

        h1_grad = h_in.gradient()

        paddle.enable_static()
        out = self.exe.run(
            self.main_program,
            feed={"x": x, "pre_h": prev_h, "seq_len": seq_len},
            fetch_list=[self.fetch_list],
        )

        y2, h2, g1, g2 = out

        np.testing.assert_allclose(h1_grad, g1, atol=1e-8, rtol=1e-5)
        self.exe._executor_cache._get_cached_program_and_executor_pir_mode.cache_clear()

    def runTest(self):
        self.test_base(True)
        self.test_base()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
