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
import sys
import time
import unittest
from os.path import dirname

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
import utils

list1 = [32]  # [32, 128, 512, 1024, 2048, 4096]
list2 = [32]  # [32, 128, 512, 1024, 2048, 4096]


class SoftmaxCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.variance_epsilon = 1e-6

    def forward(self, x):
        output = paddle.nn.functional.softmax(x, axis=-1)
        return output


def create_tensor_inputs(i, j):
    shape = [1, i, j]
    x = paddle.uniform(shape, dtype="float32", min=-0.5, max=0.5)
    x.stop_gradient = False
    inputs = x
    return inputs


class TestRMSNormSubGraphDD(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        # self.prepare_data()

    def prepare_data(self, i, j):
        self.x = create_tensor_inputs(i, j)

    def train_one_window(self, i, j):
        use_cinn = True
        net = SoftmaxCase()
        net.eval()
        input_spec = [
            InputSpec(shape=[1, None, None], dtype='float32'),
        ]
        self.prepare_data(i, j)
        net = utils.apply_to_static(net, use_cinn, input_spec)
        total_time = 0.0
        times = []
        for i in range(5000):
            if i > 100 or i < 9900:
                t0 = time.time()
                t0 = time.time()
                out = net(self.x)
                total_time += time.time() - t0
                times.append(time.time() - t0)
        sorted_times = sorted(times)

        def calculate_average(arr):
            return sum(arr) / len(arr) * 1e6

        with open('./softmax_dd_comp.csv', 'a') as fp:
            fp.write(str(calculate_average(sorted_times[500:-500])))

        return out

    def test_train(self):
        for i in list1:
            for j in list2:
                cinn_out = self.train_one_window(i, j)


class TestRMSNormSubGraphDS(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        # self.prepare_data()

    def prepare_data(self, i, j):
        self.x = create_tensor_inputs(i, j)

    def train_one_window(self, i, j):
        use_cinn = True
        net = SoftmaxCase()
        net.eval()
        input_spec = [
            InputSpec(shape=[1, None, j], dtype='float32'),
        ]
        self.prepare_data(i, j)
        net = utils.apply_to_static(net, use_cinn, input_spec)
        total_time = 0.0
        times = []
        for i in range(5000):
            if i > 100 or i < 9900:
                t0 = time.time()
                t0 = time.time()
                out = net(self.x)
                total_time += time.time() - t0
                times.append(time.time() - t0)
        sorted_times = sorted(times)

        def calculate_average(arr):
            return sum(arr) / len(arr) * 1e6

        with open('./softmax_ds_comp.csv', 'a') as fp:
            fp.write(str(calculate_average(sorted_times[500:-500])))

        return out

    def test_train(self):
        for i in list1:
            for j in list2:
                cinn_out = self.train_one_window(i, j)


class TestRMSNormSubGraphSD(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        # self.prepare_data()

    def prepare_data(self, i, j):
        self.x = create_tensor_inputs(i, j)

    def train_one_window(self, i, j):
        use_cinn = True
        net = SoftmaxCase()
        net.eval()
        input_spec = [
            InputSpec(shape=[1, i, None], dtype='float32'),
        ]
        self.prepare_data(i, j)
        net = utils.apply_to_static(net, use_cinn, input_spec)
        total_time = 0.0
        times = []
        for i in range(5000):
            if i > 100 or i < 9900:
                t0 = time.time()
                t0 = time.time()
                out = net(self.x)
                total_time += time.time() - t0
                times.append(time.time() - t0)
        sorted_times = sorted(times)

        def calculate_average(arr):
            return sum(arr) / len(arr) * 1e6

        with open('./softmax_sd_comp.csv', 'a') as fp:
            fp.write(str(calculate_average(sorted_times[500:-500])))

        return out

    def test_train(self):
        for i in list1:
            for j in list2:
                cinn_out = self.train_one_window(i, j)


if __name__ == '__main__':
    unittest.main()
