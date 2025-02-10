#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy
import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import (
    convert_np_dtype_to_dtype_,
)
from paddle.io import Dataset


class TestOptimizerDtype(unittest.TestCase):
    '''
    The dtype of optimizer should be inferred by parameters, and the learning rate
    is created with the same dtype.
    '''

    def check_with_dtype(self, dtype):
        class MyLayer(paddle.nn.Layer):
            def __init__(self, dtype):
                super().__init__()
                self._w = self.create_parameter([2, 3], dtype=dtype)
                self._b = self.create_parameter([2, 3], dtype=dtype)

            def forward(self, x):
                return x * self._w + self._b

        with paddle.base.dygraph.guard():
            model = MyLayer(dtype)
            x = paddle.rand([10, 2, 3], dtype=dtype)
            loss = model(x)
            adam = paddle.optimizer.Adam(parameters=model.parameters())
            loss.backward()
            adam.step()
            self.assertEqual(adam._dtype, convert_np_dtype_to_dtype_(dtype))

    def test_float64(self):
        self.check_with_dtype('float64')

    def test_float32(self):
        self.check_with_dtype('float32')


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestMasterWeightSaveForFP16(unittest.TestCase):
    '''
    For Amp-O2, some optimizer(Momentum, Adam ...) will create master weights for parameters to improve the accuracy.
    Master weights will be saved by optimizer::state_dict.
    '''

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def check_with_opt_state_dict(self, use_save_load=True):
        paddle.seed(100)
        numpy.random.seed(100)

        class SimpleNet(paddle.nn.Layer):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.linears = paddle.nn.LayerList(
                    [
                        paddle.nn.Linear(input_size, output_size)
                        for i in range(1)
                    ]
                )

            def forward(self, x):
                for i, l in enumerate(self.linears):
                    x = self.linears[i](x)
                return x

        input_size = 2  # 设为较大的值
        output_size = 2  # 设为较大的值
        batch_size = 2  # batch_size 为8的倍数
        nums_batch = 10

        class RandomDataset(Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                data = numpy.random.random([input_size]).astype('float16')
                label = numpy.random.random([output_size]).astype('float16')
                return data, label

            def __len__(self):
                return self.num_samples

        dataset = RandomDataset(nums_batch * batch_size)
        loader = paddle.io.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )

        mse = paddle.nn.MSELoss()
        model = SimpleNet(input_size, output_size)  # 定义模型
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.0001,
            parameters=model.parameters(),
            multi_precision=True,
        )  # 定义优化器
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        model = paddle.amp.decorate(models=model, level='O2')

        for i, (data, label) in enumerate(loader):
            with paddle.amp.auto_cast(level='O2'):
                output = model(data)
                loss = mse(output, label)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad(set_to_zero=False)

            if use_save_load and i == 5:
                model_path = os.path.join(self.temp_dir.name, "model.pdparams")
                optimizer_path = os.path.join(self.temp_dir.name, "opt.pdopt")
                paddle.save(model.state_dict(), model_path)
                paddle.save(optimizer.state_dict(), optimizer_path)
                model.set_state_dict(paddle.load(model_path))
                optimizer.set_state_dict(paddle.load(optimizer_path))

        return loss.numpy()

    def test_with_state_dict(self):
        if core.is_compiled_with_cuda():
            with base.dygraph.guard():
                out_use_state_dict = self.check_with_opt_state_dict(
                    use_save_load=True
                )
                out_no_state_dict = self.check_with_opt_state_dict(
                    use_save_load=False
                )
            np.testing.assert_array_equal(out_use_state_dict, out_no_state_dict)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
