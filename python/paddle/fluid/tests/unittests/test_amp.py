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

import unittest
import paddle.fluid as fluid
import numpy as np
import time


class SimpleConv(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(SimpleConv, self).__init__()
        self._conv = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=None,
            use_cudnn=True)

    def forward(self, inputs):
        return self._conv(inputs)


class TestAmpApi(unittest.TestCase):
    def reader_decorator(self, reader):
        def _reader_imple():
            for item in reader():
                doc = np.array(item[0]).reshape(3, 224, 224)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield doc, label

        return _reader_imple

    def test_amp(self):
        #resnet = ResNet()
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def run_model(enable_amp=True):
            with fluid.dygraph.guard():
                with fluid.dygraph.amp.autocast(enable_amp):
                    #            with fluid.dygraph.guard():
                    model = SimpleConv(
                        num_channels=3,
                        num_filters=64,
                        filter_size=7,
                        stride=2,
                        act='relu')
                    inp = fluid.dygraph.to_variable(inp_np)
                    out = model(inp)
                    return out.numpy()

        time1 = time.time()
        out1 = run_model(False)
        print(time.time() - time1)

        time2 = time.time()
        out2 = run_model(True)
        print(time.time() - time2)
        # self.assertTrue(np.allclose(out1, out2))


if __name__ == '__main__':
    unittest.main()
