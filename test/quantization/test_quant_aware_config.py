# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import logging
import unittest

from quant_aware_config_utils import TestQuantAwareBase

logging.basicConfig(level="INFO", format="%(message)s")


class TestQuantAwareNone(TestQuantAwareBase):
    def generate_config(self):
        config = None
        return config


class TestQuantAwareTRT(TestQuantAwareBase):
    def generate_config(self):
        config = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
            'onnx_format': False,
            'for_tensorrt': True,
        }
        return config


if __name__ == '__main__':
    unittest.main()
