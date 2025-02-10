# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import time
import unittest

import numpy as np

import paddle.nn.functional as F
from paddle import base
from paddle.base import core


def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


class TestDygraphDataLoaderWithException(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.batch_num = 4
        self.epoch_num = 1
        self.capacity = 5

    def test_not_capacity(self):
        with base.dygraph.guard():
            with self.assertRaisesRegex(
                ValueError, "Please give value to capacity."
            ):
                base.io.DataLoader.from_generator()

    def test_single_process_with_thread_exception(self):
        def error_sample_generator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield [[[1, 2], [1]]]

            return __reader__

        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(
                capacity=self.capacity, iterable=False, use_multiprocess=False
            )
            loader.set_batch_generator(
                error_sample_generator(self.batch_num), places=base.CPUPlace()
            )
            exception = None
            try:
                for _ in loader():
                    print("test_single_process_with_thread_exception")
            except core.EnforceNotMet as ex:
                self.assertIn("Blocking queue is killed", str(ex))
                exception = ex
            self.assertIsNotNone(exception)

    def test_multi_process_with_process_exception(self):
        def error_sample_generator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield [[[1, 2], [1]]]

            return __reader__

        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(
                capacity=self.capacity, use_multiprocess=True
            )
            loader.set_batch_generator(
                error_sample_generator(self.batch_num), places=base.CPUPlace()
            )
            exception = None
            try:
                for _ in loader():
                    print("test_multi_process_with_thread_exception")
            except core.EnforceNotMet as ex:
                exception = ex
            self.assertIsNotNone(exception)

    def test_multi_process_with_get_timeout(self):
        def slow_batch_generator_creator(batch_size, batch_num):
            def __reader__():
                for _ in range(batch_num):
                    time.sleep(80)
                    batch_image, batch_label = get_random_images_and_labels(
                        [batch_size, 784], [batch_size, 1]
                    )
                    yield batch_image, batch_label

            return __reader__

        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(
                capacity=self.capacity, use_multiprocess=True
            )
            loader.set_batch_generator(
                slow_batch_generator_creator(self.batch_size, self.batch_num),
                places=base.CPUPlace(),
            )
            exception = None
            try:
                for _ in range(self.epoch_num):
                    for image, _ in loader():
                        F.relu(image)
            except core.EnforceNotMet as ex:
                self.assertIn("Blocking queue is killed", str(ex))
                exception = ex
            self.assertIsNotNone(exception)


if __name__ == '__main__':
    unittest.main()
