# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import sys
import time
import unittest

import numpy as np
from test_multiprocess_dataloader_static import (
    BATCH_SIZE,
    CLASS_NUM,
    EPOCH_NUM,
    IMAGE_SIZE,
    SAMPLE_NUM,
    RandomBatchedDataset,
    RandomDataset,
    prepare_places,
)

import paddle
from paddle import base
from paddle.io import DataLoader
from paddle.nn import Linear

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


class SimpleFCNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        param_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.8)
        )
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.5)
        )
        self._fcs = []
        in_channel = IMAGE_SIZE
        for hidden_size in [10, 20, 30]:
            self._fcs.append(
                Linear(
                    in_channel,
                    hidden_size,
                    weight_attr=param_attr,
                    bias_attr=bias_attr,
                )
            )
            self._fcs.append(paddle.nn.Tanh())
            in_channel = hidden_size
        self._fcs.append(
            Linear(
                in_channel,
                CLASS_NUM,
                weight_attr=param_attr,
                bias_attr=bias_attr,
            )
        )
        self._fcs.append(paddle.nn.Softmax())

    def forward(self, image):
        out = image
        for fc in self._fcs:
            out = fc(out)
        return out


def collate_batch(batch_list):
    batch_size = len(batch_list)
    image = np.stack([item[0] for item in batch_list], axis=0).astype('float32')
    image = paddle.to_tensor(image).reshape([batch_size, -1])
    label = np.stack([item[1] for item in batch_list], axis=0).astype('int64')
    label = paddle.to_tensor(label).reshape([batch_size, -1])
    return image, label


class TestDygraphDataLoader(unittest.TestCase):
    def run_main(
        self,
        num_workers,
        places,
        persistent_workers,
        collate_fn,
        use_shared_memory,
    ):
        paddle.seed(1)
        with base.dygraph.guard(places[0]):
            fc_net = SimpleFCNet()
            optimizer = paddle.optimizer.Adam(parameters=fc_net.parameters())

            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                drop_last=True,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
                use_shared_memory=use_shared_memory,
            )
            assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in range(EPOCH_NUM):
                step = 0
                for image, label in dataloader():
                    out = fc_net(image)
                    loss = paddle.nn.functional.cross_entropy(
                        out, label, reduction='none', use_softmax=False
                    )
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    optimizer.minimize(avg_loss)
                    fc_net.clear_gradients()

                    loss_list.append(np.mean(avg_loss.numpy()))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list),
        }
        logging.info(f"time cost {ret['time']} step_list {ret['step']}")
        return ret

    def test_main(self):
        for p in prepare_places():
            for persistent_workers in [False, True]:
                for collate_fn in [None, collate_batch]:
                    for use_shared_memory in [False, True]:
                        results = []
                        for num_workers in [0, 2]:
                            logging.info(
                                f"{self.__class__.__name__} {p} {num_workers} {persistent_workers} {collate_fn} {use_shared_memory}"
                            )
                            sys.stdout.flush()
                            ret = self.run_main(
                                num_workers=num_workers,
                                places=p,
                                persistent_workers=persistent_workers,
                                collate_fn=collate_fn,
                                use_shared_memory=use_shared_memory,
                            )
                            results.append(ret)
                        diff = np.max(
                            np.abs(results[0]['loss'] - results[1]['loss'])
                            / np.abs(results[0]['loss'])
                        )
                        self.assertLess(diff, 1e-2)


class TestDygraphDataLoaderWithBatchedDataset(TestDygraphDataLoader):
    def run_main(
        self,
        num_workers,
        places,
        persistent_workers,
        collate_fn,
        use_shared_memory,
    ):
        paddle.seed(1)
        with base.dygraph.guard(places[0]):
            fc_net = SimpleFCNet()
            optimizer = paddle.optimizer.Adam(parameters=fc_net.parameters())

            dataset = RandomBatchedDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=None,
                drop_last=True,
                persistent_workers=persistent_workers,
                collate_fn=None,
                use_shared_memory=use_shared_memory,
            )
            assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in range(EPOCH_NUM):
                step = 0
                for image, label in dataloader():
                    out = fc_net(image)
                    loss = paddle.nn.functional.cross_entropy(
                        out, label, reduction='none', use_softmax=False
                    )
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    optimizer.minimize(avg_loss)
                    fc_net.clear_gradients()

                    loss_list.append(np.mean(avg_loss.numpy()))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list),
        }
        logging.info(f"time cost {ret['time']} step_list {ret['step']}")
        return ret


if __name__ == '__main__':
    unittest.main()
