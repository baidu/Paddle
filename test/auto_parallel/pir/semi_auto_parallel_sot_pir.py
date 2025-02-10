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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 16
CLASS_NUM = 8


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples, return_dict=False):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples
        self.return_dict = return_dict

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "image": self.images[idx],
                "label": self.labels[idx],
            }
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class DPDemoNet(nn.Layer):
    def __init__(
        self,
        mesh,
    ):
        super().__init__()
        self._mesh = mesh
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.linear_0.weight = paddle.distributed.shard_tensor(
            self.linear_0.weight,
            self._mesh,
            [paddle.distributed.Replicate()],
            stop_gradient=False,
        )
        self.linear_1.weight = paddle.distributed.shard_tensor(
            self.linear_1.weight,
            self._mesh,
            [paddle.distributed.Replicate()],
            stop_gradient=False,
        )
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        out = self.relu_0(x)
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = self.linear_1(out)
        out = self.relu_2(out)
        out = paddle.cast(out, 'float32')
        return out


class TestSimpleNetForSemiAutoParallelSOT:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self, return_dict=False):
        images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, BATCH_SIZE, return_dict)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dynamic(self, layer, opt, dist_loader):
        # create loss
        loss_fn = nn.MSELoss()
        loss_list = []
        for epoch in range(BATCH_NUM):
            for batch_id, data in enumerate(dist_loader()):
                if isinstance(data, dict):
                    image = data['image']
                    label = data['label']
                else:
                    image, label = data
                out = layer(image)
                loss = loss_fn(out, label)
                loss_list.append(loss.numpy())
                loss.backward()
                opt.step()
                opt.clear_grad()
        return np.array(loss_list)

    def test_dp_demo_net(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader, meshes=[self.mesh]
        )
        dy_layer = DPDemoNet(self.mesh)
        unified_strategy = dist.Strategy({"full_graph": False})
        dy_layer = dist.to_static(dy_layer, strategy=unified_strategy)
        dy_opt = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=dy_layer.parameters()
        )
        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_dataloader)

    def run_test_case(self):
        self.test_dp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallelSOT().run_test_case()
