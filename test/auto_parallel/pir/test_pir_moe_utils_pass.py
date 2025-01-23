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

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader


class Config:
    def __init__(
        self,
        src_shape=None,
        dst_shape=None,
        src_mesh=None,
        dst_mesh=None,
        src_placements=None,
        dst_placements=None,
    ):
        self.src_shape = src_shape
        self.dst_shape = dst_shape
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh
        self.src_placements = src_placements
        self.dst_placements = dst_placements


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples, return_dict=False):
        self.images = images
        self.labels = labels
        self.num_samples = self.images.shape[0]
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


class Criterion(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.loss_func = paddle.nn.MSELoss()

    def forward(self, logits, labels):
        loss = self.loss_func(logits, labels)
        return loss


class DemoLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mesh_1d = dist.ProcessMesh([0, 1, 2, 3])
        self.mesh_2d = dist.ProcessMesh([[0, 1], [2, 3]])
        b_shape = [config.batch_size, *config.src_shape]  # [B, S, H]
        self.dst_shape = [config.batch_size, config.dst_shape]
        param_initializer = paddle.nn.initializer.Constant(value=0.0)

        self.b = dist.shard_tensor(
            paddle.create_parameter(
                shape=b_shape,
                dtype="float32",
                default_initializer=param_initializer,
            ),
            self.mesh_2d,
            [dist.Shard(0), dist.Shard(1)],
        )

        hidden_size = config.src_shape[-1]
        self.gate = dist.shard_tensor(
            paddle.create_parameter(
                shape=[hidden_size, hidden_size],
                dtype="float32",
            ),
            self.mesh_1d,
            [dist.Replicate()],
        )
        # self.w = dist.shard_tensor(
        #     paddle.create_parameter(
        #         shape=[hidden_size, hidden_size], dtype="float32"
        #     ),
        # )

    def forward(self, x):
        y = x + self.b
        y = dist.auto_parallel.api._dist_reshape(
            y, y.shape, self.mesh_2d, [dist.Shard(1), dist.Shard(1)]
        )
        # print("==== y before reshard ====")
        # print(y)
        # print("==== local value ====")
        # print(y._local_value())
        y = dist.reshard(y, self.mesh_2d, [dist.Shard(1), dist.Shard(0)])
        # y = dist.auto_parallel.api._dist_reshape(
        #     y, y.shape, self.mesh_1d, [dist.Shard(1)]
        # )
        # y = dist.auto_parallel.moe_utils._dist_reshape(
        #     y, self.dst_shape, self.config.dst_mesh, self.config.dst_placements
        # )
        return y


class TestDistReshape(unittest.TestCase):
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self, config):
        nsamples = config.batch_size * config.batch_num
        images_shape = [nsamples, *config.src_shape]
        labels_shape = [nsamples, *config.dst_shape]
        images = np.random.rand(*images_shape).astype('float32')
        # images = (
        #     np.arange(np.prod(images_shape))
        #     .reshape(images_shape)
        #     .astype('float32')
        # )
        labels = np.random.rand(*labels_shape).astype('float32')
        train_dataset = RandomDataset(images, labels, config.batch_size)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        return train_dataloader

    def create_optimizer(self, model, lr_scheduler=None):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=model.parameters(),
        )
        # grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        # )
        return optimizer

    def build(self, config):
        model = DemoLayer(config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        criterion = Criterion()
        return model, dataloader, criterion, optimizer

    def train(self, config, model, train_dataloader, criterion, optimizer):
        tr_loss = float(0)
        global_step = 0
        model.train()

        losses = []
        for step, inputs in enumerate(train_dataloader()):
            inputs, labels = inputs
            logits = model(inputs)
            print("==== logits ====")
            print(logits)
            print("==== local value ====")
            print(logits._local_value())
            # tr_loss = criterion(logits, labels)

            # tr_loss.backward()
            # optimizer.step()
            # optimizer.clear_grad()
            # losses.append(tr_loss.numpy())

        return losses

    def run_dy(self, config, seed):
        self.set_seed(seed)
        model, dataloader, criterion, optimizer = self.build(config)
        losses = self.train(config, model, dataloader, criterion, optimizer)
        return losses

    def run_dy2st(self, config, seed):
        self.set_seed(seed)
        model, dataloader, criterion, optimizer = self.build(config)

        dist_model = dist.to_static(model, dataloader, criterion, optimizer)
        dist_model.train()
        # dist_model.predict()

        losses = []
        for step, inputs in enumerate(dataloader()):
            inputs, labels = inputs
            loss = dist_model(inputs, labels)
            # losses.append(loss)
        return np.array(losses)

    def run_test_case(self):
        seed = 1234
        config = Config()
        config.batch_size = 8
        config.batch_num = 1
        config.src_shape = [4096, 2560]
        config.dst_shape = [4096, 2560]
        config.src_mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        config.dst_mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        config.src_placements = [dist.Shard(1), dist.Replicate()]
        config.dst_placements = [dist.Shard(0), dist.Replicate()]

        # dy_loss = self.run_dy(config, seed)
        dy2st_loss = self.run_dy2st(config, seed)

        # paddle.disable_static()
        # global_mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        # pd_loss_dy2st = paddle.to_tensor(dy2st_loss)
        # pd_loss_dy2st = dist.auto_parallel.api.dtensor_from_local(
        #     pd_loss_dy2st,
        #     global_mesh,
        #     [
        #         dist.Partial(dist.ReduceType.kRedAvg),
        #         dist.Partial(dist.ReduceType.kRedAvg),
        #     ],
        # )
        # pd_loss_dy2st = dist.reshard(
        #     pd_loss_dy2st, global_mesh, [dist.Replicate(), dist.Replicate()]
        # )
        # dy2st_loss = pd_loss_dy2st.numpy()
        # np.testing.assert_equal(dy_loss, dy2st_loss)


if __name__ == "__main__":
    TestDistReshape().run_test_case()
