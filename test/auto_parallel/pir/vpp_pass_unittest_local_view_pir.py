# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle import nn
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 8
np.random.seed(2024)
paddle.seed(2024)


PP_MESH_0 = dist.ProcessMesh([0], dim_names=['pp'])
PP_MESH_1 = dist.ProcessMesh([1], dim_names=['pp'])


def is_forward_op(op):
    if int(op.op_role) == 0:
        return True
    return False


def is_backward_op(op):
    if int(op.op_role) == 1:
        return True
    return False


def is_optimize_op(op):
    if int(op.op_role) == 2:
        return True
    return False


class CustomLayer(dist.LocalLayer):
    def __init__(self, out_dist_attrs):
        super().__init__(out_dist_attrs)

    def forward(self, input):
        input += 0.1
        input -= 0.3
        input *= 0.5
        return input


class MyLinear(nn.Layer):
    def __init__(
        self,
        hidden_size=784,
        intermediate_size=4 * 784,
        dropout_ratio=0.1,
        weight_attr=None,
        mesh=None,
    ):
        super().__init__()

        self.linear0 = nn.Linear(
            hidden_size, intermediate_size, weight_attr, bias_attr=None
        )
        self.linear1 = nn.Linear(
            intermediate_size, hidden_size, weight_attr, bias_attr=None
        )
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

        self.linear0.weight = dist.shard_tensor(
            self.linear0.weight,
            mesh,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.linear1.weight = dist.shard_tensor(
            self.linear1.weight,
            mesh,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.custom_local_layer = CustomLayer([(mesh, [dist.Replicate()])])

    def forward(self, input):
        out = self.linear0(input)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.custom_local_layer(out)
        out = self.dropout(out)

        return out


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=784,
        intermediate_size=4 * 784,
        dropout_ratio=0.1,
        initializer_range=0.02,
        manual=True,
        hidden_layer=4,
        random_shard=False,
    ):
        super().__init__()

        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )

        if manual:
            self.layer_to_mesh = [PP_MESH_0, PP_MESH_1, PP_MESH_0, PP_MESH_1]
        else:
            self.layer_to_mesh = [PP_MESH_0] * (
                hidden_layer - hidden_layer // 2
            ) + [PP_MESH_1] * (hidden_layer // 2)
        if random_shard:
            self.layer_to_mesh = [PP_MESH_0] * (4) + [PP_MESH_1] * (
                hidden_layer - 4
            )

        self.layers = nn.LayerList(
            [
                MyLinear(
                    hidden_size,
                    intermediate_size,
                    dropout_ratio,
                    weight_attr,
                    mesh=self.layer_to_mesh[i],
                )
                for i in range(hidden_layer)
            ]
        )

        self.linear = nn.Linear(
            hidden_size, CLASS_NUM, weight_attr, bias_attr=None
        )
        self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)

    def forward(self, input):
        out = self.norm(input)

        for i, layer in enumerate(self.layers):
            if i > 0 and self.layer_to_mesh[i] != self.layer_to_mesh[i - 1]:
                out = dist.reshard(
                    out, self.layer_to_mesh[i], [dist.Replicate()]
                )
            out = layer(out)

        out = self.linear(out)
        return out


def loss_fn(pred, label):
    loss = F.l1_loss(pred, label)
    return loss


def apply_pass(schedule_mode, acc_step, enable_send_recv_overlap=False):
    strategy = dist.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = True
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = acc_step
    pipeline.pp_degree = 2
    pipeline.enable_send_recv_overlap = enable_send_recv_overlap

    if schedule_mode == 'VPP':
        pipeline.vpp_degree = 2
        pipeline.vpp_seg_method = "MyLinear"

    return strategy


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


class TestVPPLocalView(unittest.TestCase):
    def create_data_loader(
        self,
        batch_size=BATCH_SIZE,
        batch_num=BATCH_NUM,
        image_size=IMAGE_SIZE,
        class_num=CLASS_NUM,
    ):
        nsamples = batch_size * batch_num
        images = np.random.rand(nsamples, image_size).astype('float32')
        labels = np.random.rand(nsamples, class_num).astype('float32')
        dataset = RandomDataset(images, labels, nsamples)
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader

    def init(self):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)

    def run_pipeline(
        self,
        schedule_mode,
        acc_step,
        manual=True,
        enable_send_recv_overlap=False,
        batch_size=BATCH_SIZE,
        hidden_layer=4,
        random_shard=False,
    ):
        self.init()

        strategy = apply_pass(schedule_mode, acc_step, enable_send_recv_overlap)
        model = MLPLayer(
            manual=manual, hidden_layer=hidden_layer, random_shard=random_shard
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.00001, parameters=model.parameters()
        )

        loss_fn = nn.MSELoss()
        loader = self.create_data_loader(batch_size)
        dist_loader = dist.shard_dataloader(
            loader, meshes=[PP_MESH_0, PP_MESH_1]
        )

        dist_model = dist.to_static(model, dist_loader, loss_fn, opt, strategy)
        dist_model.train()

        loss_list = []
        for _, (image, label) in enumerate(dist_loader):
            loss = dist_model(image, label)
            if loss is not None:
                loss_list.append(np.mean(loss))

        return loss_list

    def test_pp_pass(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        # pp2-vpp-auto
        loss_vpp = self.run_pipeline(
            schedule_mode="VPP", acc_step=4, manual=False
        )
        loss_fthenb = self.run_pipeline(
            schedule_mode="FThenB", acc_step=4, manual=False
        )
        self.check_result(loss_fthenb, loss_vpp)

    def check_result(self, loss1, loss2):
        return np.array_equal(loss1, loss2)


if __name__ == "__main__":
    TestVPPLocalView().test_pp_pass()
