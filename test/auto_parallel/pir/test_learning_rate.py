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

import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 40
IMAGE_SIZE = 16
CLASS_NUM = 8
np.random.seed(2024)
paddle.seed(2024)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class SimpleDemoNet(nn.Layer):
    def __init__(self, mesh1, mesh2):
        super().__init__()
        self._mesh1 = mesh1
        self._mesh2 = mesh2
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # shard the weights of this layer
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh1,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh2,
            [dist.Replicate()],
            stop_gradient=False,
        )

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)  # trigger backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Replicate()])
        out = self.linear_1(out)
        out = self.relu_2(out)  # trigger forward partial allreduce
        return out


def create_data_loader():
    images = np.random.rand(BATCH_NUM, IMAGE_SIZE).astype('float32')
    labels = np.random.rand(BATCH_NUM, CLASS_NUM).astype('float32')
    dataset = RandomDataset(images, labels, BATCH_NUM)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return loader


class TestLearningRate(unittest.TestCase):
    def test_copy_between_mesh(self):
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        layer = SimpleDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)

        engine = dist_model._engine
        engine._build("train")
        dist_program = engine._fwd_main_progs["train"]
        apply_mix2dist_pass(dist_program)
        loss = dist_program.get_output_value_by_name(engine._loss_names[0])
        with paddle.static.program_guard(dist_program):
            params_grads = paddle.autograd.ir_backward.append_backward(loss)
            engine._optimizer._apply_optimize(
                loss, startup_program=None, params_grads=params_grads
            )
        apply_mix2dist_pass(dist_program)
        sgd_idx = 0
        ops = dist_program.global_block().ops
        for op in ops:
            if op.name() != 'pd_op.sgd_':
                continue
            param = op.operand_source(0)
            learning_rate = op.operand_source(1)
            op_dist_attr = learning_rate.get_defining_op().dist_attr
            self.assertEqual(
                learning_rate.dist_attr().process_mesh,
                param.dist_attr().process_mesh,
            )
            self.assertEqual(
                learning_rate.dist_attr().process_mesh,
                op_dist_attr.process_mesh,
            )
            if sgd_idx == 0:
                self.assertEqual(param.dist_attr().process_mesh, mesh2)
            elif sgd_idx == 1:
                self.assertEqual(param.dist_attr().process_mesh, mesh1)
            sgd_idx += 1
        self.assertEqual(sgd_idx, 2)


if __name__ == "__main__":
    unittest.main()
