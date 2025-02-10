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
import unittest

import numpy as np
from test_to_static_pir_program import create_data_loader

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.base.framework import auto_complete_op_role
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)
from paddle.distributed.auto_parallel.static.utils import set_all_ops_op_role
from paddle.distributed.fleet.meta_optimizers.common import OpRole

BATCH_SIZE = 4
BATCH_NUM = 10
IMAGE_SIZE = 16
CLASS_NUM = 8


class HybridParallelDemoNet(nn.Layer):
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
            [dist.Replicate(), dist.Shard(1)],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh2,
            [dist.Replicate(), dist.Shard(0)],
            stop_gradient=False,
        )

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)  # trigger backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Shard(0), dist.Shard(1)])
        out = self.linear_1(out)
        out = self.relu_2(out)  # trigger forward partial allreduce
        return out


class TestML3DParallel(unittest.TestCase):
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self.mesh1 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self.mesh2 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=["x", "y"])

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = self.mesh1
        mesh2 = self.mesh2
        threeD_layer = HybridParallelDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=threeD_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            batch_size=BATCH_SIZE,
            batch_num=BATCH_NUM,
            image_size=IMAGE_SIZE,
            class_num=CLASS_NUM,
        )
        dist_loader = dist.shard_dataloader(
            loader, shard_dims="x", meshes=[mesh1, mesh2]
        )
        dist_model = dist.to_static(threeD_layer, dist_loader, loss_fn, opt)

        engine = dist_model._engine
        engine._build("train")
        dist_program = engine._fwd_main_progs["train"]

        apply_mix2dist_pass(dist_program)
        set_all_ops_op_role(dist_program.global_block(), OpRole.Forward)
        loss = dist_program.get_output_value_by_name(engine._loss_names[0])
        with paddle.static.program_guard(dist_program):
            with auto_complete_op_role(dist_program, OpRole.Backward):
                params_grads = paddle.autograd.ir_backward.append_backward(loss)
            with auto_complete_op_role(dist_program, OpRole.Optimize):
                engine._optimizer._apply_optimize(
                    loss, startup_program=None, params_grads=params_grads
                )
        from paddle.distributed.auto_parallel.static.pir_pass import (
            apply_partition_pass,
        )

        apply_partition_pass(dist_program)
        rank = paddle.distributed.get_rank()
        ops = dist_program.global_block().ops
        op_names = [op.name() for op in ops]
        std_ops = [
            'pd_op.data',
            'pd_op.data',
            'builtin.parameter',
            'builtin.parameter',
            'pd_op.data',
            'pd_op.data',
            'pd_op.relu',
            'pd_op.matmul',
            'pd_op.relu',
            'dist_op.reshard',
            'pd_op.matmul',
            'dist_op.reshard',
            'pd_op.relu',
            'pd_op.subtract',
            'pd_op.square',
            'pd_op.full_int_array',
            'pd_op.mean',
            'builtin.shadow_output',
            'pd_op.full',
            'pd_op.full_like',
            'dist_op.reshard',
            'pd_op.mean_grad',
            'dist_op.reshard',
            'pd_op.square_grad',
            'pd_op.subtract_grad',
            'pd_op.relu_grad',
            'pd_op.matmul_grad',
            'dist_op.reshard',
            'pd_op.relu_grad',
            'pd_op.matmul_grad',
            'dist_op.reshard',
            'pd_op.relu_grad',
            'dist_op.reshard',
            'pd_op.sgd_',
            'dist_op.reshard',
            'pd_op.sgd_',
        ]
        assert op_names == std_ops

    def test_loss_value(self):
        paddle.disable_static()
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        data_loader = create_data_loader(
            batch_size=BATCH_SIZE,
            batch_num=BATCH_NUM,
            image_size=IMAGE_SIZE,
            class_num=CLASS_NUM,
        )

        self.set_random_seed(self._seed)
        dy_layer = HybridParallelDemoNet(self.mesh1, self.mesh2)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        loss_fn = nn.MSELoss()
        dy2st_layer = HybridParallelDemoNet(self.mesh1, self.mesh2)
        dy2st_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2st_layer.parameters()
        )

        dist_loader = dist.shard_dataloader(
            data_loader, shard_dims="x", meshes=[self.mesh1, self.mesh2]
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_loader)
        dy2st_losses = self.run_dy2static(dy2st_layer, dy2st_opt, dist_loader)

        paddle.disable_static()
        rank_id = dist.get_rank()
        if rank_id in self.mesh2.process_ids:
            pd_loss_dy2st = paddle.to_tensor(dy2st_losses)
            pd_loss_dy2st = dist.auto_parallel.api.dtensor_from_local(
                pd_loss_dy2st,
                self.mesh2,
                [dist.Partial(dist.ReduceType.kRedAvg), dist.Replicate()],
            )
            pd_loss_dy2st = dist.reshard(
                pd_loss_dy2st, self.mesh2, [dist.Replicate(), dist.Replicate()]
            )
            dy2st_losses = pd_loss_dy2st.numpy()
            np.testing.assert_equal(dy_losses, dy2st_losses)

    def run_dy2static(self, layer, opt, dist_loader):
        loss_fn = nn.MSELoss()
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        dist_model.train()
        mode = "train"
        dist_program = dist_model._engine.main_program
        dist_model._fetch_value(
            dist_program.global_block().ops[4].result(0), "fetch_value"
        )

        loss_list = []
        for batch_id, data in enumerate(dist_loader()):
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            loss = dist_model(image, label)
            assert "fetch_value" in dist_model.outs
            loss_list.append(loss)

        return np.array(loss_list)

    def run_dynamic(self, layer, opt, dist_loader, is_recompute=False):
        # create loss
        loss_fn = nn.MSELoss()
        loss_list = []
        for batch_id, data in enumerate(dist_loader()):
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            if is_recompute:
                image.stop_gradient = False
            out = layer(image)
            loss = loss_fn(out, label)
            loss_list.append(loss.numpy())
            loss.backward()

            opt.step()
            opt.clear_grad()
        return np.array(loss_list)

    def run_test_cases(self):
        self.test_to_static_program()
        self.test_loss_value()


if __name__ == "__main__":
    TestML3DParallel().run_test_cases()
