# -*- coding: UTF-8 -*-

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.nn import Linear, ReLU
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import DygraphShardingOptimizerV2

seed = 2022
epoch = 2
linear_size = 1000

np.random.seed(seed)
paddle.seed(seed)


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)
        self._relu = ReLU()

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        y = self._relu(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=200, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype('float32')
        return img

    def __len__(self):
        return self.num_samples


def optimizer_setting(model, use_pure_fp16, use_main_grad):
    if use_main_grad:
        assert use_pure_fp16
        model = mix_precision_utils.MixPrecisionLayer(model, dtype="float16")
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.00001,
        weight_decay=0.00001,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        multi_precision=use_pure_fp16,
    )
    if use_main_grad:
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    return optimizer


def train_mlp(
        model,
        sharding_stage,
        use_pure_fp16=False,
        accumulate_grad=False,
        use_main_grad=False,
        test_scaler=False,
        sharding_use_reduce_avg=False,
        comm_overlap=False,
        tensor_fusion=False,
):
    scaler = None
    scale_loss = 1024
    if test_scaler:
        assert sharding_stage == 1
        assert not accumulate_grad
        scaler = paddle.amp.GradScaler(init_loss_scaling=scale_loss)
        scaler = fleet.distributed_scaler(scaler)
    optimizer = optimizer_setting(
        model=model, use_pure_fp16=use_pure_fp16, use_main_grad=use_main_grad
    )

    strategy = fleet.DistributedStrategy()
    if use_pure_fp16:
        level = 'O2'
        custom_white_list = None

        amp_configs = {"init_loss_scaling": scale_loss, "use_pure_fp16": True}
        strategy.amp_configs = amp_configs
        strategy.amp = True
    else:
        level = 'O1'
        custom_white_list = [
            "matmul_v2",
            "elementwise_add",
            "relu",
            "reduce_mean",
        ]

    if sharding_stage == 1:
        hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 2,
        }
        strategy.hybrid_configs = hybrid_configs
        strategy.hybrid_configs[
            "sharding_configs"
        ].use_reduce_avg = sharding_use_reduce_avg
        strategy.hybrid_configs["sharding_configs"].comm_overlap = comm_overlap
        strategy.hybrid_configs[
            "sharding_configs"
        ].tensor_fusion = tensor_fusion

    fleet.init(is_collective=True, strategy=strategy)
    model = fleet.distributed_model(model)

    if sharding_stage == 1:
        # Initialize the hybrid communication group (hcg)
        hcg = fleet.get_hybrid_communicate_group()
        # Wrap the optimizer with DygraphShardingOptimizerV2
        optimizer = DygraphShardingOptimizerV2(optimizer, hcg)

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=100,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    if sharding_stage == 1:
        model.to(device="gpu")

    if not use_pure_fp16:
        for param in model.parameters():
            t = paddle.cast(
                paddle.cast(param, dtype='float16'), dtype='float32'
            )
            param.set_value(t)

    losses = []
    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            data.stop_gradient = True

            with paddle.amp.auto_cast(
                    True,
                    level=level,
                    dtype="float16",
                    custom_white_list=custom_white_list,
            ):
                out = model(data)
                loss = paddle.mean(out)

            losses.append(loss)

            if test_scaler:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()
            else:
                loss.backward()
                if not accumulate_grad:
                    optimizer.step()
                    optimizer.clear_grad()

        if accumulate_grad:
            optimizer.step()
            optimizer.clear_grad()

    return losses


def test_stage1_fp16():
    if not paddle.amp.is_float16_supported():
        return
    paddle.distributed.init_parallel_env()

    mlp = MLP()
    state_dict = mlp.state_dict()

    # stage1 fp16 O1 vs stage1 fp16 O2 main_grad
    mlp1 = MLP()
    mlp2 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    o1_losses = train_mlp(
        mlp1,
        sharding_stage=1,
        use_pure_fp16=False,
    )
    o2_losses = train_mlp(
        mlp2,
        sharding_stage=1,
        use_pure_fp16=True,
        use_main_grad=True,
    )
    for i in range(len(o1_losses)):
        o1_32_loss = paddle.cast(o1_losses[i], dtype='float32').detach()
        o2_32_loss = paddle.cast(o2_losses[i], dtype='float32').detach()
        np.testing.assert_array_equal(o1_32_loss, o2_32_loss)

    # stage1 scaler test
    mlp3 = MLP()
    mlp3.set_state_dict(state_dict)
    train_mlp(
        mlp3,
        sharding_stage=1,
        use_pure_fp16=True,
        use_main_grad=True,
        test_scaler=True,
    )

    # grad accumulation test
    mlp5 = MLP()
    mlp6 = MLP()
    mlp5.set_state_dict(state_dict)
    mlp6.set_state_dict(state_dict)
    o1_losses_grad_acc = train_mlp(
        mlp5,
        sharding_stage=1,
        use_pure_fp16=False,
        accumulate_grad=True,
    )
    o2_losses_grad_acc = train_mlp(
        mlp6,
        sharding_stage=1,
        use_pure_fp16=True,
        use_main_grad=True,
        accumulate_grad=True,
    )
    for i in range(len(o2_losses_grad_acc)):
        o2_loss_grad_acc = paddle.cast(
            o2_losses_grad_acc[i], dtype='float32'
        ).detach()
        o1_loss_grad_acc = paddle.cast(
            o1_losses_grad_acc[i], dtype='float32'
        ).detach()
        np.testing.assert_array_equal(o2_loss_grad_acc, o1_loss_grad_acc)

    # nccl reduce_avg test
    mlp7 = MLP()
    mlp8 = MLP()
    mlp7.set_state_dict(state_dict)
    mlp8.set_state_dict(state_dict)
    losses_reduce_avg = train_mlp(
        mlp7,
        sharding_stage=1,
        use_pure_fp16=True,
        use_main_grad=True,
        sharding_use_reduce_avg=True,
    )
    losses_reduce_avg_commoverlap = train_mlp(
        mlp8,
        sharding_stage=1,
        use_pure_fp16=True,
        use_main_grad=True,
        sharding_use_reduce_avg=True,
        comm_overlap=False,
        tensor_fusion=False,
    )
    for i in range(len(o2_losses)):
        loss_reduce_avg = paddle.cast(
            losses_reduce_avg[i], dtype='float32'
        ).detach()
        loss_reduce_avg_commoverlap = paddle.cast(
            losses_reduce_avg_commoverlap[i], dtype='float32'
        ).detach()
        loss = paddle.cast(o2_losses[i], dtype='float32').detach()

        np.testing.assert_array_equal(loss_reduce_avg, loss)
        np.testing.assert_array_equal(loss_reduce_avg_commoverlap, loss)

    return


if __name__ == '__main__':
    test_stage1_fp16()
