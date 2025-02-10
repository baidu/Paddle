# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionLayer,
    MixPrecisionOptimizer,
)

g_shard_split_param = int(os.environ.get("FLAGS_shard_split_param", 0))
g_shard_param_with_color = int(
    os.environ.get("FLAGS_shard_param_with_color", 0)
)

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4
STEPS = 10


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group
        )

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class SimpleMPNet(paddle.nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_size,
        output_size,
        np_fc1,
        np_fc2,
        mp_id,
    ):
        super().__init__()

        if mp_id == 0:
            init_fc1_data = np_fc1[:, : (inner_size // 2)]
            init_fc2_data = np_fc2[: (inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2) :]
            init_fc2_data = np_fc2[(inner_size // 2) :, :]

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc1_data)
            ),
            gather_output=False,
            has_bias=True,
        )

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc2_data)
            ),
            input_is_parallel=True,
            has_bias=True,
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = parallel_matmul(x, self.embedding.weight, False)
        return x


class SimpleDPNet(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

        if g_shard_param_with_color:
            for p in self.linear1.parameters():
                p.color = {'color': "linear1"}

            for p in self.linear2.parameters():
                p.color = {'color': "linear2"}

            for p in self.linear3.parameters():
                p.color = {'color': "linear3"}

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.hybrid_configs["sharding_configs"].split_param = (
            g_shard_split_param
        )

        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            for _ in range(STEPS)
        ]

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model, strategy=None, Optimizer="adam"):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        if Optimizer == "adam":
            optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=0.001,
                weight_decay=0.00001,
                grad_clip=clip,
            )
        else:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                parameters=model.parameters(),
                grad_clip=clip,
            )
        return optimizer

    def build_model_optimizer(self, Optimizer="adam", amp_level=None):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        sharding_id = hcg.get_sharding_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_a = self.build_optimizer(
            model_a,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_b = self.build_optimizer(
            model_b,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        if amp_level is not None and amp_level == "O2":
            model_a = MixPrecisionLayer(model_a)
            optimizer_a = MixPrecisionOptimizer(optimizer_a)
            model_b = MixPrecisionLayer(model_b)
            optimizer_b = MixPrecisionOptimizer(optimizer_b)

        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        return model_a, optimizer_a, model_b, optimizer_b

    def sharding_model(self, Optimizer, sharded_accumulators, amp_level=None):
        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
            Optimizer=Optimizer, amp_level=amp_level
        )
        shard_opt_cls = (
            DygraphShardingOptimizerV2
            if g_shard_split_param
            else DygraphShardingOptimizer
        )
        self.assertTrue(isinstance(optimizer_a._inner_opt, shard_opt_cls))

        for idx in range(STEPS):
            if (
                idx == 2
                and paddle.distributed.get_rank() == 0
                and not g_shard_split_param
            ):
                self.assertTrue(
                    set(optimizer_a._inner_opt._inner_opt.state_dict().keys())
                    == sharded_accumulators
                )

            if paddle.distributed.get_rank() == 0:
                batch_sharding = paddle.to_tensor(self.data[idx][:2])
            else:
                batch_sharding = paddle.to_tensor(self.data[idx][2:])

            batch_single = paddle.to_tensor(self.data[idx])
            loss_a = self.train_batch(batch_sharding, model_a, optimizer_a)
            loss_b = self.train_batch(batch_single, model_b, optimizer_b)

            for j in range(len(model_a.parameters())):
                np.testing.assert_allclose(
                    model_a.parameters()[j].numpy(),
                    model_b.parameters()[j].numpy(),
                    rtol=1e-6,
                )

    def test_sharding_adam(self):
        if not g_shard_split_param:
            sharded_accumulators = {
                'embedding_0.w_0_beta2_pow_acc_0',
                'linear_1.b_0_moment2_0',
                'linear_1.b_0_moment2_max_0',
                'linear_0.b_0_beta1_pow_acc_0',
                'linear_0.b_0_beta2_pow_acc_0',
                'linear_1.b_0_moment1_0',
                'linear_2.b_0_beta2_pow_acc_0',
                'linear_2.b_0_moment2_0',
                'linear_2.b_0_moment2_max_0',
                'embedding_0.w_0_moment1_0',
                'embedding_0.w_0_beta1_pow_acc_0',
                'linear_0.b_0_moment2_0',
                'linear_0.b_0_moment2_max_0',
                'linear_2.b_0_moment1_0',
                'linear_0.b_0_moment1_0',
                'linear_1.b_0_beta2_pow_acc_0',
                'linear_1.b_0_beta1_pow_acc_0',
                'embedding_0.w_0_moment2_0',
                'embedding_0.w_0_moment2_max_0',
                'linear_2.b_0_beta1_pow_acc_0',
            }
            self.sharding_model(
                Optimizer="adam",
                sharded_accumulators=sharded_accumulators,
            )

    def test_sharding_momentum(self):
        if not g_shard_split_param:
            sharded_accumulators = {
                'linear_7.b_0_velocity_0',
                'linear_6.b_0_velocity_0',
                'embedding_2.w_0_velocity_0',
                'linear_8.b_0_velocity_0',
            }
            self.sharding_model(
                Optimizer="Momentum", sharded_accumulators=sharded_accumulators
            )

    def test_sharding_momentum_amp(self):
        sharded_accumulators = {
            'linear_14.b_0_velocity_0',
            'linear_13.b_0_velocity_0',
            'linear_12.b_0_velocity_0',
            'embedding_4.w_0_velocity_0',
        }
        self.sharding_model(
            Optimizer="Momentum",
            sharded_accumulators=sharded_accumulators,
            amp_level="O2",
        )


class TestShardingV2AllGather(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()

        self.strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.hybrid_configs["sharding_configs"].split_param = (
            g_shard_split_param
        )
        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            for _ in range(STEPS)
        ]

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model, strategy=None, Optimizer="adam"):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        if Optimizer == "adam":
            optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=0.001,
                weight_decay=0.00001,
                grad_clip=clip,
            )
        else:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                parameters=model.parameters(),
                grad_clip=clip,
            )
        return optimizer

    def build_model_optimizer(self, Optimizer="adam", amp_level=None):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_a = self.build_optimizer(
            model_a,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_b = self.build_optimizer(
            model_b,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        if amp_level is not None and amp_level == "O2":
            model_a = MixPrecisionLayer(model_a)
            optimizer_a = MixPrecisionOptimizer(optimizer_a)
            model_b = MixPrecisionLayer(model_b)
            optimizer_b = MixPrecisionOptimizer(optimizer_b)

        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        optimizer_a._set_all_gather_overlap_forward(True, model_a)
        optimizer_b._set_all_gather_overlap_forward(False, model_b)
        return model_a, optimizer_a, model_b, optimizer_b

    def sharding_model(self, Optimizer, sharded_accumulators, amp_level=None):
        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
            Optimizer=Optimizer,
            amp_level=amp_level,
        )
        opt_cls = (
            DygraphShardingOptimizerV2
            if g_shard_split_param
            else DygraphShardingOptimizer
        )
        self.assertTrue(isinstance(optimizer_a._inner_opt, opt_cls))

        for idx in range(STEPS):
            if (
                idx == 2
                and paddle.distributed.get_rank() == 0
                and not g_shard_split_param
            ):
                self.assertTrue(
                    set(optimizer_a._inner_opt._inner_opt.state_dict().keys())
                    == sharded_accumulators
                )

            if paddle.distributed.get_rank() == 0:
                batch_sharding = paddle.to_tensor(self.data[idx][:2])
            else:
                batch_sharding = paddle.to_tensor(self.data[idx][2:])

            batch_single = paddle.to_tensor(self.data[idx])
            loss_a = self.train_batch(batch_sharding, model_a, optimizer_a)
            loss_b = self.train_batch(batch_single, model_b, optimizer_b)

            for j in range(len(model_a.parameters())):
                np.testing.assert_allclose(
                    model_a.parameters()[j].numpy(),
                    model_b.parameters()[j].numpy(),
                    rtol=1e-6,
                )

    def test_all_gather_overlap_forward(self):
        if g_shard_split_param:
            sharded_accumulators = {
                'linear_12.b_0_velocity_0',
                'linear_13.b_0_velocity_0',
                'linear_14.b_0_velocity_0',
                'embedding_4.w_0_velocity_0',
            }
            self.sharding_model(
                Optimizer="Momentum",
                sharded_accumulators=sharded_accumulators,
                amp_level="O2",
            )


if __name__ == "__main__":
    unittest.main()
