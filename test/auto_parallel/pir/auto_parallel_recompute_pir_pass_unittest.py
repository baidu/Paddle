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

import hashlib
import os
import sys

sys.path.append("../hybrid_strategy/")

import random
from functools import reduce

import numpy as np
from semi_auto_parallel_llama_model import (
    LlamaForCausalLMAuto,
    LlamaPretrainingCriterionAuto,
    get_mesh,
)

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset


class Config:
    vocab_size = 320
    hidden_size = 8
    intermediate_size = 64
    max_position_embeddings = 8
    seq_length = 8

    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 4
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    use_flash_attention = False
    sequence_parallel = False
    rope = True


class RandomDataset(Dataset):
    def __init__(self, seq_len, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.full([self.seq_len], index, dtype="int64")
        label = np.array([index] * 8)

        return input, label

    def __len__(self):
        return self.num_samples


def create_optimizer(model, lr_scheduler):
    decay_parameters = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    def apply_decay_param_fun(x):
        return x in decay_parameters

    optimizer = paddle.optimizer.adamw.AdamW(
        learning_rate=lr_scheduler,
        apply_decay_param_fun=apply_decay_param_fun,
        parameters=model.parameters(),
        weight_decay=0.01,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
    )
    return optimizer


class TestLlamaAuto:
    def __init__(self):
        self.config = Config()
        self.dp = int(os.getenv("dp"))
        self.mp = int(os.getenv("mp"))
        self.pp = int(os.getenv("pp"))

        self.strategy = dist.Strategy()

        self.run_step = 10

    def prepare_llama(self, model, model_config):
        # optimizer
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)
        optimizer = dist.shard_optimizer(optimizer)

        # dataloader
        train_dataset = RandomDataset(model_config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=[get_mesh(0), get_mesh(1)],
            shard_dims="dp",
        )
        return optimizer, dist_loader

    def run_llama(self, model_config):
        self.init_dist_env()
        # model
        model = LlamaForCausalLMAuto(model_config)
        criterion = LlamaPretrainingCriterionAuto(model_config)

        optimizer, dist_loader = self.prepare_llama(model, model_config)

        model = dist.to_static(
            model, dist_loader, criterion, optimizer, strategy=self.strategy
        )
        model.train()

        md5_losses = []
        for step, inputs in enumerate(dist_loader()):
            if step >= self.run_step:
                break
            input_ids, labels = inputs
            loss = model(input_ids, labels)
            array_bytes = np.array(loss).tobytes()
            md5_losses.append(hashlib.md5(array_bytes).hexdigest())
        return md5_losses, model

    def init_dist_env(self):
        order = ["dp", "pp", "mp"]
        dp_degree = self.dp
        mp_degree = self.mp
        pp_degree = self.pp
        degree = [dp_degree, pp_degree, mp_degree]
        mesh_dims = list(filter(lambda x: x[1] > 1, list(zip(order, degree))))
        if not mesh_dims:
            mesh_dims = [("dp", 1)]
        dim_names = [mesh_dim[0] for mesh_dim in mesh_dims]
        mesh_shape = [mesh_dim[1] for mesh_dim in mesh_dims]
        mesh_arr = np.arange(
            0, reduce(lambda x, y: x * y, mesh_shape, 1)
        ).reshape(mesh_shape)
        global_mesh = dist.ProcessMesh(mesh_arr, dim_names)
        dist.auto_parallel.set_mesh(global_mesh)
        paddle.seed(1024)
        np.random.seed(1024)
        random.seed(1024)

    def check_loss(self, losses_1, losses_2):
        np.testing.assert_equal(len(losses_1), len(losses_2))
        for idx in range(len(losses_1)):
            np.testing.assert_equal(losses_1[idx], losses_2[idx])

    def get_recompute_message(self, program):
        segment_num = 0
        rc_op_num = 0
        for block in program.blocks:
            for op in block.ops:
                if op.has_attr("fwd_recompute_id"):
                    idx = op.attrs()["fwd_recompute_id"]
                    segment_num = max(segment_num, idx)
                    rc_op_num += 1
                elif op.has_attr("bwd_recompute_id"):
                    idx = op.attrs()["bwd_recompute_id"]
                    segment_num = max(segment_num, idx)
                    rc_op_num += 1
        return segment_num, rc_op_num

    def run_test_cases(self):
        self.strategy._recompute.enable = False
        self.config.recompute = False
        base_losses, base_model = self.run_llama(self.config)

        self.strategy._recompute.enable = True
        self.config.recompute = True
        self.config.recompute_granularity = "core_attn"
        losses_1, model_1 = self.run_llama(self.config)

        self.config.recompute = True
        self.config.recompute_granularity = "full_attn"
        losses_2, model_2 = self.run_llama(self.config)

        self.config.recompute = True
        self.config.recompute_granularity = "full"
        losses_3, model_3 = self.run_llama(self.config)

        # check loss
        self.check_loss(base_losses, losses_1)
        self.check_loss(base_losses, losses_2)
        self.check_loss(base_losses, losses_3)

        # check program
        base_prog = base_model.dist_main_program()
        prog_1 = model_1.dist_main_program()
        prog_2 = model_2.dist_main_program()
        prog_3 = model_3.dist_main_program()
        base_segment_num, base_rc_op_num = self.get_recompute_message(base_prog)
        segment_num_1, rc_op_num_1 = self.get_recompute_message(prog_1)
        segment_num_2, rc_op_num_2 = self.get_recompute_message(prog_2)
        segment_num_3, rc_op_num_3 = self.get_recompute_message(prog_3)

        assert base_segment_num == 0 and base_rc_op_num == 0
        assert (
            base_segment_num < segment_num_1
            and segment_num_1 < segment_num_2
            and segment_num_2 < segment_num_3
        )
        assert (
            base_rc_op_num < rc_op_num_1
            and rc_op_num_1 < rc_op_num_2
            and rc_op_num_2 < rc_op_num_3
        )


if __name__ == '__main__':
    TestLlamaAuto().run_test_cases()
