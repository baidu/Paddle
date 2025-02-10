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

import os
from functools import reduce

import numpy as np
from semi_auto_parallel_llama_model import (
    LlamaForCausalLMAuto,
    LlamaPretrainingCriterionAuto,
    get_mesh,
)

import paddle
import paddle.distributed as dist
from paddle import LazyGuard
from paddle.io import BatchSampler, DataLoader, Dataset

np.random.seed(2024)


class Config:
    vocab_size = 32000
    hidden_size = 4096
    intermediate_size = 11008
    max_position_embeddings = 2048
    seq_length = 2048
    num_hidden_layers = 2
    num_attention_heads = 32
    num_key_value_heads = 32
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    use_flash_attention = False
    sequence_parallel = False
    rope = True
    recompute = False
    recompute_granularity = None
    use_lazy_init = False


inputs = []
labels = []

for i in range(100):
    inputs.append(
        np.random.uniform(low=0, high=32000, size=[Config().seq_length]).astype(
            "int64"
        )
    )
    labels.append(
        (np.random.uniform(size=[Config().seq_length]) * 10).astype("int64")
    )


class RandomDataset(Dataset):
    def __init__(self, seq_len, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __getitem__(self, index):
        global inputs, labels
        return inputs[index], labels[index]

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

    # test global_clip in auto_parallel
    if os.getenv("use_param_group") == "true":
        param_group = {}
        param_group["params"] = list(model.parameters())
        param_group["weight_decay"] = 0.01
        param_group["grad_clip"] = paddle.nn.ClipGradByGlobalNorm(1.0)
        optimizer = paddle.optimizer.adamw.AdamW(
            learning_rate=lr_scheduler,
            apply_decay_param_fun=apply_decay_param_fun,
            parameters=[param_group],
        )
    else:
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
        if os.getenv("use_sp") == "true":
            self.config.sequence_parallel = True
        if os.getenv("recompute") == "true":
            self.config.recompute = True
        self.config.recompute_granularity = os.getenv("recompute_granularity")
        if os.getenv("use_lazy_init") == "true":
            self.config.use_lazy_init = True
        self.gradient_accumulation_steps = int(os.getenv("acc_step"))
        self.amp = False
        self.amp_dtype = "float16"
        self.amp_level = "O1"
        self.amp_master_grad = False
        if os.getenv("amp") == "true":
            self.amp = True
        if os.getenv("amp_dtype") in ["float16", "bfloat16"]:
            self.amp_dtype = os.getenv("amp_dtype")
        if os.getenv("amp_level") in ["O0", "O1", "O2"]:
            self.amp_level = os.getenv("amp_level")
        if os.getenv("amp_master_grad") == "true":
            self.amp_master_grad = True

        self.init_dist_env()

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

    def run_llama(self, to_static=0):
        if self.config.use_lazy_init:
            with LazyGuard():
                model = LlamaForCausalLMAuto(self.config)
            for param in model.parameters():
                assert not param._is_initialized()
                param.initialize()
        else:
            model = LlamaForCausalLMAuto(self.config)
        criterion = LlamaPretrainingCriterionAuto(self.config)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)
        if self.amp and not to_static:
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=self.amp_level,
                dtype=self.amp_dtype,
                master_grad=self.amp_master_grad,
            )
        optimizer = dist.shard_optimizer(optimizer)

        train_dataset = RandomDataset(self.config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=True,
            drop_last=False,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )

        if self.pp == 1:
            meshes = [get_mesh(0)]
        elif self.pp > 1:
            meshes = [get_mesh(0), get_mesh(-1)]
        else:
            raise ValueError("pp should be greater or equal to 1")

        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=meshes,
            shard_dims="dp",
        )

        global_step = 1
        tr_loss = float(0)

        if not to_static:
            model.train()
            scaler = None
            if self.amp and self.amp_dtype == "float16":
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                scaler = dist.shard_scaler(scaler)

            for epoch_idx in range(1):
                for step, inputs in enumerate(dist_loader()):
                    input_ids, labels = inputs
                    return input_ids._local_value()._md5sum()
                    break
        else:
            strategy = dist.Strategy()
            if self.gradient_accumulation_steps > 1:
                strategy.pipeline.accumulate_steps = (
                    self.gradient_accumulation_steps
                )

            if self.amp:
                amp = strategy.amp
                amp.enable = self.amp
                amp.dtype = self.amp_dtype
                amp.level = self.amp_level.lower()
                if self.amp_master_grad:
                    amp.use_master_grad = True

            dist_model = dist.to_static(
                model,
                dist_loader,
                criterion,
                optimizer,
                strategy=strategy,
            )

            dist_model.train()
            for step, inputs in enumerate(dist_loader()):
                input_ids, labels = inputs
                return input_ids._local_value()._md5sum()
                break

    def run_test_cases(self):
        dynamic_input_md5sum = self.run_llama(to_static=0)
        static_input_md5sum = self.run_llama(to_static=1)
        if dist.get_rank() == 0:
            assert dynamic_input_md5sum == static_input_md5sum


if __name__ == '__main__':
    TestLlamaAuto().run_test_cases()
