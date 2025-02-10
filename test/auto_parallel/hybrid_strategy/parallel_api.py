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
import logging
import os
import random
from dataclasses import dataclass
from functools import reduce

import numpy as np
from single_llama_model import LlamaForCausalLM, LlamaPretrainingCriterion
from single_lora_model import LoRAModel

import paddle
import paddle.distributed as dist
from paddle import LazyGuard
from paddle.distributed.auto_parallel.intermediate.parallelize import (
    parallelize_model,
    parallelize_optimizer,
)
from paddle.io import BatchSampler, DataLoader, Dataset


def is_pp_enable():
    global_mesh = dist.auto_parallel.get_mesh()
    return "pp" in global_mesh.dim_names


def get_mesh(pp_idx=None):
    global_mesh = dist.auto_parallel.get_mesh()
    assert global_mesh is not None, "global_mesh is not initialized!"
    if pp_idx is None:
        return global_mesh
    if is_pp_enable():
        mesh = global_mesh.get_mesh_with_dim("pp")[pp_idx]
        return mesh
    else:
        return global_mesh


class Config:
    vocab_size = 8192
    hidden_size = 512
    intermediate_size = 2048
    seq_length = 512
    num_hidden_layers = 2
    num_attention_heads = 8
    rms_norm_eps = 1e-6
    use_lazy_init = False


@dataclass
class LoRaConfig:
    r = 8
    lora_alpha = 8
    lora_dropout = 0.0
    rslora = False
    lora_plus_scale = 1.0
    pissa = False
    use_quick_lora = False
    lora_use_mixer = False
    use_mora = False
    trainable_bias = False
    trainable_modules = None
    target_modules = [
        ".*q_proj.*",
        ".*v_proj.*",
        ".*k_proj.*",
        ".*o_proj.*",
        ".*qkv_proj.*",
        ".*gate_proj.*",
        ".*down_proj.*",
        ".*up_proj.*",
        ".*gate_up_fused_proj.*",
    ]


class RandomDataset(Dataset):
    def __init__(self, seq_len, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len]).astype("int64")
        label = (np.random.uniform(size=[self.seq_len]) * 10).astype("int64")
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


class TestParallelAPI:
    def __init__(self):
        self.config = Config()
        self.lora_config = LoRaConfig()
        self.dp = int(os.getenv("dp"))
        self.mp = int(os.getenv("mp"))
        self.pp = int(os.getenv("pp"))
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
        self.level = os.getenv("sharding_stage", "0")
        self.sequence_parallel = False
        if os.getenv("sequence_parallel") == "true":
            self.sequence_parallel = True
        self.prepare_input_output = False
        if os.getenv("prepare_input_output") == "true":
            self.sequence_parallel = True

        num_hidden_layers = os.getenv("num_hidden_layers")
        if num_hidden_layers:
            self.config.num_hidden_layers = int(num_hidden_layers)

        self.one_api = False
        if os.getenv("one_api") == "true":
            self.one_api = True

        seed = int(os.getenv("seed", 2024))
        self.share_embedding = int(os.getenv("test_share_embedding", "0"))
        self.position_embedding = int(os.getenv("test_position_embedding", "0"))
        self.test_lora = int(os.getenv("test_lora", "0"))
        np.random.seed(seed)
        random.seed(seed)
        paddle.seed(seed)
        self.init_dist_env()

    def init_dist_env(self):
        mesh_dims = [("dp", self.dp), ("pp", self.pp), ("mp", self.mp)]
        if self.pp * self.mp == 1:
            mesh_dims = [("dp", self.dp)]
        dim_names = [mesh_dim[0] for mesh_dim in mesh_dims]
        mesh_shape = [mesh_dim[1] for mesh_dim in mesh_dims]
        mesh_arr = np.arange(
            0, reduce(lambda x, y: x * y, mesh_shape, 1)
        ).reshape(mesh_shape)
        global_mesh = dist.ProcessMesh(mesh_arr, dim_names)
        dist.auto_parallel.set_mesh(global_mesh)

    def check_mp(self, layer):
        if self.mp == 1:
            return
        for name, sub_layer in layer.named_sublayers():
            if len(sub_layer.sublayers()) == 0:
                if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(1),
                    ]
                    assert sub_layer.bias.placements == [
                        dist.Replicate(),
                        dist.Shard(0),
                    ]
                    if self.test_lora:
                        assert sub_layer.lora_B.placements == [
                            dist.Replicate(),
                            dist.Shard(1),
                        ]
                if 'gate_proj' in name or 'up_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(1),
                    ]
                    if self.test_lora:
                        assert sub_layer.lora_B.placements == [
                            dist.Replicate(),
                            dist.Shard(1),
                        ]
                if (
                    'embed_tokens' in name or 'lm_head' in name
                ) and not self.share_embedding:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(1),
                    ]
                if 'o_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(0),
                    ], f'{name} , {sub_layer.weight.name} , {sub_layer.weight}'
                    if self.test_lora:
                        assert sub_layer.lora_A.placements == [
                            dist.Replicate(),
                            dist.Shard(0),
                        ]
                    # assert sub_layer.bias.placements is None
                if 'down_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(0),
                    ]
                    if self.test_lora:
                        assert sub_layer.lora_A.placements == [
                            dist.Replicate(),
                            dist.Shard(0),
                        ]

    def check_lora(self, layer):
        if not self.test_lora:
            return
        for name, sub_layer in layer.named_sublayers():
            if len(sub_layer.sublayers()) == 0:
                if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    assert sub_layer.weight.stop_gradient
                    assert not sub_layer.lora_A.stop_gradient
                    assert not sub_layer.lora_B.stop_gradient
                if 'gate_proj' in name or 'up_proj' in name:
                    assert sub_layer.weight.stop_gradient
                    assert not sub_layer.lora_A.stop_gradient
                    assert not sub_layer.lora_B.stop_gradient
                if (
                    'embed_tokens' in name or 'lm_head' in name
                ) and not self.share_embedding:
                    assert sub_layer.weight.stop_gradient
                if 'o_proj' in name:
                    assert (
                        sub_layer.weight.stop_gradient
                    ), f'{name} , {sub_layer.weight.name} , {sub_layer.weight}'
                    assert not sub_layer.lora_A.stop_gradient
                    assert not sub_layer.lora_B.stop_gradient
                    # assert sub_layer.bias.stop_gradient is None
                if 'down_proj' in name:
                    assert sub_layer.weight.stop_gradient
                    assert not sub_layer.lora_A.stop_gradient
                    assert not sub_layer.lora_B.stop_gradient

    def parallel_model(self, layer):
        dp_config = None
        mp_config = None
        pp_config = None
        prefix = "model." if self.test_lora else ""
        if self.pp > 1:
            # decoders_per_rank = self.config.num_hidden_layers // self.pp
            # split_spec = {
            #     ff"{prefix}llama.layers.{i * decoders_per_rank - 1}": SplitPoint.END
            #     for i in range(1, self.pp)
            # }
            pp_config = {
                'split_spec': f"{prefix}llama.layers",
                "global_spec": f"{prefix}llama.global_layer",
            }
        if self.dp > 1:
            dp_config = {'sharding_level': self.level}
        if self.mp > 1:
            if not self.sequence_parallel:
                plan = {
                    f"{prefix}llama.embed_tokens": dist.ColWiseParallel(
                        gather_output=True
                    ),
                    f"{prefix}llama.position_embedding": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(
                        gather_output=True
                    ),
                    f"{prefix}llama.layers.*.self_attn.q_proj.lora_B": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(
                        gather_output=True
                    ),
                    f"{prefix}llama.layers.*.self_attn.k_proj.lora_B": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(
                        gather_output=True
                    ),
                    f"{prefix}llama.layers.*.self_attn.v_proj.lora_B": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(
                        is_input_parallel=False
                    ),
                    f"{prefix}llama.layers.*.self_attn.o_proj.lora_A": dist.RowWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.gate_proj.lora_B": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.up_proj.lora_B": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.down_proj.lora_A": dist.RowWiseParallel(),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                }
            else:
                if self.prepare_input_output:
                    plan = {
                        f"{prefix}llama.embed_tokens": dist.ColWiseParallel(),
                        f"{prefix}llama.position_embedding": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                        f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                        f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.input_layernorm": dist.SequenceParallelEnable(),
                        f"{prefix}llama.layers.*.post_attention_layernorm": dist.SequenceParallelEnable(),
                        f"{prefix}llama.norm": dist.SequenceParallelEnable(),
                    }
                else:
                    plan = {
                        f"{prefix}llama.embed_tokens": [
                            dist.ColWiseParallel(),
                            dist.SequenceParallelBegin(),
                        ],
                        f"{prefix}llama.position_embedding": [
                            dist.ColWiseParallel(),
                            dist.SequenceParallelBegin(),
                        ],
                        f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                        f"{prefix}llama.layers.*.self_attn": dist.SequenceParallelDisable(),
                        f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                        f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                        f"{prefix}llama.layers.*.mlp": dist.SequenceParallelDisable(
                            need_transpose=False
                        ),
                        f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                        f"{prefix}lm_head": dist.SequenceParallelEnd(),
                    }
            mp_config = {'parallelize_plan': plan}

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )

        config = {
            'dp_config': dp_config,
            'mp_config': mp_config,
            'pp_config': pp_config,
        }

        if self.one_api:
            optimizer = create_optimizer(layer, lr_scheduler)
            model, optimizer = dist.parallelize(
                layer,
                optimizer,
                config=config,
            )
        else:
            layer = parallelize_model(
                layer,
                config=config,
            )
            optimizer = create_optimizer(layer, lr_scheduler)
            optimizer = parallelize_optimizer(
                optimizer,
                config=config,
            )
        self.check_mp(layer)
        self.check_lora(layer)
        return layer, optimizer, lr_scheduler

    def run_llama(self, to_static=0):
        if self.config.use_lazy_init:
            with LazyGuard():
                model = LlamaForCausalLM(
                    self.config, self.share_embedding, self.position_embedding
                )
        else:
            model = LlamaForCausalLM(
                self.config, self.share_embedding, self.position_embedding
            )
        if self.test_lora:
            if self.config.use_lazy_init:
                with LazyGuard():
                    model = LoRAModel(model, self.lora_config)
            else:
                model = LoRAModel(model, self.lora_config)
        model, optimizer, lr_scheduler = self.parallel_model(model)

        criterion = LlamaPretrainingCriterion(self.config)

        if self.config.use_lazy_init:
            for param in model.parameters():
                assert not param._is_initialized()
                param.initialize()

        if self.amp and not to_static:
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=self.amp_level,
                dtype=self.amp_dtype,
                master_grad=self.amp_master_grad,
            )

        train_dataset = RandomDataset(self.config.seq_length)
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

            for step, inputs in enumerate(dist_loader()):
                input_ids, labels = inputs
                custom_black_list = [
                    "reduce_sum",
                    "c_softmax_with_cross_entropy",
                ]
                custom_white_list = []
                if self.amp_level == "O2":
                    custom_white_list.extend(
                        ["lookup_table", "lookup_table_v2"]
                    )
                with paddle.amp.auto_cast(
                    self.amp,
                    custom_black_list=set(custom_black_list),
                    custom_white_list=set(custom_white_list),
                    level=self.amp_level,
                    dtype=self.amp_dtype,
                ):
                    logits = model(input_ids)
                    tr_loss_step = criterion(logits, labels)

                if self.gradient_accumulation_steps > 1:
                    tr_loss_step /= self.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(tr_loss_step).backward()
                else:
                    tr_loss_step.backward()
                tr_loss += tr_loss_step

                if global_step % self.gradient_accumulation_steps == 0:
                    logging.info(
                        f"step: {global_step // self.gradient_accumulation_steps}  loss: {tr_loss.numpy()}"
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.clear_grad()
                    lr_scheduler.step()
                    tr_loss = 0

                global_step += 1
                if global_step // self.gradient_accumulation_steps >= 3:
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
                loss = dist_model(input_ids, labels)
                logging.info(f"step: {step}  loss: {loss}")
                if step >= 3:
                    break

    def run_test_cases(self):
        self.run_llama(0)
        self.run_llama(1)


if __name__ == '__main__':
    TestParallelAPI().run_test_cases()
