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

import hashlib
import os
import random
import tempfile
import time
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

    num_hidden_layers = 2
    num_attention_heads = 2
    num_key_value_heads = 2
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
        if os.getenv("use_sp") == "true":
            self.config.sequence_parallel = True
        self.gradient_accumulation_steps = int(os.getenv("acc_step"))
        self.config.recompute = False
        self.config.sep_parallel_degree = 1

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
        paddle.seed(1024)
        np.random.seed(1024)
        random.seed(1024)

    def check_program_equal(self, program_a, program_b):
        assert (
            program_a.num_ops() == program_b.num_ops()
        ), f'The number of ops between two programs is different: {program_a.num_ops()} vs {program_b.num_ops()}.'
        for i in range(program_a.num_ops()):
            a_op = program_a.global_block().ops[i]
            b_op = program_a.global_block().ops[i]
            # check op name
            assert (
                a_op.name() == b_op.name()
            ), f'The name of {i} op in program is different: {a_op.name()} vs {b_op.name()}.'
            # check op inputs
            for index in range(a_op.num_operands()):
                assert (
                    a_op.operand(index)
                    .source()
                    .is_same(b_op.operand(index).source())
                ), f'The type of {index} operand is different: {a_op.operand(index).source()} vs {b_op.operand(index).source()}'
            # check op outputs
            for index in range(a_op.num_results()):
                assert a_op.result(index).is_same(
                    b_op.result(index)
                ), f'The type of {index} result is different: {a_op.result(index)} vs {b_op.result(index)}'
            # check op attrs
            for k, v in a_op.attrs().items():
                assert (
                    k in b_op.attrs()
                ), f'Can not find key of {k} attribute in other program'
                if k == 'place':
                    assert type(v) == type(
                        b_op.attrs()[k]
                    ), f'The attribute of {k} is different: {type(v)} vs {type(b_op.attrs()[k])}'
                else:
                    assert (
                        v == b_op.attrs()[k]
                    ), f'The attribute of {k} is different: {v} vs {b_op.attrs()[k]}'

    def run_dy2static(self, tmp_ckpt_path):
        model = LlamaForCausalLMAuto(self.config)
        criterion = LlamaPretrainingCriterionAuto(self.config)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)

        train_dataset = RandomDataset(self.config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=False,
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

        strategy = None
        if self.gradient_accumulation_steps > 1:
            strategy = dist.Strategy()
            strategy.pipeline.accumulate_steps = (
                self.gradient_accumulation_steps
            )

        dist_model = dist.to_static(
            model, dist_loader, criterion, optimizer, strategy=strategy
        )

        dist_model.train()

        state_dict = dist_model.state_dict()

        loss_before_save = []
        for step, inputs in enumerate(dist_loader()):
            input_ids, labels = inputs
            loss = dist_model(input_ids, labels)
            lr_scheduler.step()
            if step == 2:
                state_dict = dist_model.state_dict()
                dist.save_state_dict(state_dict, tmp_ckpt_path, async_save=True)
            if step > 2:
                numpy_array = np.array(loss)
                array_bytes = numpy_array.tobytes()
                loss_md5 = hashlib.md5(array_bytes).hexdigest()
                loss_before_save.append(loss_md5)

                if int(dist.get_rank()) in [2, 3, 6, 7]:
                    assert loss is not None
                else:
                    assert loss is None

            if step >= 9:
                break

        # check pir dist_model save&load
        paddle.enable_static()
        model_file_path = os.path.join(
            tmp_ckpt_path,
            "rank_" + str(paddle.distributed.get_rank()) + ".pd_dist_model",
        )
        paddle.save(
            dist_model._engine._pir_dist_main_progs["train"], model_file_path
        )
        loaded_model = paddle.load(model_file_path)
        self.check_program_equal(
            dist_model._engine._pir_dist_main_progs["train"], loaded_model
        )
        paddle.disable_static()
        paddle.distributed.barrier()

        time.sleep(10)

        loss_after_load = []
        for step, inputs in enumerate(dist_loader()):
            if step < 2:
                continue
            input_ids, labels = inputs
            loss = dist_model(input_ids, labels)
            lr_scheduler.step()
            if step == 2:
                state_dict = dist_model.state_dict()
                dist.load_state_dict(state_dict, tmp_ckpt_path)
            if step > 2:
                numpy_array = np.array(loss)
                array_bytes = numpy_array.tobytes()
                loss_md5 = hashlib.md5(array_bytes).hexdigest()
                loss_after_load.append(loss_md5)

                if int(dist.get_rank()) in [2, 3, 6, 7]:
                    assert loss is not None
                else:
                    assert loss is None
            if step >= 9:
                break

        return (loss_before_save, loss_after_load)

    def broadcast_ckpt_path(self, ckpt_path):
        dist.init_parallel_env()
        rank = dist.get_rank()
        if rank == 0:
            byte_array = np.frombuffer(
                ckpt_path.encode('utf-8'), dtype=np.uint8
            )
            length = np.array([len(byte_array)], dtype=np.int32)
        else:
            length = np.array([0], dtype=np.int32)

        length_tensor = paddle.to_tensor(length)
        dist.broadcast(length_tensor, src=0)
        length = length_tensor.numpy()[0]

        if rank != 0:
            byte_array = np.empty(length, dtype=np.uint8)

        byte_array_tensor = paddle.to_tensor(byte_array)

        dist.broadcast(byte_array_tensor, src=0)

        global_ckpt_path = byte_array_tensor.numpy().tobytes().decode('utf-8')

        return global_ckpt_path

    def run_test_cases(self):
        self.init_dist_env()
        ckpt_path = tempfile.TemporaryDirectory()
        tmp_ckpt_path = self.broadcast_ckpt_path(ckpt_path.name)
        loss = self.run_dy2static(tmp_ckpt_path)
        if int(dist.get_rank()) in [2, 3, 6, 7]:
            assert len(loss[0]) == len(loss[1])
            for i in range(len(loss[0])):
                assert loss[0][i] == loss[1][i]
        ckpt_path.cleanup()


if __name__ == '__main__':
    TestLlamaAuto().run_test_cases()
