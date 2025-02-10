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
import random
import time

import numpy as np
from mlp_demo import DPDemoNet

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
IMAGE_SIZE = 16
CLASS_NUM = 8


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(0, 1)
    )


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


class TestSimpleNetShardingTensorFusionSaveLoad:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self._ckpt_path = os.getenv("ckpt_path")
        self._amp = eval(os.getenv("amp", '0'))
        self._master_weight = eval(os.getenv("use_master_weight", '0'))
        self._master_grad = eval(os.getenv("use_master_grad", '0'))
        self._use_promote = eval(os.getenv("use_promote", '0'))
        self._amp_dtype = os.getenv("amp_dtype", 'float16')
        self._amp_level = os.getenv("amp_level", 'O0')
        self._init_loss_scaling = 1024.0
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]
        self.num_batch = 2
        self.save_unbalanced_param = int(
            os.getenv("save_unbalanced_param", '1')
        )

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self, return_dict=False):
        images = np.random.rand(100, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(100, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, 100, return_dict)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

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

    def run_dy2static(self):
        paddle.disable_static()
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)

        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        os.environ["FLAGS_enable_sharding_stage1_tensor_fusion"] = "1"

        self.set_random_seed(self._seed)
        layer = DPDemoNet(self.mesh)
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=layer.parameters(),
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage1("dp", self.mesh))
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
            input_keys=["image", "label"],
            shard_dims=["dp"],
        )

        loss_fn = nn.MSELoss()
        strategy = dist.Strategy()
        strategy.sharding.enable = True
        strategy.sharding.degree = 2
        strategy.sharding.stage = 1
        strategy.sharding.enable_tensor_fusion = True
        strategy.sharding.save_unbalanced_param = self.save_unbalanced_param

        if self._amp:
            layer, opt = paddle.amp.decorate(
                models=layer,
                optimizers=opt,
                level=self._amp_level,
                master_weight=self._master_weight,
                master_grad=self._master_grad,
            )
            amp = strategy.amp
            amp.enable = self._amp
            amp.dtype = self._amp_dtype
            amp.level = self._amp_level
            amp.use_master_weight = self._master_weight
            amp.use_master_weight = self._master_grad
            amp.use_promote = self._use_promote
            amp.init_loss_scaling = self._init_loss_scaling

        # static training
        dist_model = dist.to_static(
            layer, dist_loader, loss_fn, opt, strategy=strategy
        )

        dist_model.train()

        loss_before_save = []
        for step, inputs in enumerate(dist_loader()):
            input_ids, labels = inputs
            loss = dist_model(input_ids, labels)
            lr_scheduler.step()
            if step == 2:
                state_dict = dist_model.state_dict()
                if self.save_unbalanced_param:
                    state_dict = (
                        dist_model._convert_state_dict_with_rank_unique_name(
                            state_dict
                        )
                    )
                else:
                    state_dict = dist_model._convert_state_dict_without_tensor_fusion_param(
                        state_dict
                    )
                dist.save_state_dict(
                    state_dict, self._ckpt_path, async_save=True
                )
            if step > 2:
                numpy_array = np.array(loss)
                array_bytes = numpy_array.tobytes()
                loss_md5 = hashlib.md5(array_bytes).hexdigest()
                loss_before_save.append(loss_md5)

            if step >= 9:
                break

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
                if self.save_unbalanced_param:
                    state_dict = (
                        dist_model._convert_state_dict_with_rank_unique_name(
                            state_dict
                        )
                    )
                else:
                    state_dict = dist_model._convert_state_dict_without_tensor_fusion_param(
                        state_dict
                    )
                dist.load_state_dict(state_dict, self._ckpt_path)
                if self.save_unbalanced_param:
                    state_dict = (
                        dist_model._convert_state_dict_with_origin_name(
                            state_dict
                        )
                    )
                else:
                    state_dict = (
                        dist_model._convert_state_dict_with_tensor_fusion_param(
                            state_dict
                        )
                    )
                dist_model.set_state_dict(state_dict)
            if step > 2:
                numpy_array = np.array(loss)
                array_bytes = numpy_array.tobytes()
                loss_md5 = hashlib.md5(array_bytes).hexdigest()
                loss_after_load.append(loss_md5)

            if step >= 9:
                break

        return (loss_before_save, loss_after_load)

    def run_test_case(self):
        loss = self.run_dy2static()
        if int(dist.get_rank()) == 0:
            assert len(loss[0]) == len(loss[1])
            for i in range(len(loss[0])):
                assert loss[0][i] == loss[1][i]


if __name__ == '__main__':
    TestSimpleNetShardingTensorFusionSaveLoad().run_test_case()
