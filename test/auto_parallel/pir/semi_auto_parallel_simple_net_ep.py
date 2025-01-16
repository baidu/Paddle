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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import BatchSampler, DataLoader
from paddle.static import global_scope


class Config:
    def __init__(self):
        self.batch_num = 3
        self.batch_size = 4
        self.seq_len = 8
        self.input_size = 32
        self.hidden_size = 16
        self.class_num = 10
        self.run_ep = False
        self.mesh = dist.ProcessMesh([0, 1])
        self.expert_mesh_list = []
        self.expert_mesh_list.append(dist.ProcessMesh([0]))
        self.expert_mesh_list.append(dist.ProcessMesh([1]))
        self.sharding_stage = None


class All2AllConfig(Config):
    def __init__(self):
        super().__init__()
        self.flatten_mesh = dist.ProcessMesh([0, 1, 2, 3])
        self.mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        self.expert_mesh_list = []
        self.expert_mesh_list.append(dist.ProcessMesh([[0], [2]]))
        self.expert_mesh_list.append(dist.ProcessMesh([[1], [3]]))


class Config_shared:
    def __init__(self):
        self.batch_num = 5
        self.batch_size = 4
        self.input_size = 32
        self.hidden_size = 16
        self.shared_hidden_size = 32
        self.class_num = 10
        self.run_ep = False
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.num_devices = 2
        self.num_experts = 4
        self.expert_mesh_list = []
        for i in range(self.num_devices):
            for j in range(self.num_experts // self.num_devices):
                self.expert_mesh_list.append(
                    dist.ProcessMesh([i], dim_names=["x"])
                )


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples, return_dict=False):
        self.images = images
        self.labels = labels
        self.num_samples = self.images.shape[0]
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


class MLP(nn.Layer):
    def __init__(self, config, is_shared=False):
        super().__init__()
        self.config = config
        if is_shared:
            self.input_size = config.input_size * config.num_experts
        else:
            self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.class_num = config.class_num
        self.down_proj = nn.Linear(
            self.hidden_size, self.class_num, bias_attr=False
        )

    def redistribute_expert(self, mesh, placements):
        # place the experts on different devices
        self.down_proj.weight = dist.shard_tensor(
            self.down_proj.weight, mesh, placements
        )

    def forward(self, x):
        return self.down_proj(x)


class DemoLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.mesh, [dist.Replicate()]
        )

        self.experts = nn.LayerList()
        self.experts.append(MLP(config))
        self.experts.append(MLP(config))
        if config.run_ep:
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        h = self.gate(x)
        if self.config.run_ep:
            local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                h, self.config.mesh, 0, [dist.Shard(0)]
            )
        else:
            local_val_list = paddle.split(h, num_or_sections=2, axis=0)
        expert_out_list = []
        for i, expert in enumerate(self.experts):
            local_val = local_val_list[i]
            expert_out_list.append(expert(local_val))
        if self.config.run_ep:
            out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out_list, self.config.mesh, [dist.Shard(0)], 0
            )
        else:
            out = paddle.stack(expert_out_list, axis=0)
            out = out.reshape((-1, self.config.class_num))
        return out


class All2AllDemo(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = 2
        b_shape = [config.batch_size, config.seq_len, config.input_size]
        param_initializer = paddle.nn.initializer.Constant(value=0.0)
        self.b = dist.shard_tensor(
            paddle.create_parameter(
                shape=b_shape,
                dtype="float32",
                default_initializer=param_initializer,
            ),
            config.mesh,
            [dist.Replicate(), dist.Shard(1)],
        )

        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.flatten_mesh, [dist.Replicate()]
        )
        self.gate2 = nn.Linear(
            config.class_num, config.class_num, bias_attr=False
        )
        self.gate2.weight = dist.shard_tensor(
            self.gate2.weight, config.mesh, [dist.Replicate()]
        )
        self.experts = nn.LayerList()
        self.experts.append(MLP(config))
        self.experts.append(MLP(config))
        if config.run_ep:
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        h = x + self.b
        h = dist.auto_parallel.moe_utils._dist_reshape(
            h, [-1, h.shape[-1]], self.config.flatten_mesh, [dist.Shard(0)]
        )
        # print(h)
        # print("==== gate.weight ====")
        # print(self.gate.weight)
        h = self.gate(h)
        h = dist.auto_parallel.moe_utils._dist_reshape(
            h,
            [self.num_experts, -1, h.shape[-1]],
            self.config.mesh,
            [dist.Shard(1), dist.Shard(1)],
        )
        h = dist.reshard(h, self.config.mesh, [dist.Shard(1), dist.Shard(0)])
        if self.config.run_ep:
            local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                h, self.config.mesh, 1, [dist.Shard(1), dist.Shard(0)]
            )
        else:
            local_val_list = paddle.split(h, num_or_sections=2, axis=0)
        expert_out_list = []
        for i, expert in enumerate(self.experts):
            local_val = local_val_list[i]
            expert_out_list.append(expert(local_val))
        if self.config.run_ep:
            out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out_list,
                self.config.mesh,
                [dist.Shard(1), dist.Shard(0)],
                1,
            )
        else:
            out = paddle.stack(expert_out_list, axis=0)
            out = out.reshape((-1, self.config.class_num))
        out = dist.reshard(
            out, self.config.mesh, [dist.Shard(1), dist.Shard(1)]
        )
        out = dist.auto_parallel.moe_utils._dist_reshape(
            out,
            [self.config.batch_size, -1, out.shape[-1]],
            self.config.mesh,
            [dist.Shard(0), dist.Shard(1)],
        )
        out = self.gate2(out)
        return out


class DemoSharedLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.mesh, [dist.Replicate()]
        )
        self.shared_gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.shared_gate.weight = dist.shard_tensor(
            self.shared_gate.weight, config.mesh, [dist.Replicate()]
        )
        self.shared_expert = MLP(config, is_shared=True)
        self.experts = nn.LayerList()
        for i in range(self.config.num_experts):
            self.experts.append(MLP(config))
        if config.run_ep:
            self.shared_expert.redistribute_expert(
                self.config.mesh, [dist.Replicate()]
            )
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        h = self.gate(x)
        y = self.shared_gate(x)

        if self.config.run_ep:
            local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                h, self.config.mesh, 0, [dist.Shard(0)]
            )
        else:
            local_val_list = paddle.split(
                h, num_or_sections=self.config.num_experts, axis=0
            )
        expert_out_list = []
        if self.config.run_ep:
            for i in range(self.config.num_devices):
                device_input = paddle.split(
                    local_val_list[i],
                    num_or_sections=self.config.num_experts
                    // self.config.num_devices,
                    axis=0,
                )
                device_out = []
                for j in range(
                    self.config.num_experts // self.config.num_devices
                ):
                    local_val = device_input[j]
                    device_out.append(
                        self.experts[
                            i
                            * self.config.num_experts
                            // self.config.num_devices
                            + j
                        ](local_val)
                    )
                expert_out_list.append(paddle.stack(device_out, axis=0))
        else:
            for i, expert in enumerate(self.experts):
                local_val = local_val_list[i]
                expert_out_list.append(expert(local_val))
        z = self.shared_expert(y)
        if self.config.run_ep:
            out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out_list, self.config.mesh, [dist.Shard(0)], 0
            )
        else:
            out = paddle.stack(expert_out_list, axis=0)
            out = out.reshape((-1, self.config.class_num))
        out = paddle.squeeze(out)
        return out + z


class Criterion(nn.Layer):
    def __init__(self):
        super().__init__()
        self.loss_func = paddle.nn.MSELoss()

    def forward(self, logits, labels):
        loss = self.loss_func(logits, labels)
        return loss


class TestSimpleNetForEP:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))

        paddle.set_device(self._backend)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_optimizer(self, model, lr_scheduler=None):
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )
        return optimizer

    def create_data_loader(self, config):
        nsamples = config.batch_size * config.batch_num
        images = np.random.rand(
            nsamples, config.seq_len, config.input_size
        ).astype('float32')
        labels = np.random.rand(
            nsamples, config.seq_len, config.class_num
        ).astype('float32')
        train_dataset = RandomDataset(images, labels, config.batch_size)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        return train_dataloader

    def build(self, config):
        if isinstance(config, All2AllConfig):
            model = All2AllDemo(config)
        else:
            model = DemoLayer(config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        if config.sharding_stage == 1:
            print("before shard optimizer:", optimizer)
            optimizer = dist.shard_optimizer(
                optimizer, dist.ShardingStage1("d0", config.mesh)
            )
            print("after shard optimizer:", optimizer)
        criterion = Criterion()
        return model, dataloader, criterion, optimizer

    def build_shared(self, config):
        model = DemoSharedLayer(config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        criterion = Criterion()
        return model, dataloader, criterion, optimizer

    def train(self, config, model, train_dataloader, criterion, optimizer):
        tr_loss = float(0)
        global_step = 0
        model.train()

        losses = []
        for step, inputs in enumerate(train_dataloader()):
            inputs, labels = inputs
            logits = model(inputs)
            tr_loss = criterion(logits, labels)

            tr_loss.backward()
            print("==== gate.weight.grad ====")
            print(model.gate.weight.grad)
            optimizer.step()
            optimizer.clear_grad()
            losses.append(tr_loss.numpy())

        return losses

    def run_ep(self):
        self.set_seed(self._seed)
        config = Config()
        config.run_ep = True
        model, train_dataloader, criterion, optimizer = self.build(config)

        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims=0
        )
        loss = self.train(config, model, dist_dataloader, criterion, optimizer)

        return loss

    def run_replicate(self):
        self.set_seed(self._seed)
        config = Config()
        config.run_ep = False
        model, train_dataloader, criterion, optimizer = self.build(config)

        loss = self.train(config, model, train_dataloader, criterion, optimizer)
        return loss

    def run_dy2st(self):
        self.set_seed(self._seed)
        config = Config()
        config.run_ep = True
        model, train_dataloader, criterion, optimizer = self.build(config)

        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims=0
        )

        mode = "train"
        dist_model = dist.to_static(
            model, dist_dataloader, criterion, optimizer
        )
        dist_model.train()

        loss_list = []
        for batch_id, data in enumerate(dist_dataloader()):
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            loss = dist_model(image, label)
            loss_list.append(loss)

        return np.array(loss_list)

    def run_shared_ep(self):
        self.set_seed(self._seed)
        config = Config_shared()
        config.run_ep = True
        model, train_dataloader, criterion, optimizer = self.build_shared(
            config
        )

        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims="x"
        )
        loss = self.train(config, model, dist_dataloader, criterion, optimizer)

        return loss

    def run_shared_replicate(self):
        self.set_seed(self._seed)
        config = Config_shared()
        config.run_ep = False
        model, train_dataloader, criterion, optimizer = self.build_shared(
            config
        )

        loss = self.train(config, model, train_dataloader, criterion, optimizer)
        return loss

    def test_ep_shared_demo_net(self):
        ep_loss = self.run_shared_ep()
        replicate_loss = self.run_shared_replicate()
        np.testing.assert_allclose(ep_loss, replicate_loss, rtol=1e-6)

    def test_ep_demo_net(self):
        replicate_loss = self.run_replicate()
        ep_loss = self.run_ep()
        np.testing.assert_allclose(ep_loss, replicate_loss, rtol=1e-6)

        dy2st_loss = self.run_dy2st()
        paddle.disable_static()
        global_mesh = dist.ProcessMesh([0, 1])
        pd_loss_dy2st = paddle.to_tensor(dy2st_loss)
        pd_loss_dy2st = dist.auto_parallel.api.dtensor_from_local(
            pd_loss_dy2st,
            global_mesh,
            [dist.Partial(dist.ReduceType.kRedAvg)],
        )
        pd_loss_dy2st = dist.reshard(
            pd_loss_dy2st, global_mesh, [dist.Replicate()]
        )
        dy2st_loss = pd_loss_dy2st.numpy()
        np.testing.assert_equal(ep_loss, dy2st_loss)

    def run_all2all_dy(self, config):
        self.set_seed(self._seed)
        model, train_dataloader, criterion, optimizer = self.build(config)

        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims=0
        )
        loss = self.train(config, model, dist_dataloader, criterion, optimizer)
        print(loss)

        return loss

    def run_all2all_dy2st(self, config):
        self.set_seed(self._seed)
        model, train_dataloader, criterion, optimizer = self.build(config)

        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims=0
        )

        mode = "train"
        dist_model = dist.to_static(
            model, dist_dataloader, criterion, optimizer
        )
        dist_model.train()

        print("==== dense program ====")
        print(dist_model._engine.main_program)
        ops = dist_model._engine.main_program.global_block().ops
        dense_program = dist_model._engine.main_program
        # dist_model._fetch_value(ops[7].result(0), "linear_0.w_0.dist_moment1_0")
        dist_model._fetch_value(ops[19].result(0), "linear_0.w_0.dist")
        # dist_model._fetch_value(ops[18].result(0), "expert_weight")
        # dist_model._fetch_value(ops[17].result(0), "linear_2.w_0.dist")

        val = dense_program.get_value_by_op_id(513)
        dist_model._fetch_value(val, "share_data_out")
        val = dense_program.get_value_by_op_id(524)[0]
        dist_model._fetch_value(val, "adamw_out")
        val = dense_program.get_value_by_op_id(525)
        dist_model._fetch_value(val, "allgather_out")
        val = dense_program.get_value_by_op_id(526)
        dist_model._fetch_value(val, "assign_out_out")
        moment2_var = global_scope().find_var("linear_0.w_0.dist_moment2_0")
        loss_list = []
        for batch_id, data in enumerate(dist_dataloader()):
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            loss = dist_model(image, label)
            loss_list.append(loss)

        return np.array(loss_list)

    def run_all2all_case(self):
        config = All2AllConfig()
        config.run_ep = True
        config.batch_size = 2
        config.seq_len = 4
        config.input_size = 6
        config.hidden_size = 4
        config.class_num = 4
        config.sharding_stage = 1

        losses = self.run_all2all_dy2st(config)
        print(losses)

    def run_test_case(self):
        # self.test_ep_demo_net()
        # self.test_ep_shared_demo_net()
        self.run_all2all_case()


if __name__ == "__main__":
    TestSimpleNetForEP().run_test_case()
