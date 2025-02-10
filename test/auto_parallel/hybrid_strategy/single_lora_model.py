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
import math
import re

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_quick_lora: bool = False,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        pissa: bool = False,
        lora_use_mixer: bool = False,
        use_mora: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.use_mora = use_mora
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.pissa = pissa
        self.lora_use_mixer = lora_use_mixer

        # Actual trainable parameters
        if use_mora:  # reset the rank and create high rank matrix
            self.in_features = in_features
            self.out_features = out_features
            new_r = int(math.sqrt((in_features + out_features) * r) + 0.5)
            new_r = new_r // 2 * 2
            self.r = new_r
            self.lora_A = self.create_parameter(
                shape=[self.r, self.r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.cos = None
            self.sin = None
            # Count the number of tiles
            self.rb1 = (
                self.in_features // self.r
                if self.in_features % self.r == 0
                else self.in_features // self.r + 1
            )
            self.rb2 = (
                self.out_features // self.r
                if self.out_features % self.r == 0
                else self.out_features // self.r + 1
            )
            self.rope_init()
        else:
            self.lora_A = self.create_parameter(
                shape=[in_features, r],
                dtype=self._dtype,
                is_bias=False,
            )
            if self.lora_use_mixer:
                self.lora_AB = self.create_parameter(
                    shape=[r, r],
                    dtype=self._dtype,
                    is_bias=False,
                )
            self.lora_B = self.create_parameter(
                shape=[r, out_features],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )
        self.apply_pissa = False
        if use_mora or pissa:
            self.scaling = 1.0
        elif not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    def pissa_init(self, rank):
        weight = self.weight
        dtype = weight.dtype
        if dtype != paddle.float32:
            weight = weight.astype(paddle.float32)

        U, S, Vh = paddle.linalg.svd(weight.data, full_matrices=False)
        Ur = U[:, :rank]
        Sr = S[:rank]
        Vhr = Vh[:rank]

        lora_A = Ur @ paddle.diag(paddle.sqrt(Sr))
        lora_B = paddle.diag(paddle.sqrt(Sr)) @ Vhr
        self.lora_A.set_value(lora_A.astype(dtype))
        self.lora_B.set_value(lora_B.astype(dtype))
        res = weight.data - lora_A @ lora_B
        weight = res.astype(dtype)
        self.weight.set_value(weight)

    def rope_init(self):
        if self.cos is None or self.sin is None:
            inv_freq = 1.0 / (
                10000
                ** (paddle.arange(0, self.r, 2, dtype=paddle.float32) / self.r)
            )
            t = paddle.arange(self.rb1, dtype=paddle.float32)
            freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos = paddle.unsqueeze(paddle.cos(emb), axis=0).astype(
                self._dtype
            )
            self.sin = paddle.unsqueeze(paddle.sin(emb), axis=0).astype(
                self._dtype
            )

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def _apply_mora(self, x):
        r = self.r

        # Calculate grouping
        sum_inter = self.in_features // r

        # padding
        if self.in_features % r != 0:
            pad_size = r - self.in_features % r
            x = paddle.concat([x, x[..., :pad_size]], axis=-1)
            sum_inter += 1

        # reshape the input to apply RoPE
        in_x = x.reshape([*x.shape[:-1], sum_inter, r])

        # apply RoPE rotation
        rh_in_x = paddle.concat(
            [-in_x[..., r // 2 :], in_x[..., : r // 2]], axis=-1
        )
        in_x = in_x * self.cos + rh_in_x * self.sin

        # matmul with high rank matrix
        out_x = in_x @ self.lora_A

        # reshape the output
        out_x = out_x.reshape([*x.shape[:-1], -1])[..., : self.out_features]
        if out_x.shape[-1] < self.out_features:
            repeat_time = self.out_features // out_x.shape[-1]
            if self.out_features % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = paddle.concat([out_x] * repeat_time, axis=-1)[
                ..., : self.out_features
            ]

        return out_x

    def get_delta_weight(self, lora_A=None, lora_B=None, lora_AB=None):
        # compute the delta weight，which is used to merge weights
        if self.lora_use_mixer:
            lora_A = lora_A if lora_A is not None else self.lora_A
            lora_B = lora_B if lora_B is not None else self.lora_B
            lora_AB = lora_AB if lora_AB is not None else self.lora_AB
            delta_weight = lora_A @ lora_AB @ lora_B * self.scaling
        elif self.use_mora:
            lora_A = lora_A if lora_A is not None else self.lora_A
            r = self.r
            # compute padding
            pad_size = (
                r - self.in_features % r if self.in_features % r != 0 else 0
            )
            # initialize weights
            w = paddle.zeros(
                [self.in_features + pad_size, self.in_features],
                dtype=lora_A.dtype,
            )

            # create the weights after rotation
            aw2 = paddle.concat(
                [lora_A[:, r // 2 :], -lora_A[:, : r // 2]], axis=-1
            )
            # apply RoPE
            for i in range(self.rb1 - 1):
                w[i * r : (i + 1) * r, i * r : (i + 1) * r] = (
                    aw2 * self.sin[:, i] + lora_A * self.cos[:, i]
                )
            # Process the last chunk that may be incomplete
            i = self.rb1 - 1
            w[i * r :, i * r :] = (
                aw2 * self.sin[:, i] + lora_A * self.cos[:, i]
            )[:, : r - pad_size]
            # padding
            if pad_size > 0:
                w[i * r :, :pad_size] = (
                    aw2 * self.sin[:, i] + lora_A * self.cos[:, i]
                )[:, r - pad_size :]
            # reshape the weights
            if self.in_features < self.out_features:
                w = paddle.concat([w] * self.rb2, axis=0)[: self.out_features]
            else:
                w = w[: self.out_features]
            final_weight = w
            delta_weight = final_weight.T
        else:
            lora_A = lora_A if lora_A is not None else self.lora_A
            lora_B = lora_B if lora_B is not None else self.lora_B
            delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        if not self.apply_pissa and self.pissa:
            self.pissa_init(self.r)
            self.apply_pissa = True
        if self.disable_lora or self.merged:
            result = F.linear(
                x=input, weight=self.weight, bias=self.bias, name=self.name
            )
        elif self.use_mora:
            result = F.linear(
                x=input, weight=self.weight, bias=self.bias, name=self.name
            )
            input = self.lora_dropout(input)
            mora_out = self._apply_mora(input)
            result += mora_out
        else:
            result = F.linear(
                x=input, weight=self.weight, bias=self.bias, name=self.name
            )
            if self.lora_use_mixer:
                result += (
                    self.lora_dropout(input)
                    @ self.lora_A
                    @ self.lora_AB
                    @ self.lora_B
                ) * self.scaling
            else:
                result += (
                    self.lora_dropout(input) @ self.lora_A @ self.lora_B
                ) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


lora_layers = {
    "LoRALinear": LoRALinear,
}
LoRALinear = lora_layers["LoRALinear"]
AVAILABLE_LAYERS = [
    LoRALinear,
]


class LoRAModel(nn.Layer):

    def __init__(self, model, lora_config) -> None:
        super().__init__()
        self.model = self.get_lora_model(model, lora_config)

        self.lora_config = lora_config
        logging.info("Mark only lora and trainable_module as trainable.")
        self.mark_only_lora_as_trainable()

    def forward(self, input_ids):
        return self.model(input_ids)

    def _find_and_replace_module(self, model, module_name, lora_config):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lora_module = None
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                rslora=lora_config.rslora,
                lora_plus_scale=lora_config.lora_plus_scale,
                pissa=lora_config.pissa,
                bias_attr=False if module.bias is None else None,
                use_quick_lora=lora_config.use_quick_lora,
                lora_use_mixer=lora_config.lora_use_mixer,
                use_mora=lora_config.use_mora,
            )
        if lora_module is None:
            raise ValueError(
                f"LoRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear or paddlenlp.transformers.sequence_utils. {module}({module_name} {type(module).__name__}) is not supported。"
            )
        lora_module.weight = module.weight
        if module.bias is not None:
            lora_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], lora_module)

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for _, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += np.prod(weight.shape)
            else:
                trainable_numel += np.prod(weight.shape)
        logging.debug(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel + trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel + trainable_numel):.2%}"
        )

    def mark_only_lora_as_trainable(self) -> None:
        for _, layer in self.model.named_sublayers():
            if isinstance(layer, LoRALinear):
                for name, weight in layer.state_dict().items():
                    if (
                        self.lora_config.trainable_bias in ["lora", "all"]
                        and "bias" in name
                    ):
                        weight.stop_gradient = False
                    elif "lora" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if (
                        self.lora_config.trainable_bias == "all"
                        and "bias" in name
                    ):
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.lora_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name)
                    for trainable_module in self.lora_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_lora_model(self, model, lora_config):
        if lora_config.target_modules is None:
            return model
        elif isinstance(lora_config.target_modules, str):
            target_modules = [lora_config.target_modules]
        else:
            target_modules = lora_config.target_modules
        for target_module in target_modules:
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(
                        model, module_name, lora_config
                    )
        return model

    def train(self):
        self.training = True
        self.model.training = True
        for layer in self.model.sublayers():
            layer.training = True
            layer.train()

    def eval(self):
        self.training = False
        self.model.training = False
        for layer in self.model.sublayers():
            layer.training = False
            layer.eval()

    def disable_lora(self):
        for _, layer in self.model.named_sublayers():
            if any(
                isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS
            ):
                layer.disable_lora = True

    def enable_lora(self):
        for _, layer in self.model.named_sublayers():
            if any(
                isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS
            ):
                layer.disable_lora = False

    def merge(self):
        for _, layer in self.model.named_sublayers():
            if any(
                isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS
            ):
                layer.merge()

    def unmerge(self):
        for _, layer in self.model.named_sublayers():
            if any(
                isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS
            ):
                layer.unmerge()
