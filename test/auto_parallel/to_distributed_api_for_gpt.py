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

import math
import os

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.auto_parallel.high_level_api import (
    ToDistributedConfig,
    to_distributed,
)

VOCAB_SIZE = 8000
BATCH_NUM = 3
BATCH_SIZE = 4
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 4096
SEQ_LENGTH = 1024
N_HEAD = 32
NUM_HIDDEN_LAYERS = 4


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(-0.1, 0.1)
    )


def scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    # merge with the next tranpose
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])

    # matmul and devide by sqrt(head_dim)
    attn_weights = paddle.matmul(
        query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2])
    )

    attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

    attn_weights = attn_weights + attention_mask
    if not paddle.in_dynamic_mode():
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
            query_states.dtype
        )
    else:
        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(
                attn_weights, axis=-1, dtype="float32"
            ).astype(query_states.dtype)

    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose([0, 2, 1, 3])

    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

    return attn_output


class GPTAttention(nn.Layer):
    def __init__(self, param_prefix="", hidden_size=HIDDEN_SIZE, n_head=N_HEAD):
        super().__init__()
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")
        weight_attr_2 = create_numpy_like_random(param_prefix + "_2")
        weight_attr_3 = create_numpy_like_random(param_prefix + "_3")
        self.hidden_size = hidden_size
        self.num_heads = n_head
        self.head_dim = hidden_size // n_head
        self.q_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_0, bias_attr=False
        )
        self.k_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_1, bias_attr=False
        )
        self.v_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_2, bias_attr=False
        )
        self.o_proj = nn.Linear(
            hidden_size, hidden_size, weight_attr_3, bias_attr=False
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # mix_layer = self.qkv_proj(x)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # target_shape = [0, 0, self.num_heads, 3 * self.head_dim]
        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_heads, self.head_dim]

        # mix_layer = paddle.reshape(mix_layer, target_shape)
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)

        output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = output
        attn_output = self.o_proj(attn_output)

        return attn_output


class GPTMlp(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")

        self.linear1 = nn.Linear(
            hidden_size, intermediate_size, weight_attr_0, bias_attr=True
        )
        self.linear2 = nn.Linear(
            intermediate_size, hidden_size, weight_attr_1, bias_attr=True
        )
        self.activation = paddle.nn.functional.gelu

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class GPTLayerNorm(paddle.nn.LayerNorm):
    def __init__(self, hidden_size=HIDDEN_SIZE, epsilon=1e-5):
        super().__init__(
            normalized_shape=hidden_size,
            epsilon=epsilon,
        )

    def forward(self, hidden_states):
        return super().forward(hidden_states)


class GPTDecoderLayer(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.self_attn = GPTAttention(param_prefix + "_att", hidden_size)
        self.dropout1 = nn.Dropout(mode="upscale_in_train")
        self.dropout2 = nn.Dropout(mode="upscale_in_train")
        self.mlp = GPTMlp(param_prefix + "_mlp")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        residual = hidden_states

        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + self.dropout1(hidden_states)
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout2(hidden_states)

        return hidden_states


class GPTEmbedding(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
        )
        self.pos_embeddings = nn.Embedding(
            SEQ_LENGTH,
            hidden_size,
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
    ):
        input_shape = input_ids.shape
        input_embeddings = self.word_embeddings(input_ids)
        if position_ids is None:
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + position_embeddings
        return embeddings


class GPTModel(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.bias = paddle.tril(
            paddle.ones([1, 1, SEQ_LENGTH, SEQ_LENGTH], dtype="int64")
        )

        self.embeddings = GPTEmbedding(
            param_prefix + "embedding",
            vocab_size,
            hidden_size,
            intermediate_size,
        )

        self.layers = nn.LayerList(
            [
                GPTDecoderLayer(param_prefix + "_decoder_" + str(i))
                for i in range(NUM_HIDDEN_LAYERS)
            ]
        )

        self.norm = GPTLayerNorm(hidden_size, epsilon=1e-5)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
    ):
        input_shape = input_ids.shape
        input_ids = input_ids.reshape((-1, input_shape[-1]))

        if position_ids is None:
            past_length = 0
            position_ids = paddle.arange(
                past_length, input_shape[-1] + past_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.expand(position_ids, input_shape)

        inputs_embeds = self.embeddings(
            input_ids=input_ids, position_ids=position_ids
        )

        length = input_shape[-1]
        cache_length = 0
        causal_mask = self.bias[:, :, cache_length:length, :length]
        attention_mask = (1.0 - causal_mask) * -1e4
        # The tensor returned by triu not in static graph.
        attention_mask.stop_gradient = True

        hidden_states = inputs_embeds

        for idx, (decoder_layer) in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.
    It calculates the final loss.
    """

    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.ignore_index = -100
        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

    def forward(self, prediction_scores, masked_lm_labels):
        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(
                prediction_scores.astype("float32"),
                masked_lm_labels.unsqueeze(2),
            )

            binary_sequence = paddle.where(
                masked_lm_loss > 0,
                paddle.ones_like(masked_lm_loss),
                paddle.zeros_like(masked_lm_loss),
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count

        return loss


class GPTLMHead(paddle.nn.Layer):
    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )

    def forward(self, hidden_states, tensor_parallel_output=None):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits


class GPTForCausalLM(paddle.nn.Layer):

    def __init__(
        self,
        param_prefix="",
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.GPT = GPTModel(
            param_prefix + "_GPT", vocab_size, hidden_size, intermediate_size
        )
        self.lm_head = GPTLMHead(
            param_prefix + "_lm_head",
            vocab_size,
            hidden_size,
            intermediate_size,
        )
        self.criterion = GPTPretrainingCriterion(
            param_prefix + "_criterion",
            vocab_size,
            hidden_size,
            intermediate_size,
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
    ):

        outputs = self.GPT(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return (loss, logits)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class TestGPTDecoderForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype", "float32")
        self._backend = os.getenv("backend", "gpu")
        self._seed = eval(os.getenv("seed", "2023"))

        self._mesh = mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["pp", "dp", "mp"]
        )
        self._model = GPTForCausalLM("demo_GPT")

        self._input_seqs = np.random.randint(
            low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
        ).astype("int64")
        self._labels = np.random.randint(
            low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
        ).astype("int64")
        self._dataset = RandomDataset(
            self._input_seqs, self._labels, BATCH_SIZE * BATCH_NUM
        )
        self._loader = paddle.io.DataLoader(
            self._dataset, batch_size=BATCH_SIZE
        )
        self._opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=self._model.parameters()
        )

        paddle.set_device(self._backend)

    def test_to_distributed_api(self):
        # # config: input_spec
        input_seq_spec = paddle.static.InputSpec(
            [BATCH_SIZE, SEQ_LENGTH], 'float32', 'input_seq', True
        )
        dist_config = ToDistributedConfig()
        dist_config.input_spec = [input_seq_spec]
        dist_config.sequence_parallel = True

        # # wrap model by using **to_distributed**
        dist_model, dist_loader, dist_opt = to_distributed(
            self._model, self._loader, self._opt, self._mesh, dist_config
        )

    def run_test_case(self):
        if self._backend == "gpu":
            cuda_version_main = int(paddle.version.cuda().split(".")[0])
            device_prop_main = paddle.device.cuda.get_device_capability()[0]
            if cuda_version_main >= 11 and device_prop_main >= 8:
                self.test_to_distributed_api()


if __name__ == '__main__':
    TestGPTDecoderForSemiAutoParallel().run_test_case()
