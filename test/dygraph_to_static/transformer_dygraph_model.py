# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle import base
from paddle.nn import Layer, Linear


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = np.log(1e4 / float(1)) / (num_timescales - 1)
    inv_timescales = (
        np.exp(np.arange(num_timescales)) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(
        inv_timescales, 0
    )
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


class PrePostProcessLayer(Layer):
    def __init__(self, process_cmd, d_model, dropout_rate):
        super().__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y is not None else x)
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_sublayer(
                        f"layer_norm_{len(list(self.children()))}",
                        paddle.nn.LayerNorm(
                            normalized_shape=d_model,
                            weight_attr=base.ParamAttr(
                                initializer=paddle.nn.initializer.Constant(1.0)
                            ),
                            bias_attr=base.ParamAttr(
                                initializer=paddle.nn.initializer.Constant(0.0)
                            ),
                        ),
                    )
                )
            elif cmd == "d":  # add dropout
                if dropout_rate:
                    # TODO(zhangliujie) fix dropout error
                    self.dropout = paddle.nn.Dropout(
                        p=dropout_rate, mode="downscale_in_infer"
                    )
                    self.functors.append(lambda x: self.dropout(x))

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class MultiHeadAttention(Layer):
    def __init__(
        self,
        d_key,
        d_value,
        d_model,
        n_head=1,
        dropout_rate=0.0,
        param_initializer=None,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = Linear(
            in_features=d_model,
            out_features=d_key * n_head,
            bias_attr=False,
            weight_attr=base.ParamAttr(initializer=param_initializer),
        )
        self.k_fc = Linear(
            in_features=d_model,
            out_features=d_key * n_head,
            bias_attr=False,
            weight_attr=base.ParamAttr(initializer=param_initializer),
        )
        self.v_fc = Linear(
            in_features=d_model,
            out_features=d_value * n_head,
            bias_attr=False,
            weight_attr=base.ParamAttr(initializer=param_initializer),
        )
        self.proj_fc = Linear(
            in_features=d_value * n_head,
            out_features=d_model,
            bias_attr=False,
            weight_attr=base.ParamAttr(initializer=param_initializer),
        )

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q = self.q_fc(queries)
        k = self.k_fc(keys)
        v = self.v_fc(values)
        # split head

        q = paddle.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])
        k = paddle.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
        k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
        v = paddle.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
        v = paddle.transpose(x=v, perm=[0, 2, 1, 3])

        if cache is not None:
            cache_k, cache_v = cache["k"], cache["v"]
            k = paddle.concat([cache_k, k], axis=2)
            v = paddle.concat([cache_v, v], axis=2)
            cache["k"], cache["v"] = k, v
        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=self.d_model**-0.5)
        if attn_bias is not None:
            product += attn_bias
        weights = paddle.nn.functional.softmax(product)
        if self.dropout_rate:
            # TODO(zhangliujie) fix dropout error
            weights = paddle.nn.functional.dropout(
                weights,
                p=self.dropout_rate,
                training=self.training,
                mode="downscale_in_infer",
            )
            out = paddle.matmul(weights, v)

        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        out = self.proj_fc(out)
        return out


class FFN(Layer):
    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = Linear(d_model, d_inner_hid)
        self.fc2 = Linear(d_inner_hid, d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = paddle.nn.functional.relu(hidden)
        if self.dropout_rate:
            # TODO(zhangliujie) fix dropout error
            hidden = paddle.nn.functional.dropout(
                hidden,
                p=self.dropout_rate,
                training=self.training,
                mode="downscale_in_infer",
            )
        out = self.fc2(hidden)
        return out


class EncoderLayer(Layer):
    def __init__(
        self,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd="n",
        postprocess_cmd="da",
    ):
        super().__init__()

        self.preprocessor1 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.self_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout
        )
        self.postprocessor1 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )

        self.preprocessor2 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocessor2 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(
            self.preprocessor1(enc_input), None, None, attn_bias
        )
        attn_output = self.postprocessor1(attn_output, enc_input)
        ffn_output = self.ffn(self.preprocessor2(attn_output))
        ffn_output = self.postprocessor2(ffn_output, attn_output)
        return ffn_output


class Encoder(Layer):
    def __init__(
        self,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd="n",
        postprocess_cmd="da",
    ):
        super().__init__()

        self.encoder_layers = []
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_sublayer(
                    f"layer_{i}",
                    EncoderLayer(
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        preprocess_cmd,
                        postprocess_cmd,
                    ),
                )
            )
        self.processor = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output

        return self.processor(enc_output)


class Embedder(Layer):
    def __init__(self, vocab_size, emb_dim, bos_idx=0):
        super().__init__()
        self.word_embedder = paddle.nn.Embedding(
            vocab_size,
            emb_dim,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Normal(0.0, emb_dim**-0.5)
            ),
        )

    def forward(self, word):
        word_emb = self.word_embedder(word)
        return word_emb


class WrapEncoder(Layer):
    def __init__(
        self,
        src_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        word_embedder,
    ):
        super().__init__()
        self.emb_dropout = prepostprocess_dropout
        self.emb_dim = d_model
        self.word_embedder = word_embedder
        self.pos_encoder = paddle.nn.Embedding(
            max_length,
            self.emb_dim,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    position_encoding_init(max_length, self.emb_dim)
                ),
                trainable=False,
            ),
        )
        self.encoder = Encoder(
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
        )

    def forward(self, src_word, src_pos, src_slf_attn_bias):
        word_emb = self.word_embedder(src_word)
        word_emb = paddle.scale(x=word_emb, scale=self.emb_dim**0.5)
        pos_enc = self.pos_encoder(src_pos)
        pos_enc.stop_gradient = True
        emb = word_emb + pos_enc
        # TODO(zhangliujie) fix dropout error
        enc_input = (
            paddle.nn.functional.dropout(
                emb,
                p=self.emb_dropout,
                training=self.training,
                mode="downscale_in_infer",
            )
            if self.emb_dropout
            else emb
        )
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class DecoderLayer(Layer):
    def __init__(
        self,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd="n",
        postprocess_cmd="da",
    ):
        super().__init__()

        self.preprocessor1 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.self_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout
        )
        self.postprocessor1 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )
        self.preprocessor2 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.cross_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout
        )
        self.postprocessor2 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )
        self.preprocessor3 = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocessor3 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout
        )

    def forward(
        self, dec_input, enc_output, self_attn_bias, cross_attn_bias, cache=None
    ):
        self_attn_output = self.self_attn(
            self.preprocessor1(dec_input), None, None, self_attn_bias, cache
        )
        self_attn_output = self.postprocessor1(self_attn_output, dec_input)
        cross_attn_output = self.cross_attn(
            self.preprocessor2(self_attn_output),
            enc_output,
            enc_output,
            cross_attn_bias,
        )
        cross_attn_output = self.postprocessor2(
            cross_attn_output, self_attn_output
        )
        ffn_output = self.ffn(self.preprocessor3(cross_attn_output))
        ffn_output = self.postprocessor3(ffn_output, cross_attn_output)
        return ffn_output


class Decoder(Layer):
    def __init__(
        self,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
    ):
        super().__init__()

        self.decoder_layers = []
        for i in range(n_layer):
            self.decoder_layers.append(
                self.add_sublayer(
                    f"layer_{i}",
                    DecoderLayer(
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        preprocess_cmd,
                        postprocess_cmd,
                    ),
                )
            )
        self.processor = PrePostProcessLayer(
            preprocess_cmd, d_model, prepostprocess_dropout
        )

    def forward(
        self,
        dec_input,
        enc_output,
        self_attn_bias,
        cross_attn_bias,
        caches=None,
    ):
        for i, decoder_layer in enumerate(self.decoder_layers):
            dec_output = decoder_layer(
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                None if caches is None else caches[i],
            )
            dec_input = dec_output
        return self.processor(dec_output)


class WrapDecoder(Layer):
    def __init__(
        self,
        trg_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        share_input_output_embed,
        word_embedder,
    ):
        super().__init__()

        self.emb_dropout = prepostprocess_dropout
        self.emb_dim = d_model
        self.word_embedder = word_embedder
        self.pos_encoder = paddle.nn.Embedding(
            max_length,
            self.emb_dim,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    position_encoding_init(max_length, self.emb_dim)
                ),
                trainable=False,
            ),
        )
        self.decoder = Decoder(
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
        )
        if share_input_output_embed:
            self.linear = lambda x: paddle.matmul(
                x=x, y=self.word_embedder.word_embedder.weight, transpose_y=True
            )
        else:
            self.linear = Linear(
                input_dim=d_model, output_dim=trg_vocab_size, bias_attr=False
            )

    def forward(
        self,
        trg_word,
        trg_pos,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        enc_output,
        caches=None,
    ):
        word_emb = self.word_embedder(trg_word)
        word_emb = paddle.scale(x=word_emb, scale=self.emb_dim**0.5)
        pos_enc = self.pos_encoder(trg_pos)
        pos_enc.stop_gradient = True
        emb = word_emb + pos_enc
        # TODO(zhangliujie) fix dropout error
        dec_input = (
            paddle.nn.functional.dropout(
                emb,
                p=self.emb_dropout,
                training=self.training,
                mode="downscale_in_infer",
            )
            if self.emb_dropout
            else emb
        )
        dec_output = self.decoder(
            dec_input, enc_output, trg_slf_attn_bias, trg_src_attn_bias, caches
        )
        dec_output = paddle.reshape(
            dec_output,
            shape=[-1, dec_output.shape[-1]],
        )
        logits = self.linear(dec_output)
        return logits


class CrossEntropyCriterion:
    def __init__(self, label_smooth_eps):
        self.label_smooth_eps = label_smooth_eps

    def __call__(self, predict, label, weights):
        if self.label_smooth_eps:
            label_out = F.label_smooth(
                label=paddle.squeeze(
                    paddle.nn.functional.one_hot(label, predict.shape[-1])
                ),
                epsilon=self.label_smooth_eps,
            )

        cost = paddle.nn.functional.cross_entropy(
            input=predict,
            label=label_out,
            soft_label=True if self.label_smooth_eps else False,
            reduction="none",
        )
        weighted_cost = cost * weights
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num


class Transformer(Layer):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        bos_id=0,
        eos_id=1,
    ):
        super().__init__()
        src_word_embedder = Embedder(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_idx=bos_id
        )
        self.encoder = WrapEncoder(
            src_vocab_size,
            max_length,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
            src_word_embedder,
        )
        if weight_sharing:
            assert (
                src_vocab_size == trg_vocab_size
            ), "Vocabularies in source and target should be same for weight sharing."
            trg_word_embedder = src_word_embedder
        else:
            trg_word_embedder = Embedder(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_idx=bos_id
            )
        self.decoder = WrapDecoder(
            trg_vocab_size,
            max_length,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
            weight_sharing,
            trg_word_embedder,
        )

        self.trg_vocab_size = trg_vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value

    def forward(
        self,
        src_word,
        src_pos,
        src_slf_attn_bias,
        trg_word,
        trg_pos,
        trg_slf_attn_bias,
        trg_src_attn_bias,
    ):
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        predict = self.decoder(
            trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, enc_output
        )
        return predict

    def beam_search(
        self,
        src_word,
        src_pos,
        src_slf_attn_bias,
        trg_word,
        trg_src_attn_bias,
        bos_id=0,
        eos_id=1,
        beam_size=4,
        max_len=256,
    ):
        def expand_to_beam_size(tensor, beam_size):
            tensor = paddle.reshape(
                tensor, [tensor.shape[0], 1, *list(tensor.shape[1:])]
            )
            tile_dims = [-1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return paddle.expand(tensor, tile_dims)

        def merge_batch_beams(tensor):
            var_dim_in_state = 2  # count in beam dim
            tensor = paddle.transpose(
                tensor,
                list(range(var_dim_in_state, len(tensor.shape)))
                + list(range(0, var_dim_in_state)),
            )

            tensor = paddle.reshape(
                tensor,
                [0] * (len(tensor.shape) - var_dim_in_state)
                + [batch_size * beam_size],
            )
            res = paddle.transpose(
                tensor,
                list(
                    range(
                        (len(tensor.shape) + 1 - var_dim_in_state),
                        len(tensor.shape),
                    )
                )
                + list(range(0, (len(tensor.shape) + 1 - var_dim_in_state))),
            )
            return res

        def split_batch_beams(tensor):
            var_dim_in_state = 1
            tensor = paddle.transpose(
                tensor,
                list(range(var_dim_in_state, len(tensor.shape)))
                + list(range(0, var_dim_in_state)),
            )
            tensor = paddle.reshape(
                tensor,
                [0] * (len(tensor.shape) - var_dim_in_state)
                + [batch_size, beam_size],
            )
            res = paddle.transpose(
                tensor,
                list(
                    range(
                        (len(tensor.shape) - 1 - var_dim_in_state),
                        len(tensor.shape),
                    )
                )
                + list(range(0, (len(tensor.shape) - 1 - var_dim_in_state))),
            )
            return res

        def mask_probs(probs, finished, noend_mask_tensor):
            finished = paddle.cast(finished, dtype=probs.dtype)
            probs = paddle.multiply(
                paddle.expand(
                    paddle.unsqueeze(finished, [2]),
                    [-1, -1, self.trg_vocab_size],
                ),
                noend_mask_tensor,
            ) - paddle.tensor.math._multiply_with_axis(
                probs, (finished - 1), axis=0
            )
            return probs

        def gather(input, indices, batch_pos):
            topk_coordinates = paddle.stack([batch_pos, indices], axis=2)
            return paddle.gather_nd(input, topk_coordinates)

        # run encoder
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        batch_size = enc_output.shape[0]

        # constant number
        inf = float(1.0 * 1e7)
        max_len = (enc_output.shape[1] + 20) if max_len is None else max_len
        vocab_size_tensor = paddle.tensor.fill_constant(
            shape=[1], dtype="int64", value=self.trg_vocab_size
        )
        end_token_tensor = paddle.to_tensor(
            np.full([batch_size, beam_size], eos_id, dtype="int64")
        )
        noend_array = [-inf] * self.trg_vocab_size
        noend_array[eos_id] = 0
        noend_mask_tensor = paddle.to_tensor(
            np.array(noend_array, dtype="float32")
        )
        batch_pos = paddle.expand(
            paddle.unsqueeze(
                paddle.to_tensor(np.arange(0, batch_size, 1, dtype="int64")),
                [1],
            ),
            [-1, beam_size],
        )
        predict_ids = []
        parent_ids = []
        # initialize states of beam search
        log_probs = paddle.to_tensor(
            np.array(
                [[0.0] + [-inf] * (beam_size - 1)] * batch_size, dtype="float32"
            )
        )

        finished = paddle.to_tensor(
            np.full([batch_size, beam_size], 0, dtype="bool")
        )

        trg_word = paddle.tensor.fill_constant(
            shape=[batch_size * beam_size, 1], dtype="int64", value=bos_id
        )

        trg_src_attn_bias = merge_batch_beams(
            expand_to_beam_size(trg_src_attn_bias, beam_size)
        )
        enc_output = merge_batch_beams(
            expand_to_beam_size(enc_output, beam_size)
        )

        # init states (caches) for transformer, need to be updated according to selected beam
        caches = [
            {
                "k": paddle.tensor.fill_constant(
                    shape=[batch_size, beam_size, self.n_head, 0, self.d_key],
                    dtype=enc_output.dtype,
                    value=0,
                ),
                "v": paddle.tensor.fill_constant(
                    shape=[batch_size, beam_size, self.n_head, 0, self.d_value],
                    dtype=enc_output.dtype,
                    value=0,
                ),
            }
            for i in range(self.n_layer)
        ]

        for i in range(paddle.to_tensor(max_len)):
            trg_pos = paddle.tensor.fill_constant(
                shape=trg_word.shape, dtype="int64", value=i
            )
            caches = paddle.utils.map_structure(
                merge_batch_beams, caches
            )  # TODO: modified for dygraph2static
            logits = self.decoder(
                trg_word, trg_pos, None, trg_src_attn_bias, enc_output, caches
            )
            caches = paddle.utils.map_structure(split_batch_beams, caches)
            step_log_probs = split_batch_beams(
                paddle.log(paddle.nn.functional.softmax(logits))
            )

            step_log_probs = mask_probs(
                step_log_probs, finished, noend_mask_tensor
            )
            log_probs = paddle.tensor.math._add_with_axis(
                x=step_log_probs, y=log_probs, axis=0
            )
            log_probs = paddle.reshape(
                log_probs, [-1, beam_size * self.trg_vocab_size]
            )
            scores = log_probs
            topk_scores, topk_indices = paddle.topk(x=scores, k=beam_size)
            beam_indices = paddle.floor_divide(topk_indices, vocab_size_tensor)
            token_indices = paddle.remainder(topk_indices, vocab_size_tensor)

            # update states
            caches = paddle.utils.map_structure(
                lambda x: gather(x, beam_indices, batch_pos), caches
            )
            log_probs = gather(log_probs, topk_indices, batch_pos)
            finished = gather(finished, beam_indices, batch_pos)
            finished = paddle.logical_or(
                finished, paddle.equal(token_indices, end_token_tensor)
            )
            trg_word = paddle.reshape(token_indices, [-1, 1])

            predict_ids.append(token_indices)
            parent_ids.append(beam_indices)

            if paddle.all(finished).numpy():
                break

        predict_ids = paddle.stack(predict_ids, axis=0)
        parent_ids = paddle.stack(parent_ids, axis=0)
        finished_seq = paddle.transpose(
            paddle.nn.functional.gather_tree(predict_ids, parent_ids), [1, 2, 0]
        )
        finished_scores = topk_scores

        return finished_seq, finished_scores
