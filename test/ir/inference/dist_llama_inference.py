# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from dist_llama_inference_model import LlamaInferenceModel

import paddle
import paddle.distributed as dist
from paddle import LazyGuard
from paddle.distributed import fleet


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
    rope_theta = 10000
    tensor_parallel_degree = 1
    tensor_parallel_rank = 0
    dtype = 'bfloat16'


class TestLlamaExportAndPredict:
    def __init__(self):
        self.config = Config()
        self.dp = int(os.getenv("dp")) if os.getenv("dp") is not None else 1
        self.mp = int(os.getenv("mp")) if os.getenv("dp") is not None else 1
        self.pp = int(os.getenv("pp")) if os.getenv("dp") is not None else 1
        if os.getenv("use_sp") == "true":
            self.config.sequence_parallel = True
        if os.getenv("recompute") == "true":
            self.config.recompute = True
        if os.getenv("use_lazy_init") == "true":
            self.config.use_lazy_init = True

        self.init_dist_env()

    def init_dist_env(self):
        tensor_parallel_degree = paddle.distributed.get_world_size()

        self.mp = paddle.distributed.get_world_size()
        self.config.tensor_parallel_degree = self.mp
        self.tensor_parallel_rank = dist.get_rank()
        if tensor_parallel_degree > 1:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)

    def run_export(self):
        if self.config.use_lazy_init:
            with LazyGuard():
                model = LlamaInferenceModel(self.config)
            for param in model.parameters():
                assert not param._is_initialized()
                param.initialize()
        else:
            model = LlamaInferenceModel(self.config)

        cache_kvs = []
        for i in range(self.config.num_hidden_layers):
            cache_kvs.append(
                paddle.static.InputSpec(
                    shape=[
                        None,
                        self.config.num_key_value_heads // self.mp,
                        None,
                        None,
                    ],
                    dtype=self.config.dtype,
                    name=f"key_caches_{i}",
                )
            )
            cache_kvs.append(
                paddle.static.InputSpec(
                    shape=[
                        None,
                        self.config.num_key_value_heads // self.mp,
                        None,
                        None,
                    ],
                    dtype=self.config.dtype,
                    name=f"value_caches_{i}",
                )
            )

        precache_input_spec = None

        input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name="input_ids"
            ),  # input_ids
            None,  # attention_mask
            None,  # inputs_embeds
            False,  # use_cache
            cache_kvs,  # cache_kvs
            None,  # pre_caches,
            paddle.static.InputSpec(
                shape=[None, 1], dtype="int32", name="seq_lens_encoder"
            ),  # seq_len_encoder
            paddle.static.InputSpec(
                shape=[None, 1], dtype="int32", name="seq_lens_decoder"
            ),  # seq_len_decoder
            None,  # past_key_values
            False,  # output_attentions
            False,  # output_hidden_states
            False,  # return_dict
        ]

        model = paddle.jit.to_static(model, input_spec=input_spec)
        paddle.jit.save(
            model, "./inference_model_unitest", skip_prune_program=True
        )

    def run_inference(self):
        infer_model_path = "./inference_model_unitest"
        if paddle.base.framework.use_pir_api():
            config = paddle.inference.Config(
                infer_model_path + ".json", infer_model_path + ".pdiparams"
            )
        else:
            config = paddle.inference.Config(
                infer_model_path + ".pdmodel", infer_model_path + ".pdiparams"
            )
        config.switch_ir_optim(False)
        device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
        config.enable_use_gpu(100, device_id)
        config.enable_new_executor()
        predictor = paddle.inference.create_predictor(config)

        model_inputs = {}
        model_inputs["input_ids"] = paddle.to_tensor(
            [[10002] * 10], dtype='int64'
        )
        model_inputs["seq_lens_encoder"] = paddle.to_tensor(
            [10], dtype='int32'
        ).reshape((-1, 1))
        model_inputs["seq_lens_decoder"] = paddle.to_tensor(
            [0], dtype='int32'
        ).reshape((-1, 1))
        for i in range(self.config.num_hidden_layers):
            model_inputs[f"key_caches_{i}"] = paddle.full(
                shape=[
                    96,
                    self.config.num_key_value_heads,
                    64,
                    self.config.hidden_size // self.config.num_attention_heads,
                ],
                fill_value=0,
                dtype=self.config.dtype,
            )
            model_inputs[f"value_caches_{i}"] = paddle.full(
                shape=[
                    96,
                    self.config.num_key_value_heads,
                    64,
                    self.config.hidden_size // self.config.num_attention_heads,
                ],
                fill_value=0,
                dtype=self.config.dtype,
            )

        for name in predictor.get_input_names():
            input_tensor = predictor.get_input_handle(name)
            input_tensor.share_external_data(model_inputs[name])

        predictor.run()
        outputs_handle = predictor.get_output_handle(
            predictor.get_output_names()[0]
        )
        result = outputs_handle.copy_to_cpu()

    def run_test_cases(self):
        self.run_export()
        self.run_inference()
        os.system("rm -rf ./inference_model_unitest*")  # 删除模型文件


if __name__ == '__main__':
    TestLlamaExportAndPredict().run_test_cases()
