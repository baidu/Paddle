# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing.dtype_like import _DTypeLiteral

# _g_default_config[category][field] = default_value
_g_default_config = defaultdict(dict)


def get_category_default_config(category):
    return _g_default_config[category]


def set_category_default_config(category, default_value):
    _g_default_config[category] = default_value


def get_field_default_config(category, field):
    return _g_default_config[category][field]


def set_field_default_config(category, field, default_value):
    _g_default_config[category][field] = default_value


NOT_FOUND = "not_found"

#########################################
# base configuration
#########################################
BASE = "base"
set_field_default_config(BASE, "auto_mode", "semi")
set_field_default_config(BASE, "gradient_scale", True)
set_field_default_config(BASE, "gradient_scale_using_allreduce_avg", False)
set_field_default_config(BASE, "use_cache", True)
set_field_default_config(BASE, "return_numpy", True)
set_field_default_config(BASE, "all_ranks", False)
set_field_default_config(BASE, "split_data", True)
set_field_default_config(BASE, "seed", None)
set_field_default_config(BASE, "reinit", False)  # Only for debug

if TYPE_CHECKING:

    class _BaseConfig(TypedDict, total=False):  # noqa: PYI049
        auto_mode: str
        gradient_scale: bool
        gradient_scale_using_allreduce_avg: bool
        use_cache: bool
        return_numpy: bool
        all_ranks: bool
        split_data: bool
        seed: int | None
        reinit: bool


#########################################
# recompute configuration
#########################################
RECOMPUTE = "recompute"
set_field_default_config(RECOMPUTE, "enable", False)
set_field_default_config(RECOMPUTE, "checkpoints", [])
set_field_default_config(RECOMPUTE, "no_recompute_segments", [])
set_field_default_config(RECOMPUTE, "sr", 0)
set_field_default_config(RECOMPUTE, "refined_ops_patterns", [])  # List[Dict]
set_field_default_config(RECOMPUTE, "enable_tuning", False)

if TYPE_CHECKING:

    class _RefinedOpsPatterns(TypedDict, total=False):
        main_ops: list[str]
        num: int
        pre_ops: list[str]
        suf_ops: list[str]

    class _RecomputeConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        checkpoints: list[Tensor]
        no_recompute_segments: list[int]
        sr: int
        refined_ops_patterns: list[_RefinedOpsPatterns]
        enable_tuning: bool


#########################################
# AMP configuration
#########################################
AMP = "amp"
set_field_default_config(AMP, "enable", False)
set_field_default_config(AMP, "dtype", "float16")
set_field_default_config(AMP, "level", "o1")
set_field_default_config(AMP, "init_loss_scaling", 32768.0)
set_field_default_config(AMP, "incr_every_n_steps", 1000)
set_field_default_config(AMP, "decr_every_n_nan_or_inf", 2)
set_field_default_config(AMP, "incr_ratio", 2.0)
set_field_default_config(AMP, "decr_ratio", 0.8)
set_field_default_config(AMP, "use_dynamic_loss_scaling", True)
set_field_default_config(AMP, "custom_white_list", [])
set_field_default_config(AMP, "custom_black_list", [])
set_field_default_config(AMP, "custom_black_varnames", [])
set_field_default_config(AMP, "use_fp16_guard", False)
set_field_default_config(AMP, "use_bf16_guard", False)
set_field_default_config(AMP, "use_master_grad", False)
set_field_default_config(AMP, "use_promote", True)

if TYPE_CHECKING:

    class _AMPConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        dtype: _DTypeLiteral
        level: str
        init_loss_scaling: float
        incr_every_n_steps: int
        decr_every_n_nan_or_inf: int
        incr_ratio: float
        decr_ratio: float
        use_dynamic_loss_scaling: bool
        custom_white_list: list[str]
        custom_black_list: list[str]
        custom_black_varnames: list[str]
        use_fp16_guard: bool
        use_bf16_guard: bool
        use_master_grad: bool
        use_promote: bool


#########################################
# sharding configuration
#########################################
SHARDING = "sharding"
set_field_default_config(SHARDING, "enable", False)
set_field_default_config(SHARDING, "stage", 1)
set_field_default_config(SHARDING, "degree", 8)
set_field_default_config(SHARDING, "enable_overlap", False)
set_field_default_config(SHARDING, "param_comm_stream_num", 1)
set_field_default_config(SHARDING, "grad_comm_stream_num", 1)
set_field_default_config(SHARDING, "param_bucket_size_numel", 1)
set_field_default_config(SHARDING, "grad_bucket_size_numel", 1)
set_field_default_config(SHARDING, "enable_hierarchical_comm", False)
set_field_default_config(SHARDING, "partition_algor", "greedy_even")
set_field_default_config(SHARDING, "enable_tuning", False)
set_field_default_config(SHARDING, "tuning_range", [])
set_field_default_config(SHARDING, "release_gradients", False)
set_field_default_config(SHARDING, "comm_buffer_size_MB", 256)
set_field_default_config(SHARDING, "enable_tensor_fusion", False)
set_field_default_config(SHARDING, "save_unbalanced_param", True)


if TYPE_CHECKING:

    class _ShardingConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        stage: int
        degree: int
        enable_overlap: bool
        param_comm_stream_num: int
        grad_comm_stream_num: int
        param_bucket_size_numel: int
        grad_bucket_size_numel: int
        enable_hierarchical_comm: bool
        partition_algor: str
        enable_tuning: bool
        tuning_range: list[int] | tuple[int, int]


#########################################
# gradient merge configuration
#########################################
GRADIENT_MERGE = "gradient_merge"
set_field_default_config(GRADIENT_MERGE, "enable", False)
set_field_default_config(GRADIENT_MERGE, "k_steps", 1)
set_field_default_config(GRADIENT_MERGE, "avg", True)

if TYPE_CHECKING:

    class _GradientMergeConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        k_steps: int
        avg: bool


#########################################
# pipeline configuration
#########################################
PIPELINE = "pipeline"
set_field_default_config(PIPELINE, "enable", False)
set_field_default_config(PIPELINE, "schedule_mode", "1F1B")
set_field_default_config(PIPELINE, "pp_degree", 1)
set_field_default_config(PIPELINE, "vpp_degree", 1)
set_field_default_config(PIPELINE, "vpp_seg_method", "")
set_field_default_config(PIPELINE, "micro_batch_size", 1)
set_field_default_config(PIPELINE, "accumulate_steps", 1)
set_field_default_config(PIPELINE, "generation_batch_size", 1)
set_field_default_config(PIPELINE, "enable_send_recv_overlap", False)
set_field_default_config(PIPELINE, "job_schedule_profiler_start", -1)
set_field_default_config(PIPELINE, "job_schedule_profiler_stop", -1)
set_field_default_config(PIPELINE, "program_runtimes", [61, 72, 71, 34, 3])
set_field_default_config(PIPELINE, "memory_limit_times", -1)
set_field_default_config(PIPELINE, "split_backward", False)

if TYPE_CHECKING:

    class _PipelineConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        schedule_mode: str
        pp_degree: int
        vpp_degree: int
        vpp_seg_method: str
        micro_batch_size: int
        accumulate_steps: int
        generation_batch_size: int
        enable_send_recv_overlap: bool
        job_schedule_profiler_start: int
        job_schedule_profiler_stop: int
        split_backward: bool


#########################################
# quantization configuration
#########################################
QAT = "qat"
set_field_default_config(QAT, "enable", False)
set_field_default_config(QAT, "channel_wise_abs_max", True)
set_field_default_config(QAT, "weight_bits", 8)
set_field_default_config(QAT, "activation_bits", 8)
set_field_default_config(QAT, "not_quant_pattern", ['skip_quant'])
set_field_default_config(QAT, "algo", None)
set_field_default_config(QAT, "onnx_format", True)

if TYPE_CHECKING:

    class _QATConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        channel_wise_abs_max: bool
        weight_bits: int
        activation_bits: int
        not_quant_pattern: list[str]
        algo: str | None
        onnx_format: bool


#########################################
# auto tuning configuration
#########################################
TUNING = "tuning"
set_field_default_config(TUNING, "enable", False)
set_field_default_config(TUNING, "profile_start_step", 1)
set_field_default_config(TUNING, "profile_end_step", 1)
set_field_default_config(TUNING, "run_after_tuning", True)
set_field_default_config(TUNING, "debug", False)

if TYPE_CHECKING:

    class _TuningConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        profile_start_step: int
        profile_end_step: int
        run_after_tuning: bool
        debug: bool


#########################################
# dataset configuration
#########################################
DATASET = "dataset"
set_field_default_config(DATASET, "enable", False)
set_field_default_config(DATASET, "num_shards", 1)

if TYPE_CHECKING:

    class _DatasetConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        num_shards: int


# #########################################
# # offload configuration
# #########################################
FUSEDLINEARPROMOTION = "fused_linear_promotion"
set_field_default_config(FUSEDLINEARPROMOTION, "enable", False)

if TYPE_CHECKING:

    class _FusedLinearPromotionConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool


#########################################
# fused passes configuration
#########################################
FUSED_PASSES = "fused_passes"
set_field_default_config(FUSED_PASSES, "enable", False)
set_field_default_config(FUSED_PASSES, "fused_passes_list", [])

if TYPE_CHECKING:

    class _FusedPassesConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        fused_passes_list: list[str]


#########################################
# data parallel configuration
#########################################
DP_OPTIMIZATION = "dp_optimization"
set_field_default_config(DP_OPTIMIZATION, "enable", False)
set_field_default_config(DP_OPTIMIZATION, "fuse_all_reduce_ops", True)
set_field_default_config(DP_OPTIMIZATION, "fuse_grad_size_in_MB", 32)
set_field_default_config(DP_OPTIMIZATION, "overlap_comm_cacl", True)
set_field_default_config(
    DP_OPTIMIZATION, "gradient_sync_after_accumulate", False
)

if TYPE_CHECKING:

    class _DPOptimizationConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
        fuse_all_reduce_ops: bool
        fuse_grad_size_in_MB: int
        overlap_comm_cacl: bool
        gradient_sync_after_accumulate: bool


#########################################
# model parallel configuration
#########################################
MP_OPTIMIZATION = "mp_optimization"
set_field_default_config(
    MP_OPTIMIZATION, "allreduce_matmul_grad_overlapping", False
)
set_field_default_config(MP_OPTIMIZATION, "replace_with_c_embedding", False)

set_field_default_config(
    MP_OPTIMIZATION, "replace_with_parallel_cross_entropy", False
)
if TYPE_CHECKING:

    class _MPOptimizationConfig(TypedDict, total=False):  # noqa: PYI049
        allreduce_matmul_grad_overlapping: bool


#########################################
# sequence parallel configuration
#########################################
SP_OPTIMIZATION = "sp_optimization"
set_field_default_config(SP_OPTIMIZATION, "enable", True)

if TYPE_CHECKING:

    class _SPOptimizationConfig(TypedDict, total=False):  # noqa: PYI049
        enable: bool
