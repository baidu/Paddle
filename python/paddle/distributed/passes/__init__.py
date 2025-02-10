# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .allreduce_matmul_grad_overlapping import (  # noqa: F401
    AllreduceMatmulGradOverlappingPass,
)
from .auto_parallel_amp import (  # noqa: F401
    AMPLists,
    AMPPass,
    AMPState,
)
from .auto_parallel_c_embedding import (  # noqa: F401
    AutoParallelCEmbeddingPass,
)
from .auto_parallel_data_parallel_optimization import (  # noqa: F401
    DataParallelOptimizationPass,
    GradientsGroup,
)
from .auto_parallel_fp16 import (  # noqa: F401
    FP16Pass,
    FP16State,
    cast_startup_program,
    set_auto_cast_attr,
    set_op_dtype_to_fp16,
)
from .auto_parallel_fused_linear_promotion import (  # noqa: F401
    FusedLinearPromotionPass,
)
from .auto_parallel_grad_clip import (  # noqa: F401
    ClipGradByGlobalNormPass,
    ClipHelper,
)
from .auto_parallel_gradient_merge import (  # noqa: F401
    GradientMergePass,
    parse_program,
)
from .auto_parallel_master_grad import (  # noqa: F401
    MasterGradPass,
    get_output_in_varlist,
)
from .auto_parallel_quantization import QuantizationPass  # noqa: F401
from .auto_parallel_recompute import (  # noqa: F401
    RecomputePass,
    RecomputeState,
)
from .auto_parallel_recompute_pir import (  # noqa: F401
    AutoParallelRecomputePIRPass,
)
from .auto_parallel_replace_with_parallel_cross_entropy import (  # noqa: F401
    AutoParallelReplaceWithParallelCrossEntropyPass,
)
from .auto_parallel_sequence_parallel_optimization import (  # noqa: F401
    SequenceParallelOptimizationPass,
)
from .auto_parallel_sharding import (  # noqa: F401
    ShardingInfo,
    ShardingPass,
    VarGroup,
    group_param,
    is_sharding_param_broadcast_op,
    partition_by_greedy_even,
    partition_by_use_order,
    partition_parameters,
    re_order_program,
)
from .auto_parallel_supplement_explicit_dependencies import (  # noqa: F401
    AutoParalSupplementDepPass,
)
from .cpp_pass import (  # noqa: F401
    BuildCINNPass,
    FuseAdamWPass,
    FuseBatchNormActPass,
    FuseBatchNormAddActPass,
    FusedAttentionPass,
    FusedFeedforwardPass,
    FuseDotProductAttentionPass,
    FuseElementwiseAddActPass,
    FuseGemmEpiloguePass,
    FuseOptimizerPass,
    FuseReluDepthwiseConvPass,
    FuseResUnitPass,
)

# InplaceAddtoOpPass,
from .fuse_all_reduce import (  # noqa: F401
    FuseAllReducePass,
    filter_all_collective_op_indices,
    find_adjacent_match_sequences,
    find_all_fuse_all_reduce_groups,
    has_same_attrs,
    insert_coalesce_tensor_ops,
    insert_fuse_all_reduce_by_memory_size,
    insert_fuse_all_reduce_ops,
    split_fuse_all_reduce_groups_by_deps,
)
from .pass_base import PassContext, PassManager, new_pass
from .pipeline_scheduler_pass import (  # noqa: F401
    Pipeline1F1BPass,
    PipelineEager1F1BPass,
    PipelineFThenBPass,
    PipelineVirtualPipelinePass,
    PipelineZeroBubblePipelinePass,
    apply_pass,
)
from .ps_server_pass import (  # noqa: F401
    AddGeoOptimizerPass,
    AddListenAndServPass,
    AddLrDecayTablePass,
    AddOptimizerPass,
    AddRpcGlobalFlagsPass,
    BuildPserverStartupProgramPass,
    DeleteUnusedInStartupPass,
)
from .ps_trainer_pass import (  # noqa: F401
    AppendSendOpsPass,
    DeleteExtraOptimizerPass,
    DeleteOptimizesPass,
    DistributedOpsPass,
    FakeInitOpsPass,
    PsGpuPass,
    PsTranspilePass,
    SetHeterPipelineOptPass,
    SplitFlOpsPass,
    SplitHeterWorkerOpsPass,
    SplitTrainerOpsPass,
)

__all__ = [
    'new_pass',
    'PassManager',
    'PassContext',
]
