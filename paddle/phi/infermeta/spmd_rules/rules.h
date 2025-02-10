/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/infermeta/spmd_rules/add_n.h"
#include "paddle/phi/infermeta/spmd_rules/amp_ops.h"
#include "paddle/phi/infermeta/spmd_rules/argmax.h"
#include "paddle/phi/infermeta/spmd_rules/c_embedding.h"
#include "paddle/phi/infermeta/spmd_rules/c_softmax_with_cross_entropy.h"
#include "paddle/phi/infermeta/spmd_rules/c_softmax_with_multi_label_cross_entropy.h"
#include "paddle/phi/infermeta/spmd_rules/cast.h"
#include "paddle/phi/infermeta/spmd_rules/clip.h"
#include "paddle/phi/infermeta/spmd_rules/coalesce_tensor.h"
#include "paddle/phi/infermeta/spmd_rules/concat.h"
#include "paddle/phi/infermeta/spmd_rules/conv2d.h"
#include "paddle/phi/infermeta/spmd_rules/cross_entropy_with_softmax.h"
#include "paddle/phi/infermeta/spmd_rules/cumsum.h"
#include "paddle/phi/infermeta/spmd_rules/default_data_parallel.h"
#include "paddle/phi/infermeta/spmd_rules/dropout.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/embedding.h"
#include "paddle/phi/infermeta/spmd_rules/expand_as.h"
#include "paddle/phi/infermeta/spmd_rules/flash_attention.h"
#include "paddle/phi/infermeta/spmd_rules/flatten.h"
#include "paddle/phi/infermeta/spmd_rules/full_like.h"
#include "paddle/phi/infermeta/spmd_rules/fused_dropout_add.h"
#include "paddle/phi/infermeta/spmd_rules/fused_linear_param_grad_add.h"
#include "paddle/phi/infermeta/spmd_rules/fused_rope.h"
#include "paddle/phi/infermeta/spmd_rules/gather.h"
#include "paddle/phi/infermeta/spmd_rules/gather_nd.h"
#include "paddle/phi/infermeta/spmd_rules/layer_norm.h"
#include "paddle/phi/infermeta/spmd_rules/logsumexp.h"
#include "paddle/phi/infermeta/spmd_rules/matmul.h"
#include "paddle/phi/infermeta/spmd_rules/moe_combine.h"
#include "paddle/phi/infermeta/spmd_rules/moe_gate_dispatch.h"
#include "paddle/phi/infermeta/spmd_rules/nonzero.h"
#include "paddle/phi/infermeta/spmd_rules/numel.h"
#include "paddle/phi/infermeta/spmd_rules/one_hot.h"
#include "paddle/phi/infermeta/spmd_rules/optimizer.h"
#include "paddle/phi/infermeta/spmd_rules/p_norm.h"
#include "paddle/phi/infermeta/spmd_rules/pad.h"
#include "paddle/phi/infermeta/spmd_rules/pow.h"
#include "paddle/phi/infermeta/spmd_rules/reduction.h"
#include "paddle/phi/infermeta/spmd_rules/replicated.h"
#include "paddle/phi/infermeta/spmd_rules/reshape.h"
#include "paddle/phi/infermeta/spmd_rules/rms_norm.h"
#include "paddle/phi/infermeta/spmd_rules/scale.h"
#include "paddle/phi/infermeta/spmd_rules/scatter.h"
#include "paddle/phi/infermeta/spmd_rules/slice.h"
#include "paddle/phi/infermeta/spmd_rules/softmax.h"
#include "paddle/phi/infermeta/spmd_rules/split.h"
#include "paddle/phi/infermeta/spmd_rules/squared_l2_norm.h"
#include "paddle/phi/infermeta/spmd_rules/squeeze.h"
#include "paddle/phi/infermeta/spmd_rules/stack.h"
#include "paddle/phi/infermeta/spmd_rules/tile.h"
#include "paddle/phi/infermeta/spmd_rules/transpose.h"
#include "paddle/phi/infermeta/spmd_rules/triu.h"
#include "paddle/phi/infermeta/spmd_rules/unbind.h"
#include "paddle/phi/infermeta/spmd_rules/unsqueeze.h"
#include "paddle/phi/infermeta/spmd_rules/where.h"
