// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/mkldnn/matmul_transpose_reshape_fuse_pass.h"
#include <paddle/fluid/string/pretty_log.h>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void MatmulTransposeReshapePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::NotFound(
                 "MatmulTransposeReshapePass graph parameter cannot be null"));
  FusePassBase::Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::MatmulTransposeReshapePattern mtrp(gpd.mutable_pattern(),
                                               name_scope_);

  mtrp();

  int found_matmul_transpose_reshape_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle matmul_transpose_reshape fuse";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out_xshape, transpose_out_xshape, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape_op, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out_xshape, reshape_out_xshape, mtrp);
    auto reshape_shape =
        boost::get<std::vector<int>>(reshape_op->Op()->GetAttr("shape"));
    auto transpose_axis =
        boost::get<std::vector<int>>(transpose_op->Op()->GetAttr("axis"));

    OpDesc *matmul_desc = matmul_op->Op();
    matmul_desc->SetOutput("Out", {reshape_out->Name()});
    matmul_desc->SetAttr("reshape_Out", reshape_shape);
    matmul_desc->SetAttr("transpose_Out", transpose_axis);

    GraphSafeRemoveNodes(graph,
                         {matmul_out, transpose_op, transpose_out, reshape_op,
                          transpose_out_xshape, reshape_out_xshape});

    IR_OP_VAR_LINK(matmul_op, reshape_out);

    found_matmul_transpose_reshape_count++;
  };

  gpd(graph, handler);
  AddStatis(found_matmul_transpose_reshape_count);
  std::stringstream msg_ss;
  msg_ss << "---    Fused " << found_matmul_transpose_reshape_count
         << " MatmulTransposeReshape patterns";
  paddle::string::PrettyLogDetail(msg_ss.str().c_str());
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(matmul_transpose_reshape_fuse_pass,
              paddle::framework::ir::MatmulTransposeReshapePass);
