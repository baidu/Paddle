// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <string>
#include "paddle/fluid/inference/analysis/data_flow_graph_to_fluid_pass.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include "paddle/fluid/inference/analysis/fluid_to_ir_pass.h"
#include "paddle/fluid/inference/analysis/model_store_pass.h"
#include "paddle/fluid/inference/analysis/pass_manager.h"
#include "paddle/fluid/inference/analysis/tensorrt_subgraph_node_mark_pass.h"
#include "paddle/fluid/inference/analysis/tensorrt_subgraph_pass.h"

DEFINE_bool(IA_enable_tensorrt_subgraph_engine, false,
            "Enable subgraph to TensorRT engine for acceleration");

DEFINE_bool(IA_enable_ir, false, "Turn on IR support");

DEFINE_string(IA_graphviz_log_root, "./",
              "Graphviz debuger for data flow graphs.");

DEFINE_string(IA_output_storage_path, "", "optimized model output path");

namespace paddle {
namespace inference {
namespace analysis {

class DfgPassManagerImpl final : public DfgPassManager {
 public:
  DfgPassManagerImpl() {
    // TODO(Superjomn) set the key with pass reprs.
    LOG(INFO)
        << "-----------------------------------------------------------------";
    if (FLAGS_IA_enable_ir) {
      AddPass("fluid-to-ir-pass", new FluidToIrPass);
    } else {
      AddPass("fluid-to-data-flow-graph", new FluidToDataFlowGraphPass);
    }
    TryAddTensorRtPass();
    AddPass("data-flow-graph-to-fluid", new DataFlowGraphToFluidPass);
    if (!FLAGS_IA_output_storage_path.empty()) {
      AddPass("model-store-pass", new ModelStorePass);
    }
    LOG(INFO)
        << "-----------------------------------------------------------------";
  }

  std::string repr() const override { return "dfg-pass-manager"; }
  std::string description() const override { return "DFG pass manager."; }

 private:
  void AddPass(const std::string& name, Pass* pass) {
    VLOG(3) << "Adding pass " << name;
    Register(name, pass);
    AddGraphvizDebugerPass(pass);
  }

  void TryAddTensorRtPass() {
    if (FLAGS_IA_enable_tensorrt_subgraph_engine) {
      auto trt_teller = [&](const Node* node) {
        std::unordered_set<std::string> teller_set(
            {"elementwise_add", "mul", "conv2d", "pool2d", "relu", "softmax",
             "depthwise_conv2d", "batch_norm", "concat"});
        if (!node->IsFunction()) return false;

        const auto* func = static_cast<const Function*>(node);
        if (teller_set.count(func->func_type())) {
          return true;
        } else {
          return false;
        }
      };

      AddPass("tensorrt-subgraph-marker",
              new TensorRTSubgraphNodeMarkPass(trt_teller));
      AddPass("tensorrt-subgraph", new TensorRTSubGraphPass(trt_teller));
    }
  }

  // Add the graphviz debuger pass if the parent pass has one.
  void AddGraphvizDebugerPass(Pass* pass) {
    auto* debuger_pass = pass->CreateGraphvizDebugerPass();
    if (debuger_pass) {
      LOG(INFO) << " - register debug pass [" << debuger_pass->repr() << "]";
      Register(debuger_pass->repr(), debuger_pass);
    }
  }
};

Analyzer::Analyzer() { Register("manager1", new DfgPassManagerImpl); }

void Analyzer::Run(Argument* argument) {
  // Ugly support fluid-to-ir-pass
  argument->Set(kFluidToIrPassesAttr,
                new std::vector<std::string>({
                    // Manual update the passes here.
                    "graph_viz_pass",                              //
                    "infer_clean_graph_pass", "graph_viz_pass",    //
                    "attention_lstm_fuse_pass", "graph_viz_pass",  //
                    "fc_lstm_fuse_pass", "graph_viz_pass",         //
                    "seq_concat_fc_fuse_pass", "graph_viz_pass",   //
                    "fc_fuse_pass", "graph_viz_pass"               //

                }));

  for (auto& x : data_) {
    PADDLE_ENFORCE(x->Initialize(argument));
    x->RunAll();
    PADDLE_ENFORCE(x->Finalize());
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
