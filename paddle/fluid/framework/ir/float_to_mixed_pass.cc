// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/float_to_mixed_pass.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {

using VarType = FloatToMixedPass::VarType;

bool PhiKernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  const auto& kernels = phi::KernelFactory::Instance().kernels();
  if (kernels.count(op_type) == 0) {
    return false;
  }
  phi::KernelKey kernel_key(backend, layout, data_type);
  return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
}

bool GpuKernelSupportPrecision(
    const std::string& op_type,
    phi::DataType precision,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);
  bool support = PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPU, precision, layout);
  support |= PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPUDNN, precision, layout);

  if (!support) {
    const auto& all_kernels = framework::OperatorWithKernel::AllOpKernels();
    auto it = all_kernels.find(op_type);
    if (it != all_kernels.end()) {
      for (const auto& kern_pair : it->second) {
        if (platform::is_gpu_place(kern_pair.first.place_) &&
            kern_pair.first.data_type_ ==
                framework::TransToProtoVarType(precision)) {
          support = true;
          break;
        }
      }
    }
  }
  return support;
}

void DoInsertCastOp(Graph* graph,
                    Node* var_node,
                    Node* op_node,
                    VarType::Type from_type,
                    VarType::Type to_type,
                    framework::BlockDesc* block_desc,
                    int* suffix,
                    std::unordered_map<Node*, Node*>* cache) {
  if (from_type == to_type) return;

  auto update_cast_desc = [&](framework::OpDesc& desc,
                              const std::string& x_name,
                              const std::string& out_name,
                              const int in_dtype,
                              const int out_dtype) {
    desc.SetType("cast");
    desc.SetInput("X", {x_name});
    desc.SetOutput("Out", {out_name});
    desc.SetAttr("in_dtype", in_dtype);
    desc.SetAttr("out_dtype", out_dtype);
    desc.SetAttr("use_mkldnn", false);
    desc.SetAttr("with_quant_attr", false);
    desc.Flush();
  };

  if (cache->count(var_node) == 0) {
    // insert cast op before node.
    std::string cast_input_name = var_node->Var()->Name();
    std::string cast_output_name =
        var_node->Var()->Name() + "_cast.tmp_" + std::to_string((*suffix)++);
    framework::OpDesc cast_op_desc(block_desc);
    update_cast_desc(cast_op_desc,
                     cast_input_name,
                     cast_output_name,
                     static_cast<int>(from_type),
                     static_cast<int>(to_type));
    auto* cast_op_node = graph->CreateOpNode(&cast_op_desc);
    auto* cast_output_vardesc = block_desc->Var(cast_output_name);
    cast_output_vardesc->SetPersistable(false);
    cast_output_vardesc->SetDataType(to_type);
    cast_output_vardesc->SetShape(var_node->Var()->GetShape());
    auto* cast_output_node = graph->CreateVarNode(cast_output_vardesc);
    IR_NODE_LINK_TO(cast_op_node, cast_output_node);
    (*cache)[var_node] = cast_output_node;
  }
  op_node->Op()->Rename(var_node->Name(), cache->at(var_node)->Name());
  IR_NODE_LINK_TO(var_node, cache->at(var_node)->inputs[0]);
  IR_NODE_LINK_TO(cache->at(var_node), op_node);

  IR_NODE_UNLINK(var_node, op_node);
}

inline bool VarNodeHasDtype(Node* var_node) {
  CHECK_EQ(var_node->IsVar(), true);
  auto type = var_node->Var()->GetType();
  return (type == VarType::SELECTED_ROWS) || (type == VarType::LOD_TENSOR) ||
         (type == VarType::LOD_TENSOR_ARRAY) || (type == VarType::STRINGS) ||
         (type == VarType::VOCAB);
}

inline bool IsFloatType(VarType::Type type) {
  return (type == VarType::FP64) || (type == VarType::FP32);
}

inline bool IsMixedType(VarType::Type type) {
  return (type == VarType::FP16) || (type == VarType::BF16);
}

};  // namespace

void FloatToMixedPass::SetDefaultBlacklist() const {
  // Copy from python/paddle/fluid/contrib/mixed_precision/fp16_lists.py

  // The set of ops that support fp16 calculation and are considered
  // numerically-dangerous and whose effects may also be observed in downstream
  // ops.
  black_list_.insert({
      "exp",
      "square",
      "log",
      "mean",
      "sum",
      "cos_sim",
      "softmax",
      "softmax_with_cross_entropy",
      "sigmoid_cross_entropy_with_logits",
      "c_softmax_with_cross_entropy",
      "cross_entropy",
      "cross_entropy2",
      // fp16 is slower than fp32, though fp16 is supported.
      "lookup_table",
      "lookup_table_v2",
      "linear_interp_v2",
      "nearest_interp_v2",
      "bilinear_interp_v2",
      "bicubic_interp_v2",
      "trilinear_interp_v2",
      // default fp32 can avoid return inf when the sum value large than 65504.
      "reduce_sum",
      // special op for this pass
  });
}

void FloatToMixedPass::Init(Graph* graph) const {
  keep_io_types_ = true;
  mixed_precision_ =
      static_cast<phi::DataType>(Get<int>("mixed_precision_mode"));
  black_list_ = Get<std::unordered_set<std::string>>("mixed_black_list");
  SetDefaultBlacklist();

  auto graph_size = graph->SubGraphsSize();
  VLOG(4) << "graph size: " << graph_size;
  subgraphes_.resize(graph_size);
  all_op_nodes_.resize(graph_size);

  for (size_t i = 0; i < graph_size; i++) {
    subgraphes_[i] = graph->GetSubGraph(i);
    all_op_nodes_[i] = TopologySortOperations(*subgraphes_[i]);
    VLOG(4) << "subgraph " << i << " has " << all_op_nodes_[i].size()
            << "op nodes";
    for (auto* var_node : subgraphes_[i]->Nodes()) {
      if (!var_node->IsVar()) continue;

      auto var_name = var_node->Var()->Name();
      if (real_vars_.count(var_name) == 0) {
        real_vars_[var_name] = var_node;
        VLOG(4) << var_name << " is in graph " << i;
      }
    }
  }
}

void FloatToMixedPass::ApplyImpl(Graph* graph) const {
  auto enable_gpu_fp16 = Get<bool>("enable_gpu_fp16");
  if (!enable_gpu_fp16) return;

  CHECK_NOTNULL(graph);
  CHECK_EQ(graph->IsMainGraph(), true);

  FusePassBase::Init("float_to_mixed", graph);

  Init(graph);
  VLOG(4) << "Init done";
  SetOpUniqueType();
  VLOG(4) << "SetOpUniqueType done";
  GetOpPrecision();
  VLOG(4) << "GetOpPrecision done";
  UpdateOpPrecision();
  VLOG(4) << "UpdateOpPrecision done";
  SetVarPrecision();
  VLOG(4) << "SetVarPrecision done";
  ConvertWeightsData();
  VLOG(4) << "ConvertWeightsData done";
  ProcessOpWithDtypeAttr();
  VLOG(4) << "ProcessOpWithDtypeAttr done";
  InsertCastOp();
  VLOG(4) << "InsertCastOp done";
  RestoreOpOriginType();
  VLOG(4) << "RestoreOpOriginType done";

  for (auto* graph : subgraphes_) {
    CHECK_EQ(HasCircle(*graph), false);
    CHECK_EQ(VarDescIsConsistency(*graph), true);
  }
}

bool FloatToMixedPass::OpSupportPrecision(const std::string& op_type,
                                          phi::DataType precision,
                                          phi::Backend backend) const {
  bool support = false;
  if (black_list_.count(op_type) == 0) {
    if (backend == phi::Backend::GPU) {
      support = GpuKernelSupportPrecision(op_type, precision);
    }
  }
  return support;
}

void FloatToMixedPass::SetOpUniqueType() const {
  int suffix = 0;
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      CHECK_EQ(op_node->IsOp(), true);

      auto op_type = op_node->Op()->Type();

      if (op_type == "feed" || op_type == "fetch") continue;

      std::string unique_type = op_type + "_" + std::to_string(suffix++);
      op_original_type_[unique_type] = op_type;
      op_node->Op()->SetType(unique_type);
      op_node->Op()->Flush();
      VLOG(4) << "change op type: " << op_type << " ---> " << unique_type;
    }
  }
}

void FloatToMixedPass::RestoreOpOriginType() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      CHECK_EQ(op_node->IsOp(), true);

      auto op_type = op_node->Op()->Type();
      if (op_original_type_.count(op_type) &&
          op_original_type_[op_type] != std::string()) {
        op_node->Op()->SetType(op_original_type_[op_type]);
        op_node->Op()->Flush();
        VLOG(4) << "restore op type: " << op_type << " ---> "
                << op_original_type_[op_type];
      }
    }
  }
}

void FloatToMixedPass::ProcessOpWithDtypeAttr() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      CHECK_EQ(op_node->IsOp(), true);

      auto op_type = op_node->Op()->Type();
      if (op_run_mixed_.count(op_type) == 0) continue;

      if (op_node->Op()->HasAttr("dtype")) {
        auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
        if (IsFloatType(static_cast<VarType::Type>(dtype))) {
          op_node->Op()->SetAttr(
              "dtype",
              static_cast<int>(
                  framework::TransToProtoVarType(mixed_precision_)));
          op_node->Op()->Flush();
          VLOG(4) << "process op with dtype attr: " << op_type << " ( " << dtype
                  << " --->" << static_cast<int>(mixed_precision_) << " )";
        }
      }
      if (op_node->Op()->HasAttr("out_dtype")) {
        auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
        if (IsFloatType(static_cast<VarType::Type>(out_dtype))) {
          op_node->Op()->SetAttr(
              "out_dtype",
              static_cast<int>(
                  framework::TransToProtoVarType(mixed_precision_)));
          op_node->Op()->Flush();
          VLOG(4) << "process op with out_dtype attr: " << op_type << " ( "
                  << out_dtype << " --->" << static_cast<int>(mixed_precision_)
                  << " )";
        }
      }
    }
  }
}

void FloatToMixedPass::GetOpPrecision() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      CHECK_EQ(op_node->IsOp(), true);

      auto op_type = op_node->Op()->Type();
      bool support_mixed = true;
      if (op_type == "feed" || op_type == "fetch") {
        support_mixed = !keep_io_types_;
      } else {
        support_mixed =
            OpSupportPrecision(op_original_type_[op_type], mixed_precision_);
      }

      if (op_node->Op()->HasAttr("dtype")) {
        auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
        support_mixed =
            support_mixed && IsFloatType(static_cast<VarType::Type>(dtype));
      } else if (op_node->Op()->HasAttr("out_dtype")) {
        auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
        support_mixed =
            support_mixed && IsFloatType(static_cast<VarType::Type>(out_dtype));
      } else {
        // if op's input var and output var is not dense tensor, the op should
        // not run mixed.
        for (auto* in_var_node : op_node->inputs) {
          CHECK_EQ(in_var_node->IsVar(), true);
          auto* real_in_var_node = real_vars_[in_var_node->Var()->Name()];
          if (real_in_var_node->Var()->Persistable()) continue;

          support_mixed =
              support_mixed &&
              (real_in_var_node->Var()->GetType() == VarType::LOD_TENSOR);
        }

        for (auto* out_var_node : op_node->outputs) {
          CHECK_EQ(out_var_node->IsVar(), true);
          auto* real_out_var_node = real_vars_[out_var_node->Var()->Name()];
          if (real_out_var_node->Var()->Persistable()) continue;

          support_mixed =
              support_mixed &&
              (real_out_var_node->Var()->GetType() == VarType::LOD_TENSOR);
        }
      }

      if (support_mixed) {
        op_run_mixed_.insert(op_type);
        VLOG(4) << "support precision: " << op_type << " run at mixed";
      } else {
        VLOG(4) << "support precision: " << op_type << " not run at mixed";
      }
    }
  }
}

void FloatToMixedPass::UpdateOpPrecision() const {
  std::unordered_set<std::string> vars_should_not_mixed;

  // var -> the var's all input op
  std::unordered_map<std::string, std::vector<Node*>> var_input_ops;
  auto GetVarInputOps = [&] {
    for (const auto& nodes : all_op_nodes_) {
      for (auto* op_node : nodes) {
        CHECK_EQ(op_node->IsOp(), true);

        auto op_type = op_node->Op()->Type();

        if (op_type == "fetch") continue;
        if (op_node->Op()->HasAttr("sub_block")) continue;

        for (auto* var_node : op_node->outputs) {
          CHECK_EQ(var_node->IsVar(), true);
          if (var_node->Var()->Persistable()) continue;
          if (!VarNodeHasDtype(var_node)) continue;

          var_input_ops[var_node->Var()->Name()].push_back(op_node);
          VLOG(4) << "var input ops: " << var_node->Var()->Name()
                  << " is output of " << op_type;
        }

        // the select_input op's input var should not convert to mixed. when
        // op's output var is select_input op's input var, the op should not run
        // mixed.
        if (op_original_type_[op_node->Op()->Type()] == "select_input") {
          for (auto* var_node : op_node->inputs) {
            CHECK_EQ(var_node->IsVar(), true);
            if (var_node->Var()->Persistable()) continue;
            if (!VarNodeHasDtype(var_node)) continue;

            vars_should_not_mixed.insert(var_node->Var()->Name());
          }
        }
      }
    }
  };
  GetVarInputOps();

  bool precision_updated = false;
  do {
    precision_updated = false;
    for (const auto& nodes : all_op_nodes_) {
      for (auto* op_node : nodes) {
        CHECK_EQ(op_node->IsOp(), true);

        if (op_run_mixed_.count(op_node->Op()->Type()) == 0) continue;

        for (auto* var_node : op_node->outputs) {
          CHECK_EQ(var_node->IsVar(), true);
          if (!VarNodeHasDtype(var_node)) continue;

          auto* real_var_node = real_vars_[var_node->Var()->Name()];
          if (real_var_node->Var()->Persistable()) continue;

          bool not_run_mixed = false;
          const auto& input_op_nodes =
              var_input_ops[real_var_node->Var()->Name()];
          if (vars_should_not_mixed.count(real_var_node->Var()->Name())) {
            not_run_mixed = true;
          } else {
            for (auto* node : input_op_nodes) {
              if (op_run_mixed_.count(node->Op()->Type()) == 0) {
                not_run_mixed = true;
                break;
              }
            }
          }
          if (not_run_mixed) {
            op_run_mixed_.erase(op_node->Op()->Type());
            precision_updated = true;
            VLOG(4) << op_node->Op()->Type()
                    << " should not support mixed precision.";
            break;
          }
        }
      }
    }
  } while (precision_updated);
}

bool FloatToMixedPass::InputVarsNotConvert(Node* op_node,
                                           const std::string& var_name) const {
  CHECK_EQ(op_node->IsOp(), true);
  auto* op_desc = op_node->Op();
  if (op_original_type_[op_desc->Type()] == "batch_norm") {
    auto vecs = op_desc->Input("Bias");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("Mean");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("Scale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("Variance");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  } else if (op_original_type_[op_desc->Type()] == "fused_multi_transformer") {
    auto vecs = op_desc->Input("LnScale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("LnBias");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("FFNLnScale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("FFNLnBias");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  }
  return false;
}

bool FloatToMixedPass::OutputVarsNotConvert(Node* op_node,
                                            const std::string& var_name) const {
  CHECK_EQ(op_node->IsOp(), true);
  auto* op_desc = op_node->Op();
  // batch_norm's input and output (variance and mean) are the same.
  if (op_original_type_[op_desc->Type()] == "batch_norm") {
    auto vecs = op_desc->Output("MeanOut");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("VarianceOut");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("SavedMean");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("SavedVariance");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  }
  return false;
}

void FloatToMixedPass::SetVarPrecision() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      CHECK_EQ(op_node->IsOp(), true);

      if (op_run_mixed_.count(op_node->Op()->Type())) {
        for (auto* in_var_node : op_node->inputs) {
          CHECK_EQ(in_var_node->IsVar(), true);

          auto* real_in_var_node = real_vars_[in_var_node->Var()->Name()];
          auto in_var_name = real_in_var_node->Var()->Name();

          if (!IsFloatType(real_in_var_node->Var()->GetDataType())) continue;
          if (!VarNodeHasDtype(real_in_var_node)) continue;
          if (InputVarsNotConvert(op_node, in_var_name)) continue;

          if (real_in_var_node->Var()->Persistable()) {
            real_in_var_node->Var()->SetDataType(
                framework::TransToProtoVarType(mixed_precision_));
            vars_convert_to_mixed_.insert(in_var_name);
          }
        }

        for (auto* out_var_node : op_node->outputs) {
          CHECK_EQ(out_var_node->IsVar(), true);

          auto* real_out_var_node = real_vars_[out_var_node->Var()->Name()];
          auto out_var_name = real_out_var_node->Var()->Name();

          if (!IsFloatType(real_out_var_node->Var()->GetDataType())) continue;
          if (!VarNodeHasDtype(real_out_var_node)) continue;
          if (OutputVarsNotConvert(op_node, out_var_name)) continue;

          real_out_var_node->Var()->SetDataType(
              framework::TransToProtoVarType(mixed_precision_));
          if (real_out_var_node->Var()->Persistable()) {
            vars_convert_to_mixed_.insert(out_var_name);
          }
        }
      }
    }
  }

  // This code used to precess vars with the same name. Vars with the same
  // name should have the same data type.
  for (auto* subgraph : subgraphes_) {
    for (auto* var_node : subgraph->Nodes()) {
      if (!var_node->IsVar() || !var_node->Var()->Persistable()) continue;
      if (!VarNodeHasDtype(var_node)) continue;

      auto var_name = var_node->Var()->Name();
      if (vars_convert_to_mixed_.count(var_name)) {
        var_node->Var()->SetDataType(
            framework::TransToProtoVarType(mixed_precision_));
      }
    }
  }
}

void FloatToMixedPass::ConvertWeightsData() const {
  auto* scope = param_scope();
  CHECK_NOTNULL(scope);

  auto var_names = scope->LocalVarNames();
  for (const auto& var_name : var_names) {
    if (vars_convert_to_mixed_.count(var_name)) {
      VLOG(4) << var_name << "'s data type was convert to mixed";
#define CONVERT_TENSOR_DTYPE(DTYPE, dtype)                                   \
  mixed_tensor.set_type(DTYPE);                                              \
  auto* mixed_data = mixed_tensor.mutable_data<dtype>(platform::CPUPlace()); \
  for (int64_t i = 0; i < origin_tensor->numel(); i++) {                     \
    mixed_data[i] = static_cast<dtype>(origin_data[i]);                      \
  }                                                                          \
  origin_tensor->clear();                                                    \
  paddle::framework::TensorCopySync(                                         \
      mixed_tensor, platform::CPUPlace(), origin_tensor)

      auto* var = scope->FindLocalVar(var_name);

      if (var->IsType<phi::DenseTensor>()) {
        auto* origin_tensor = var->GetMutable<phi::DenseTensor>();
        phi::DenseTensor mixed_tensor;
        mixed_tensor.Resize(origin_tensor->dims());
        auto* origin_data =
            origin_tensor->mutable_data<float>(platform::CPUPlace());
        if (mixed_precision_ == phi::DataType::FLOAT16) {
          CONVERT_TENSOR_DTYPE(paddle::experimental::DataType::FLOAT16,
                               phi::dtype::float16);
        } else if (mixed_precision_ == phi::DataType::BFLOAT16) {
          CONVERT_TENSOR_DTYPE(paddle::experimental::DataType::BFLOAT16,
                               phi::dtype::bfloat16);
        }
      }
    }
#undef CONVERT_TENSOR_DTYPE
  }
}

void FloatToMixedPass::InsertCastOp() const {
  int suffix = 0;
  std::unordered_map<Node*, Node*> cache;

  for (size_t i = 0; i < all_op_nodes_.size(); i++) {
    auto* block_desc = all_op_nodes_[i][0]->Op()->Block();
    CHECK_NOTNULL(block_desc);
    for (auto* op_node : all_op_nodes_[i]) {
      CHECK_EQ(op_node->IsOp(), true);

      auto op_type = op_node->Op()->Type();

      if (op_type == "feed") continue;
      if (op_node->Op()->HasAttr("sub_block")) continue;

      VLOG(4) << "process op: " << op_type
              << " run mixed: " << op_run_mixed_.count(op_type);

      auto inputs = op_node->inputs;
      for (auto* in_var_node : inputs) {
        if (!in_var_node->IsVar()) continue;
        if (!VarNodeHasDtype(in_var_node)) continue;
        if (in_var_node->Var()->Persistable()) continue;

        auto* real_in_var_node = real_vars_[in_var_node->Var()->Name()];

        auto in_var_type = real_in_var_node->Var()->GetDataType();

        VLOG(4) << "process var: " << real_in_var_node->Var()->Name()
                << " with type " << in_var_type;

        if (IsFloatType(in_var_type) && op_run_mixed_.count(op_type)) {
          DoInsertCastOp(subgraphes_[i],
                         in_var_node,
                         op_node,
                         in_var_type,
                         framework::TransToProtoVarType(mixed_precision_),
                         block_desc,
                         &suffix,
                         &cache);
        } else if (IsMixedType(in_var_type) &&
                   op_run_mixed_.count(op_type) == 0) {
          DoInsertCastOp(subgraphes_[i],
                         in_var_node,
                         op_node,
                         in_var_type,
                         VarType::FP32,
                         block_desc,
                         &suffix,
                         &cache);
        }
      }

      // Special op.
      // fused_multi_transformer's input(CacheKV) and output(CacheKVOut) vars
      // have same name.
      if (op_original_type_[op_type] == "fused_multi_transformer") {
        auto cache_kv_inputs = op_node->Op()->Input("CacheKV");
        auto cache_kv_outputs = op_node->Op()->Output("CacheKVOut");
        CHECK_EQ(cache_kv_inputs.size(), cache_kv_outputs.size());
        for (size_t i = 0; i < cache_kv_inputs.size(); ++i) {
          op_node->Op()->RenameOutput(cache_kv_outputs[i], cache_kv_inputs[i]);
        }
      }
    }
  }
  VLOG(4) << "insert number of cast op: " << cache.size();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(float_to_mixed_pass, paddle::framework::ir::FloatToMixedPass);
