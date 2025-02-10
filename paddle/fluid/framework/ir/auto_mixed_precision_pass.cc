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

#include "paddle/fluid/framework/ir/auto_mixed_precision_pass.h"

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

namespace paddle::framework::ir {

namespace {

using VarType = AutoMixedPrecisionPass::VarType;

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

static phi::Backend ConvertPlaceToBackend(const phi::Place& place) {
  switch (place.GetType()) {
    case phi::AllocationType::CPU:
      return phi::Backend::CPU;
    case phi::AllocationType::GPU:
      return phi::Backend::GPU;
    case phi::AllocationType::XPU:
      return phi::Backend::XPU;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Cannot convert place(%d).", static_cast<int>(place.GetType())));
  }
  return phi::Backend::UNDEFINED;
}

bool KernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType precision,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);

  bool support =
      PhiKernelSupportPrecision(phi_op_type, backend, precision, layout);
  if (backend == phi::Backend::GPU) {
    support |= PhiKernelSupportPrecision(
        phi_op_type, phi::Backend::GPUDNN, precision, layout);
  }
  if (!support) {
    const auto& all_kernels = framework::OperatorWithKernel::AllOpKernels();
    auto it = all_kernels.find(op_type);
    if (it != all_kernels.end()) {
      for (const auto& kern_pair : it->second) {
        if (ConvertPlaceToBackend(kern_pair.first.place_) == backend &&
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

inline bool VarNodeHasDtype(Node* var_node) {
  auto type = var_node->Var()->GetType();
  return (type == VarType::SELECTED_ROWS) || (type == VarType::DENSE_TENSOR) ||
         (type == VarType::DENSE_TENSOR_ARRAY) || (type == VarType::STRINGS) ||
         (type == VarType::VOCAB) || (type == VarType::SPARSE_COO) ||
         (type == VarType::SPARSE_CSR);
}

inline bool IsFP32(VarType::Type type) { return type == VarType::FP32; }

inline bool IsFP64(VarType::Type type) { return type == VarType::FP64; }

inline bool IsFP16AndBFP16(VarType::Type type) {
  return (type == VarType::FP16) || (type == VarType::BF16);
}

};  // namespace

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
                              const int out_dtype,
                              const VarType::Type t) {
    if (t == VarType::SPARSE_COO || t == VarType::SPARSE_CSR) {
      desc.SetType("sparse_cast");
      desc.SetInput("x", {x_name});
      desc.SetOutput("out", {out_name});
      desc.SetAttr("index_dtype", -1);
      desc.SetAttr("value_dtype", to_type);
    } else {
      desc.SetType("cast");
      desc.SetInput("X", {x_name});
      desc.SetOutput("Out", {out_name});
      desc.SetAttr("in_dtype", in_dtype);
      desc.SetAttr("out_dtype", out_dtype);
    }
    desc.SetAttr("use_mkldnn", false);
    desc.SetAttr("with_quant_attr", false);
    desc.Flush();
  };

  if (cache->count(var_node) == 0) {
    // insert cast op between var_node and op_node
    std::string cast_input_name = var_node->Var()->Name();
    std::string cast_output_name = var_node->Var()->Name() +
                                   "_cast_auto_mixed.tmp_" +
                                   std::to_string((*suffix)++);
    VarType::Type var_type = var_node->Var()->GetType();
    framework::OpDesc cast_op_desc(block_desc);
    update_cast_desc(cast_op_desc,
                     cast_input_name,
                     cast_output_name,
                     static_cast<int>(from_type),
                     static_cast<int>(to_type),
                     var_type);
    auto* cast_op_node = graph->CreateOpNode(&cast_op_desc);
    auto* cast_output_vardesc = block_desc->Var(cast_output_name);
    cast_output_vardesc->SetType(var_type);
    cast_output_vardesc->SetPersistable(false);
    cast_output_vardesc->SetDataType(to_type);
    cast_output_vardesc->SetShape(var_node->Var()->GetShape());
    cast_output_vardesc->Flush();
    auto* cast_output_node = graph->CreateVarNode(cast_output_vardesc);
    IR_NODE_LINK_TO(cast_op_node, cast_output_node);
    (*cache)[var_node] = cast_output_node;
  }
  op_node->Op()->Rename(var_node->Name(), cache->at(var_node)->Name());
  IR_NODE_LINK_TO(var_node, cache->at(var_node)->inputs[0]);
  IR_NODE_LINK_TO(cache->at(var_node), op_node);

  IR_NODE_UNLINK(var_node, op_node);
}

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& black_list,
                        const std::unordered_set<std::string>& white_list) {
  if (white_list.count(op_type)) return true;
  return black_list.count(op_type) == 0 &&
         KernelSupportPrecision(op_type, backend, precision);
}

// The set of ops that support fp16 calculation and are considered
// numerically-dangerous, slower and whose effects may also be observed in
// downstream ops.
// ref to python/paddle/base/contrib/mixed_precision/fp16_lists.py
void AutoMixedPrecisionPass::SetDefaultBlacklist() const {
  black_list_.insert({
      "cast",
      // numerically-dangerous
      "exp",
      "square",
      "log",
      "mean",
      "sum",
      "softmax_with_cross_entropy",
      "sigmoid_cross_entropy_with_logits",
      "c_softmax_with_cross_entropy",
      "c_softmax_with_multi_label_cross_entropy",
      "cross_entropy",
      "cross_entropy2",
#ifndef PADDLE_WITH_XPU
      // slower than fp32
      "conv2d_transpose",
#endif
      // default fp32 can avoid return inf when the sum value large than 65504
      "reduce_sum",
  });
}

void AutoMixedPrecisionPass::Init(Graph* graph) const {
  if (Has("enable_gpu_mixed") && Get<bool>("enable_gpu_mixed")) {
    backend_ = phi::Backend::GPU;
  } else if (Has("enable_xpu_mixed") && Get<bool>("enable_xpu_mixed")) {
    backend_ = phi::Backend::XPU;
  } else if (Has("enable_custom_device_mixed") &&
             Get<bool>("enable_custom_device_mixed")) {
    // transform Backend::CUSTOM to actual backend.
// Here, we only consider one custom backend.
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto device_type = phi::DeviceManager::GetAllCustomDeviceTypes()[0];
    backend_ = static_cast<phi::Backend>(
        static_cast<size_t>(phi::Backend::NUM_BACKENDS) +
        phi::CustomRegisteredDeviceMap::Instance()
            .GetOrRegisterGlobalDeviceTypeId(device_type));
#else
    PADDLE_THROW(
        common::errors::Unavailable("Paddle is not compiled with CustomDevice. "
                                    "Cannot enable custom_device_mixed."));
#endif
  }

  if (Has("mixed_precision_mode")) {
    low_precision_ =
        static_cast<phi::DataType>(Get<int>("mixed_precision_mode"));
  }

  skip_pass_ = (backend_ == phi::Backend::UNDEFINED) ||
               (low_precision_ == phi::DataType::UNDEFINED);

  if (skip_pass_) return;

  black_list_ = Get<std::unordered_set<std::string>>("mixed_black_list");
  white_list_ = Get<std::unordered_set<std::string>>("mixed_white_list");
  SetDefaultBlacklist();
  VLOG(4) << "black_list has ";
  for (const auto& name : black_list_) {
    VLOG(4) << " - " << name;
  }
  VLOG(4) << "white_list has ";
  for (const auto& name : white_list_) {
    VLOG(4) << " - " << name;
  }

  if (Has("enable_low_precision_io")) {
    enable_low_precision_io_ = Get<bool>("enable_low_precision_io");
  }

  auto graph_size = graph->SubGraphsSize();
  VLOG(4) << "graph size: " << graph_size;
  subgraphes_.resize(graph_size);
  all_op_nodes_.resize(graph_size);

  for (size_t i = 0; i < graph_size; i++) {
    subgraphes_[i] = graph->GetSubGraph(i);
    all_op_nodes_[i] = TopologySortOperations(*subgraphes_[i]);
    VLOG(4) << "subgraph " << i << " has " << all_op_nodes_[i].size()
            << " op nodes";
    for (auto* var_node : subgraphes_[i]->Nodes()) {
      if (!var_node->IsVar()) continue;

      auto var_name = var_node->Var()->Name();
      if (real_vars_.count(var_name) == 0) {
        real_vars_[var_name] = std::vector<Node*>();
      }
      real_vars_[var_name].push_back(var_node);
    }
  }
}

void AutoMixedPrecisionPass::ApplyImpl(Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::PreconditionNotMet(
                              "During the auto_mixed_precision_pass, the graph "
                              "should not be nullptr."));
  PADDLE_ENFORCE_EQ(graph->IsMainGraph(),
                    true,
                    common::errors::PreconditionNotMet(
                        "During the auto_mixed_precision_pass, the graph "
                        "should be main graph."));

  FusePassBase::Init("auto_mixed_precision", graph);

  Init(graph);
  VLOG(4) << "Init done";

  if (skip_pass_) {
    VLOG(3) << "Skip auto_mixed_precision_pass.";
    return;
  }

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
  InsertCastOp();
  VLOG(4) << "InsertCastOp done";
  ProcessOpWithDtypeAttr();
  VLOG(4) << "ProcessOpWithDtypeAttr done";
  RestoreOpOriginType();

  VLOG(4) << "RestoreOpOriginType done";
  LOG(INFO) << "The number of ops run at low precision ["
            << op_run_low_precision_.size() << "/"
            << op_original_type_.size() + 2 << "]";
}

void AutoMixedPrecisionPass::SetOpUniqueType() const {
  int suffix = 0;
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
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

void AutoMixedPrecisionPass::RestoreOpOriginType() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      auto op_type = op_node->Op()->Type();
      op_node->Op()->SetType(GetOpOriginalType(op_type));
      op_node->Op()->Flush();
      VLOG(4) << "restore op type: " << op_type << " ---> "
              << op_node->Op()->Type();
    }
  }
}

inline std::string AutoMixedPrecisionPass::GetOpOriginalType(
    const std::string& op_type) const {
  if (op_original_type_.count(op_type)) {
    return op_original_type_.at(op_type);
  }
  return op_type;
}

void AutoMixedPrecisionPass::ProcessOpWithDtypeAttr() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      auto op_type = op_node->Op()->Type();

      if (op_node->Op()->HasAttr("in_dtype")) {
        auto* var_node = op_node->inputs[0];
        auto* real_var_node = real_vars_.count(var_node->Var()->Name())
                                  ? real_vars_.at(var_node->Var()->Name())[0]
                                  : var_node;
        if (IsFP16AndBFP16(real_var_node->Var()->GetDataType())) {
          op_node->Op()->SetAttr(
              "in_dtype",
              static_cast<int>(framework::TransToProtoVarType(low_precision_)));
          op_node->Op()->Flush();
          VLOG(4) << "process op with in_dtype attr: " << op_type << " ( "
                  << static_cast<int>(real_var_node->Var()->GetDataType())
                  << " --->" << static_cast<int>(low_precision_) << " )";
        }
      }

      if (op_run_low_precision_.count(op_type) == 0) continue;

      if (op_node->Op()->HasAttr("dtype")) {
        auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
        if (IsFP32(static_cast<VarType::Type>(dtype))) {
          op_node->Op()->SetAttr(
              "dtype",
              static_cast<int>(framework::TransToProtoVarType(low_precision_)));
          op_node->Op()->Flush();
          VLOG(4) << "process op with dtype attr: " << op_type << " ( " << dtype
                  << " --->" << static_cast<int>(low_precision_) << " )";
        }
      } else if (op_node->Op()->HasAttr("out_dtype")) {
        auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
        if (IsFP32(static_cast<VarType::Type>(out_dtype))) {
          op_node->Op()->SetAttr(
              "out_dtype",
              static_cast<int>(framework::TransToProtoVarType(low_precision_)));
          op_node->Op()->Flush();
          VLOG(4) << "process op with out_dtype attr: " << op_type << " ( "
                  << out_dtype << " --->" << static_cast<int>(low_precision_)
                  << " )";
        }
      }
    }
  }
}

void AutoMixedPrecisionPass::GetOpPrecision() const {
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      auto op_type = op_node->Op()->Type();
      bool support_low_precision = true;
      if (GetOpOriginalType(op_type) == "feed" ||
          GetOpOriginalType(op_type) == "fetch") {
        support_low_precision = enable_low_precision_io_;
      } else if (GetOpOriginalType(op_type) == "tensorrt_engine") {
        auto enable_fp16 = op_node->Op()->GetAttrIfExists<bool>("enable_fp16");
        auto enable_int8 = op_node->Op()->GetAttrIfExists<bool>("enable_int8");
        auto low_precision_io =
            op_node->Op()->GetAttrIfExists<bool>("enable_low_precision_io");
        support_low_precision = enable_fp16 && !enable_int8 && low_precision_io;
      } else {
        support_low_precision = OpSupportPrecision(GetOpOriginalType(op_type),
                                                   backend_,
                                                   low_precision_,
                                                   black_list_,
                                                   white_list_);

        std::unordered_set<std::string> check_dtype_op_blacklist(
            {"arg_max", "arg_min"});
        if (op_node->Op()->HasAttr("dtype") &&
            !check_dtype_op_blacklist.count(GetOpOriginalType(op_type))) {
          auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
          support_low_precision = support_low_precision &&
                                  IsFP32(static_cast<VarType::Type>(dtype));
        } else if (op_node->Op()->HasAttr("out_dtype")) {
          auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
          support_low_precision =
              support_low_precision &&
              (IsFP32(static_cast<VarType::Type>(out_dtype)) ||
               out_dtype == -1);
        }

        // If scale op's "scale" and "bias" attr value exceed the range of
        // fp16 and bf16, it cannot run at low precision.
        if (GetOpOriginalType(op_node->Op()->Type()) == "scale") {
          auto scale = op_node->Op()->GetAttrIfExists<float>("scale");
          auto bias = op_node->Op()->GetAttrIfExists<float>("bias");
          if (low_precision_ == phi::DataType::FLOAT16) {
            support_low_precision =
                support_low_precision &&
                phi::dtype::isfinite(static_cast<phi::dtype::float16>(scale)) &&
                phi::dtype::isfinite(static_cast<phi::dtype::float16>(bias));
          } else if (low_precision_ == phi::DataType::BFLOAT16) {
            support_low_precision =
                support_low_precision &&
                phi::dtype::isfinite(
                    static_cast<phi::dtype::bfloat16>(scale)) &&
                phi::dtype::isfinite(static_cast<phi::dtype::bfloat16>(bias));
          }
        }

        // op's input var and output var only support
        // dense/sparse_coo/sparse_csr tensor.
        for (auto* in_var_node : op_node->inputs) {
          PADDLE_ENFORCE_EQ(
              in_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "in_var_node->IsVar() is False, which means that "
                  "inputs may be not a valid variable."));
          auto* real_in_var_node = real_vars_.at(in_var_node->Var()->Name())[0];
          if (real_in_var_node->Var()->Persistable()) continue;

          support_low_precision =
              support_low_precision &&
              (real_in_var_node->Var()->GetType() == VarType::DENSE_TENSOR ||
               real_in_var_node->Var()->GetType() == VarType::SPARSE_COO ||
               real_in_var_node->Var()->GetType() == VarType::SPARSE_CSR);
        }
        for (auto* out_var_node : op_node->outputs) {
          PADDLE_ENFORCE_EQ(
              out_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "out_var_node->IsVar() is False, which means that "
                  "outputs may be not a valid variable."));
          auto* real_out_var_node =
              real_vars_.at(out_var_node->Var()->Name())[0];
          if (real_out_var_node->Var()->Persistable()) continue;

          support_low_precision =
              support_low_precision &&
              (real_out_var_node->Var()->GetType() == VarType::DENSE_TENSOR ||
               real_out_var_node->Var()->GetType() == VarType::SPARSE_COO ||
               real_out_var_node->Var()->GetType() == VarType::SPARSE_CSR);
        }
      }

      if (support_low_precision) {
        op_run_low_precision_.insert(op_type);
        VLOG(4) << "support precision: " << op_type << " run at low precision";
      } else {
        VLOG(4) << "support precision: " << op_type
                << " not run at low precision";
      }
    }
  }
}

void AutoMixedPrecisionPass::UpdateOpPrecision() const {
  std::unordered_set<std::string> vars_should_not_low_precision;

  // var -> the var's all input op
  std::unordered_map<std::string, std::vector<Node*>> var_input_ops;

  auto GetVarInputOps = [&] {
    for (const auto& nodes : all_op_nodes_) {
      for (auto* op_node : nodes) {
        auto op_type = op_node->Op()->Type();

        if (GetOpOriginalType(op_type) == "fetch") continue;
        if (op_node->Op()->HasAttr("sub_block") &&
            GetOpOriginalType(op_type) != "tensorrt_engine")
          continue;

        for (auto* var_node : op_node->outputs) {
          PADDLE_ENFORCE_EQ(var_node->IsVar(),
                            true,
                            common::errors::InvalidArgument(
                                "var_node->IsVar() is False, which means that "
                                "outputs may be not a valid variable."));
          if (var_node->Var()->Persistable()) continue;
          if (!VarNodeHasDtype(var_node)) continue;

          var_input_ops[var_node->Var()->Name()].push_back(op_node);
          VLOG(4) << "var input ops: " << var_node->Var()->Name()
                  << " is output of " << op_type;
          if (IsFP64(var_node->Var()->GetDataType())) {
            // All op involving float64 precision must not run in low precision
            // mode.
            vars_should_not_low_precision.insert(var_node->Var()->Name());
          }
        }

        // the select_input op's input var should not convert to low
        // precision. when op's output var is select_input op's input var, the
        // op should not run at low precision.
        if (GetOpOriginalType(op_node->Op()->Type()) == "select_input") {
          for (auto* in_var_node : op_node->inputs) {
            PADDLE_ENFORCE_EQ(
                in_var_node->IsVar(),
                true,
                common::errors::InvalidArgument(
                    "in_var_node->IsVar() is False, which means that "
                    "inputs may be not a valid variable."));
            if (in_var_node->Var()->Persistable()) continue;
            if (!VarNodeHasDtype(in_var_node)) continue;

            vars_should_not_low_precision.insert(in_var_node->Var()->Name());
          }
        }
        // when op_1 only support cpu kernel. if op_2's input var is op_1's
        // output var, then op_2 should not run at low precision.
        if (GetOpOriginalType(op_type) != "feed" &&
            GetOpOriginalType(op_type) != "tensorrt_engine" &&
            white_list_.count(GetOpOriginalType(op_type)) == 0 &&
            !KernelSupportPrecision(
                GetOpOriginalType(op_type), backend_, phi::DataType::FLOAT32)) {
          for (auto* out_var_node : op_node->outputs) {
            PADDLE_ENFORCE_EQ(
                out_var_node->IsVar(),
                true,
                common::errors::InvalidArgument(
                    "out_var_node->IsVar() is False, which means that "
                    "outputs may be not a valid variable."));
            if (out_var_node->Var()->Persistable()) continue;
            if (!VarNodeHasDtype(out_var_node)) continue;

            vars_should_not_low_precision.insert(out_var_node->Var()->Name());
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
        if (op_run_low_precision_.count(op_node->Op()->Type()) == 0) continue;

        for (auto* in_var_node : op_node->inputs) {
          PADDLE_ENFORCE_EQ(
              in_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "in_var_node->IsVar() is False, which means that "
                  "inputs may be not a valid variable."));
          if (!VarNodeHasDtype(in_var_node)) continue;

          auto* real_in_var_node = real_vars_.at(in_var_node->Var()->Name())[0];
          if (real_in_var_node->Var()->Persistable()) continue;

          if (vars_should_not_low_precision.count(
                  real_in_var_node->Var()->Name())) {
            op_run_low_precision_.erase(op_node->Op()->Type());
            precision_updated = true;
            VLOG(4) << op_node->Op()->Type()
                    << " should not run at low precision.";
            break;
          }
        }

        if (op_run_low_precision_.count(op_node->Op()->Type()) == 0) continue;

        for (auto* out_var_node : op_node->outputs) {
          PADDLE_ENFORCE_EQ(
              out_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "out_var_node->IsVar() is False, which means that "
                  "outputs may be not a valid variable."));
          if (!VarNodeHasDtype(out_var_node)) continue;

          auto* real_out_var_node =
              real_vars_.at(out_var_node->Var()->Name())[0];
          if (real_out_var_node->Var()->Persistable()) continue;

          bool not_run_low_precision = false;
          const auto& input_op_nodes =
              var_input_ops[real_out_var_node->Var()->Name()];
          if (vars_should_not_low_precision.count(
                  real_out_var_node->Var()->Name())) {
            not_run_low_precision = true;
          } else {
            for (auto* node : input_op_nodes) {
              if (op_run_low_precision_.count(node->Op()->Type()) == 0) {
                not_run_low_precision = true;
                break;
              }
            }
          }
          if (not_run_low_precision) {
            op_run_low_precision_.erase(op_node->Op()->Type());
            precision_updated = true;
            VLOG(4) << op_node->Op()->Type()
                    << " should not run at low precision.";
            break;
          }
        }
      }
    }
  } while (precision_updated);
}

// special ops, its weights should not be low precision.
bool AutoMixedPrecisionPass::InputVarsNotConvert(
    Node* op_node, const std::string& var_name) const {
  auto* op_desc = op_node->Op();
  if (GetOpOriginalType(op_desc->Type()) == "tensorrt_engine") {
    auto vecs = op_desc->Input("Xs");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  } else if (GetOpOriginalType(op_desc->Type()) == "batch_norm") {
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
  } else if (GetOpOriginalType(op_desc->Type()) == "sparse_batch_norm") {
    auto vecs = op_desc->Input("bias");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("mean");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("scale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("variance");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  } else if (GetOpOriginalType(op_desc->Type()) == "instance_norm" ||
             GetOpOriginalType(op_desc->Type()) == "layer_norm") {
    auto vecs = op_desc->Input("Bias");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("Scale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  } else if (GetOpOriginalType(op_desc->Type()) == "fused_multi_transformer") {
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
  } else if (GetOpOriginalType(op_desc->Type()) ==
             "fused_bias_dropout_residual_layer_norm") {
    auto vecs = op_desc->Input("LnScale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("LnBias");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  } else if (GetOpOriginalType(op_desc->Type()) == "quantize_linear" ||
             GetOpOriginalType(op_desc->Type()) == "dequantize_linear") {
    auto vecs = op_desc->Input("Scale");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Input("ZeroPoint");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  }

  return false;
}

bool AutoMixedPrecisionPass::OutputVarsNotConvert(
    Node* op_node, const std::string& var_name) const {
  auto* op_desc = op_node->Op();
  // batch_norm's input and output (variance and mean) are the same.
  if (GetOpOriginalType(op_desc->Type()) == "batch_norm") {
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
  } else if (GetOpOriginalType(op_desc->Type()) == "sparse_batch_norm") {
    auto vecs = op_desc->Output("mean_out");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("variance_out");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("saved_mean");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("saved_variance");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("reserve_space");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  } else if (GetOpOriginalType(op_desc->Type()) == "layer_norm" ||
             GetOpOriginalType(op_desc->Type()) == "group_norm") {
    auto vecs = op_desc->Output("Mean");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("Variance");
    if (std::find(vecs.begin(), vecs.end(), var_name) != vecs.end()) {
      return true;
    }
  }

  return false;
}

void AutoMixedPrecisionPass::SetVarPrecision() const {
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(scope,
                          common::errors::PreconditionNotMet(
                              "During the auto_mixed_precision_pass, the scope "
                              "should not be null."));
  for (const auto& nodes : all_op_nodes_) {
    for (auto* op_node : nodes) {
      if (op_run_low_precision_.count(op_node->Op()->Type()) == 0) {
        continue;
      }

      if (GetOpOriginalType(op_node->Op()->Type()) != "feed") {
        for (auto* in_var_node : op_node->inputs) {
          PADDLE_ENFORCE_EQ(
              in_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "in_var_node->IsVar() is False, which means that "
                  "inputs may be not a valid variable."));

          auto* real_in_var_node = real_vars_.at(in_var_node->Var()->Name())[0];
          auto in_var_name = real_in_var_node->Var()->Name();

          if (!IsFP32(real_in_var_node->Var()->GetDataType())) continue;
          if (!VarNodeHasDtype(real_in_var_node)) continue;
          if (InputVarsNotConvert(op_node, in_var_name)) continue;
          // Judge the real tensor is same to variable, Paddle-Slim weight use
          // fp32 variable to save int8 tensor.
          if (real_in_var_node->Var()->Persistable()) {
            auto* tensor = scope->Var(real_in_var_node->Name())
                               ->GetMutable<phi::DenseTensor>();
            if (framework::TransToProtoVarType(tensor->type()) !=
                real_in_var_node->Var()->GetDataType()) {
              VLOG(3) << "[AutoMixedPrecisionPass] variable "
                      << real_in_var_node->Name() << "'s proto data type "
                      << real_in_var_node->Var()->GetDataType()
                      << " is different from real dense tensor "
                      << framework::TransToProtoVarType(tensor->type());
              continue;
            }
          }
          if (real_in_var_node->Var()->Persistable()) {
            for (auto* in_var_node :
                 real_vars_.at(in_var_node->Var()->Name())) {
              in_var_node->Var()->SetDataType(
                  framework::TransToProtoVarType(low_precision_));
            }

            VLOG(4) << real_in_var_node->Var()->Name()
                    << "'s data type was set to low precision";
            vars_convert_to_low_precision_.insert(in_var_name);
          }
        }
      }

      if (GetOpOriginalType(op_node->Op()->Type()) != "fetch") {
        for (auto* out_var_node : op_node->outputs) {
          PADDLE_ENFORCE_EQ(
              out_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "out_var_node->IsVar() is False, which means that "
                  "outputs may be not a valid variable."));

          auto* real_out_var_node =
              real_vars_.at(out_var_node->Var()->Name())[0];
          auto out_var_name = real_out_var_node->Var()->Name();

          if (!IsFP32(real_out_var_node->Var()->GetDataType())) continue;
          if (!VarNodeHasDtype(real_out_var_node)) continue;
          if (OutputVarsNotConvert(op_node, out_var_name)) continue;

          for (auto* out_var_node :
               real_vars_.at(out_var_node->Var()->Name())) {
            out_var_node->Var()->SetDataType(
                framework::TransToProtoVarType(low_precision_));
          }
          VLOG(4) << real_out_var_node->Var()->Name()
                  << "'s data type was set to low precision";
          if (real_out_var_node->Var()->Persistable()) {
            vars_convert_to_low_precision_.insert(out_var_name);
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
      if (vars_convert_to_low_precision_.count(var_name)) {
        var_node->Var()->SetDataType(
            framework::TransToProtoVarType(low_precision_));
        VLOG(4) << var_node->Var()->Name()
                << "'s data type was set to low precision";
      }
    }
  }
}

void AutoMixedPrecisionPass::ConvertWeightsData() const {
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(scope,
                          common::errors::PreconditionNotMet(
                              "During the auto_mixed_precision_pass, the scope "
                              "should not be null."));

  auto var_names = scope->LocalVarNames();
  for (const auto& var_name : var_names) {
    if (vars_convert_to_low_precision_.count(var_name)) {
      VLOG(4) << var_name << "'s data type was convert to low precision";

      auto* var = scope->FindLocalVar(var_name);
      PADDLE_ENFORCE_EQ(
          var->IsType<phi::DenseTensor>(),
          true,
          common::errors::InvalidArgument(
              "var->IsType<phi::DenseTensor>() is False, which means the "
              "variable has invalid type instead of <phi::DenseTensor>."));

      auto* origin_tensor = var->GetMutable<phi::DenseTensor>();

      phi::DenseTensor low_precision_tensor;
      low_precision_tensor.Resize(origin_tensor->dims());
      low_precision_tensor.set_type(low_precision_);

      if (low_precision_ == phi::DataType::FLOAT16) {
        auto* low_precision_data =
            low_precision_tensor.mutable_data<phi::dtype::float16>(
                phi::CPUPlace{});
        for (int64_t i = 0; i < origin_tensor->numel(); i++) {
          if (origin_tensor->dtype() == phi::DataType::FLOAT64) {
            auto* origin_data = origin_tensor->data<double>();
            low_precision_data[i] =
                static_cast<phi::dtype::float16>(origin_data[i]);
          } else if (origin_tensor->dtype() == phi::DataType::FLOAT32) {
            auto* origin_data = origin_tensor->data<float>();
            low_precision_data[i] =
                static_cast<phi::dtype::float16>(origin_data[i]);
          }
        }
      } else if (low_precision_ == phi::DataType::BFLOAT16) {
        auto* low_precision_data =
            low_precision_tensor.mutable_data<phi::dtype::bfloat16>(
                phi::CPUPlace{});
        for (int64_t i = 0; i < origin_tensor->numel(); i++) {
          if (origin_tensor->dtype() == phi::DataType::FLOAT64) {
            auto* origin_data = origin_tensor->data<double>();
            low_precision_data[i] =
                static_cast<phi::dtype::bfloat16>(origin_data[i]);
          } else if (origin_tensor->dtype() == phi::DataType::FLOAT32) {
            auto* origin_data = origin_tensor->data<float>();
            low_precision_data[i] =
                static_cast<phi::dtype::bfloat16>(origin_data[i]);
          }
        }
      }
      origin_tensor->clear();
      paddle::framework::TensorCopySync(
          low_precision_tensor, phi::CPUPlace{}, origin_tensor);
    }
  }
}

void AutoMixedPrecisionPass::InsertCastOp() const {
  int suffix = 0;
  std::unordered_map<Node*, Node*> cache;

  for (size_t i = 0; i < all_op_nodes_.size(); i++) {
    auto* block_desc = all_op_nodes_[i][0]->Op()->Block();
    PADDLE_ENFORCE_NOT_NULL(
        block_desc,
        common::errors::PreconditionNotMet(
            "During the auto_mixed_precision_pass, the block description "
            "should not be null."));
    for (auto* op_node : all_op_nodes_[i]) {
      auto op_type = op_node->Op()->Type();

      if (GetOpOriginalType(op_type) == "feed") continue;
      if (op_node->Op()->HasAttr("sub_block") &&
          GetOpOriginalType(op_type) != "tensorrt_engine")
        continue;

      VLOG(4) << "process op: " << op_type
              << " run low precision: " << op_run_low_precision_.count(op_type);

      auto inputs = op_node->inputs;
      for (auto* in_var_node : inputs) {
        if (!in_var_node->IsVar()) continue;
        if (!VarNodeHasDtype(in_var_node)) continue;
        if (in_var_node->Var()->Persistable()) continue;

        auto* real_in_var_node = real_vars_.at(in_var_node->Var()->Name())[0];

        auto in_var_type = real_in_var_node->Var()->GetDataType();

        VLOG(4) << "process var: " << real_in_var_node->Var()->Name()
                << " with type " << in_var_type;

        if (IsFP32(in_var_type) && op_run_low_precision_.count(op_type)) {
          auto to_type = framework::TransToProtoVarType(low_precision_);
          auto* prev_op =
              in_var_node->inputs.empty() ? nullptr : in_var_node->inputs[0];
          if (prev_op && GetOpOriginalType(prev_op->Op()->Type()) == "cast") {
            in_var_node->Var()->SetDataType(to_type);
            prev_op->Op()->SetAttr("out_dtype", static_cast<int>(to_type));
            prev_op->Op()->Flush();
          } else {
            DoInsertCastOp(subgraphes_[i],
                           in_var_node,
                           op_node,
                           in_var_type,
                           to_type,
                           block_desc,
                           &suffix,
                           &cache);
          }
        } else if (IsFP16AndBFP16(in_var_type) &&
                   op_run_low_precision_.count(op_type) == 0) {
          auto to_type = VarType::FP32;
          auto* prev_op =
              in_var_node->inputs.empty() ? nullptr : in_var_node->inputs[0];
          if (prev_op && GetOpOriginalType(prev_op->Op()->Type()) == "cast") {
            in_var_node->Var()->SetDataType(to_type);
            prev_op->Op()->SetAttr("out_dtype", static_cast<int>(to_type));
            prev_op->Op()->Flush();
          } else {
            DoInsertCastOp(subgraphes_[i],
                           in_var_node,
                           op_node,
                           in_var_type,
                           to_type,
                           block_desc,
                           &suffix,
                           &cache);
          }
        }
      }

      // Special op.
      // fused_multi_transformer's input(CacheKV) and output(CacheKVOut) vars
      // have same name.
      if (GetOpOriginalType(op_type) == "fused_multi_transformer") {
        auto cache_kv_inputs = op_node->Op()->Input("CacheKV");
        auto cache_kv_outputs = op_node->Op()->Output("CacheKVOut");
        PADDLE_ENFORCE_EQ(
            cache_kv_inputs.size(),
            cache_kv_outputs.size(),
            common::errors::InvalidArgument(
                "Cache inputs should be the same size with cache outputs, but "
                "received %d as inputs and %d as outputs.",
                cache_kv_inputs.size(),
                cache_kv_outputs.size()));
        for (size_t i = 0; i < cache_kv_inputs.size(); ++i) {
          op_node->Op()->RenameOutput(cache_kv_outputs[i], cache_kv_inputs[i]);
        }
      }
    }
  }
  VLOG(4) << "insert number of cast op: " << cache.size();
}

}  // namespace paddle::framework::ir

REGISTER_PASS(auto_mixed_precision_pass,
              paddle::framework::ir::AutoMixedPrecisionPass);
