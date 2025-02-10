
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

#include "paddle/fluid/inference/analysis/ir_passes/tensorrt_subgraph_pass.h"
#include <fcntl.h>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"
#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::inference::analysis {
namespace {

// if in mixed model precision, we should make all tensorrt_engine's output
// floats dtype to float32 dtype.
void OutputProcess(framework::ir::Graph *graph,
                   const std::unordered_set<framework::ir::Node *> &trt_outputs,
                   phi::Backend backend,
                   phi::DataType precision,
                   const std::unordered_set<std::string> &blacklist,
                   const std::unordered_set<std::string> &whitelist) {
  framework::BlockDesc *block_desc{nullptr};
  int suffix = 0;
  std::unordered_map<framework::ir::Node *, framework::ir::Node *>
      var_to_cast_op_map;

  framework::proto::VarType::Type to_type;
  if (precision == phi::DataType::FLOAT16) {
    to_type = framework::proto::VarType::FP16;
  } else if (precision == phi::DataType::BFLOAT16) {
    to_type = framework::proto::VarType::BF16;
  } else if (precision == phi::DataType::FLOAT32) {
    return;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only support "
        "fp16 and bf16.",
        static_cast<int>(precision)));
  }

  for (auto *op_node : framework::ir::TopologySortOperations(*graph)) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed") block_desc = op_node->Op()->Block();
    if (op_type != "tensorrt_engine") continue;
    for (auto *var_node : op_node->outputs) {
      if (!trt_outputs.count(var_node)) continue;
      if (!var_node->Var()->Persistable() &&
          IsFloatVar(var_node->Var()->GetDataType()) &&
          var_node->Var()->GetDataType() != framework::proto::VarType::FP32) {
        for (auto *next_op : var_node->outputs) {
          // if next_op support mixed_precision, we need to add cast op.
          if (OpSupportPrecision(
                  phi::TransToPhiKernelName(next_op->Op()->Type()),
                  backend,
                  precision,
                  blacklist,
                  whitelist)) {
            InsertCastOp(graph,
                         var_node,
                         next_op,
                         framework::proto::VarType::FP32,
                         to_type,
                         block_desc,
                         &suffix,
                         &var_to_cast_op_map);
            var_node->Var()->SetDataType(framework::proto::VarType::FP32);
          }
        }
      }
    }
  }
}

// Determine whether the whole graph offload to tensorrt. If so we can try to
// enable optimization such as cudaGraph.
bool AllNodesLowerToTrtPostProcess(framework::ir::Graph *graph) {
  std::unordered_set<std::string> trt_nodes_set{
      "feed", "fetch", "tensorrt_engine"};
  bool all_nodes_offload_to_trt = true;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp()) {
      if (!trt_nodes_set.count(node->Op()->Type())) {
        all_nodes_offload_to_trt = false;
        break;
      }
    }
  }
  return all_nodes_offload_to_trt;
}
}  // namespace

using framework::ir::Node;

void analysis::TensorRtSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("tensorrt_subgraph_pass", graph);

  VLOG(3) << "Running tensorrt_subgraph_pass.";
  if (graph->IsMainGraph()) {
    VLOG(3)
        << "The ID of block running tensorrt_subgraph_pass is: 0(main_graph)";
  } else {
    VLOG(3) << "The ID of block running tensorrt_subgraph_pass is: "
            << graph->GetBlockId();
  }

  auto model_precision =
      static_cast<phi::DataType>(Get<int>("model_precision"));
  if (model_precision == phi::DataType::BFLOAT16) {
    LOG(WARNING)
        << "Paddle-TRT not support bf16 mixed precision, just fallback.";
    return;
  }

  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  bool use_cuda_graph = Get<bool>("use_cuda_graph");
  bool no_calib_int8 = enable_int8 && !(use_calib_mode);
  auto trt_disabled_ops = Get<std::vector<std::string>>("trt_disabled_ops");
  auto with_dynamic_shape = Get<bool>("with_dynamic_shape");
  auto use_explicit_quantization = Get<bool>("use_explicit_quantization");
  auto forbid_dynamic_op = Get<bool>("forbid_dynamic_op");
  auto teller = [&](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    if (find(trt_disabled_ops.begin(),
             trt_disabled_ops.end(),
             node->Op()->Type()) != trt_disabled_ops.end()) {
      VLOG(3) << node->Op()->Type().c_str()

              << " is disabled by config in TensorRT";
      return false;
    }
    for (const auto &out_var : node->Op()->OutputNames()) {
      for (const auto &var_name : node->Op()->Output(out_var)) {
        if (find(trt_disabled_ops.begin(), trt_disabled_ops.end(), var_name) !=
            trt_disabled_ops.end()) {
          VLOG(3) << node->Op()->Type().c_str()
                  << " is disabled by config in TensorRT";
          return false;
        }
      }
    }
    bool is_ok = tensorrt::OpTeller::Global().Tell(node,
                                                   no_calib_int8,
                                                   with_dynamic_shape,
                                                   forbid_dynamic_op,
                                                   use_explicit_quantization);
    if (!is_ok)
      VLOG(3) << node->Op()->Type().c_str() << " op is not in TensorRT";
    return is_ok;
  };

  framework::ir::SubGraphFuser fuser(
      graph,
      teller,
      Get<int>("min_subgraph_size") /*min subgraph size*/,
      Get<std::vector<std::string>>("trt_exclude_var_names"),
      "tensorrt_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in trt, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;
  std::vector<std::string> engine_names;
  for (auto *node : graph->Nodes()) {
    // load optimized model may update shape_range_info_path
    auto shape_range_info_path = Get<std::string>("trt_shape_range_info_path");
    if (node->IsOp() && node->Op()->Type() == "tensorrt_engine" &&
        !shape_range_info_path.empty()) {
      node->Op()->SetAttr("shape_range_info_path", shape_range_info_path);
    }
    if (node->IsOp() && !framework::ir::Agent(node).subgraph()->empty()) {
      engine_names.push_back(CreateTensorRTOp(
          node, graph, graph_param_names, &repetitive_params, use_cuda_graph));
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && framework::ir::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));

  bool all_nodes_offload_to_trt = AllNodesLowerToTrtPostProcess(graph);
  if (all_nodes_offload_to_trt) {
    LOG(INFO) << "The entire graph is offloaded to TensorRT.";
  }
  if (use_cuda_graph && !all_nodes_offload_to_trt) {
    LOG_FIRST_N(WARNING, 1)
        << "You have enabled CudaGraph, but not the entire graph offload to "
           "trt, now return to normal mode.";
    use_cuda_graph = false;
  }
  if (use_cuda_graph && all_nodes_offload_to_trt) {
    for (auto &name : engine_names) {
      PADDLE_ENFORCE_EQ(
          paddle::inference::Singleton<
              inference::tensorrt::TRTEngineManager>::Global()
              .Has(name),
          true,
          common::errors::PreconditionNotMet(
              "TRTEngineManager should has engine %s, but not found.", name));
      paddle::inference::Singleton<
          inference::tensorrt::TRTEngineManager>::Global()
          .Get(name)
          ->SetAllNodesLowerToTrt(use_cuda_graph);
    }
  }

  // some ops are only implemented in paddle-trt,
  // but not in paddle ,we should revert it.
  for (auto *op_node : framework::ir::TopologyVariantSort(
           *graph, static_cast<framework::ir::SortKind>(0))) {
    if (op_node->Op()->Type() == "matrix_multiply") {
      auto origin_type =
          op_node->Op()->GetAttrIfExists<std::string>("original_type");
      LOG(WARNING) << "matrix_multiply can't enter into paddle-trt,"
                   << "we will revert to " << origin_type;
      op_node->Op()->SetType(origin_type);
      op_node->RenameOp(origin_type);
    }
  }
}

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &predictor_id,
                              const std::string &max_batch_size,
                              const std::string &precision,
                              bool use_cuda_graph,
                              const bool for_calibration) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
    engine_hash_key += "#";
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
    engine_hash_key += "#";
  }
  engine_hash_key += predictor_id;
  if (!for_calibration) {
    engine_hash_key += "#";
    engine_hash_key += max_batch_size;
  }
  engine_hash_key += "#";
  engine_hash_key += precision;

  engine_hash_key += "#";
  engine_hash_key += std::to_string(use_cuda_graph);

  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  VLOG(2) << "TRT engine hash key: " << engine_hash_key;
  VLOG(2) << "TRT engine key: " << engine_key;
  return engine_key;
}

std::string TensorRtSubgraphPass::CreateTensorRTOp(
    framework::ir::Node *node,
    framework::ir::Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params,
    bool use_cuda_graph) const {
  auto *op_desc = node->Op();
  auto &subgraph = *framework::ir::Agent(node).subgraph();
  PADDLE_ENFORCE_EQ(
      subgraph.empty(),
      false,
      common::errors::PreconditionNotMet("The subgraph should not be empty."));

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  // Add new block for TensorRTEngineOP
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  // A fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  LOG(INFO) << "---  detect a sub-graph with " << subgraph.size() << " nodes";
  for (auto node : subgraph) {
    if (node->NodeType() == Node::Type::kOperation) {
      VLOG(5) << "trt subgraph has op: " << (node->Op()->Type());
    }
  }

  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the engine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> parameters;
  // if we delete fluid copy of parameters shared by more than 1 ops, there will
  // be problem, so we filter them out.
  std::vector<std::string> params_not_shared;

  auto *scope = param_scope();
  // The node->inputs contains input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(
        RenameVarBeUnique(x->Name(), std::to_string(x->id())));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      parameters.push_back(x->Name());
    }
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0 &&
        x->outputs.size() <= 1) {
      params_not_shared.push_back(x->Name());
    }
    // When TRT Engine's input is INT64 or FP64, we need do some extra work.
    // So we reserved a name for later use when casting INT64 -> INT32 or
    // FP64->FP32. We must check whether scope has had the same name var!
    if (x->Var()->GetDataType() == framework::proto::VarType::INT64) {
      LOG(WARNING)
          << "tensorrt_subgraph's input named " << x->Name()
          << " having int64 dtype in pdmodel description, we will cast them to "
             "int32 dtype to feed them into paddle-trt.";
    } else if (x->Var()->GetDataType() == framework::proto::VarType::FP64) {
      LOG(WARNING) << "tensorrt_subgraph's input named " << x->Name()
                   << " having float64 dtype in pdmodel description, we will "
                      "cast them to "
                      "float32 dtype to feed them into paddle-trt.";
    }
  }

  // var may have the same name but not have the same id.
  // e.g., var(batch_norm2d_0.w_1) may have id: 10, 13, 25.... in a graph.
  // so we must find all the var_name+id.
  // https://github.com/PaddlePaddle/Paddle/pull/53184
  for (auto *n : graph->Nodes()) {
    if (n->IsVar() &&
        find(graph_params.begin(), graph_params.end(), n->Name()) !=
            graph_params.end()) {
      input_names_with_id.insert(
          RenameVarBeUnique(n->Name(), std::to_string(n->id())));
    }
  }

  auto model_precision =
      static_cast<phi::DataType>(Get<int>("model_precision"));
  auto mixed_black_list =
      Get<std::unordered_set<std::string>>("mixed_black_list");
  auto mixed_white_list =
      Get<std::unordered_set<std::string>>("mixed_white_list");

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  std::map<std::string, int> origin_name_output_rank;
  std::unordered_set<Node *> trt_outputs;
  // record the origin output data type
  std::vector<int> origin_outputs_dtype;
  std::map<std::string, int> map_origin_outputs_dtype;

  // rename output names in trt_ops_run_float
  auto trt_ops_run_float =
      Get<std::unordered_set<std::string>>("trt_ops_run_float");
  for (auto node : subgraph) {
    if (node->NodeType() == Node::Type::kOperation) {
      for (auto *x : node->outputs) {
        if (std::count(parameters.begin(), parameters.end(), x->Name()) > 0)
          continue;
        if (trt_ops_run_float.count(x->Name()) > 0) {
          trt_ops_run_float.erase(x->Name());
          trt_ops_run_float.insert(
              RenameVarBeUnique(x->Name(), std::to_string(x->id())));
        }
      }
    }
  }

  // Mark TensorRT output nodes as trt outputs
  auto mark_output = Get<bool>("mark_output");
  auto output_tensor_name =
      Get<std::vector<std::string>>("output_tensor_names");

  if (mark_output) {
    VLOG(1) << "begin to mark output ...";
    for (auto node : subgraph) {
      if (node->NodeType() == Node::Type::kOperation) {
        for (auto *x : node->outputs) {
          if (std::count(parameters.begin(), parameters.end(), x->Name()) > 0)
            continue;
          if ((std::count(output_tensor_name.begin(),
                          output_tensor_name.end(),
                          x->Name()) > 0) &&
              !x->outputs.empty()) {
            VLOG(3) << "output " << x->Name() << " has been marked";
            output_names.insert(x->Name());
            output_names_with_id.insert(
                RenameVarBeUnique(x->Name(), std::to_string(x->id())));
            origin_name_output_rank[x->Name()] = x->Var()->GetShape().size();
            trt_outputs.insert(x);
            map_origin_outputs_dtype[x->Name()] =
                static_cast<int>(x->Var()->GetDataType());
          }
        }
      }
    }
  }

  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(
        RenameVarBeUnique(x->Name(), std::to_string(x->id())));
    origin_name_output_rank[x->Name()] = x->Var()->GetShape().size();
    trt_outputs.insert(x);
    map_origin_outputs_dtype[x->Name()] =
        static_cast<int>(x->Var()->GetDataType());
  }

  OutputProcess(graph,
                trt_outputs,
                phi::Backend::GPU,
                model_precision,
                mixed_black_list,
                mixed_white_list);

  std::unordered_map<std::string, std::string> output_name_map;
  std::unordered_map<std::string, framework::ir::Node *> graph_var_map;

  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      graph_var_map[node->Name()] = node;
    }
  }
  auto precision_mode =
      static_cast<phi::DataType>(Get<int>("trt_precision_mode"));
  auto trt_params_run_fp16 =
      Get<std::vector<std::string>>("trt_parameter_run_fp16");
  auto trt_params_run_int8 =
      Get<std::vector<std::string>>("trt_parameter_run_int8");
  auto trt_params_run_bfp16 =
      Get<std::vector<std::string>>("trt_parameter_run_bfp16");

  for (const auto &para : parameters) {
    if (std::find(trt_params_run_fp16.begin(),
                  trt_params_run_fp16.end(),
                  para) != trt_params_run_fp16.end()) {
      precision_mode = phi::DataType::FLOAT16;
      break;
    }
  }

  bool enable_fp16 = false;
  if (precision_mode == phi::DataType::FLOAT16) enable_fp16 = true;
  auto enable_int8 = Get<bool>("enable_int8");

  for (const auto &para : parameters) {
    if (std::find(trt_params_run_int8.begin(),
                  trt_params_run_int8.end(),
                  para) != trt_params_run_int8.end()) {
      enable_int8 = true;
      precision_mode = phi::DataType::INT8;
      break;
    }
  }

  for (const auto &para : parameters) {
    if (std::find(trt_params_run_bfp16.begin(),
                  trt_params_run_bfp16.end(),
                  para) != trt_params_run_bfp16.end()) {
      precision_mode = phi::DataType::BFLOAT16;
      break;
    }
  }
  bool enable_bfp16 = false;
  if (precision_mode == phi::DataType::BFLOAT16) enable_bfp16 = true;

  auto use_calib_mode = Get<bool>("use_calib_mode");
  auto &subgraph_nodes = *framework::ir::Agent(node).subgraph();
  auto min_input_shape =
      Get<std::map<std::string, std::vector<int>>>("min_input_shape");
  auto max_input_shape =
      Get<std::map<std::string, std::vector<int>>>("max_input_shape");
  auto optim_input_shape =
      Get<std::map<std::string, std::vector<int>>>("optim_input_shape");

  auto min_shape_tensor =
      Get<std::map<std::string, std::vector<int>>>("min_shape_tensor");
  auto max_shape_tensor =
      Get<std::map<std::string, std::vector<int>>>("max_shape_tensor");
  auto optim_shape_tensor =
      Get<std::map<std::string, std::vector<int>>>("optim_shape_tensor");

  auto allow_build_at_runtime = Get<bool>("trt_allow_build_at_runtime");
  auto with_dynamic_shape = Get<bool>("with_dynamic_shape");
  auto shape_range_info_path = Get<std::string>("trt_shape_range_info_path");
  auto trt_tuned_dynamic_shape = Get<bool>("trt_tuned_dynamic_shape");
  int max_batch_size = Get<int>("max_batch_size");
  if (trt_tuned_dynamic_shape) {
    if (!shape_range_info_path.empty()) {
      VLOG(1) << "trt dynamic_shape deserialize from " << shape_range_info_path;
      inference::DeserializeShapeRangeInfo(shape_range_info_path,
                                           &min_input_shape,
                                           &max_input_shape,
                                           &optim_input_shape,
                                           &min_shape_tensor,
                                           &max_shape_tensor,
                                           &optim_shape_tensor);
    } else {
      shape_range_info_path = Get<std::string>("model_opt_cache_dir") + "/" +
                              "shape_range_info.pbtxt";
      if (open(shape_range_info_path.c_str(), O_RDONLY) != -1) {
        VLOG(1) << "trt dynamic_shape deserialize from "
                << shape_range_info_path;
        inference::DeserializeShapeRangeInfo(shape_range_info_path,
                                             &min_input_shape,
                                             &max_input_shape,
                                             &optim_input_shape,
                                             &min_shape_tensor,
                                             &max_shape_tensor,
                                             &optim_shape_tensor);
      } else {
        int fd = open(shape_range_info_path.c_str(), O_WRONLY | O_CREAT, 0644);
        close(fd);
      }
    }
  }

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to tensorrt OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the corresponding ITensor (trt data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.
  RenameAndGetOutputs(subgraph_nodes,
                      &block_desc,
                      input_names_with_id,
                      &output_names_with_id,
                      &output_names,
                      &output_name_map,
                      graph_var_map,
                      !enable_int8);

  // When tensorrt engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  std::vector<int> renamed_output_rank;
  for (auto name : output_names) {
    PADDLE_ENFORCE_NE(output_name_map.count(name),
                      0,
                      common::errors::PreconditionNotMet(
                          "The output_name_map should have %s", name));
    output_mapping.push_back(output_name_map[name]);
    renamed_output_rank.push_back(origin_name_output_rank[name]);
    origin_outputs_dtype.push_back(map_origin_outputs_dtype[name]);

    // When TRT Engine's output is INT64 or FP64, we need do some extra work.
    // So we reserved a name for later use when casting INT32 -> INT64 or FP32
    // -> FP64. We must check whether scope has had the same name var!
    if (static_cast<framework::proto::VarType_Type>(
            map_origin_outputs_dtype[name]) ==
        framework::proto::VarType::INT64) {
      LOG(WARNING) << "tensorrt_subgraph's output named " << name
                   << " having int64 dtype in pdmodel description, but in fact "
                      "it is int32 "
                      "dtype after executing this tensorrt_subgraph, so we "
                      "need cast them into int64.";
    } else if (static_cast<framework::proto::VarType_Type>(
                   map_origin_outputs_dtype[name]) ==
               framework::proto::VarType::FP64) {
      LOG(WARNING)
          << "tensorrt_subgraph's output named " << name
          << " having float64 dtype in pdmodel description, but in fact "
             "it is float32 "
             "dtype after executing this tensorrt_subgraph, so we "
             "need cast them into float64.";
    }
  }
  PADDLE_ENFORCE_EQ(output_mapping.empty(),
                    false,
                    common::errors::PreconditionNotMet(
                        "The output_mapping should not be empty."));
  PADDLE_ENFORCE_EQ(
      !block_desc.Proto()->vars().empty(),
      true,
      common::errors::PreconditionNotMet("the block has no var-desc"));

  // Get pass attrs.
  auto use_varseqlen = Get<bool>("use_varseqlen");
  auto with_interleaved = Get<bool>("with_interleaved");
  auto tensorrt_transformer_posid =
      Get<std::string>("tensorrt_transformer_posid");
  auto tensorrt_transformer_maskid =
      Get<std::string>("tensorrt_transformer_maskid");
  auto use_dla = Get<bool>("trt_use_dla");
  auto refit_params_path = Get<std::string>("refit_params_path");
  LOG(INFO) << "refit_params_path" << refit_params_path;
  auto dla_core = Get<int>("trt_dla_core");
  auto use_inspector = Get<bool>("use_inspector");
  auto inspector_serialize = Get<bool>("inspector_serialize");
  auto disable_trt_plugin_fp16 = Get<bool>("disable_trt_plugin_fp16");
  auto context_memory_sharing = Get<bool>("context_memory_sharing");
  if (context_memory_sharing && TRT_VERSION < 7200) {
    // https://forums.developer.nvidia.com/t/nvinfer1-createexecutioncontextwithoutdevicememory-returns-nullptr/111878/2
    // when trt version less than 7.2,
    // createExecutionContextWithoutDeviceMemory() has bug.
    // so, we cannot enable engine context memory sharing.
    context_memory_sharing = false;
  }
  auto enable_low_precision_io = Get<bool>("enable_low_precision_io");
  auto workspace_size = Get<int64_t>("workspace_size");
  auto gpu_device_id = Get<int>("gpu_device_id");
  auto optimization_level = Get<int>("optimization_level");
  auto use_explicit_quantization = Get<bool>("use_explicit_quantization");

  // Set op's attrs.
  op_desc->SetType("tensorrt_engine");
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));
  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));
  op_desc->SetBlockAttr("sub_block", new_block);
  op_desc->SetAttr("subgraph", block_desc.Proto()->SerializeAsString());
  op_desc->SetAttr("origin_outputs_dtype", origin_outputs_dtype);
  op_desc->SetAttr("max_batch_size", max_batch_size);
  op_desc->SetAttr("workspace_size", workspace_size);
  op_desc->SetAttr("gpu_device_id", gpu_device_id);
  op_desc->SetAttr("output_name_mapping", output_mapping);
  op_desc->SetAttr("origin_output_rank", renamed_output_rank);
  op_desc->SetAttr("parameters", parameters);
  op_desc->SetAttr("allow_build_at_runtime", allow_build_at_runtime);
  op_desc->SetAttr("shape_range_info_path", shape_range_info_path);
  op_desc->SetAttr("with_dynamic_shape", with_dynamic_shape);
  op_desc->SetAttr("enable_low_precision_io", enable_low_precision_io);

  if (!trt_tuned_dynamic_shape) {
    std::vector<std::string> dynamic_shape_names;
    std::vector<int> dynamic_shape_lens;
    std::vector<int> min_input_shape_vector;
    std::vector<int> max_input_shape_vector;
    std::vector<int> opt_input_shape_vector;
    for (const auto &it : min_input_shape) {
      dynamic_shape_names.push_back(it.first);
      dynamic_shape_lens.push_back(it.second.size());
      for (const auto &value : it.second) {
        min_input_shape_vector.push_back(value);
      }
    }
    for (const auto &it : max_input_shape) {
      for (const auto &value : it.second) {
        max_input_shape_vector.push_back(value);
      }
    }
    for (const auto &it : optim_input_shape) {
      for (const auto &value : it.second) {
        opt_input_shape_vector.push_back(value);
      }
    }

    op_desc->SetAttr("dynamic_shape_names", dynamic_shape_names);
    op_desc->SetAttr("dynamic_shape_lens", dynamic_shape_lens);
    op_desc->SetAttr("min_input_shape_vector", min_input_shape_vector);
    op_desc->SetAttr("max_input_shape_vector", max_input_shape_vector);
    op_desc->SetAttr("opt_input_shape_vector", opt_input_shape_vector);
  }

  op_desc->SetAttr("optimization_level", Get<int>("optimization_level"));

  // we record all inputs' shapes in attr to check if they are consistent
  // with the real inputs' shapes retrieved from scope when trt runs.
  for (auto *x : node->inputs) {
    if (x->IsVar() && x->Var()) {
      framework::VarDesc *var = x->Var();
      op_desc->SetAttr(var->Name() + "_shape", var->GetShape());
    }
  }

  auto use_static_engine = Get<bool>("use_static_engine");
  op_desc->SetAttr("use_static_engine", use_static_engine);
  if (use_static_engine)
    op_desc->SetAttr("model_opt_cache_dir",
                     Get<std::string>("model_opt_cache_dir"));

  // TODO(NHZlX)
  // There are models with the same structure but the different parameters,
  // when running in the 'use_serialize' mode, there is a bug.
  // serialization is affected by max_batch_size, but calibration is not.
  // So we use separate engine keys in serialization and calibration.
  auto engine_key =
      GenerateEngineKey(input_names_with_id,
                        output_names_with_id,
                        std::to_string(0),
                        std::to_string(max_batch_size),
                        std::to_string(static_cast<int>(precision_mode)),
                        use_cuda_graph,
                        false);
  auto calibration_engine_key =
      GenerateEngineKey(input_names_with_id,
                        output_names_with_id,
                        std::to_string(0),
                        std::to_string(max_batch_size),
                        std::to_string(static_cast<int>(precision_mode)),
                        use_cuda_graph,
                        true);
  auto predictor_id = Get<int>("predictor_id");

  // Get "" when there is no cached calibration table data.
  std::string calibration_data = "";
  if (enable_int8 && use_calib_mode) {
    calibration_data =
        GetTrtCalibTableData(Get<std::string>("model_opt_cache_dir"),
                             calibration_engine_key,
                             enable_int8);
  }
  op_desc->SetAttr("calibration_data", calibration_data);
  op_desc->SetAttr("enable_int8", enable_int8);
  op_desc->SetAttr("enable_fp16", enable_fp16);
  op_desc->SetAttr("enable_bfp16", enable_bfp16);
  op_desc->SetAttr("use_calib_mode", use_calib_mode);
  op_desc->SetAttr("engine_key", engine_key);
  op_desc->SetAttr("calibration_engine_key", calibration_engine_key);
  op_desc->SetAttr("predictor_id", predictor_id);
  op_desc->SetAttr("use_varseqlen", use_varseqlen);
  op_desc->SetAttr("with_interleaved", with_interleaved);
  op_desc->SetAttr("use_dla", use_dla);
  op_desc->SetAttr("refit_params_path", refit_params_path);
  op_desc->SetAttr("dla_core", dla_core);
  op_desc->SetAttr("disable_trt_plugin_fp16", disable_trt_plugin_fp16);
  op_desc->SetAttr("context_memory_sharing", context_memory_sharing);
  std::string trt_engine_serialized_data;
  op_desc->SetAttr("engine_serialized_data", trt_engine_serialized_data);

  // serialization engine info
  std::string engine_info_path;
  if (inspector_serialize) {
    engine_info_path = Get<std::string>("model_opt_cache_dir") +
                       "engine_info_" + engine_key + ".json";
    LOG(INFO) << "Serialize engine info to " << engine_info_path;
  }
  op_desc->SetAttr("use_inspector", use_inspector);
  op_desc->SetAttr("engine_info_path", engine_info_path);
  op_desc->Flush();

  std::unique_ptr<tensorrt::TRTInt8Calibrator> calibrator;
  if (enable_int8 && !calibration_data.empty()) {
    calibrator =
        std::make_unique<tensorrt::TRTInt8Calibrator>(calibration_data);
    LOG(INFO) << "RUN Paddle TRT int8 calibration mode...";
  }
  // When in int8 mode and calibration_mode, the program just produce the
  // calibration table data.
  bool calibration_mode =
      (enable_int8 && calibration_data.empty() && use_calib_mode);
  if (calibration_mode) {
    // calibration mode means generate int8 calibration table data process.
    return calibration_engine_key;
  }

  std::copy(params_not_shared.begin(),
            params_not_shared.end(),
            std::back_inserter(*repetitive_params));

  // Check trt version for dynamic shape input.

  if (!min_input_shape.empty() && TRT_VERSION < 6000) {
    LOG_FIRST_N(WARNING, 1) << "You are using the dynamic size input mode of "
                               "Paddle-TRT, but we found that the version of "
                               "the TensorRT is less than 6.0, so we use the "
                               "static shape mode instead.";
    min_input_shape = {};
    max_input_shape = {};
    optim_input_shape = {};
  }

  const float trt_compile_version = tensorrt::TrtMajorVersion(TRT_VERSION);
  const float trt_runtime_version =
      tensorrt::TrtMajorVersion(tensorrt::GetInferLibVersion());
  if (trt_compile_version != trt_runtime_version) {
    LOG_FIRST_N(WARNING, 1)
        << "The Paddle Inference library is compiled with "
        << trt_compile_version << " version TensorRT, "
        << "but the runtime TensorRT you are using is " << trt_runtime_version
        << " version. "
           "This might cause serious compatibility issues. We strongly "
           "recommend using the same TRT version at runtime.";
  }

  std::unordered_set<const Node *> nodes2remove(
      framework::ir::Agent(node).subgraph()->begin(),
      framework::ir::Agent(node).subgraph()->end());
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);

  // Adding new parameters must set a new op attribute by "op_desc->SetAttr()"
  // first and syncing it to tensorrt_engine_op.h
  tensorrt::TensorRTEngine::ConstructionParams params;
  params.max_batch_size = max_batch_size;
  params.max_workspace_size = workspace_size;
  params.calibrator = calibrator.get();
  params.device_id = gpu_device_id;
  params.with_dynamic_shape = with_dynamic_shape;
  params.min_input_shape = min_input_shape;
  params.max_input_shape = max_input_shape;
  params.optim_input_shape = optim_input_shape;
  params.min_shape_tensor = min_shape_tensor;
  params.max_shape_tensor = max_shape_tensor;
  params.optim_shape_tensor = optim_shape_tensor;
  params.disable_trt_plugin_fp16 = disable_trt_plugin_fp16;
  params.precision = precision_mode;
  params.use_varseqlen = use_varseqlen;
  params.use_dla = use_dla;
  params.refit_params_path = refit_params_path;
  params.dla_core = dla_core;
  params.with_interleaved = with_interleaved;
  params.tensorrt_transformer_posid = tensorrt_transformer_posid;
  params.tensorrt_transformer_maskid = tensorrt_transformer_maskid;
  params.context_memory_sharing = context_memory_sharing;
  params.use_inspector = use_inspector;
  params.engine_info_path = engine_info_path;
  params.enable_low_precision_io = enable_low_precision_io;
  params.optimization_level = optimization_level;
  params.use_explicit_quantization = use_explicit_quantization;

  tensorrt::TensorRTEngine *trt_engine =
      inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
          .Create(engine_key + std::to_string(predictor_id), params);

  // support force ops to run in FP32 precision
  trt_engine->SetRunFloat(trt_ops_run_float);

  if (use_static_engine) {
    trt_engine_serialized_data = GetTrtEngineSerializedData(
        Get<std::string>("model_opt_cache_dir"), engine_key);
    // we can load the engine info serialized before from the disk.
    if (!trt_engine_serialized_data.empty()) {
      try {
        trt_engine->Deserialize(trt_engine_serialized_data);
        LOG(INFO) << "Load TRT Optimized Info from "
                  << GetTrtEngineSerializedPath(
                         Get<std::string>("model_opt_cache_dir"), engine_key);
        return engine_key + std::to_string(predictor_id);
      } catch (const std::exception &exp) {
        LOG(WARNING)
            << "Fail to load TRT Optimized Info from "
            << GetTrtEngineSerializedPath(
                   Get<std::string>("model_opt_cache_dir"), engine_key)
            << ". Engine deserialization failed: Serialized Engine Version "
               "does not match Current Version, TRT engine will be rebuilt";
      }
    }
  }

  // If with_dynamic_shape is configured, but min_input_shape is empty,
  // create trt engine in runtime instead of in pass.
  if (with_dynamic_shape && min_input_shape.empty()) {
    return engine_key + std::to_string(predictor_id);
  }

  // the following code will NOT run in following situation:
  // 1. calibration mode (generate trt int8 calibration table data)
  // 2. already load serialized trt engine info.
  LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
               "kernel etc). This process may cost a lot of time.";

  framework::BlockDesc block_desc_temp(nullptr, block_desc.Proto());
  std::unordered_set<std::string> parameters_set(parameters.begin(),
                                                 parameters.end());
  inference::Singleton<inference::tensorrt::OpConverter>::Global()
      .ConvertBlockToTRTEngine(
          &block_desc_temp,
          *scope,
          std::vector<std::string>(input_names.begin(), input_names.end()),
          parameters_set,
          output_mapping,
          trt_engine);

  if (use_static_engine) {
    nvinfer1::IHostMemory *serialized_engine_data = trt_engine->Serialize();
    trt_engine_serialized_data =
        std::string((const char *)serialized_engine_data->data(),
                    serialized_engine_data->size());
    SaveTrtEngineSerializedDataToFile(
        GetTrtEngineSerializedPath(Get<std::string>("model_opt_cache_dir"),
                                   engine_key),
        trt_engine_serialized_data);
    LOG(INFO) << "Save TRT Optimized Info to "
              << GetTrtEngineSerializedPath(
                     Get<std::string>("model_opt_cache_dir"), engine_key);
  }

  return engine_key + std::to_string(predictor_id);
}

}  // namespace paddle::inference::analysis

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");

REGISTER_PASS_CAPABILITY(tensorrt_subgraph_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("pool2d", 0)
            .EQ("relu", 0)
            .EQ("softmax", 0)
            .EQ("sigmoid", 0)
            .EQ("hard_swish", 0)
            .LE("depthwise_conv2d", 1)
            .EQ("batch_norm", 0)
            .EQ("concat", 0)
            .EQ("tanh", 0)
            .EQ("pad", 0)
            .LE("elementwise_add", 1)
            .LE("elementwise_mul", 1)
            .EQ("prelu", 0)
            .LE("conv2d_transpose", 2)
            .LE("leaky_relu", 1)
            .EQ("fc", 0)
            .EQ("shuffle_channel", 0)
            .EQ("swish", 0)
            .EQ("silu", 0)
            .EQ("split", 0)
            .LE("instance_norm", 1)
            .EQ("gelu", 0)
            .EQ("layer_norm", 0)
            .EQ("scale", 0)
            .LE("matmul", 1));
