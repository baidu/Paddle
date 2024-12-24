// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/fluid/pybind/eager_generator.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/utils/string/string_helper.h"

// phi
#include "paddle/phi/kernels/declarations.h"

#define NUM_CREATED_DUP_INPUTS 4

namespace paddle::framework {

// To handle append_op at python-level
std::unordered_map<std::string, std::vector<std::string>>
    core_ops_legacy_returns_info = {};
std::unordered_map<std::string, std::vector<std::string>>
    core_ops_legacy_args_info = {};
std::unordered_map<std::string, std::vector<std::string>>
    core_ops_legacy_args_type_info = {};

/* --- Static maps to handle corner cases --- */
static std::unordered_map<std::string, paddle::framework::AttributeMap>
    operators_with_attrs = {};

static std::unordered_set<std::string> ops_to_fill_zero_for_empty_grads = {
    "split", "rnn"};

/* --- Black Ops list that's NO NEED to apply code generation --- */
static std::unordered_set<std::string> black_ops_list = {
    "run_program",
    "fused_gate_attention",
    "fused_feedforward",
    "fused_attention",
    "fused_gemm_epilogue",
    "fused_bias_dropout_residual_layer_norm",
    "sparse_divide_scalar",
    "sparse_scale"};

static std::string LegalizeVariableName(const std::string& var_name) {
  std::string ret = var_name;
  std::replace(ret.begin(), ret.end(), '-', '_');  // replace all '-' to '_'
  std::replace(ret.begin(), ret.end(), '@', '_');  // replace all '-' to '_'
  return ret;
}

static std::string LegalizeVarName(const std::string& var_name) {
  std::string ret = var_name;
  std::replace(ret.begin(), ret.end(), '@', '_');  // replace all '-' to '_'
  return ret;
}

static std::string HandleDynamicGradAttributes(const std::string& fwd_op_type,
                                               const std::string& attrs_name) {
  std::string additional_grad_attrs_str = "";

  if (fwd_op_type == "sum") {
    const char* GRAD_ATTRS_TEMPLATE = "  %s[\"%s\"] = %s;\n";
    additional_grad_attrs_str = paddle::string::Sprintf(
        GRAD_ATTRS_TEMPLATE, attrs_name, "scale", "float(1.0)");
    additional_grad_attrs_str += paddle::string::Sprintf(
        GRAD_ATTRS_TEMPLATE, attrs_name, "bias", "float(0.0f)");
    additional_grad_attrs_str += paddle::string::Sprintf(
        GRAD_ATTRS_TEMPLATE, attrs_name, "bias_after_scale", "bool(true)");

  } else if (fwd_op_type == "scale") {
    const char* GRAD_ATTRS_TEMPLATE = "  %s[\"%s\"] = %s;\n";

    additional_grad_attrs_str += paddle::string::Sprintf(
        GRAD_ATTRS_TEMPLATE, attrs_name, "bias", "float(0.0f)");
    additional_grad_attrs_str += paddle::string::Sprintf(
        GRAD_ATTRS_TEMPLATE, attrs_name, "bias_after_scale", "bool(true)");
  }

  return additional_grad_attrs_str;
}

static void PrepareAttrMapForOps() {
  // Handle "fused_elemwise_add_activation"
  std::vector<std::string> functor_list = {"a", "b"};
  operators_with_attrs["fused_elemwise_add_activation"] = {};
  operators_with_attrs["fused_elemwise_add_activation"]["functor_list"] =
      functor_list;

  // Handle "fused_elemwise_activation"
  operators_with_attrs["fused_elemwise_activation"] = {};
  operators_with_attrs["fused_elemwise_activation"]["functor_list"] =
      functor_list;

  // Handle "reverse"
  std::vector<int> axis = {0};
  operators_with_attrs["reverse"] = {};
  operators_with_attrs["reverse"]["axis"] = axis;

  // Handle "flip"
  operators_with_attrs["flip"] = {};
  operators_with_attrs["flip"]["axis"] = axis;

  // Handle "cast"
  operators_with_attrs["cast"] = {};
  operators_with_attrs["cast"]["out_dtype"] = 5;
  operators_with_attrs["cast"]["in_dtype"] = 5;

  // Handle "c_split"
  operators_with_attrs["c_split"] = {};
  operators_with_attrs["c_split"]["nranks"] = 1;
}

/* --- Helper Objects --- */
class ForwardGenerationInfo {
 public:
  ForwardGenerationInfo()
      : fwd_inputs_name_pos_map_(),
        fwd_outputs_name_pos_map_(),
        in_vars_(),
        out_vars_() {}
  const std::string& GetOpType() const { return op_type_; }
  void SetOpType(const std::string& op_type) { op_type_ = op_type; }

  const std::unordered_map<std::string, size_t>& GetFwdInputsNamePosMap()
      const {
    return fwd_inputs_name_pos_map_;
  }
  std::unordered_map<std::string, size_t>* GetMutableFwdInputsNamePosMap() {
    return &fwd_inputs_name_pos_map_;
  }

  const std::unordered_map<std::string, size_t>& GetFwdOutputsNamePosMap()
      const {
    return fwd_outputs_name_pos_map_;
  }
  std::unordered_map<std::string, size_t>* GetMutableFwdOutputsNamePosMap() {
    return &fwd_outputs_name_pos_map_;
  }

  const std::vector<proto::OpProto::Var>& GetInVars() const { return in_vars_; }
  std::vector<proto::OpProto::Var>* GetMutableInVars() { return &in_vars_; }

  const std::vector<proto::OpProto::Var>& GetOutVars() const {
    return out_vars_;
  }
  std::vector<proto::OpProto::Var>* GetMutableOutVars() { return &out_vars_; }

 private:
  std::string op_type_;
  std::unordered_map<std::string, size_t> fwd_inputs_name_pos_map_;
  std::unordered_map<std::string, size_t> fwd_outputs_name_pos_map_;
  std::vector<proto::OpProto::Var> in_vars_;
  std::vector<proto::OpProto::Var> out_vars_;
};

class GradNodeGenerationInfo {
  class OpBaseGenerationInfo {
   public:
    const std::string& GetOpBaseType() const { return op_base_type_; }
    void SetOpBaseType(const std::string& op_type) { op_base_type_ = op_type; }

    const std::map<std::string, std::string>& GetGradOutsSlotnameMap() const {
      return grad_outs_slotname_map_;
    }
    std::map<std::string, std::string>* GetMutableGradOutsSlotnameMap() {
      return &grad_outs_slotname_map_;
    }

    const std::map<std::string, std::string>& GetGradInsFwdSlotnameMap() const {
      return grad_ins_fwd_slotname_map_;
    }
    std::map<std::string, std::string>* GetMutableGradInsFwdSlotnameMap() {
      return &grad_ins_fwd_slotname_map_;
    }

    const std::map<std::string, std::string>& GetGradInsGradSlotnameMap()
        const {
      return grad_ins_grad_slotname_map_;
    }
    std::map<std::string, std::string>* GetMutableGradInsGradSlotnameMap() {
      return &grad_ins_grad_slotname_map_;
    }

    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
    GetGradIns() const {
      return grad_ins_;
    }
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
    GetMutableGradIns() {
      return &grad_ins_;
    }

    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
    GetGradOuts() const {
      return grad_outs_;
    }
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
    GetMutableGradOuts() {
      return &grad_outs_;
    }

    const paddle::framework::AttributeMap& GetGradAttrs() const {
      return grad_attrs_;
    }
    paddle::framework::AttributeMap* GetMutableGradAttrs() {
      return &grad_attrs_;
    }

    const std::unordered_set<std::string>& GetNoNeedBufferInputs() const {
      return no_need_buffer_ins_;
    }
    std::unordered_set<std::string>* GetMutableNoNeedBufferInputs() {
      return &no_need_buffer_ins_;
    }

    const std::unordered_map<std::string, std::string>& GetBackwardInplaceMap()
        const {
      return backward_inplace_map_;
    }
    std::unordered_map<std::string, std::string>*
    GetMutableBackwardInplaceMap() {
      return &backward_inplace_map_;
    }

   private:
    std::string op_base_type_;
    std::map<std::string, std::string> grad_outs_slotname_map_;
    std::map<std::string, std::string> grad_ins_fwd_slotname_map_;
    std::map<std::string, std::string> grad_ins_grad_slotname_map_;
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
        grad_ins_;
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
        grad_outs_;
    paddle::framework::AttributeMap grad_attrs_;
    std::unordered_set<std::string> no_need_buffer_ins_;
    std::unordered_map<std::string, std::string> backward_inplace_map_;
  };

 public:
  GradNodeGenerationInfo() : op_base_infos_() {}
  const std::string& GetFwdOpType() const { return fwd_op_type_; }
  void SetFwdOpType(const std::string& op_type) { fwd_op_type_ = op_type; }

  bool GenerateForwardOnly() const { return generate_forward_only_; }
  void SetGenerateForwardOnly(bool generate_forward_only) {
    generate_forward_only_ = generate_forward_only;
  }

  const std::vector<OpBaseGenerationInfo>& GetOpBaseInfos() const {
    return op_base_infos_;
  }
  std::vector<OpBaseGenerationInfo>* GetMutableOpBaseInfos() {
    return &op_base_infos_;
  }

 private:
  std::string fwd_op_type_;
  bool generate_forward_only_ = false;
  std::vector<OpBaseGenerationInfo> op_base_infos_;
};

/* --- Helper Functions --- */
static std::string AttrTypeToString(const proto::AttrType& type) {
  std::string ret;
  switch (type) {
    case (proto::AttrType::INT): {
      ret = "int";
      break;
    }
    case (proto::AttrType::FLOAT): {
      ret = "float";
      break;
    }
    case (proto::AttrType::STRING): {
      ret = "std::string&";
      break;
    }
    case (proto::AttrType::INTS): {
      ret = "std::vector<int>&";
      break;
    }
    case (proto::AttrType::FLOATS): {
      ret = "std::vector<float>&";
      break;
    }
    case (proto::AttrType::STRINGS): {
      ret = "std::vector<std::string>&";
      break;
    }
    case (proto::AttrType::BOOLEAN): {
      ret = "bool";
      break;
    }
    case (proto::AttrType::BOOLEANS): {
      ret = "std::vector<bool>&";
      break;
    }
    case (proto::AttrType::LONG): {
      ret = "int64_t";
      break;
    }
    case (proto::AttrType::LONGS): {
      ret = "std::vector<int64_t>&";
      break;
    }
    case (proto::AttrType::BLOCK): {
      ret = "paddle::framework::BlockDesc*";
      break;
    }
    case (proto::AttrType::BLOCKS): {
      ret = "std::vector<paddle::framework::BlockDesc*>&";
      break;
    }
    case (proto::AttrType::FLOAT64S): {
      ret = "std::vector<double>&";
      break;
    }
    default: {
      PADDLE_THROW(common::errors::Fatal(
          "AttrType of type paddle::variant only supports specific data types."
          "However, detected unrecognized AttrType: %d",
          type));
    }
  }
  return ret;
}

template <typename T, bool IsVector>
static typename std::enable_if<IsVector, std::string>::type GetAttrValue(
    const framework::Attribute& attr) {
  std::string val = "";
  val += "{";
  for (auto x : PADDLE_GET_CONST(std::vector<T>, attr)) {
    val += std::to_string(x) + ",";
  }
  if (val.size() > 1) val.pop_back();
  val += "}";
  return val;
}

template <typename T, bool IsVector>
static typename std::enable_if<!IsVector, std::string>::type GetAttrValue(
    const framework::Attribute& attr) {
  return std::to_string(PADDLE_GET_CONST(T, attr));
}

static std::pair<std::string, std::string> GetAttrType(
    const framework::Attribute& attr, bool is_arg) {
  std::string ret = "";
  std::string val = "";
  size_t variant_pos = attr.index();
  switch (variant_pos) {
    case (1): {
      ret = "int";
      val = GetAttrValue<int, false>(attr);
      break;
    }
    case (2): {
      ret = "float";
      val = GetAttrValue<float, false>(attr);
      break;
    }
    case (3): {
      ret = "std::string";
      if (is_arg) ret += "&";
      val = "\"" + PADDLE_GET_CONST(std::string, attr) + "\"";
      break;
    }
    case (4): {
      ret = "std::vector<int>";
      if (is_arg) ret += "&";
      val = GetAttrValue<int, true>(attr);
      break;
    }
    case (5): {
      ret = "std::vector<float>";
      if (is_arg) ret += "&";
      val = GetAttrValue<float, true>(attr);
      break;
    }
    case (6): {
      ret = "std::vector<std::string>";
      if (is_arg) ret += "&";
      val += "{";
      for (auto const& x : PADDLE_GET_CONST(std::vector<std::string>, attr)) {
        val += "\"" + x + "\"" + ",";
      }
      if (val.size() > 1) val.pop_back();
      val += "};";
      break;
    }
    case (7): {
      ret = "bool";
      val = GetAttrValue<bool, false>(attr);
      break;
    }
    case (8): {
      ret = "std::vector<bool>";
      if (is_arg) ret += "&";
      val = GetAttrValue<bool, true>(attr);
      break;
    }
    case (9): {
      ret = "BlockDesc*";
      break;
    }
    case (10): {
      ret = "int64_t";
      val = GetAttrValue<int64_t, false>(attr);
      break;
    }
    case (11): {
      ret = "std::vector<BlockDesc*>";
      if (is_arg) ret += "&";
      break;
    }
    case (12): {
      ret = "std::vector<int64_t>";
      if (is_arg) ret += "&";
      val = GetAttrValue<int64_t, true>(attr);
      break;
    }
    case (13): {
      ret = "std::vector<double>";
      if (is_arg) ret += "&";
      val = GetAttrValue<double, true>(attr);
      break;
    }
    default: {
      PADDLE_THROW(common::errors::Fatal(
          "AttrType of type paddle::variant only supports specific data types."
          "However, detected unrecognized AttrType: %d",
          variant_pos));
    }
  }
  return {ret, val};
}

static void SlotNameMatching(
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_map,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        fwd_ins,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        fwd_outs,
    std::map<std::string, std::string>* grad_fwd_slotname_map_ptr,
    std::map<std::string, std::string>* grad_grad_slotname_map_ptr) {
  std::map<std::string, std::string>& grad_fwd_slotname_map =
      *grad_fwd_slotname_map_ptr;
  std::map<std::string, std::string>& grad_grad_slotname_map =
      *grad_grad_slotname_map_ptr;
  for (const auto& iter : grad_map) {
    const std::string& grad_slot_name = iter.first;
    const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
        grad_vars = iter.second;

    // Find matching fwd_slot_name
    bool found_matching = false;
    for (const std::shared_ptr<paddle::imperative::VariableWrapper>& grad_var :
         grad_vars) {
      for (const auto& fwd_iter : fwd_ins) {
        const std::string& fwd_slot_name = fwd_iter.first;
        const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
            fwd_vars = fwd_iter.second;
        for (const std::shared_ptr<paddle::imperative::VariableWrapper>&
                 fwd_var : fwd_vars) {
          if (grad_var == fwd_var) {
            if (grad_fwd_slotname_map.count(grad_slot_name) &&
                grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(common::errors::Fatal(
                  "Detected mismatched slot names."
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name,
                  grad_fwd_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }

          if (fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
            if (grad_grad_slotname_map.count(grad_slot_name) &&
                grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(common::errors::Fatal(
                  "Detected mismatched slot names."
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name,
                  grad_grad_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }
        }
      }
      for (const auto& fwd_iter : fwd_outs) {
        const std::string& fwd_slot_name = fwd_iter.first;
        const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
            fwd_vars = fwd_iter.second;
        for (const std::shared_ptr<paddle::imperative::VariableWrapper>&
                 fwd_var : fwd_vars) {
          if (grad_var == fwd_var) {
            if (grad_fwd_slotname_map.count(grad_slot_name) &&
                grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(common::errors::Fatal(
                  "Detected mismatched slot names"
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name,
                  grad_fwd_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }

          if (fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
            if (grad_grad_slotname_map.count(grad_slot_name) &&
                grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(common::errors::Fatal(
                  "Detected mismatched slot names."
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name,
                  grad_grad_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }
        }
      }
    }

    if (!found_matching) {
      PADDLE_THROW(common::errors::Fatal(
          "Detected mismatched slot names."
          "Found no matching fwd_slot_name for grad_slot_name: %s",
          grad_slot_name));

    } else {
      std::string fwd_slot_name = grad_grad_slotname_map.count(grad_slot_name)
                                      ? grad_grad_slotname_map[grad_slot_name]
                                      : grad_fwd_slotname_map[grad_slot_name];
      VLOG(6) << "Found matching fwd_slot_name: " << fwd_slot_name
              << " for grad_slot_name: " << grad_slot_name;
    }
  }
}

static bool CheckOpProto(proto::OpProto* op_proto) {
  if (op_proto == nullptr) {
    return false;
  }
  const std::string& op_type = op_proto->type();

  // Skip operator which is not inherit form OperatorWithKernel, like while,
  // since only OperatorWithKernel can run in dygraph mode.
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
  if (!all_kernels.count(op_type) &&
      !phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type)) {
    return false;
  }

  // Only handle matmul_v2 for now
  VLOG(3) << "------ Analyzing Op ------: " << op_type;

  return true;
}

static bool BeSameAsInput(const std::string& output_name,
                          const std::set<std::string>& input_names) {
  if (output_name.size() < 4) {
    return false;
  }

  if (output_name.substr(output_name.size() - 3, 3) == "Out") {
    if (input_names.count(output_name.substr(0, output_name.size() - 3))) {
      return true;
    }
  }

  return false;
}

/* --------------------------------------- */
/* --------- Preprocess Ins/Outs --------- */
/* --------------------------------------- */
static void PurifyForwardOpProto(const proto::OpProto& op_proto,
                                 ForwardGenerationInfo* fwd_info) {
  // Op Name
  const std::string& op_name = op_proto.type();

  auto* in_vars = fwd_info->GetMutableInVars();
  auto* out_vars = fwd_info->GetMutableOutVars();
  auto* fwd_inputs_name_pos_map = fwd_info->GetMutableFwdInputsNamePosMap();
  auto* fwd_outputs_name_pos_map = fwd_info->GetMutableFwdOutputsNamePosMap();

  // Handle dispensable inputs
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    std::string input_name = input.name();

    // Delete dispensable tensor unless specified in op_ins_map
    if (input.dispensable()) {
      if (!op_ins_map.count(op_name) ||
          !op_ins_map[op_name].count(input_name)) {
        VLOG(6) << "Removing Dispensable Input: " << input_name;

        // in_vars
        auto iter = in_vars->begin();
        for (iter = in_vars->begin(); iter != in_vars->end(); iter++) {
          if (iter->name() == input_name) {
            break;
          }
        }
        in_vars->erase(iter);
      }
    }
  }

  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    std::string output_name = output.name();

    // Delete dispensable tensor unless specified in op_outs_map
    if (output.dispensable()) {
      if (!op_outs_map.count(op_name) ||
          !op_outs_map[op_name].count(output_name)) {
        VLOG(6) << "Removing Dispensable Output: " << output_name;

        // out_vars
        auto iter = out_vars->begin();
        for (iter = out_vars->begin(); iter != out_vars->end(); iter++) {
          if (iter->name() == output_name) {
            break;
          }
        }
        out_vars->erase(iter);
      }
    }
  }

  /* ------ Mapping forward slot name to fwd position ------ */
  size_t in_pos = 0;
  for (const auto& var : *in_vars) {
    VLOG(6) << "Mapping input tensor: " << var.name()
            << " To position: " << in_pos;
    (*fwd_inputs_name_pos_map)[var.name()] = in_pos;
    in_pos++;
  }

  size_t out_pos = 0;
  for (const auto& var : *out_vars) {
    VLOG(6) << "Mapping output tensor: " << var.name()
            << " To position: " << out_pos;
    (*fwd_outputs_name_pos_map)[var.name()] = out_pos;
    out_pos++;
  }
}

static void PurifyGradNodeGenerationInfo(const proto::OpProto& op_proto,
                                         GradNodeGenerationInfo* bwd_info) {
  auto* op_base_infos = bwd_info->GetMutableOpBaseInfos();
  for (auto& iter : *op_base_infos) {
    std::map<std::string, std::string>* grad_outs_slotname_map =
        iter.GetMutableGradOutsSlotnameMap();
    std::map<std::string, std::string>* grad_ins_fwd_slotname_map =
        iter.GetMutableGradInsFwdSlotnameMap();
    std::map<std::string, std::string>* grad_ins_grad_slotname_map =
        iter.GetMutableGradInsGradSlotnameMap();
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_ins = iter.GetMutableGradIns();
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_outs = iter.GetMutableGradOuts();

    // Op Name
    const std::string op_name = op_proto.type();

    // Handle dispensable inputs
    for (const proto::OpProto::Var& input : op_proto.inputs()) {
      std::string input_name = input.name();

      // Delete dispensable tensor unless specified in op_ins_map
      if (input.dispensable()) {
        if (!op_ins_map.count(op_name) ||
            !op_ins_map[op_name].count(input_name)) {
          VLOG(6) << "Removing Dispensable Input: " << input_name;

          // grad_outs_slotname_map
          auto grad_outs_slotname_map_purified = *grad_outs_slotname_map;
          for (const auto& iter : *grad_outs_slotname_map) {
            const std::string& grad_output_name = iter.first;
            const std::string& matched_input_name = iter.second;
            if (matched_input_name == input_name) {
              grad_outs_slotname_map_purified.erase(grad_output_name);

              PADDLE_ENFORCE(
                  grad_outs->count(grad_output_name) > 0,
                  common::errors::Fatal(
                      "Unable to find gradient output name in grad_outs."));
              // grad_outs
              grad_outs->erase(grad_output_name);
            }
          }
          *grad_outs_slotname_map = grad_outs_slotname_map_purified;

          // grad_ins_fwd_slotname_map: output as tensorwrapper
          if (grad_ins_fwd_slotname_map->count(input_name))
            grad_ins_fwd_slotname_map->erase(input_name);

          // grad_ins: output as tensorwrapper
          if (grad_ins->count(input_name)) grad_ins->erase(input_name);
        }
      }
    }

    for (const proto::OpProto::Var& output : op_proto.outputs()) {
      std::string output_name = output.name();

      // Delete dispensable tensor unless specified in op_outs_map
      if (output.dispensable()) {
        if (!op_outs_map.count(op_name) ||
            !op_outs_map[op_name].count(output_name)) {
          VLOG(6) << "Removing Dispensable Output: " << output_name;

          // grad_ins_grad_slotname_map
          auto grad_ins_grad_slotname_map_purified =
              *grad_ins_grad_slotname_map;
          for (const auto& iter : *grad_ins_grad_slotname_map) {
            const std::string& grad_input_name = iter.first;
            const std::string& matched_output_name = iter.second;
            if (matched_output_name == output_name) {
              grad_ins_grad_slotname_map_purified.erase(grad_input_name);

              PADDLE_ENFORCE(
                  grad_ins->count(grad_input_name) > 0,
                  common::errors::Fatal(
                      "Unable to find gradient input name in grad_ins."));
              // grad_ins
              grad_ins->erase(grad_input_name);
            }
          }
          *grad_ins_grad_slotname_map = grad_ins_grad_slotname_map_purified;

          // grad_ins_fwd_slotname_map: output as tensorwrapper
          if (grad_ins_fwd_slotname_map->count(output_name))
            grad_ins_fwd_slotname_map->erase(output_name);

          // grad_ins: output as tensorwrapper
          if (grad_ins->count(output_name)) grad_ins->erase(output_name);
        }
      }
    }
  }
}

/* -------------------------------- */
/* --------- Collect Info --------- */
/* -------------------------------- */
static void CollectForwardInformationFromOpInfo(
    const paddle::framework::OpInfo& op_info, ForwardGenerationInfo* fwd_info) {
  const proto::OpProto& op_proto = *op_info.proto_;

  fwd_info->SetOpType(op_proto.type());

  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    fwd_info->GetMutableInVars()->push_back(input);
  }
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    fwd_info->GetMutableOutVars()->push_back(output);
  }
}

static bool CollectGradInformationFromOpInfo(
    const paddle::framework::OpInfo& op_info,
    GradNodeGenerationInfo* bwd_info) {
  const proto::OpProto& op_proto = *op_info.proto_;
  const std::string& op_type = op_proto.type();
  std::vector<int64_t> dims = {1, 1, 1, 1};

  /* ------ Prepare "ins" ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VarBase>>>
      ins;

  if (op_proto.inputs().size() == 1 && op_proto.outputs().size() == 1 &&
      op_proto.inputs()[0].duplicable() &&
      !op_proto.outputs()[0].duplicable()) {
    VLOG(6) << "Handle op with special op_bases: " << op_type;
    // @special case (sum_op): for ops with single duplicable input and single
    // non-duplicable output
    //                         feed in NUM_CREATED_DUP_INPUTS inputs to detect a
    //                         special scenario.
    const std::string& in_name = op_proto.inputs()[0].name();
    ins[in_name] = {};
    for (size_t i = 0; i < NUM_CREATED_DUP_INPUTS; i++) {
      ins[in_name].emplace_back(std::make_shared<paddle::imperative::VarBase>(
          "auto_" + in_name + "_" + std::to_string(i)));
      ins[in_name][i]->SetOverriddenStopGradient(false);
      ins[in_name][i]->MutableVar()->GetMutable<phi::DenseTensor>();
    }
  } else {
    for (const proto::OpProto::Var& input : op_proto.inputs()) {
      const std::string& in_name = input.name();

      // Handle dispensable input:
      // 1. At python level, dispensable input will be detected at Python-C
      // interface and filled with an empty vector
      // 2. At C++ level, customers should always pass an empty vector for any
      // dispensable input
      // 3. During further lowering, there will always be a placeholder VarBase
      // in ins/outs no matter whether it's dispensable or not
      // As a result, we always create input VarBase regardless of its
      // dispensability.

      // Handle duplicable input: list(VarBase) or VarBase
      // We dont know the exact number of inputs expected,
      // but we only need to identify the slot name order,
      // therefore fill in 1 single input VarBase is enough in this scenario

      ins[in_name] = {
          std::make_shared<paddle::imperative::VarBase>("auto_" + in_name)};
      ins[in_name][0]->SetOverriddenStopGradient(false);
      ins[in_name][0]->MutableVar()->GetMutable<phi::DenseTensor>();
    }
  }
  VLOG(6) << "Prepared Forward Ins Map, size = " << ins.size();

  /* ------ Prepare "outs" ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VarBase>>>
      outs;
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    const std::string& out_name = output.name();

    // We always create output VarBase regardless of its dispensability.
    // We dont know the exact number of outputs during code generation,
    // however, simply identifying the slot name order would be enough
    outs[out_name] = {
        std::make_shared<paddle::imperative::VarBase>("auto_" + out_name)};
    outs[out_name][0]->SetOverriddenStopGradient(false);
    outs[out_name][0]->MutableVar()->GetMutable<phi::DenseTensor>();
  }
  VLOG(6) << "Prepared Forward Outs Map, size = " << outs.size();

  framework::AttributeMap attrs;
  paddle::framework::AttributeMap default_attrs;
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    VLOG(6) << "Checking AttributeMap Settings";
    attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
    default_attrs = attr_checker->GetDefaultAttrMap();
  } else {
    VLOG(6) << "Detected Null Attribute Checker, use empty default_attrs";
  }

  if (operators_with_attrs.count(op_type)) {
    VLOG(6) << "Found operator " << op_type << " using special AttributeMap";
    attrs = operators_with_attrs[op_type];
  }

  VLOG(6) << "Prepared Default Attributes Map, size = " << default_attrs.size();
  for (const auto& iter : default_attrs) {
    VLOG(6) << iter.first;
  }

  /* ---------------------------- */
  /* --------- Backward --------- */
  /* ---------------------------- */
  /* ------ Fwd paddle::imperative::VariableWrapper Map ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
      fwd_ins;
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
      fwd_outs;
  for (const auto& iter : ins) {
    fwd_ins[iter.first] = {};
    for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
         iter.second) {
      fwd_ins[iter.first].push_back(var_base->SharedVar());
    }
  }
  for (const auto& iter : outs) {
    fwd_outs[iter.first] = {};
    for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
         iter.second) {
      fwd_outs[iter.first].push_back(var_base->SharedVar());
    }
  }
  VLOG(6) << "Constructed Forward paddle::imperative::VariableWrapper Map";

  /* ------ Run GradOpMaker ------ */
  if (!op_info.dygraph_grad_op_maker_) {
    VLOG(6) << op_type << " has no GradOpMaker";
    bwd_info->SetGenerateForwardOnly(true);
    return false;
  }

  std::shared_ptr<paddle::imperative::GradOpNode> grad_node =
      op_info.dygraph_grad_op_maker_(
          op_type, ins, outs, attrs, default_attrs, {});

  if (!grad_node) {
    VLOG(6) << "Got nullptr GradOpNode for " << op_type
            << " likely registered EmptyGradOpMaker";
    bwd_info->SetGenerateForwardOnly(true);
    return false;
  }

  VLOG(6) << "Prepared GradOpNode";

  /* ---- Collect OpBase's op_types ---- */
  bwd_info->SetFwdOpType(op_type);
  auto* op_base_infos = bwd_info->GetMutableOpBaseInfos();
  op_base_infos->resize(grad_node->size());
  for (auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
    // Each OpBase
    int index = static_cast<int>(std::distance(grad_node->begin(), iter));
    paddle::imperative::OpBase& op_base = *iter;
    (*op_base_infos)[index].SetOpBaseType(op_base.Type());
  }

  /* ------ Get Grad ins/outs/attrs ---- */
  VLOG(6) << "In function size: " << grad_node->size();
  for (auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
    int index = static_cast<int>(std::distance(grad_node->begin(), iter));
    auto* op_base_grad_ins = (*op_base_infos)[index].GetMutableGradIns();
    auto* op_base_grad_outs = (*op_base_infos)[index].GetMutableGradOuts();
    auto* op_base_grad_attrs = (*op_base_infos)[index].GetMutableGradAttrs();

    const paddle::imperative::OpBase& op_base = *iter;
    const std::map<std::string, paddle::imperative::SavedVariableWrapperList>&
        g_ins = op_base.GetInsMap();
    const std::map<std::string, paddle::imperative::SavedVariableWrapperList>&
        g_outs = op_base.GetOutsMap();

    *op_base_grad_attrs = op_base.Attrs();

    for (const auto& it : g_ins) {
      if (!op_base_grad_ins->count(it.first))
        (*op_base_grad_ins)[it.first] = {};

      for (auto vw_iter = it.second.begin(); vw_iter != it.second.end();
           vw_iter++) {
        std::shared_ptr<paddle::imperative::VariableWrapper> vw = *vw_iter;

        (*op_base_grad_ins)[it.first].push_back(vw);

        VLOG(6) << "GradIns Name: " << it.first;
      }
    }

    for (const auto& it : g_outs) {
      if (!op_base_grad_outs->count(it.first))
        (*op_base_grad_outs)[it.first] = {};

      for (auto vw_iter = it.second.begin(); vw_iter != it.second.end();
           vw_iter++) {
        std::shared_ptr<paddle::imperative::VariableWrapper> vw = *vw_iter;

        (*op_base_grad_outs)[it.first].push_back(vw);

        VLOG(6) << "GradOuts Name: " << it.first;
      }
    }

    auto& inferer = op_base.Info().NoNeedBufferVarsInferer();
    if (inferer && !special_no_need_buffer_op_set.count(op_type)) {
      *(*op_base_infos)[index].GetMutableNoNeedBufferInputs() =
          inferer(g_ins, g_outs, *op_base_grad_attrs);
    }

    auto& infer_backward_inplace = op_base.Info().infer_inplace_;
    if (infer_backward_inplace) {
      *(*op_base_infos)[index].GetMutableBackwardInplaceMap() =
          infer_backward_inplace(true);
    }
  }

  /* ------ Slot Name Matching ---- */
  for (auto& iter : *op_base_infos) {
    // grad_ins -> fwd_ins, fwd_outs
    SlotNameMatching(iter.GetGradIns(),
                     fwd_ins,
                     fwd_outs,
                     iter.GetMutableGradInsFwdSlotnameMap(),
                     iter.GetMutableGradInsGradSlotnameMap());

    // grad_outs -> fwd_ins, fwd_outs
    SlotNameMatching(iter.GetGradOuts(),
                     fwd_ins,
                     fwd_outs,
                     iter.GetMutableGradOutsSlotnameMap(),
                     iter.GetMutableGradOutsSlotnameMap());
  }
  VLOG(6) << "Finished Slotname Matching";

  return true;
}

/* --------------------------------------------------- */
/* --------- CodeGen: Forward GradNode Creation ------ */
/* --------------------------------------------------- */
static std::string GenerateGradNodeCreationContent(
    const ForwardGenerationInfo& fwd_info,
    const GradNodeGenerationInfo& bwd_info,
    const std::string& trace_op_body_str,
    std::map<std::string, std::string> forward_inplace_map = {}) {
  VLOG(6) << "Generating GradNode Creation codes";

  const std::string& op_type = fwd_info.GetOpType();
  const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map =
      fwd_info.GetFwdInputsNamePosMap();
  const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map =
      fwd_info.GetFwdOutputsNamePosMap();
  const std::vector<proto::OpProto::Var>& in_vars = fwd_info.GetInVars();
  const std::vector<proto::OpProto::Var>& out_vars = fwd_info.GetOutVars();

  const auto& op_base_infos = bwd_info.GetOpBaseInfos();

  // [Generation] Construct GradOpNode
  // Run ComputeRequiredGrad

  // If single output slotname and not duplicable,
  // then generate: "egr::AutogradMeta* p_autograd_out =
  // egr::EagerUtils::autograd_meta("op_proto->outputs()[0].name()")"
  std::string get_input_autograd_meta_str = "  // Prepare Autograd Meta\n";
  std::string get_output_autograd_meta_str = "";
  // If single output slotname and not duplicable,
  // then generate: "egr::AutogradMeta* p_autograd_out =
  // egr::EagerUtils::autograd_meta("op_proto.outputs()[0].name()")"
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    const std::string& output_autograd_name =
        "p_autograd_" + LegalizeVarName(output_name);

    // output autograd_meta should be got after running TraceOP.
    if (output.duplicable()) {
      const char* GET_MULTI_AUTOGRAD_META_TEMPLATE =
          "    std::vector<egr::AutogradMeta*> %s = "
          "egr::EagerUtils::autograd_meta(&%s);\n";
      get_output_autograd_meta_str +=
          paddle::string::Sprintf(GET_MULTI_AUTOGRAD_META_TEMPLATE,
                                  output_autograd_name,
                                  LegalizeVarName(output_name));
    } else {
      // In inplace op, the case where output is duplicable is not considered.
      // Replace output directly with input in inplace op.
      if (!forward_inplace_map.empty() &&
          forward_inplace_map.count(output_name)) {
        auto inplace_input_name =
            LegalizeVarName(forward_inplace_map[output_name]);
        const std::string& inplace_input_autograd_name =
            "p_autograd_" + inplace_input_name;
        const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
            "    %s = egr::EagerUtils::autograd_meta(&%s);\n";
        get_output_autograd_meta_str +=
            paddle::string::Sprintf(GET_SINGLE_AUTOGRAD_META_TEMPLATE,
                                    inplace_input_autograd_name,
                                    inplace_input_name);
      } else {
        const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
            "    egr::AutogradMeta* %s = "
            "egr::EagerUtils::autograd_meta(&%s);\n";
        get_output_autograd_meta_str +=
            paddle::string::Sprintf(GET_SINGLE_AUTOGRAD_META_TEMPLATE,
                                    output_autograd_name,
                                    LegalizeVarName(output_name));
      }
    }
  }
  VLOG(6) << "Generated outputs autograd_meta";

  // input autograd_meta should be got before running TraceOP (for checking
  // inplace).
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    const std::string& input_autograd_name =
        "p_autograd_" + LegalizeVarName(input_name);

    if (input.duplicable()) {
      const char* GET_MULTI_AUTOGRAD_META_TEMPLATE =
          "  std::vector<egr::AutogradMeta*> %s = "
          "egr::EagerUtils::nullable_autograd_meta(%s);\n";
      get_input_autograd_meta_str +=
          paddle::string::Sprintf(GET_MULTI_AUTOGRAD_META_TEMPLATE,
                                  input_autograd_name,
                                  LegalizeVarName(input_name));

    } else if (input.dispensable()) {
      const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
          "  egr::AutogradMeta* %s = "
          "egr::EagerUtils::nullable_autograd_meta(%s);\n";
      get_input_autograd_meta_str +=
          paddle::string::Sprintf(GET_SINGLE_AUTOGRAD_META_TEMPLATE,
                                  input_autograd_name,
                                  LegalizeVarName(input_name));

    } else {
      const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
          "  egr::AutogradMeta* %s = "
          "egr::EagerUtils::nullable_autograd_meta(%s);\n";
      get_input_autograd_meta_str +=
          paddle::string::Sprintf(GET_SINGLE_AUTOGRAD_META_TEMPLATE,
                                  input_autograd_name,
                                  LegalizeVarName(input_name));
    }
  }
  VLOG(6) << "Generated inputs autograd_meta";

  // check inplace input to avoid inplace operations on leaf nodes with
  // stop_gradient=False.
  std::string check_inplace_str = "";
  if (!forward_inplace_map.empty()) {
    const char* CHECKING_INPLACE_TEMPLATE =
        "  // Check Inplace\n"
        "  egr::EagerUtils::CheckInplace(%s, p_autograd_%s, "
        "require_any_grad);\n";
    for (auto& inplace_pair : forward_inplace_map) {
      std::string inplace_name = LegalizeVarName(inplace_pair.second);
      check_inplace_str += paddle::string::Sprintf(
          CHECKING_INPLACE_TEMPLATE, inplace_name, inplace_name);
    }
    VLOG(6) << "Check Inplace Input";
  }

  std::string prepare_autograd_meta_str = "";
  // only generate input autograd_meta in temporary.
  // output autograd_meta will be generated after running TraceOP.
  prepare_autograd_meta_str += get_input_autograd_meta_str;
  prepare_autograd_meta_str += "\n";

  // [GradOpNode] GetTraceBackward
  std::string trace_backward_str =
      "  bool trace_backward = egr::Controller::Instance().HasGrad();\n";
  prepare_autograd_meta_str += trace_backward_str;
  prepare_autograd_meta_str += "\n";

  // [GradOpNode] Generation
  std::string grad_node_creation_str = "";

  size_t bwd_in_slot_num = out_vars.size();
  size_t bwd_out_slot_num = in_vars.size();
  const char* GRAD_OP_NODE_TEMPLATE =
      "      auto grad_node = std::shared_ptr<%sGradNodeCompat>(new "
      "%sGradNodeCompat(%d, "
      "%d)); // NOLINT\n";
  grad_node_creation_str += "    // Create GradOpNode\n";
  grad_node_creation_str += paddle::string::Sprintf(GRAD_OP_NODE_TEMPLATE,
                                                    op_type,
                                                    op_type,
                                                    bwd_in_slot_num,
                                                    bwd_out_slot_num);
  grad_node_creation_str += "\n";

  VLOG(6) << "Generated GradOpNode construction";

  // [GradOpNode] Set Attrs
  grad_node_creation_str += "      // Set Attributes\n";
  grad_node_creation_str += "      grad_node->SetAttrMap(std::move(attrs));\n";
  grad_node_creation_str +=
      "      grad_node->SetDefaultAttrMap(std::move(default_attrs));\n";
  grad_node_creation_str += "\n";

  // [GradOpNode] Set TensorWrappers
  grad_node_creation_str += "      // Set Tensor Wrappers\n";
  for (const auto& iter : op_base_infos) {
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map =
        iter.GetGradInsFwdSlotnameMap();
    for (auto& kv : grad_ins_fwd_slotname_map) {
      const std::string& tensor_wrapper_name = kv.second;
      const char* SET_TENSOR_WRAPPER_TEMPLATE =
          "      grad_node->SetTensorWrapper_%s(%s);\n";
      // Replace output directly with input in inplace op.
      if (!forward_inplace_map.empty() &&
          forward_inplace_map.count(tensor_wrapper_name)) {
        auto inplace_input_name = forward_inplace_map[tensor_wrapper_name];
        grad_node_creation_str +=
            paddle::string::Sprintf(SET_TENSOR_WRAPPER_TEMPLATE,
                                    LegalizeVarName(tensor_wrapper_name),
                                    LegalizeVarName(inplace_input_name));
      } else {
        grad_node_creation_str +=
            paddle::string::Sprintf(SET_TENSOR_WRAPPER_TEMPLATE,
                                    LegalizeVarName(tensor_wrapper_name),
                                    LegalizeVarName(tensor_wrapper_name));
      }
    }
  }
  grad_node_creation_str += "\n";
  VLOG(6) << "Generated SetTensorWrapper";

  // [GradOpNode] SetGradOutMeta
  // [GradOpNode] Add Edges
  std::string compute_require_grad_args = "trace_backward";
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    const std::string& input_autograd_name =
        "p_autograd_" + LegalizeVarName(input_name);

    if (!input.duplicable()) {
      compute_require_grad_args += ", " + input_autograd_name;
      size_t input_position = fwd_inputs_name_pos_map.at(input_name);
      bool found_target_name = false;
      for (const auto& iter : op_base_infos) {
        const auto& grad_outs_slot_map = iter.GetGradOutsSlotnameMap();
        for (auto const& iter : grad_outs_slot_map) {
          if ((!found_target_name) && (input_name == iter.second)) {
            const char* SET_GRAD_OUT_META_TEMPLATE =
                "      grad_node->SetGradOutMeta(%s, %d);\n";
            grad_node_creation_str +=
                paddle::string::Sprintf(SET_GRAD_OUT_META_TEMPLATE,
                                        LegalizeVarName(input_name),
                                        input_position);
            found_target_name = true;
          }
        }
      }
    } else {
      compute_require_grad_args += ", &" + input_autograd_name;
      size_t input_position = fwd_inputs_name_pos_map.at(input_name);
      bool found_target_name = false;
      for (const auto& iter : op_base_infos) {
        const auto& grad_outs_slot_map = iter.GetGradOutsSlotnameMap();
        for (auto const& iter : grad_outs_slot_map) {
          if ((!found_target_name) && (input_name == iter.second)) {
            const char* SET_GRAD_OUT_META_TEMPLATE =
                "      grad_node->SetGradOutMeta(%s, %d);\n";
            grad_node_creation_str +=
                paddle::string::Sprintf(SET_GRAD_OUT_META_TEMPLATE,
                                        LegalizeVarName(input_name),
                                        input_position);
            found_target_name = true;
          }
        }
      }
    }
  }

  // [GradOpNode] SetGradInMeta
  // [AutogradMeta] SetOutRank
  // [AutogradMeta] SetHistory
  std::string pass_stop_gradient_args = "false";
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    // Replace output directly with input in inplace op.
    if (!forward_inplace_map.empty() &&
        forward_inplace_map.count(output_name)) {
      auto inplace_input_name = forward_inplace_map[output_name];
      const std::string& inplace_input_autograd_name =
          "p_autograd_" + LegalizeVarName(inplace_input_name);
      size_t output_position = fwd_outputs_name_pos_map.at(output_name);

      // Intermediate Tensor does not require SetHistory, nor RetainGrad
      pass_stop_gradient_args += ", " + inplace_input_autograd_name;
      const char* SET_OUT_RANK_TEMPLATE =
          "      egr::EagerUtils::SetOutRankWithSlot(%s, %d);\n";
      grad_node_creation_str += paddle::string::Sprintf(
          SET_OUT_RANK_TEMPLATE, inplace_input_autograd_name, output_position);

      // Intermediate Tensor does not require SetHistory
      if (!output.intermediate()) {
        const char* SET_HISTORY_TEMPLATE =
            "      egr::EagerUtils::SetHistory(%s, grad_node);\n";
        grad_node_creation_str += paddle::string::Sprintf(
            SET_HISTORY_TEMPLATE, inplace_input_autograd_name);
      }
      const char* SET_GRAD_IN_META_TEMPLATE =
          "      grad_node->SetGradInMeta(%s, %d);\n";
      grad_node_creation_str +=
          paddle::string::Sprintf(SET_GRAD_IN_META_TEMPLATE,
                                  LegalizeVarName(inplace_input_name),
                                  output_position);
    } else {
      const std::string& output_autograd_name =
          "p_autograd_" + LegalizeVarName(output_name);
      size_t output_position = fwd_outputs_name_pos_map.at(output_name);

      // Intermediate Tensor does not require SetHistory, nor RetainGrad

      if (output.duplicable()) {
        pass_stop_gradient_args += ", &" + output_autograd_name;
        const char* SET_OUT_RANK_TEMPLATE =
            "      egr::EagerUtils::SetOutRankWithSlot(&%s, %d);\n";
        grad_node_creation_str += paddle::string::Sprintf(
            SET_OUT_RANK_TEMPLATE, output_autograd_name, output_position);

        // Intermediate Tensor does not require SetHistory
        if (!output.intermediate()) {
          const char* SET_HISTORY_TEMPLATE =
              "      egr::EagerUtils::SetHistory(&%s, grad_node);\n";
          grad_node_creation_str += paddle::string::Sprintf(
              SET_HISTORY_TEMPLATE, output_autograd_name);
        }
        const char* SET_GRAD_IN_META_TEMPLATE =
            "      grad_node->SetGradInMeta(%s, %d);\n";
        grad_node_creation_str +=
            paddle::string::Sprintf(SET_GRAD_IN_META_TEMPLATE,
                                    LegalizeVarName(output_name),
                                    output_position);

      } else {
        pass_stop_gradient_args += ", " + output_autograd_name;
        const char* SET_OUT_RANK_TEMPLATE =
            "      egr::EagerUtils::SetOutRankWithSlot(%s, %d);\n";
        grad_node_creation_str += paddle::string::Sprintf(
            SET_OUT_RANK_TEMPLATE, output_autograd_name, output_position);

        // Intermediate Tensor does not require SetHistory
        if (!output.intermediate()) {
          const char* SET_HISTORY_TEMPLATE =
              "      egr::EagerUtils::SetHistory(%s, grad_node);\n";
          grad_node_creation_str += paddle::string::Sprintf(
              SET_HISTORY_TEMPLATE, output_autograd_name);
        }
        const char* SET_GRAD_IN_META_TEMPLATE =
            "      grad_node->SetGradInMeta(%s, %d);\n";
        grad_node_creation_str +=
            paddle::string::Sprintf(SET_GRAD_IN_META_TEMPLATE,
                                    LegalizeVarName(output_name),
                                    output_position);
      }
    }
  }
  VLOG(6) << "Generated SetGradIn/OutMeta";

  // [Generation] GradNode Creation
  // After getting require_any_grad, firstly use CheckInplace method for inplace
  // op.
  // Then execute TraceOp and generate output autograd_meta.
  // Finally, Construct GradNode. (Replace output directly with input in inplace
  // op.)
  // Add event record
  std::string event_name = op_type + " node_creation";
  const char* GRAD_NODE_CREATION_TEMPLATE =
      "%s"
      "  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(%s);\n"
      "%s\n"
      "%s"
      "  {\n"
      "    phi::RecordEvent node_creation_record_event(\"%s\", "
      "phi::TracerEventType::OperatorInner, 1);\n"
      "%s"
      "    if(require_any_grad) {\n"
      "      VLOG(6) << \" Construct Grad for %s \";\n"
      "      egr::EagerUtils::PassStopGradient(%s);\n"
      "  %s\n"
      "    }\n"
      "  }";
  std::string grad_node_creation_body_str =
      paddle::string::Sprintf(GRAD_NODE_CREATION_TEMPLATE,
                              prepare_autograd_meta_str,
                              compute_require_grad_args,
                              check_inplace_str,
                              trace_op_body_str,
                              event_name,
                              get_output_autograd_meta_str,
                              op_type,
                              pass_stop_gradient_args,
                              grad_node_creation_str);

  return grad_node_creation_body_str;
}

/* -------------------------------- */
/* --------- CodeGen: Forward ----- */
/* -------------------------------- */
static std::pair<std::string, std::string> GenerateForwardFunctionContents(
    const ForwardGenerationInfo& fwd_info,
    const GradNodeGenerationInfo& bwd_info,
    std::map<std::string, std::string> forward_inplace_map = {}) {
  /* --- Process Forward Info ---*/
  const std::string& op_type = fwd_info.GetOpType();
  const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map =
      fwd_info.GetFwdInputsNamePosMap();
  const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map =
      fwd_info.GetFwdOutputsNamePosMap();
  const std::vector<proto::OpProto::Var>& in_vars = fwd_info.GetInVars();
  const std::vector<proto::OpProto::Var>& out_vars = fwd_info.GetOutVars();

  /*
    // Forward Function Example:
  std::tuple<vector<Tensor>, Tensor, vector<Tensor>>
  kernel_function(vector<Tensor>& X, Tensor& Y, const paddle::AttributeMap&
  attr_map, size_t
  Out0Num, size_t Out1Num) {

        // Forward Function Body
        // According to fwd_inputs_name_pos_map
        std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>>
  ins =
                { {"X" , TrySyncToVars(X)}, { "Y" , TrySyncToVars(Y)} };

        std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>>
  outs =
  {
          {"Out0" , CreateVars(Out0Num)}, {"Out1"
  ,CreateVars(Out1Num)} };

        // According to op_proto->attrs()

        Controller.Instance().GetCurrentTracer()->TraceOp("op_type", ins, outs,
  attr_map,
  Controller.Instance().GetExpectedPlace(), {});

        // According to fwd_outputs_names
        std::vector<paddle::Tensor> Out0 =
  GetOutputs(outs["Out0"]);
        paddle::Tensor Out1 = GetOutputs(outs["Out1"][0]);
        std::vector<paddle::Tensor> Out2 =
  GetOutputs(outs["Out2"]);

        // Grad Node Generation Codes
        ...

        return std::make_tuple(Out0, Out1, Out2);
    }
  */
  VLOG(6) << "Generating Dygraph Forward Function";

  const char* FORWARD_FUNCTION_TEMPLATE =
      "  VLOG(3) << \"Running Eager Forward Op: %s\";\n";
  std::string generated_function_body =
      paddle::string::Sprintf(FORWARD_FUNCTION_TEMPLATE, op_type);

  std::string dygraph_function_args_str = "";
  std::string amp_function_call_args_str = "";
  core_ops_legacy_args_info[op_type] = {};
  core_ops_legacy_args_type_info[op_type] = {};
  core_ops_legacy_args_info[op_type].resize(in_vars.size());
  core_ops_legacy_args_type_info[op_type].resize(in_vars.size());

  /* ------ Dygraph forward function generation ------ */
  generated_function_body += "  // Dygraph Forward Pass\n";
  generated_function_body += "\n";

  // [Generation] Get Ins Map
  std::string ins_contents_str = "";
  std::vector<std::string> input_args_str_list(in_vars.size());
  std::vector<std::string> amp_function_call_args_str_list(in_vars.size());
  std::string amp_tensors_vector_str = "";
  std::string amp_auto_cast_str = "";
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    size_t input_position = fwd_inputs_name_pos_map.at(input_name);

    if (input.duplicable()) {
      const char* FWD_INS_ARG_TEMPLATE =
          "const std::vector<paddle::Tensor>& %s";
      input_args_str_list[input_position] = paddle::string::Sprintf(
          FWD_INS_ARG_TEMPLATE, LegalizeVarName(input_name));
      amp_function_call_args_str_list[input_position] =
          " NEW_" + LegalizeVarName(input_name);

      core_ops_legacy_args_type_info[op_type][input_position] = "list";
    } else {
      // inplace tensor can't be const
      const char* FWD_INS_ARG_TEMPLATE;
      bool flag_find_input_name = false;
      if (!forward_inplace_map.empty()) {
        for (auto& inplace_pair : forward_inplace_map) {
          if (inplace_pair.second == input_name) {
            flag_find_input_name = true;
            FWD_INS_ARG_TEMPLATE = "paddle::Tensor& %s";
            break;
          }
        }
      }
      if (!flag_find_input_name) {
        FWD_INS_ARG_TEMPLATE = "const paddle::Tensor& %s";
      }
      input_args_str_list[input_position] = paddle::string::Sprintf(
          FWD_INS_ARG_TEMPLATE, LegalizeVarName(input_name));
      amp_function_call_args_str_list[input_position] =
          " NEW_" + LegalizeVarName(input_name);

      core_ops_legacy_args_type_info[op_type][input_position] = "tensor";
    }
    core_ops_legacy_args_info[op_type][input_position] = input_name;

    if (input.dispensable()) continue;

    const char* FWD_INS_CONTENT_TEMPLATE =
        "{ \"%s\", egr::EagerUtils::TrySyncToVars(%s) },";
    ins_contents_str += paddle::string::Sprintf(
        FWD_INS_CONTENT_TEMPLATE, input_name, LegalizeVarName(input_name));
    if (input.duplicable()) {
      const char* AMP_TENSORS_VECTOR_TEMPLATE = "%s,";
      amp_tensors_vector_str +=
          paddle::string::Sprintf(AMP_TENSORS_VECTOR_TEMPLATE, input_name);
      const char* AMP_AUTO_CAST_TEMPLATE =
          "    auto NEW_%s = egr::AmpAutoCasts(\"%s\", %s, amp_dst_dtype, "
          "\"%s\");\n";
      amp_auto_cast_str += paddle::string::Sprintf(AMP_AUTO_CAST_TEMPLATE,
                                                   LegalizeVarName(input_name),
                                                   input_name,
                                                   LegalizeVarName(input_name),
                                                   op_type);
    } else {
      const char* AMP_TENSORS_VECTOR_TEMPLATE = "{%s},";
      amp_tensors_vector_str += paddle::string::Sprintf(
          AMP_TENSORS_VECTOR_TEMPLATE, LegalizeVarName(input_name));
      const char* AMP_AUTO_CAST_TEMPLATE =
          "    auto NEW_%s = egr::AmpAutoCast(\"%s\", %s, amp_dst_dtype, "
          "\"%s\");\n";
      amp_auto_cast_str += paddle::string::Sprintf(AMP_AUTO_CAST_TEMPLATE,
                                                   LegalizeVarName(input_name),
                                                   input_name,
                                                   LegalizeVarName(input_name),
                                                   op_type);
    }
  }
  if (!ins_contents_str.empty())
    ins_contents_str.pop_back();  // // Remove trailing ","

  if (!amp_tensors_vector_str.empty()) amp_tensors_vector_str.pop_back();

  for (const std::string& arg : input_args_str_list) {
    dygraph_function_args_str += arg;
    dygraph_function_args_str += ",";
  }
  if (!dygraph_function_args_str.empty()) dygraph_function_args_str.pop_back();

  for (const std::string& arg : amp_function_call_args_str_list) {
    amp_function_call_args_str += arg;
    amp_function_call_args_str += ",";
  }
  if (!amp_function_call_args_str.empty())
    amp_function_call_args_str.pop_back();

  // Handle Dispensable Inputs
  std::string dispensable_ins_contents_str = "";
  std::string dispensable_amp_tensors_vector_str = "";
  std::string dispensable_amp_auto_cast_str = "";
  std::set<std::string> input_names;
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    input_names.insert(input_name);
    if (input.dispensable()) {
      if (input.duplicable()) {
        const char* FWD_INS_CONTENT_TEMPLATE =
            "  if(%s.size() > 0) "
            "ins[\"%s\"] = egr::EagerUtils::TrySyncToVars(%s);\n";
        dispensable_ins_contents_str +=
            paddle::string::Sprintf(FWD_INS_CONTENT_TEMPLATE,
                                    LegalizeVarName(input_name),
                                    input_name,
                                    LegalizeVarName(input_name));
        const char* FWD_AMP_TENSORS_VECTOR_TEMPLATE =
            "    if(%s.size() > 0) "
            "amp_tensors_vector.push_back(%s);\n";
        dispensable_amp_tensors_vector_str +=
            paddle::string::Sprintf(FWD_AMP_TENSORS_VECTOR_TEMPLATE,
                                    LegalizeVarName(input_name),
                                    LegalizeVarName(input_name));
        const char* DISPENSABLE_AMP_AUTO_CAST_TEMPLATE =
            "    auto NEW_%s = ((%s.size() > 0) ? egr::AmpAutoCasts(\"%s\", "
            "%s, amp_dst_dtype, \"%s\") : %s);\n";
        dispensable_amp_auto_cast_str +=
            paddle::string::Sprintf(DISPENSABLE_AMP_AUTO_CAST_TEMPLATE,
                                    LegalizeVarName(input_name),
                                    LegalizeVarName(input_name),
                                    input_name,
                                    LegalizeVarName(input_name),
                                    op_type,
                                    LegalizeVarName(input_name));
      } else {
        const char* FWD_INS_CONTENT_TEMPLATE =
            "  if(%s.has_allocation()) "
            "ins[\"%s\"] = egr::EagerUtils::TrySyncToVars(%s);\n";
        dispensable_ins_contents_str +=
            paddle::string::Sprintf(FWD_INS_CONTENT_TEMPLATE,
                                    LegalizeVarName(input_name),
                                    input_name,
                                    LegalizeVarName(input_name));
        const char* FWD_AMP_TENSORS_VECTOR_TEMPLATE =
            "    if(%s.has_allocation()) "
            "amp_tensors_vector.push_back({ %s });\n";
        dispensable_amp_tensors_vector_str +=
            paddle::string::Sprintf(FWD_AMP_TENSORS_VECTOR_TEMPLATE,
                                    LegalizeVarName(input_name),
                                    LegalizeVarName(input_name));
        const char* DISPENSABLE_AMP_AUTO_CAST_TEMPLATE =
            "    auto NEW_%s = ((%s.has_allocation()) ? "
            "egr::AmpAutoCast(\"%s\", "
            "%s, amp_dst_dtype, \"%s\") : %s);\n";
        dispensable_amp_auto_cast_str +=
            paddle::string::Sprintf(DISPENSABLE_AMP_AUTO_CAST_TEMPLATE,
                                    LegalizeVarName(input_name),
                                    LegalizeVarName(input_name),
                                    input_name,
                                    LegalizeVarName(input_name),
                                    op_type,
                                    LegalizeVarName(input_name));
      }
    }
  }

  VLOG(6) << "Generated Ins Map";

  // [Generation] Get Outs Map
  std::string outs_contents_str = "";
  std::string inplace_mapping_str = "";
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    std::string outnum = "1";
    if (op_passing_outs_map[op_type].count(output_name)) {
      const std::string output_var_name = output_name + "Var";

      // Pass Output from function
      // argument(EagerVariable*/vector<EagerVariable*>&),
      // in form of shared_ptr<EagerVariable>/vector<shared_ptr<EagerVariable>>
      if (output.duplicable()) {
        const char* FWD_NUM_ARG_TEMPLATE = ", std::vector<paddle::Tensor*>& %s";
        std::string arg_str = paddle::string::Sprintf(
            FWD_NUM_ARG_TEMPLATE, LegalizeVarName(output_var_name));
        dygraph_function_args_str += arg_str;
        amp_function_call_args_str += (", " + LegalizeVarName(output_var_name));

        core_ops_legacy_args_type_info[op_type].push_back("list");
      } else {
        const char* FWD_NUM_ARG_TEMPLATE = ", paddle::Tensor* %s";
        std::string arg_str = paddle::string::Sprintf(
            FWD_NUM_ARG_TEMPLATE, LegalizeVarName(output_var_name));
        dygraph_function_args_str += arg_str;
        amp_function_call_args_str += (", " + LegalizeVarName(output_var_name));

        core_ops_legacy_args_type_info[op_type].push_back("tensor");
      }

      if (BeSameAsInput(output_name, input_names)) {
        if (!output.dispensable()) {
          std::string input_name =
              output_name.substr(0, output_name.size() - 3);
          const char* FWD_OUTS_CONTENT_TEMPLATE = R"({ "%s", ins["%s"] },)";
          outs_contents_str += paddle::string::Sprintf(
              FWD_OUTS_CONTENT_TEMPLATE, output_name, input_name);
        }
      } else {
        const char* FWD_OUTS_CONTENT_TEMPLATE =
            "{ \"%s\", egr::EagerUtils::TrySyncToVars(%s) },";
        outs_contents_str +=
            paddle::string::Sprintf(FWD_OUTS_CONTENT_TEMPLATE,
                                    output_name,
                                    LegalizeVarName(output_var_name));
      }
      core_ops_legacy_args_info[op_type].push_back(output_name);

    } else if (!forward_inplace_map.empty() &&
               forward_inplace_map.count(output_name)) {
      // In inplace op, replace the output with the input directly.
      PADDLE_ENFORCE_NE(
          forward_inplace_map[output_name],
          "",
          common::errors::InvalidArgument(
              "Inplace op %s has no input corresponding to output %s.",
              op_type,
              output_name));
      const char* FWD_OUTS_CONTENT_TEMPLATE = R"({ "%s", ins["%s"] },)";
      auto inplace_input_name = forward_inplace_map[output_name];
      outs_contents_str += paddle::string::Sprintf(
          FWD_OUTS_CONTENT_TEMPLATE, output_name, inplace_input_name);

      // inplace_map used in TraceOp.
      const char* INPLACE_MAPPING_TEMPLATE = R"({"%s", "%s"},)";
      inplace_mapping_str += paddle::string::Sprintf(
          INPLACE_MAPPING_TEMPLATE, inplace_input_name, output_name);
    } else {
      if (output.duplicable()) {
        outnum = output_name + "Num";

        const char* FWD_NUM_ARG_TEMPLATE = ", size_t %s";
        std::string arg_str =
            paddle::string::Sprintf(FWD_NUM_ARG_TEMPLATE, outnum);
        dygraph_function_args_str += arg_str;
        amp_function_call_args_str += (", " + outnum);
        const char* FWD_OUTS_CONTENT_TEMPLATE =
            "{ \"%s\", egr::EagerUtils::CreateVars(%s) },";
        outs_contents_str += paddle::string::Sprintf(
            FWD_OUTS_CONTENT_TEMPLATE, output_name, outnum);
        core_ops_legacy_args_info[op_type].push_back(outnum);
        core_ops_legacy_args_type_info[op_type].push_back("int");
      } else {
        const char* FWD_OUTS_CONTENT_TEMPLATE =
            "{ \"%s\", "
            "{std::make_shared<egr::EagerVariable>(egr::Controller::Instance()."
            "GenerateUniqueName())}},";
        outs_contents_str +=
            paddle::string::Sprintf(FWD_OUTS_CONTENT_TEMPLATE, output_name);
      }
    }
  }
  if (!outs_contents_str.empty())
    outs_contents_str.pop_back();  // Remove trailing ","
  if (!inplace_mapping_str.empty())
    inplace_mapping_str.pop_back();  // Remove trailing ","

  if ((op_type != "cast") && (forward_inplace_map.empty())) {
    VLOG(6) << "Generating Dygraph Forward AMP";
    const char* AMP_LOGIC_CONTEXT =
        "  if (egr::Controller::Instance().GetAMPLevel() != "
        "paddle::imperative::AmpLevel::O0) {\n"
        "    VLOG(5) << \"Check and Prepare For AMP\";\n"
        " \n"
        "%s\n"
        "  }\n";
    std::string amp_logic_str = "";
    if (!in_vars.empty()) {
      const char* AMP_TENSORS_VECTOR_TEMPLATE =
          "    paddle::small_vector<std::vector<paddle::Tensor>, "
          "egr::kSlotSmallVectorSize> "
          "amp_tensors_vector = { "
          "%s };\n";
      std::string amp_tensors_vector = paddle::string::Sprintf(
          AMP_TENSORS_VECTOR_TEMPLATE, amp_tensors_vector_str);
      amp_tensors_vector += dispensable_amp_tensors_vector_str;
      amp_logic_str += amp_tensors_vector;
      amp_logic_str += "\n";
      const char* GET_AMP_GET_DST_DTYPE_CONTEXT =
          "    auto amp_dst_dtype = "
          "paddle::imperative::GetAmpDestDtype(\"%s\", "
          "amp_tensors_vector);\n";
      amp_logic_str +=
          paddle::string::Sprintf(GET_AMP_GET_DST_DTYPE_CONTEXT, op_type);
      amp_logic_str += "\n";
      amp_logic_str += amp_auto_cast_str;
      amp_logic_str += dispensable_amp_auto_cast_str;
      amp_logic_str += "\n";
    }
    const char* CALL_BACK_TEMPLATE =
        "    {\n"
        "      paddle::imperative::AutoCastGuard "
        "guard(egr::Controller::Instance().GetCurrentAmpAttrs(), "
        "paddle::imperative::AmpLevel::O0);\n"
        "      return %s_dygraph_function(%s);\n"
        "    }";
    amp_function_call_args_str += ", attr_map ";
    if (!amp_function_call_args_str.empty()) {
      auto iter = amp_function_call_args_str.begin();
      if ((*iter) == ',') amp_function_call_args_str.erase(iter);
    }
    std::string call_back_str = paddle::string::Sprintf(
        CALL_BACK_TEMPLATE, op_type, amp_function_call_args_str);
    amp_logic_str += call_back_str;
    amp_logic_str += "\n";
    std::string amp_context =
        paddle::string::Sprintf(AMP_LOGIC_CONTEXT, amp_logic_str);
    generated_function_body += amp_context;
    generated_function_body += "\n";
  }

  if (!forward_inplace_map.empty()) {
    generated_function_body +=
        "  auto current_level = egr::Controller::Instance().GetAMPLevel();\n";
    generated_function_body +=
        "  "
        "egr::Controller::Instance().SetAMPLevel(paddle::imperative::AmpLevel::"
        "O0);\n";
  }
  // forward ins insert
  const char* FWD_INS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerVariable>>> ins = { "
      "%s };\n";
  std::string ins_map_str =
      paddle::string::Sprintf(FWD_INS_MAP_TEMPLATE, ins_contents_str);
  ins_map_str += dispensable_ins_contents_str;
  generated_function_body += ins_map_str;
  generated_function_body += "\n";
  // forward outs insert
  const char* FWD_OUTS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerVariable>>> outs = { "
      "%s };\n";
  std::string outs_map_str =
      paddle::string::Sprintf(FWD_OUTS_MAP_TEMPLATE, outs_contents_str);
  generated_function_body += outs_map_str;
  generated_function_body += "\n";

  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    if (op_passing_outs_map[op_type].count(output_name)) {
      if (BeSameAsInput(output_name, input_names)) {
        if (output.dispensable()) {
          std::string input_name =
              output_name.substr(0, output_name.size() - 3);
          const char* FWD_OUTS_CONTENT_TEMPLATE =
              "  if (ins.count(\"%s\")) outs[\"%s\"] = ins[\"%s\"];\n";
          generated_function_body += paddle::string::Sprintf(
              FWD_OUTS_CONTENT_TEMPLATE, input_name, output_name, input_name);
        }
      }
    }
  }

  VLOG(6) << "Generated Outs Map";

  // [Generation] Apply View Strategy (Tensor)
  if (forward_inplace_map.empty() && view_op_map.count(op_type)) {
    const char* HANDLE_VIEW_BETWEEN_INPUT_AND_OUTPUT =
        "  if (ins.count(\"%s\") && outs.count(\"%s\")) {\n"
        "    egr::EagerUtils::HandleViewBetweenInputAndOutput(ins[\"%s\"][0], "
        "outs[\"%s\"][0]);\n"
        "  };\n";

    std::string view_strategy_str = "";
    std::string view_input_name = view_op_map[op_type].first;
    std::string view_output_name = view_op_map[op_type].second;
    view_strategy_str +=
        paddle::string::Sprintf(HANDLE_VIEW_BETWEEN_INPUT_AND_OUTPUT,
                                view_input_name,
                                view_output_name,
                                view_input_name,
                                view_output_name);

    generated_function_body += view_strategy_str;
    generated_function_body += "\n";

    VLOG(6) << "Generated View Strategy";
  }
  generated_function_body += "\n";

  // [Generation] Get Attrs
  dygraph_function_args_str +=
      ", const paddle::framework::AttributeMap& attr_map";

  /* --------- Generate TraceOp ----- */
  // TraceOp should be run after compute require_any_grad. (for checking
  // inplace)
  // `trace_op_body_str` will be passed as a parameter to
  // `GenerateGradNodeCreationContent`.
  std::string trace_op_body_str = "";
  // [Generation] Get TraceOp
  const char* FWD_TRACE_OP_TEMPLATE =
      "  paddle::framework::AttributeMap attrs = attr_map;\n"
      "  paddle::framework::AttributeMap default_attrs;\n"
      "  egr::Controller::Instance().GetCurrentTracer()->TraceOp(\"%s\", ins, "
      "outs, attrs,\n"
      "     egr::Controller::Instance().GetExpectedPlace(),\n"
      "     &default_attrs, true, {%s});\n";
  std::string trace_op_str = paddle::string::Sprintf(
      FWD_TRACE_OP_TEMPLATE, op_type, inplace_mapping_str);

  trace_op_body_str += trace_op_str;
  trace_op_body_str += "\n";

  // [Generation] Log memory information
  const char* LOG_MEMORY_INFO_TEMPLATE =
      " // Log memory information\n"
      "  "
      "paddle::memory::LogDeviceMemoryStats(egr::Controller::Instance()."
      "GetExpectedPlace(), \"%s\");\n";
  std::string log_memory_info_str =
      paddle::string::Sprintf(LOG_MEMORY_INFO_TEMPLATE, op_type);

  trace_op_body_str += log_memory_info_str;
  trace_op_body_str += "\n";

  VLOG(6) << "Generated AttrMap & TraceOp";

  // [Generation] Convert output VarBase to Vector/Tensor
  size_t output_size = out_vars.size();
  std::vector<std::string> return_contents(output_size);
  std::vector<std::string> return_types(output_size);
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    const std::string output_var_args_name =
        LegalizeVariableName(output_name + "Var");
    std::string out_tensor_str;
    size_t return_position = fwd_outputs_name_pos_map.at(output_name);
    std::string output_varname = LegalizeVariableName(output_name);

    if (output.duplicable()) {
      if (op_passing_outs_map[op_type].count(output_name)) {
        if (output.dispensable()) {
          const char* FWD_OUT_TENSORS_TEMPLATE =
              "  std::vector<paddle::Tensor> %s;\n"
              "  if (outs.count(\"%s\"))  "
              "egr::EagerUtils::GetOutputs(outs[\"%s\"], %s);\n"
              "  egr::EagerUtils::Output2Result(%s, &%s);\n";
          out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSORS_TEMPLATE,
                                                   output_varname,
                                                   output_name,
                                                   output_name,
                                                   output_var_args_name,
                                                   output_var_args_name,
                                                   output_varname);
        } else {
          const char* FWD_OUT_TENSORS_TEMPLATE =
              "  std::vector<paddle::Tensor> %s;\n"
              "  egr::EagerUtils::GetOutputs(outs[\"%s\"], %s);\n"
              "  egr::EagerUtils::Output2Result(%s, &%s);\n";
          out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSORS_TEMPLATE,
                                                   output_varname,
                                                   output_name,
                                                   output_var_args_name,
                                                   output_var_args_name,
                                                   output_varname);
        }
      } else {
        const char* FWD_OUT_TENSORS_TEMPLATE =
            "  std::vector<paddle::Tensor> %s;\n"
            "  egr::EagerUtils::GetOutputs(outs[\"%s\"], &%s);\n";
        out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSORS_TEMPLATE,
                                                 output_varname,
                                                 output_name,
                                                 output_varname);
      }
      return_types[return_position] = "std::vector<paddle::Tensor>";
    } else {
      if (op_passing_outs_map[op_type].count(output_name)) {
        if (output.dispensable()) {
          const char* FWD_OUT_TENSOR_TEMPLATE =
              "  if (outs.count(\"%s\"))  "
              "egr::EagerUtils::GetOutput(outs[\"%s\"][0], %s);\n"
              "  paddle::Tensor& %s = *%s;\n";
          out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSOR_TEMPLATE,
                                                   output_name,
                                                   output_name,
                                                   output_var_args_name,
                                                   output_varname,
                                                   output_var_args_name);
        } else {
          const char* FWD_OUT_TENSOR_TEMPLATE =
              "  egr::EagerUtils::GetOutput(outs[\"%s\"][0], %s);\n"
              "  paddle::Tensor& %s = *%s;\n"
              "  (void)%s; // To avoid error: unused variable\n";
          out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSOR_TEMPLATE,
                                                   output_name,
                                                   output_var_args_name,
                                                   output_varname,
                                                   output_var_args_name,
                                                   output_varname);
        }
      } else {
        if (!forward_inplace_map.empty() &&
            forward_inplace_map.count(output_name)) {
          // Modify meta info of inplace tensor.
          // Bump inplace version of inplace tensor.
          auto inplace_input_name = forward_inplace_map[output_name];
          const char* FWD_OUT_TENSOR_TEMPLATE =
              "  egr::EagerUtils::GetOutput(outs[\"%s\"][0], &%s);\n"
              "  %s.bump_inplace_version();\n"
              "  VLOG(3) << \"Tensor(\" << %s.name() << \") uses Inplace "
              "Strategy.\";\n";
          out_tensor_str =
              paddle::string::Sprintf(FWD_OUT_TENSOR_TEMPLATE,
                                      output_name,
                                      LegalizeVarName(inplace_input_name),
                                      LegalizeVarName(inplace_input_name),
                                      LegalizeVarName(inplace_input_name));
        } else {
          const char* FWD_OUT_TENSOR_TEMPLATE =
              "  paddle::Tensor %s;\n"
              "  egr::EagerUtils::GetOutput(outs[\"%s\"][0], &%s);\n";
          out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSOR_TEMPLATE,
                                                   output_varname,
                                                   output_name,
                                                   output_varname);
        }
      }
      return_types[return_position] = "paddle::Tensor";
    }

    if (!forward_inplace_map.empty() &&
        forward_inplace_map.count(output_name)) {
      // Replace output directly with input in inplace op.
      return_contents[return_position] =
          LegalizeVarName(forward_inplace_map[output_name]);
    } else {
      return_contents[return_position] = output_varname;
    }
    trace_op_body_str += out_tensor_str;
  }
  if (!forward_inplace_map.empty()) {
    trace_op_body_str +=
        "  egr::Controller::Instance().SetAMPLevel(current_level);\n";
  }
  trace_op_body_str += "\n";
  VLOG(6) << "Converted Output VarBase to EagerVariable(s)";
  /* ------ END Generate TraceOp ----- */

  // [Generation] Handle core_ops_legacy_returns_info
  // avoid inplace op changing core_ops_legacy_returns_info
  if (core_ops_legacy_returns_info.empty() ||
      !core_ops_legacy_returns_info.count(op_type)) {
    core_ops_legacy_returns_info[op_type] = return_contents;
  }

  // [Generation] ComputeRequireGrad -> GradNodeCreation

  if (!bwd_info.GenerateForwardOnly()) {
    // If GradNode needs to be generated, pass `trace_op_body_str`
    // into `GenerateGradNodeCreationContent`.
    std::string grad_node_creation_body_str = GenerateGradNodeCreationContent(
        fwd_info, bwd_info, trace_op_body_str, forward_inplace_map);

    generated_function_body += grad_node_creation_body_str;
    generated_function_body += "\n";

    // [Generation] Call RetainGradForTensor
    VLOG(6) << "Generated GradNode Creation codes";
  } else {
    // If GradNode doesn't need to be generated, generate TraceOP directly.
    generated_function_body += trace_op_body_str;
  }

  // [Generation] Handle return: Tuple/Vector/Tensor
  generated_function_body += "\n";
  std::string return_str = "";
  std::string return_type_str = "";
  std::string function_proto_return_type_str = "";
  if (return_contents.size() > 1) {
    // Return tuple
    std::string return_content_str = "";
    for (const std::string& s : return_contents) {
      return_content_str += s + ",";
    }
    return_content_str.pop_back();  // Remove trailing ","

    for (const std::string& s : return_types) {
      return_type_str += s + ",";
    }
    return_type_str.pop_back();  // Remove trailing ","

    const char* FWD_TUPLE_RETURN_TEMPLATE = "  return std::make_tuple(%s);";
    return_str =
        paddle::string::Sprintf(FWD_TUPLE_RETURN_TEMPLATE, return_content_str);

    const char* FWD_FUNCTION_PROTO_RETURN_TEMPLATE = "std::tuple<%s>";
    function_proto_return_type_str = paddle::string::Sprintf(
        FWD_FUNCTION_PROTO_RETURN_TEMPLATE, return_type_str);

  } else if (return_contents.size() == 1) {
    // Return vector<Tensor> or Tensor
    return_type_str = return_types[0];
    const char* FWD_TENSOR_RETURN_TEMPLATE = "  return %s;";
    return_str =
        paddle::string::Sprintf(FWD_TENSOR_RETURN_TEMPLATE, return_contents[0]);
    function_proto_return_type_str = return_type_str;

  } else {
    return_str = "return nullptr;";
    function_proto_return_type_str = "void*";
  }

  generated_function_body += return_str;
  generated_function_body += "\n";
  VLOG(6) << "Generated return codes";

  // [Generation] Get Full Function
  std::string function_name;
  if (forward_inplace_map.empty()) {
    function_name = op_type + "_dygraph_function";
  } else {
    // change function_name for inplace op.
    function_name = op_type + "__dygraph_function";
  }

  if (!dygraph_function_args_str.empty()) {
    auto iter = dygraph_function_args_str.begin();
    if ((*iter) == ',') dygraph_function_args_str.erase(iter);
  }

  const char* DYGRAPH_FUNCTION_EVENT_RECORD_FUNCTION_TEMPLATE =
      "  phi::RecordEvent dygraph_entrance_record_event(\"%s\", "
      "phi::TracerEventType::Operator, 1);";
  std::string event_name = op_type + " dygraph";
  std::string fwd_record_event_str = paddle::string::Sprintf(
      DYGRAPH_FUNCTION_EVENT_RECORD_FUNCTION_TEMPLATE, event_name);
  const char* FWD_FUNCTION_TEMPLATE =
      "TEST_API %s %s(%s) {\n\n"
      "%s\n"
      "%s\n"
      "}\n\n";
  std::string fwd_function_str =
      paddle::string::Sprintf(FWD_FUNCTION_TEMPLATE,
                              function_proto_return_type_str,
                              function_name,
                              dygraph_function_args_str,
                              fwd_record_event_str,
                              generated_function_body);

  // [Generation] Generate forward functions header
  const char* FWD_HEADER_TEMPLATE = "TEST_API %s %s(%s);\n";
  std::string dygraph_function_declaration_str =
      paddle::string::Sprintf(FWD_HEADER_TEMPLATE,
                              function_proto_return_type_str,
                              function_name,
                              dygraph_function_args_str);

  return {fwd_function_str, dygraph_function_declaration_str};
}

static std::string GenerateSingleOpBase(
    const std::string& fwd_op_type,
    const std::string& op_base_type,
    const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map,
    const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map,
    const std::vector<proto::OpProto::Var>& in_vars,
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map,
    const std::map<std::string, std::string>& grad_ins_grad_slotname_map,
    const std::map<std::string, std::string>& grad_outs_slotname_map,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_ins,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_outs,
    const paddle::framework::AttributeMap& grad_attrs,
    const std::unordered_map<std::string, std::string>& backward_inplace_map,
    bool is_op_base_per_duplicable_input,
    size_t* outs_size) {
  std::string generated_grad_function_body = "";

  const std::string& ins_name = "ins" + std::to_string(*outs_size);
  const std::string& outs_name = "outs" + std::to_string(*outs_size);
  const std::string& attrs_name = "attrs_map" + std::to_string(*outs_size);
  const std::string& hooked_grads = "hooked_grads" + std::to_string(*outs_size);

  // [Generation] Get Full Zero
  std::string fill_zero_str = "";
  if (ops_to_fill_zero_for_empty_grads.count(fwd_op_type)) {
    for (auto const& iter : grad_ins) {
      const std::string& grad_input_name = iter.first;
      if (grad_ins_grad_slotname_map.count(grad_input_name)) {
        size_t fwd_output_position = fwd_outputs_name_pos_map.at(
            grad_ins_grad_slotname_map.at(grad_input_name));
        const char* FILL_ZERO_TEMPLATE =
            "  egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[%d], "
            "this->InputMeta()[%d]);\n";
        fill_zero_str += paddle::string::Sprintf(
            FILL_ZERO_TEMPLATE, fwd_output_position, fwd_output_position);
      }
    }
  }
  generated_grad_function_body += fill_zero_str;
  generated_grad_function_body +=
      "  paddle::small_vector<std::vector<paddle::Tensor>, "
      "egr::kSlotSmallVectorSize> " +
      hooked_grads + " = " + fwd_op_type +
      "GradNodeCompat::ApplyGradientHooks(grads);\n";

  // [Generation] Get Ins Map
  std::unordered_set<std::string> dispensable_input_name_set;
  for (const auto& in : in_vars) {
    if (in.dispensable()) dispensable_input_name_set.insert(in.name());
  }
  std::unordered_set<std::string> duplicable_input_name_set;
  for (const auto& in : in_vars) {
    if (in.duplicable()) duplicable_input_name_set.insert(in.name());
  }
  const char* CHECK_BACKWARD_INPLACE_TEMPLATE =
      "  // Check backward inplace info\n"
      "  bool %s = false;\n"
      "  %s\n"
      "  if (%s.has_allocation()) {\n"
      "    VLOG(10) << %s.name() << \"(%s) use_count: \" << "
      "%s.impl().use_count();\n"
      "    if (%s.impl().use_count() == 1 || (%s.impl().use_count() == 2 && "
      "%s.impl().get() == %s.impl().get())) {\n"
      "      %s = true;\n"
      "    }\n"
      "  }\n";
  const std::string& can_be_inplaced_name =
      "can_be_inplaced" + std::to_string(*outs_size);
  const std::string& bwd_inplace_input_name =
      "backward_inplace_tensor" + std::to_string(*outs_size);
  bool process_backward_inplace = false;
  std::string ins_contents_str = "";
  for (auto const& iter : grad_ins) {
    const std::string& grad_input_name = iter.first;

    if (grad_ins_fwd_slotname_map.count(grad_input_name)) {
      // Fwd Tensor
      const std::string& fwd_name =
          grad_ins_fwd_slotname_map.at(grad_input_name);
      if (dispensable_input_name_set.count(fwd_name)) {
        continue;
      }
      std::string struct_fwd_input_name =
          grad_ins_fwd_slotname_map.at(grad_input_name) + "_";
      const char* GRAD_INS_FWD_CONTENT_TEMPLATE =
          "{ \"%s\", "
          "egr::EagerUtils::TrySyncToVars(egr::EagerUtils::"
          "RecoverTensorWrapper("
          "&"
          "this->%s)) },";
      ins_contents_str += paddle::string::Sprintf(GRAD_INS_FWD_CONTENT_TEMPLATE,
                                                  grad_input_name,
                                                  struct_fwd_input_name);
      if (!backward_inplace_map.empty() &&
          backward_inplace_map.count(grad_input_name)) {
        process_backward_inplace = true;
        const char* GRAD_INS_FWD_TENSOR_WRAPPER_TEMPLATE =
            "auto %s = egr::EagerUtils::RecoverTensorWrapper(&this->%s);";
        std::string tensor_wrapper_str =
            paddle::string::Sprintf(GRAD_INS_FWD_TENSOR_WRAPPER_TEMPLATE,
                                    bwd_inplace_input_name,
                                    struct_fwd_input_name);
        const char* GRAD_INS_FWD_TENSOR_TEMPLATE =
            "(&this->%s)->get_intermediate_tensor()";
        std::string tensor_wrapper_intermediate_tensor_str =
            paddle::string::Sprintf(GRAD_INS_FWD_TENSOR_TEMPLATE,
                                    struct_fwd_input_name);
        generated_grad_function_body +=
            paddle::string::Sprintf(CHECK_BACKWARD_INPLACE_TEMPLATE,
                                    can_be_inplaced_name,
                                    tensor_wrapper_str,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    grad_input_name,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    tensor_wrapper_intermediate_tensor_str,
                                    can_be_inplaced_name);
      }
    } else if (grad_ins_grad_slotname_map.count(grad_input_name)) {
      // Fwd Tensor's Grad
      size_t fwd_output_position = fwd_outputs_name_pos_map.at(
          grad_ins_grad_slotname_map.at(grad_input_name));
      const char* GRAD_INS_GRAD_CONTENT_TEMPLATE =
          "{ \"%s\", egr::EagerUtils::TrySyncToVars(%s[%d]) },";
      ins_contents_str +=
          paddle::string::Sprintf(GRAD_INS_GRAD_CONTENT_TEMPLATE,
                                  grad_input_name,
                                  hooked_grads,
                                  fwd_output_position);
      if (!backward_inplace_map.empty() &&
          backward_inplace_map.count(grad_input_name)) {
        process_backward_inplace = true;
        const char* GRAD_INS_HOOKED_GRAD_TEMPLATE = "auto& %s = %s[%d][0];";
        std::string hooked_grads_tensor_str =
            paddle::string::Sprintf(GRAD_INS_HOOKED_GRAD_TEMPLATE,
                                    bwd_inplace_input_name,
                                    hooked_grads,
                                    fwd_output_position);
        const char* GRAD_INS_GRAD_TENSOR_TEMPLATE = "grads[%d][0]";
        std::string grads_tensor_str = paddle::string::Sprintf(
            GRAD_INS_GRAD_TENSOR_TEMPLATE, fwd_output_position);
        generated_grad_function_body +=
            paddle::string::Sprintf(CHECK_BACKWARD_INPLACE_TEMPLATE,
                                    can_be_inplaced_name,
                                    hooked_grads_tensor_str,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    grad_input_name,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    bwd_inplace_input_name,
                                    grads_tensor_str,
                                    can_be_inplaced_name);
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "Detected mismatched slot names."
          "Unable to find forward slot name that matches %s",
          grad_input_name));
    }
  }
  if (!ins_contents_str.empty())
    ins_contents_str.pop_back();  // // Remove trailing ","

  const char* BWD_INS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerVariable>>> %s = { "
      "%s };\n";
  std::string ins_map_str =
      paddle::string::Sprintf(BWD_INS_MAP_TEMPLATE, ins_name, ins_contents_str);
  generated_grad_function_body += ins_map_str;

  for (auto const& iter : grad_ins) {
    const std::string& grad_input_name = iter.first;

    if (grad_ins_fwd_slotname_map.count(grad_input_name)) {
      // Fwd Tensor
      const std::string& fwd_name =
          grad_ins_fwd_slotname_map.at(grad_input_name);
      if (dispensable_input_name_set.count(fwd_name)) {
        std::string struct_fwd_input_name =
            grad_ins_fwd_slotname_map.at(grad_input_name) + "_";
        if (duplicable_input_name_set.count(fwd_name)) {
          const char* DISPENSABLE_GRAD_INS_FWD_CONTENT_TEMPLATE =
              "  if(this->%s.size() > 0) %s[\"%s\"] = "
              "egr::EagerUtils::TrySyncToVars(egr::EagerUtils::"
              "RecoverTensorWrapper(&this->%s));\n";
          generated_grad_function_body +=
              paddle::string::Sprintf(DISPENSABLE_GRAD_INS_FWD_CONTENT_TEMPLATE,
                                      struct_fwd_input_name,
                                      ins_name,
                                      grad_input_name,
                                      struct_fwd_input_name);
        } else {
          const char* DISPENSABLE_GRAD_INS_FWD_CONTENT_TEMPLATE =
              "  auto %s = egr::EagerUtils::RecoverTensorWrapper(&this->%s);\n"
              "  if(%s.defined()) %s[\"%s\"] = "
              "     egr::EagerUtils::TrySyncToVars(%s);\n";
          generated_grad_function_body +=
              paddle::string::Sprintf(DISPENSABLE_GRAD_INS_FWD_CONTENT_TEMPLATE,
                                      grad_input_name,
                                      struct_fwd_input_name,
                                      grad_input_name,
                                      ins_name,
                                      grad_input_name,
                                      grad_input_name);
        }
      }
    }
  }

  VLOG(6) << "Generated Ins Map";
  // [Generation] Get Outs Map
  std::string outs_contents_str = "";
  for (auto const& iter : grad_outs) {
    const std::string& grad_output_name = iter.first;

    if (grad_outs_slotname_map.count(grad_output_name)) {
      // Fwd Tensor
      const std::string& fwd_name = grad_outs_slotname_map.at(grad_output_name);

      /* Handle Special Case: "PullSparseOp", etc

          Forward:

             Ids  W
              |   |
           PullSparseOp
                |
               Out

          Backward:

             Ids  GradOut  W
              |      |     |
             PullSparseGradOp
                     |
                  GradOut

          Its grad output "GradOut" corresponds to forward output "Out",
          where there is a hidden inplace involved. So we find "GradOut"'s
         index
         in
          grads, and perform the inplace operation by constructing outs =
         {{"Out", grads[i]}}

          GradOut -> Out -> fwd_output_pos -> grads position -> grads[i]
          outs = {{"Out", grads[i]}}

          For returns, append "GradOut" to the very end of return list.
      */
      if (!fwd_inputs_name_pos_map.count(fwd_name)) {
        PADDLE_ENFORCE(fwd_outputs_name_pos_map.count(fwd_name),
                       common::errors::Fatal(
                           "fwd_name not found in fwd_inputs_name_pos_map nor "
                           "fwd_outputs_name_pos_map"));

        size_t grads_position = fwd_outputs_name_pos_map.at(fwd_name);

        const char* GRAD_OUTS_CONTENT_TEMPLATE =
            "  if((!out_metas[%d].empty()) && "
            "(!(out_metas[%d][0].IsStopGradient()))){ %s.insert({ \"%s\", "
            "egr::EagerUtils::TrySyncToVars(%s[%d])});}\n";
        outs_contents_str += paddle::string::Sprintf(GRAD_OUTS_CONTENT_TEMPLATE,
                                                     grads_position,
                                                     grads_position,
                                                     outs_name,
                                                     grad_output_name,
                                                     hooked_grads,
                                                     grads_position);

      } else {
        if (dispensable_input_name_set.count(fwd_name) &&
            grad_ins_fwd_slotname_map.count(fwd_name)) {
          continue;
        }
        size_t fwd_input_position = fwd_inputs_name_pos_map.at(fwd_name);
        if (duplicable_input_name_set.count(fwd_name) &&
            !is_op_base_per_duplicable_input) {
          const char* GRAD_OUTS_CONTENT_TEMPLATE =
              " if(!out_metas[%d].empty()){ %s.insert({ \"%s\", "
              "egr::EagerUtils::CreateVars(out_metas[%d].size())});}\n";
          outs_contents_str +=
              paddle::string::Sprintf(GRAD_OUTS_CONTENT_TEMPLATE,
                                      fwd_input_position,
                                      outs_name,
                                      grad_output_name,
                                      fwd_input_position);
        } else {
          const char* GRAD_OUTS_CONTENT_TEMPLATE =
              "  if((!out_metas[%d].empty()) && "
              "(!(out_metas[%d][0].IsStopGradient()))){ %s.insert({ \"%s\", "
              "{std::make_shared<egr::EagerVariable>(egr::Controller::Instance("
              ").GenerateUniqueName())}});}\n";
          outs_contents_str +=
              paddle::string::Sprintf(GRAD_OUTS_CONTENT_TEMPLATE,
                                      fwd_input_position,
                                      fwd_input_position,
                                      outs_name,
                                      grad_output_name);
        }
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "Detected mismatched slot names."
          "Unable to find forward slot name that matches %s",
          grad_output_name));
    }
  }

  const char* BWD_OUTS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerVariable>>> %s;\n";
  std::string outs_map_str =
      paddle::string::Sprintf(BWD_OUTS_MAP_TEMPLATE, outs_name);

  generated_grad_function_body += outs_map_str;
  generated_grad_function_body += outs_contents_str;
  generated_grad_function_body += "\n";
  for (auto const& iter : grad_outs) {
    const std::string& grad_output_name = iter.first;

    if (grad_outs_slotname_map.count(grad_output_name)) {
      // Fwd Tensor
      const std::string& fwd_name = grad_outs_slotname_map.at(grad_output_name);
      if (fwd_inputs_name_pos_map.count(fwd_name)) {
        if (dispensable_input_name_set.count(fwd_name) &&
            grad_ins_fwd_slotname_map.count(fwd_name)) {
          if (duplicable_input_name_set.count(fwd_name) &&
              !is_op_base_per_duplicable_input) {
            size_t fwd_input_position = fwd_inputs_name_pos_map.at(fwd_name);
            const char* DISPENSABLE_GRAD_OUTS_FWD_CONTENT_TEMPLATE =
                "  if((%s.size() > 0) && (!out_metas[%d].empty()) && "
                "(!out_metas[%d][0].IsStopGradient())) %s[\"%s\"] = "
                "egr::EagerUtils::CreateVars( "
                "out_metas[%d].size() );\n";
            generated_grad_function_body += paddle::string::Sprintf(
                DISPENSABLE_GRAD_OUTS_FWD_CONTENT_TEMPLATE,
                fwd_name,
                outs_name,
                grad_output_name,
                fwd_input_position);
          } else {
            size_t fwd_input_position = fwd_inputs_name_pos_map.at(fwd_name);
            const char* DISPENSABLE_GRAD_OUTS_FWD_CONTENT_TEMPLATE =
                "  if(%s.defined() && (!out_metas[%d].empty()) && "
                "(!out_metas[%d][0].IsStopGradient())) %s[\"%s\"] = "
                "{std::make_shared<egr::EagerVariable>(egr::Controller::"
                "Instance().GenerateUniqueName())};\n";
            generated_grad_function_body += paddle::string::Sprintf(
                DISPENSABLE_GRAD_OUTS_FWD_CONTENT_TEMPLATE,
                fwd_name,
                fwd_input_position,
                fwd_input_position,
                outs_name,
                grad_output_name);
          }
        }
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "Detected mismatched slot names."
          "Unable to find forward slot name that matches %s",
          grad_output_name));
    }
  }

  VLOG(6) << "Generated Outs Map";

  // [Generation] Process Backward Inplace
  if (process_backward_inplace) {
    const char* HANDLE_BACKWARD_INPLACE_BETWEEN_INPUT_AND_OUTPUT =
        "  if (%s && %s.count(\"%s\") && %s.count(\"%s\")) {\n"
        "    egr::EagerUtils::HandleViewBetweenInputAndOutput(%s[\"%s\"][0], "
        "%s[\"%s\"][0]);\n"
        "  };\n";
    std::string backward_inplace_map_str = "";
    for (auto const& iter : backward_inplace_map) {
      std::string backward_inplace_input_name = iter.first;
      std::string backward_inplace_output_name = iter.second;
      backward_inplace_map_str += paddle::string::Sprintf(
          HANDLE_BACKWARD_INPLACE_BETWEEN_INPUT_AND_OUTPUT,
          can_be_inplaced_name,
          ins_name,
          backward_inplace_input_name,
          outs_name,
          backward_inplace_output_name,
          ins_name,
          backward_inplace_input_name,
          outs_name,
          backward_inplace_output_name);
    }
    generated_grad_function_body += backward_inplace_map_str;
    VLOG(6) << "Process Backward Inplace";
  }

  // [Generation] Get Attrs Map
  const char* ATTRS_TEMPLATE = "  auto& %s = this->attr_map_;\n";
  std::string grad_attrs_str =
      paddle::string::Sprintf(ATTRS_TEMPLATE, attrs_name);
  if (fwd_op_type == "cast") {
    // switch in out dtype
    const char* CAST_GRAD =
        "  auto temp_type = %s[\"in_dtype\"];\n"
        "  %s[\"in_dtype\"] = %s[\"out_dtype\"];\n"
        "  %s[\"out_dtype\"] = temp_type;\n";
    grad_attrs_str += paddle::string::Sprintf(
        CAST_GRAD, attrs_name, attrs_name, attrs_name, attrs_name);
  }

  // Handle dynamic grad attributes
  grad_attrs_str += HandleDynamicGradAttributes(fwd_op_type, attrs_name);
  generated_grad_function_body += grad_attrs_str;

  const char* TRACE_OP_TEMPLATE =
      "  // Pass the entire attribute map to TraceOp\n"
      "  // The underlying kernel will pickup whatever attribute they need "
      "at runtime\n"
      "  egr::Controller::Instance().GetCurrentTracer()->TraceOp(\"%s\", %s, "
      "%s, %s,\n"
      "      egr::Controller::Instance().GetExpectedPlace(),\n"
      "      &this->default_attr_map_, false, {});\n";
  std::string trace_opbase_str = paddle::string::Sprintf(
      TRACE_OP_TEMPLATE, op_base_type, ins_name, outs_name, attrs_name);

  generated_grad_function_body += trace_opbase_str;

  VLOG(6) << "Generated Attrs Map";

  // [Generation] Get Return
  std::string outputs_str = "";
  size_t num_appended_outputs = 0;
  for (auto const& iter : grad_outs) {
    const std::string& grad_out_name = iter.first;
    const std::string& fwd_name = grad_outs_slotname_map.at(grad_out_name);

    if (fwd_inputs_name_pos_map.count(fwd_name)) {
      size_t fwd_input_position = fwd_inputs_name_pos_map.at(fwd_name);
      if (!is_op_base_per_duplicable_input) {
        const char* BWD_OUTPUT_TEMPLATE =
            "  if (%s.find(\"%s\") != %s.end()) { outputs[%d] = "
            "egr::EagerUtils::GetOutputs(%s[\"%s\"]); }\n";
        outputs_str += paddle::string::Sprintf(BWD_OUTPUT_TEMPLATE,
                                               outs_name,
                                               grad_out_name,
                                               outs_name,
                                               fwd_input_position,
                                               outs_name,
                                               grad_out_name);
      } else {
        const char* BWD_OUTPUT_TEMPLATE =
            "  "
            "if (%s.find(\"%s\") != %s.end()) { "
            "outputs[0].emplace_back(egr::EagerUtils::GetOutputs(%s[\"%s\"])[0]"
            "); }\n";
        outputs_str += paddle::string::Sprintf(BWD_OUTPUT_TEMPLATE,
                                               outs_name,
                                               grad_out_name,
                                               outs_name,
                                               outs_name,
                                               grad_out_name);
      }
      num_appended_outputs++;
    } else {
      PADDLE_ENFORCE(fwd_outputs_name_pos_map.count(fwd_name),
                     common::errors::Fatal(
                         "fwd_name not found in fwd_inputs_name_pos_map nor "
                         "fwd_outputs_name_pos_map"));
    }
  }

  /* Handle Special Case: "PullSparseOp", etc
     For returns, append "GradOut" to the very end of return list. */
  for (auto const& iter : grad_outs) {
    const std::string& grad_out_name = iter.first;
    const std::string& fwd_name = grad_outs_slotname_map.at(grad_out_name);

    if (fwd_outputs_name_pos_map.count(fwd_name)) {
      const char* BWD_OUTPUT_TEMPLATE =
          "  if (%s.find(\"%s\") != %s.end()) { outputs[%d] = "
          "egr::EagerUtils::GetOutputs(%s[\"%s\"]); }\n";
      outputs_str += paddle::string::Sprintf(BWD_OUTPUT_TEMPLATE,
                                             outs_name,
                                             grad_out_name,
                                             outs_name,
                                             num_appended_outputs,
                                             outs_name,
                                             grad_out_name);
      num_appended_outputs++;
    }
  }

  generated_grad_function_body += outputs_str;
  generated_grad_function_body += "\n";

  *outs_size += grad_outs.size();

  return generated_grad_function_body;
}

/* ---------------------------------------------- */
/* --------- CodeGen: GradNode::operator() ------ */
/* ---------------------------------------------- */
static std::string GenerateGradNodeCCContents(
    const ForwardGenerationInfo& fwd_info,
    const GradNodeGenerationInfo& bwd_info) {
  /* --- Process Forward Info --- */
  const std::string& fwd_op_type = fwd_info.GetOpType();
  const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map =
      fwd_info.GetFwdInputsNamePosMap();
  const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map =
      fwd_info.GetFwdOutputsNamePosMap();
  const std::vector<proto::OpProto::Var>& in_vars = fwd_info.GetInVars();
  const std::vector<proto::OpProto::Var>& out_vars = fwd_info.GetOutVars();

  VLOG(6) << "Generating Grad Node CC";

  /* [Outline]

  vector<vector<Tensor>> GradNodeXXX::operator()(vector<vector<Tensor>>& grads)
  {

    const std::shared_ptr<Tracer>& tracer = imperative::GetCurrentTracer();

    // Comes from "grad_ins"
    std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins =
            {
            "X" : this->"X", "Y" : this->"Y",
            "Out0@Grad":
  TrySyncToVars(hooked_grads["fwd_outputs_name_pos_map[grad_ins_grad_slotname_map["Out0@Grad"]]"]),
            "Out1@Grad":
  TensorsToVarBases(hooked_grads["fwd_outputs_name_pos_map[grad_ins_grad_slotname_map["Out1@Grad"]]"])
             };

    // Comes from "grad_outs"
    std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs =
            {
            "X@Grad" :
  CreateVars(this->OutputMeta()["fwd_inputs_name_pos_map[grad_outs_slotname_map["X@Grad"]]"].Size()),
            "Y@Grad" :
  CreateVars(this->OutputMeta()["fwd_inputs_name_pos_map[grad_outs_slotname_map["Y@Grad"]]"].Size())
             };

    // Visit each OpBase
    for(auto iter = "grad_node->begin()"; iter < "grad_node->end()"; iter++) {
        // Simply pass entire attribute map to kernels
        Controller.Instance().GetCurrentTracer()->TraceOp("iter->Type()", ins,
  outs, this->attr_map_,
            egr::Controller::Instance().ExpectedPlace(), false, {});
    }

    vector<vector<paddle::Tensor>> outputs(outs.size());
    for(auto& kv : outs) {
        outputs["fwd_inputs_name_pos_map[grad_outs_slotname_map[kv.first]]"] =
  GetOutputs(outs["kv.first"]);
    }

    return outputs;
  }
  */

  const char* EAGER_LOG_TEMPLATE =
      "  VLOG(3) << \"Running Eager Backward Node: %sGradNodeCompat\";\n";
  std::string generated_grad_function_body =
      paddle::string::Sprintf(EAGER_LOG_TEMPLATE, fwd_op_type);

  // This is a Copy
  auto op_base_infos = bwd_info.GetOpBaseInfos();

  /* Special Case: ops such as sum_grad_op is implemented abnormally,
                   where it unpacked duplicable GradX and created one OpBase
                   corresponds to each member of GradX[i]
     */
  bool is_op_base_per_duplicable_input = false;
  if (in_vars.size() == 1 && out_vars.size() == 1 && in_vars[0].duplicable() &&
      !out_vars[0].duplicable() &&
      op_base_infos.size() == NUM_CREATED_DUP_INPUTS) {
    is_op_base_per_duplicable_input = true;
    // Only keep the first op_base
    auto op_base_info = op_base_infos[0];
    op_base_infos.clear();
    op_base_infos.emplace_back(std::move(op_base_info));
  }

  size_t outs_size = 0;
  for (const auto& op_base_info : op_base_infos) {
    const auto& grad_ins_fwd_slotname_map =
        op_base_info.GetGradInsFwdSlotnameMap();
    const auto& grad_ins_grad_slotname_map =
        op_base_info.GetGradInsGradSlotnameMap();
    const auto& grad_outs_slotname_map = op_base_info.GetGradOutsSlotnameMap();
    const auto& grad_ins = op_base_info.GetGradIns();
    const auto& grad_outs = op_base_info.GetGradOuts();
    const auto& grad_attrs = op_base_info.GetGradAttrs();
    const auto& backward_inplace_map = op_base_info.GetBackwardInplaceMap();

    const std::string& op_base_type = op_base_info.GetOpBaseType();
    generated_grad_function_body +=
        GenerateSingleOpBase(fwd_op_type,
                             op_base_type,
                             fwd_inputs_name_pos_map,
                             fwd_outputs_name_pos_map,
                             in_vars,
                             grad_ins_fwd_slotname_map,
                             grad_ins_grad_slotname_map,
                             grad_outs_slotname_map,
                             grad_ins,
                             grad_outs,
                             grad_attrs,
                             backward_inplace_map,
                             is_op_base_per_duplicable_input,
                             &outs_size);
  }

  if (is_op_base_per_duplicable_input) {
    const char* OP_BASE_PER_DUP_INPUT_TEMPLATE =
        "  for(size_t i = 0; i < this->OutputMeta()[0].size(); i++) {\n"
        "    %s\n"
        "  }\n";
    generated_grad_function_body = paddle::string::Sprintf(
        OP_BASE_PER_DUP_INPUT_TEMPLATE, generated_grad_function_body);
  }

  const char* BWD_RETURN_TEMPLATE =
      "  const auto& out_metas = OutputMeta();\n"
      "  paddle::small_vector<std::vector<paddle::Tensor>, "
      "egr::kSlotSmallVectorSize> outputs(%d);\n"
      "%s\n"
      "  if(NeedComplexToRealConversion()) "
      "HandleComplexGradToRealGrad(&outputs);\n"
      "  return outputs;\n";
  generated_grad_function_body = paddle::string::Sprintf(
      BWD_RETURN_TEMPLATE, in_vars.size(), generated_grad_function_body);

  // [Generation] Get Full Grad Function
  const char* GRAD_FUNCTION_TEMPLATE =
      "paddle::small_vector<std::vector<paddle::Tensor>, "
      "egr::kSlotSmallVectorSize> "
      "%sGradNodeCompat::operator()("
      "paddle::small_vector<std::vector<paddle::Tensor>, "
      "egr::kSlotSmallVectorSize>& grads, bool "
      "create_graph, bool is_new_grad) {\n"
      "%s"
      "\n}";
  std::string grad_function_str = paddle::string::Sprintf(
      GRAD_FUNCTION_TEMPLATE, fwd_op_type, generated_grad_function_body);

  VLOG(6) << "Generated returns";

  return grad_function_str;
}

/* ----------------------------------------- */
/* --------- CodeGen: GradNode Header ------ */
/* ----------------------------------------- */
static std::string GenerateGradNodeHeaderContents(
    const ForwardGenerationInfo& fwd_info,
    const GradNodeGenerationInfo& bwd_info) {
  const std::string& op_type = fwd_info.GetOpType();
  const std::vector<proto::OpProto::Var>& in_vars = fwd_info.GetInVars();
  const std::vector<proto::OpProto::Var>& out_vars = fwd_info.GetOutVars();

  const auto& op_base_infos = bwd_info.GetOpBaseInfos();

  VLOG(6) << "Generating Grad Node Header";

  const char* GRAD_NODE_TEMPLATE =
      "class %sGradNodeCompat : public egr::GradNodeBase {\n"
      " public:\n"
      "  %sGradNodeCompat() : egr::GradNodeBase() { VLOG(7) << \" Construct "
      "%sGradNodeCompat \"; }\n"
      "  %sGradNodeCompat(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : "
      "egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) { VLOG(7) << \" "
      "Construct %sGradNodeCompat \"; }\n"
      "  ~%sGradNodeCompat() override { VLOG(6) << \" Destruct "
      "%sGradNodeCompat \"; }\n"
      "\n"
      "  virtual "
      "paddle::small_vector<std::vector<paddle::Tensor>, "
      "egr::kSlotSmallVectorSize> "
      "operator()("
      "paddle::small_vector<std::vector<paddle::Tensor>, "
      "egr::kSlotSmallVectorSize>& grads, bool "
      "create_graph = false, bool is_new_grad = false) "
      "override;\n"
      "\n"
      "  void ClearTensorWrappers() override {\n"
      "%s\n"
      "    SetIsTensorWrappersCleared(true);\n"
      "  }\n"
      "  std::string name() override { return \"%sGradNodeCompat\"; }\n"
      "\n"
      "std::shared_ptr<GradNodeBase> Copy() const override {{\n"
      "    auto copied_node = std::shared_ptr<%sGradNodeCompat>(new "
      "%sGradNodeCompat(*this));\n"
      "    return copied_node;\n"
      "}}\n"
      "\n"
      "  // SetX, SetY, ...\n"
      "%s\n"
      "  // SetAttrMap\n"
      "%s\n"
      " private:\n"
      "   // TensorWrappers\n"
      "%s\n"
      "   // Attribute Map\n"
      "%s\n"
      "};";

  // [Generation] Handle Attributes
  std::string set_attr_map_str =
      "   void SetAttrMap(paddle::framework::AttributeMap&& attr_map) {\n    "
      "attr_map_ = std::move(attr_map);\n  }\n";
  set_attr_map_str +=
      "   void SetDefaultAttrMap(paddle::framework::AttributeMap&& "
      "default_attr_map) {\n    default_attr_map_ = "
      "std::move(default_attr_map);\n  }\n";
  std::string attr_members_str =
      "   paddle::framework::AttributeMap attr_map_;\n";
  attr_members_str += "   paddle::framework::AttributeMap default_attr_map_;";

  VLOG(6) << "Generated SetAttr";

  // [Generation] Handle TensorWrappers
  std::unordered_set<std::string> duplicable_tensors;
  for (const proto::OpProto::Var& input : in_vars) {
    if (input.duplicable()) {
      duplicable_tensors.insert(input.name());
    }
  }
  for (const proto::OpProto::Var& output : out_vars) {
    if (output.duplicable()) {
      duplicable_tensors.insert(output.name());
    }
  }

  std::string set_tensor_wrappers_str = "";
  std::string tensor_wrapper_members_str = "";
  std::string clear_tensor_wrappers_str = "";
  for (const auto& iter : op_base_infos) {
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map =
        iter.GetGradInsFwdSlotnameMap();
    const std::unordered_set<std::string>& no_need_buffer_ins =
        iter.GetNoNeedBufferInputs();

    for (const auto& kv : grad_ins_fwd_slotname_map) {
      const std::string& tensor_wrapper_name = kv.second;
      const std::string& struct_tensor_wrapper_name = kv.second + "_";

      std::string tensor_wrapper_arg_str;
      std::string tensor_wrapper_body_str;
      std::string no_need_buffer_str = "false";
      if (no_need_buffer_ins.count(tensor_wrapper_name)) {
        no_need_buffer_str = "true";
      }
      if (duplicable_tensors.count(tensor_wrapper_name)) {
        const char* ATTR_TENSOR_WRAPPER_ARG_TEMPLATE =
            "const std::vector<paddle::Tensor>& %s";
        tensor_wrapper_arg_str = paddle::string::Sprintf(
            ATTR_TENSOR_WRAPPER_ARG_TEMPLATE, tensor_wrapper_name);

        const char* TENSOR_WRAPPER_MEMBER_TEMPLATE =
            "   std::vector<egr::TensorWrapper> %s;\n";
        tensor_wrapper_members_str += paddle::string::Sprintf(
            TENSOR_WRAPPER_MEMBER_TEMPLATE, struct_tensor_wrapper_name);

        const char* SET_TENSOR_WRAPPER_BODY_TEMPLATE =
            "for(const auto& eager_tensor : %s) {\n"
            "          %s.emplace_back( egr::TensorWrapper(eager_tensor "
            ", %s) );\n"
            "      }\n";
        tensor_wrapper_body_str =
            paddle::string::Sprintf(SET_TENSOR_WRAPPER_BODY_TEMPLATE,
                                    tensor_wrapper_name,
                                    struct_tensor_wrapper_name,
                                    no_need_buffer_str);

        const char* CLEAR_TENSOR_WRAPPER_TEMPLATE =
            "for (auto tw: %s)   {\n"
            "       tw.clear();\n"
            "     }\n";
        clear_tensor_wrappers_str += paddle::string::Sprintf(
            CLEAR_TENSOR_WRAPPER_TEMPLATE, struct_tensor_wrapper_name);

      } else {
        const char* ATTR_TENSOR_WRAPPER_ARG_TEMPLATE =
            "const paddle::Tensor& %s";
        tensor_wrapper_arg_str = paddle::string::Sprintf(
            ATTR_TENSOR_WRAPPER_ARG_TEMPLATE, tensor_wrapper_name);

        const char* TENSOR_WRAPPER_MEMBER_TEMPLATE =
            "   egr::TensorWrapper %s;\n";
        tensor_wrapper_members_str += paddle::string::Sprintf(
            TENSOR_WRAPPER_MEMBER_TEMPLATE, struct_tensor_wrapper_name);

        const char* SET_TENSOR_WRAPPER_BODY_TEMPLATE =
            "%s = egr::TensorWrapper(%s, %s);\n";
        tensor_wrapper_body_str =
            paddle::string::Sprintf(SET_TENSOR_WRAPPER_BODY_TEMPLATE,
                                    struct_tensor_wrapper_name,
                                    tensor_wrapper_name,
                                    no_need_buffer_str);

        const char* CLEAR_TENSOR_WRAPPER_TEMPLATE = "   %s.clear();\n";
        clear_tensor_wrappers_str += paddle::string::Sprintf(
            CLEAR_TENSOR_WRAPPER_TEMPLATE, struct_tensor_wrapper_name);
      }
      const char* SET_TENSOR_WRAPPER_TEMPLATE =
          "   void SetTensorWrapper_%s(%s) {\n    %s\n  }\n";
      set_tensor_wrappers_str +=
          paddle::string::Sprintf(SET_TENSOR_WRAPPER_TEMPLATE,
                                  tensor_wrapper_name,
                                  tensor_wrapper_arg_str,
                                  tensor_wrapper_body_str);
    }
  }
  VLOG(6) << "Generated TensorWrapper";

  std::string grad_node_str =
      paddle::string::Sprintf(GRAD_NODE_TEMPLATE,
                              op_type,
                              op_type,
                              op_type,
                              op_type,
                              op_type,
                              op_type,
                              op_type,
                              clear_tensor_wrappers_str,
                              op_type,
                              op_type,
                              op_type,
                              set_tensor_wrappers_str,
                              set_attr_map_str,
                              tensor_wrapper_members_str,
                              attr_members_str);

  return grad_node_str;
}

/* --------------------------------- */
/* --------- FileGeneration --------- */
/* ---------------------------------- */
static std::string GenerateDygraphHFileIncludes() {
  std::string dygraph_forward_api_includes_str =
      "#pragma once\n"
      "#include \"glog/logging.h\"\n"
      "#include \"paddle/fluid/eager/autograd_meta.h\"\n"
      "#include \"paddle/phi/core/memory/stats.h\"\n"
      "#include \"paddle/phi/api/all.h\"\n"
      "#include \"paddle/fluid/eager/utils.h\"\n"
      "#include \"paddle/fluid/imperative/tracer.h\"\n"
      "#include \"paddle/fluid/framework/op_registry.h\"\n"
      "#include "
      "\"paddle/fluid/eager/api/manual/fluid_manual/"
      "dygraph_forward_api.h\"\n\n";

  dygraph_forward_api_includes_str +=
      "extern std::unordered_map<std::string, std::vector<std::string>> "
      "core_ops_legacy_args_info;\n";
  dygraph_forward_api_includes_str +=
      "extern std::unordered_map<std::string, std::vector<std::string>> "
      "core_ops_legacy_args_type_info;\n";
  dygraph_forward_api_includes_str +=
      "extern std::unordered_map<std::string, std::vector<std::string>> "
      "core_ops_legacy_returns_info;\n\n";

  return dygraph_forward_api_includes_str;
}

static void GenerateForwardHFile(const std::string& dygraph_forward_api_path,
                                 const std::string& dygraph_forward_api_str) {
  std::ofstream forward_header_stream(dygraph_forward_api_path, std::ios::out);
  forward_header_stream << dygraph_forward_api_str;
  forward_header_stream.close();
}

static void GenerateForwardDygraphFile(const std::string& forward_cc_path,
                                       const std::string& fwd_function_str) {
  const char* FORWARD_INCLUDE_TEMPLATE =
      "#include "
      "\"paddle/fluid/eager/api/generated/fluid_generated/"
      "dygraph_forward_api.h\"\n"
      "#include "
      "\"paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes.h\"\n"
      "#include \"paddle/fluid/eager/api/utils/global_utils.h\"\n"
      "#include \"paddle/fluid/imperative/amp_utils.h\"\n"
      "#include \"paddle/fluid/eager/amp_auto_cast.h\"\n"
      "#include \"paddle/phi/core/platform/profiler/event_tracing.h\"\n\n";

  std::string forward_cc_include_str =
      paddle::string::Sprintf(FORWARD_INCLUDE_TEMPLATE);
  std::ofstream forward_cc_stream(forward_cc_path, std::ios::out);
  forward_cc_stream << forward_cc_include_str;
  forward_cc_stream << fwd_function_str;
  forward_cc_stream.close();
}

static void GenerateNodeHFile(const std::string& node_h_path,
                              const std::string& grad_node_str) {
  std::string node_h_include_str =
      "#pragma once\n"
      "#include \"paddle/fluid/eager/tensor_wrapper.h\"\n"
      "#include \"paddle/fluid/imperative/tracer.h\"\n"
      "#include \"paddle/fluid/eager/grad_node_info.h\"\n"
      "#include "
      "\"paddle/fluid/eager/api/manual/fluid_manual/nodes/nodes.h\"\n\n";

  std::ofstream node_h_stream(node_h_path, std::ios::out);
  node_h_stream << node_h_include_str;
  node_h_stream << grad_node_str;
  node_h_stream.close();
}

static void GenerateNodeCCFile(const std::string& node_cc_path,
                               const std::string& grad_function_str) {
  const char* NODE_CC_INCLUDE_TEMPLATE =
      "#include \"glog/logging.h\"\n"
      "#include \"paddle/phi/api/all.h\"\n"
      "#include \"paddle/fluid/imperative/tracer.h\"\n"
      "#include \"paddle/fluid/framework/op_registry.h\"\n"
      "#include \"paddle/fluid/eager/utils.h\"\n"
      "#include \"paddle/fluid/eager/api/utils/global_utils.h\"\n"
      "#include "
      "\"paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes.h\"\n\n";
  std::string node_cc_include_str =
      paddle::string::Sprintf(NODE_CC_INCLUDE_TEMPLATE);
  std::ofstream node_cc_stream(node_cc_path, std::ios::out);
  node_cc_stream << node_cc_include_str;
  node_cc_stream << grad_function_str;
  node_cc_stream.close();
}

static std::string ConvertCoreOpsInfosToString(
    const std::unordered_map<std::string, std::vector<std::string>>&
        core_ops_info) {
  std::string core_ops_legacy_returns_info_init_str = "";
  for (const auto& iter : core_ops_info) {
    const char* Core_Ops_Returns_TEMPLATE = "{ \"%s\", { %s } },\n";
    const std::string& op_type = iter.first;

    std::string returns_str = "";
    for (const auto& vector_iter : iter.second) {
      returns_str += "\"" + vector_iter + "\" ,";
    }

    // Remove trailing ','
    if (!returns_str.empty()) returns_str.pop_back();
    std::string op_type_init_str = paddle::string::Sprintf(
        Core_Ops_Returns_TEMPLATE, op_type, returns_str);
    core_ops_legacy_returns_info_init_str += op_type_init_str;
  }

  // Remove trailing ','
  if (!core_ops_legacy_returns_info_init_str.empty())
    core_ops_legacy_returns_info_init_str.pop_back();

  return core_ops_legacy_returns_info_init_str;
}

static std::string GenerateCoreOpsArgsInfo() {
  const char* Core_Ops_Returns_MAP_TEMPLATE =
      "std::unordered_map<std::string, std::vector<std::string>> "
      "core_ops_legacy_args_info = { %s };\n";

  std::string core_ops_args_info_init_str =
      ConvertCoreOpsInfosToString(core_ops_legacy_args_info);

  std::string core_ops_info_str = paddle::string::Sprintf(
      Core_Ops_Returns_MAP_TEMPLATE, core_ops_args_info_init_str);

  return core_ops_info_str;
}

static std::string GenerateCoreOpsArgsTypeInfo() {
  const char* Core_Ops_Returns_MAP_TEMPLATE =
      "std::unordered_map<std::string, std::vector<std::string>> "
      "core_ops_legacy_args_type_info = { %s };\n";

  std::string core_ops_args_type_info_init_str =
      ConvertCoreOpsInfosToString(core_ops_legacy_args_type_info);

  std::string core_ops_info_str = paddle::string::Sprintf(
      Core_Ops_Returns_MAP_TEMPLATE, core_ops_args_type_info_init_str);

  return core_ops_info_str;
}

static std::string GenerateCoreOpsReturnsInfo() {
  const char* Core_Ops_Returns_MAP_TEMPLATE =
      "std::unordered_map<std::string, std::vector<std::string>> "
      "core_ops_legacy_returns_info = { %s };\n";

  std::string core_ops_legacy_returns_info_init_str =
      ConvertCoreOpsInfosToString(core_ops_legacy_returns_info);

  std::string core_ops_info_str = paddle::string::Sprintf(
      Core_Ops_Returns_MAP_TEMPLATE, core_ops_legacy_returns_info_init_str);

  return core_ops_info_str;
}

static void DygraphCodeGeneration(const std::string& output_dir,
                                  int split_count) {
  std::string dygraph_forward_api_str = GenerateDygraphHFileIncludes();
  std::string fwd_function_str = "";
  std::string grad_node_h_str = "";
  std::string grad_node_cc_str = "";

  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  paddle::flat_hash_map<std::string, OpInfo> op_info_map_need_gen;

  for (auto& pair : op_info_map) {
    const OpInfo& op_info = pair.second;
    proto::OpProto* op_proto = op_info.proto_;

    if (!CheckOpProto(op_proto)) continue;
    const std::string& op_type = op_proto->type();
    if (black_ops_list.count(op_type)) {
      continue;
    }

    // Skip the sparse op
    if (op_type.compare(0, 7, "sparse_") == 0 && op_type != "sparse_momentum" &&
        op_type != "sparse_attention") {
      continue;
    }

    GradNodeGenerationInfo bwd_info;

    bool is_available = CollectGradInformationFromOpInfo(op_info, &bwd_info);

    if (!is_available && !bwd_info.GenerateForwardOnly()) {
      VLOG(6) << "Skipped operator: " << op_type;
      continue;
    }

    op_info_map_need_gen.emplace(pair);
  }

  int each_cc_file_api_size =
      static_cast<int>(op_info_map_need_gen.size() / split_count);
  if (op_info_map_need_gen.size() % split_count != 0) {
    each_cc_file_api_size++;
  }
  int api_index = 0;
  int file_index = 0;

  for (auto& pair : op_info_map_need_gen) {
    const OpInfo& op_info = pair.second;
    proto::OpProto* op_proto = op_info.proto_;

    const std::string& op_type = op_proto->type();

    /* ----------------------------- */
    /* ---- Collect Information ---- */
    /* ----------------------------- */

    ForwardGenerationInfo fwd_info;
    GradNodeGenerationInfo bwd_info;

    VLOG(6) << "-------- CollectInformationFromOpInfo -------";

    CollectForwardInformationFromOpInfo(op_info, &fwd_info);

    CollectGradInformationFromOpInfo(op_info, &bwd_info);

    VLOG(6) << "-------- PurifyOpProto -------";
    PurifyForwardOpProto(*op_proto, &fwd_info);
    if (!bwd_info.GenerateForwardOnly()) {
      PurifyGradNodeGenerationInfo(*op_proto, &bwd_info);
    }

    /* --------------------------- */
    /* --------- CodeGen --------- */
    /* --------------------------- */
    VLOG(6) << "-------- GenerateForwardFunctionContents -------";
    std::pair<std::string, std::string> body_and_declaration =
        GenerateForwardFunctionContents(fwd_info, bwd_info, {});

    fwd_function_str += body_and_declaration.first + "\n";

    VLOG(6) << "-------- GenerateDygraphForwardAPIContents -------";
    std::string fwd_function_declare_str = body_and_declaration.second;
    dygraph_forward_api_str += fwd_function_declare_str;

    auto& infer_inplace =
        paddle::framework::OpInfoMap::Instance().Get(op_type).infer_inplace_;
    std::map<std::string, std::string> forward_inplace_map;
    // Inplace Function Generator.
    // `sum` op has duplicate input. Don't consider adding inplace strategy
    // for `sum` in temporary.
    if (infer_inplace && !special_inplace_op_set.count(op_type)) {
      auto in_to_outs = infer_inplace(true);
      for (auto& inplace_pair : in_to_outs) {
        forward_inplace_map[inplace_pair.second] = inplace_pair.first;
      }

      VLOG(6) << "-------- GenerateInplaceForwardFunctionContents -------";
      std::pair<std::string, std::string> inplace_body_and_declaration =
          GenerateForwardFunctionContents(
              fwd_info, bwd_info, forward_inplace_map);

      fwd_function_str += inplace_body_and_declaration.first + "\n";

      VLOG(6) << "-------- GenerateInplaceDygraphForwardAPIContents -------";
      std::string inplace_fwd_function_declare_str =
          inplace_body_and_declaration.second;
      dygraph_forward_api_str += inplace_fwd_function_declare_str;
    }

    if (!bwd_info.GenerateForwardOnly()) {
      VLOG(6) << "-------- GenerateGradNodeHeaderContents -------";
      grad_node_h_str += GenerateGradNodeHeaderContents(fwd_info, bwd_info);
      grad_node_h_str += "\n";

      VLOG(6) << "-------- GenerateGradNodeCCContents -------";
      grad_node_cc_str += GenerateGradNodeCCContents(fwd_info, bwd_info);
      grad_node_cc_str += "\n";
    }

    VLOG(6) << op_type << ": Finished Generating Op: " << op_type;

    api_index++;
    if (api_index / each_cc_file_api_size > file_index) {
      file_index++;
      VLOG(6) << "-------- GenerateDygraphForwardCCFile -------";
      std::string forward_cc_path = output_dir +
                                    "/forwards/dygraph_forward_functions" +
                                    std::to_string(file_index) + ".tmp.cc";
      fwd_function_str += "\n";
      GenerateForwardDygraphFile(forward_cc_path, fwd_function_str);
      fwd_function_str = "";

      VLOG(6) << "-------- GenerateNodeCCFile -------";
      std::string node_cc_path =
          output_dir + "/nodes/nodes" + std::to_string(file_index) + ".tmp.cc";
      GenerateNodeCCFile(node_cc_path, grad_node_cc_str);
      grad_node_cc_str = "";
    }
  }

  file_index++;
  VLOG(6) << "-------- GenerateDygraphForwardCCFile -------";
  std::string forward_cc_path = output_dir +
                                "/forwards/dygraph_forward_functions" +
                                std::to_string(file_index) + ".tmp.cc";
  GenerateForwardDygraphFile(forward_cc_path, fwd_function_str);
  fwd_function_str = "";

  GenerateForwardDygraphFile(
      output_dir + "/forwards/dygraph_forward_functions_args_info.tmp.cc",
      GenerateCoreOpsArgsInfo());
  GenerateForwardDygraphFile(
      output_dir + "/forwards/dygraph_forward_functions_args_type_info.tmp.cc",
      GenerateCoreOpsArgsTypeInfo());
  GenerateForwardDygraphFile(
      output_dir + "/forwards/dygraph_forward_functions_returns_info.tmp.cc",
      GenerateCoreOpsReturnsInfo());

  VLOG(6) << "-------- GenerateNodeCCFile -------";
  std::string node_cc_path =
      output_dir + "/nodes/nodes" + std::to_string(file_index) + ".tmp.cc";
  GenerateNodeCCFile(node_cc_path, grad_node_cc_str);
  grad_node_cc_str = "";

  VLOG(6) << "-------- GenerateForwardHFile -------";
  std::string dygraph_forward_api_path =
      output_dir + "/dygraph_forward_api.tmp.h";
  GenerateForwardHFile(dygraph_forward_api_path, dygraph_forward_api_str);

  VLOG(6) << "-------- GenerateNodeHFile -------";
  std::string node_h_path = output_dir + "/nodes/nodes.tmp.h";
  GenerateNodeHFile(node_h_path, grad_node_h_str);
}

}  // namespace paddle::framework

std::map<std::string, std::pair<std::string, std::string>> view_op_map = {
    {"squeeze2", {"X", "Out"}},
    {"unsqueeze2", {"X", "Out"}},
    {"reshape2", {"X", "Out"}},
    {"flatten_contiguous_range", {"X", "Out"}},
};

std::map<std::string, std::set<std::string>> op_passing_outs_map = {
    {"sgd", {"ParamOut", "MasterParamOut"}},
    {"rmsprop",
     {"ParamOut",
      "MomentOut",
      "MeanSquareOut",
      "MeanGradOut",
      "MasterParamOut"}},
    {"ftrl", {"ParamOut", "SquaredAccumOut", "LinearAccumOut"}},
    {"adadelta",
     {"ParamOut",
      "AvgSquaredGradOut",
      "AvgSquaredUpdateOut",
      "MasterParamOut"}},
    {"adagrad", {"ParamOut", "MomentOut", "MasterParamOut"}},
    {"adamax", {"ParamOut", "MomentOut", "InfNormOut", "MasterParamOut"}},
    {"dpsgd", {"ParamOut"}},
    {"decayed_adagrad", {"ParamOut", "MomentOut"}},
    {"lars_momentum", {"ParamOut", "VelocityOut"}},
    {"coalesce_tensor", {"Output", "FusedOutput"}},
    {"adam",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Moment2MaxOut",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"merged_adam",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Moment2MaxOut",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"fused_adam",
     {"ParamsOut",
      "Moments1Out",
      "Moments2Out",
      "Moments2MaxOut",
      "Beta1PowsOut",
      "Beta2PowsOut",
      "MasterParamsOut"}},
    {"adamw",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Moment2MaxOut",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"lamb",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"average_accumulates",
     {"out_sum_1",
      "out_sum_2",
      "out_sum_3",
      "out_num_accumulates",
      "out_old_num_accumulates",
      "out_num_updates"}},
    {"momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"merged_momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"sparse_momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"batch_norm", {"MeanOut", "VarianceOut"}},
    {"sync_batch_norm", {"MeanOut", "VarianceOut"}},
    {"accuracy", {"Correct", "Total"}},
    {"fill_constant", {"Out"}},
    {"p_recv", {"Out"}},
    {"partial_recv", {"Out"}},
    {"matmul", {"Out"}},
    {"c_broadcast", {"Out"}},
    {"c_sync_calc_stream", {"Out"}},
    {"c_sync_comm_stream", {"Out"}},
    {"c_reduce_sum", {"Out"}},
    {"c_reduce_max", {"Out"}},
    {"c_reduce_min", {"Out"}},
    {"c_reduce_prod", {"Out"}},
    {"c_reduce", {"Out"}},
    {"c_scatter", {"Out"}},
    {"barrier", {"Out"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"fake_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"fake_channel_wise_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"check_finite_and_unscale", {"Out", "FoundInfinite"}},
    {"update_loss_scaling",
     {"Out", "LossScaling", "OutGoodSteps", "OutBadSteps"}},
    {"moving_average_abs_max_scale",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"rnn", {"DropoutState"}},
    {"run_program", {"Out", "DOut", "OutScope", "CUDAGraph"}},
    {"clear_float_status", {"FloatStatusOut"}},
    {"get_float_status", {"FloatStatusOut"}},
    {"assign", {"Out"}},
    {"assign_value", {"Out"}},
    {"split", {"Out"}},
    {"concat", {"Out"}},
    {"fused_multi_transformer_int8", {"CacheKVOut"}},
    {"group_norm", {"Mean", "Variance"}},
    {"resnet_basic_block",
     {"Mean1Out", "Var1Out", "Mean2Out", "Var2Out", "Mean3Out", "Var3Out"}},
};

std::map<std::string, std::set<std::string>> op_ins_map = {
    {"fc", {"Input", "W", "Bias"}},
    {"precision_recall",
     {"MaxProbs", "Indices", "Labels", "Weights", "StatesInfo"}},
    {"layer_norm", {"X", "Scale", "Bias"}},
    {"fused_conv2d_add_act", {"Input", "Filter", "Bias", "ResidualData"}},
    {"bincount", {"X", "Weights"}},
    {"fused_attention",
     {"X",
      "LnScale",
      "LnBias",
      "QKVW",
      "QKVBias",
      "CacheKV",
      "SrcMask",
      "OutLinearW",
      "OutLinearBias",
      "Ln2Scale",
      "Ln2Bias"}},
    {"fused_gate_attention",
     {"Query",
      "Key",
      "QueryWeight",
      "KeyWeight",
      "ValueWeight",
      "QKVWeight",
      "NonbatchedBias",
      "SrcMask",
      "GateWeight",
      "GateBias",
      "OutLinearWeight",
      "OutLinearBias"}},
    {"fused_multi_transformer_int8",
     {"X",           "LnScale",           "LnBias",       "QKVW",
      "QKVBias",     "CacheKV",           "TimeStep",     "SrcMask",
      "OutLinearW",  "OutLinearBias",     "FFNLnScale",   "FFNLnBias",
      "FFN1Weight",  "FFN1Bias",          "FFN2Weight",   "FFN2Bias",
      "QKVOutScale", "OutLinearOutScale", "FFN1OutScale", "FFN2OutScale"}},
    {"fused_bias_dropout_residual_layer_norm",
     {"X", "Residual", "Bias", "LnScale", "LnBias"}},
    {"instance_norm", {"X", "Scale", "Bias"}},
    {"gru_unit", {"Input", "HiddenPrev", "Weight", "Bias"}},
    {"label_smooth", {"X", "PriorDist"}},
    {"assign", {"X"}},
    {"crop", {"X", "Y", "Offsets"}},
    {"crop_tensor", {"X", "Shape", "Offsets"}},
    {"reshape2", {"X", "Shape"}},
    {"expand", {"X", "ExpandTimes"}},
    {"slice",
     {"Input",
      "StartsTensor",
      "EndsTensor",
      "StartsTensorList",
      "EndsTensorList"}},
    {"strided_slice",
     {"Input",
      "StartsTensor",
      "EndsTensor",
      "StridesTensor",
      "StartsTensorList",
      "EndsTensorList",
      "StridesTensorList"}},
    {"set_value",
     {"Input",
      "ValueTensor",
      "StartsTensorList",
      "EndsTensorList",
      "StepsTensorList"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"X", "InScale", "InAccum", "InState"}},
    {"nll_loss", {"X", "Label", "Weight"}},
    {"bilinear_tensor_product", {"X", "Y", "Weight", "Bias"}},
    {"gather", {"X", "Index", "Axis"}},
    {"repeat_interleave", {"X", "RepeatsTensor"}},
    {"roi_pool", {"X", "ROIs", "RoisNum"}},
    {"roi_align", {"X", "ROIs", "RoisNum"}},
    {"prroi_pool", {"X", "ROIs", "BatchRoINums"}},
    {"psroi_pool", {"X", "ROIs", "RoisNum"}},
    {"collect_fpn_proposals",
     {"MultiLevelRois", "MultiLevelScores", "MultiLevelRoIsNum"}},
    {"distribute_fpn_proposals", {"FpnRois", "RoisNum"}},
    {"warpctc", {"Logits", "Label", "LogitsLength", "LabelLength"}},
    {"hierarchical_sigmoid",
     {"X", "W", "Label", "PathTable", "PathCode", "Bias"}},
    {"moving_average_abs_max_scale", {"X", "InAccum", "InState"}},
    {"multiclass_nms3", {"BBoxes", "Scores", "RoisNum"}},
    {"box_coder", {"PriorBox", "PriorBoxVar", "TargetBox"}},
    {"momentum", {"Param", "Grad", "Velocity", "LearningRate", "MasterParam"}},
    {"merged_momentum",
     {"Param", "Grad", "Velocity", "LearningRate", "MasterParam"}},
    {"sparse_momentum",
     {"Param", "Grad", "Velocity", "Index", "LearningRate", "MasterParam"}},
    {"rnn", {"Input", "PreState", "WeightList", "SequenceLength"}},
    {"run_program", {"X", "Params"}},
    {"fused_feedforward",
     {"Dropout1Seed",
      "Dropout2Seed",
      "Linear1Bias",
      "Linear2Bias",
      "Ln1Scale",
      "Ln1Bias",
      "Ln2Scale",
      "Ln2Bias"}},
    {"faster_tokenizer", {"Text", "Vocab", "TextPair"}},
    {"matrix_rank", {"X", "TolTensor"}},
    {"rmsprop",
     {"Param",
      "MeanSquare",
      "Grad",
      "Moment",
      "LearningRate",
      "MeanGrad",
      "MasterParam"}},
    {"adam",
     {"Param",
      "Grad",
      "LearningRate",
      "Moment1",
      "Moment2",
      "Moment2Max",
      "Beta1Pow",
      "Beta2Pow",
      "MasterParam"}},
    {"merged_adam",
     {"Param",
      "Grad",
      "LearningRate",
      "Moment1",
      "Moment2",
      "Moment2Max",
      "Beta1Pow",
      "Beta2Pow",
      "MasterParam"}},
    {"fused_adam",
     {"Params",
      "Grads",
      "LearningRate",
      "Moments1",
      "Moments2",
      "Moments2Max",
      "Beta1Pows",
      "Beta2Pows",
      "MasterParams",
      "SkipUpdate"}},
    {"adamw",
     {"Param",
      "Grad",
      "LearningRate",
      "Moment1",
      "Moment2",
      "Moment2Max",
      "Beta1Pow",
      "Beta2Pow",
      "MasterParam"}},
    {"adamax",
     {"Param",
      "Grad",
      "LearningRate",
      "Moment",
      "InfNorm",
      "Beta1Pow",
      "MasterParam"}},
    {"lamb",
     {"Param",
      "Grad",
      "LearningRate",
      "Moment1",
      "Moment2",
      "Beta1Pow",
      "Beta2Pow",
      "MasterParam"}},
    {"sparse_attention",
     {"Q", "K", "V", "Offset", "Columns", "KeyPaddingMask", "AttnMask"}},
    {"sgd", {"Param", "LearningRate", "Grad", "MasterParam"}},
    {"adagrad", {"Param", "Grad", "Moment", "LearningRate", "MasterParam"}},
    {"adadelta",
     {"Param",
      "Grad",
      "AvgSquaredGrad",
      "AvgSquaredUpdate",
      "LearningRate",
      "MasterParam"}},
    {"graph_khop_sampler", {"Row", "Eids", "Col_Ptr", "X"}},
    {"nce",
     {"Input",
      "Label",
      "Weight",
      "Bias",
      "SampleWeight",
      "CustomDistProbs",
      "CustomDistAlias",
      "CustomDistAliasProbs"}},
    {"yolov3_loss", {"X", "GTBox", "GTLabel", "GTScore"}},
    {"check_finite_and_unscale", {"X", "Scale", "FloatStatus"}},
    {"group_norm", {"X", "Scale", "Bias"}},
    {"linear_chain_crf", {"Emission", "Transition", "Label", "Length"}},
    {"crf_decoding", {"Emission", "Transition", "Label", "Length"}},
    {"chunk_eval", {"Inference", "Label", "SeqLength"}},
    {"sequence_mask", {"X", "MaxLenTensor"}},
    {"graph_reindex",
     {"X", "Neighbors", "Count", "HashTable_Value", "HashTable_Index"}},
    {"graph_sample_neighbors", {"Row", "Col_Ptr", "X", "Eids", "Perm_Buffer"}},
    {"crop", {"X", "Y", "Offsets"}},
    {"batch_norm",
     {"X", "Scale", "Bias", "Mean", "Variance", "MomentumTensor"}},
    {"linear_interp", {"X", "OutSize"}},
    {"bilinear_interp", {"X", "OutSize"}},
    {"trilinear_interp", {"X", "OutSize"}},
    {"nearest_interp", {"X", "OutSize"}},
    {"bicubic_interp", {"X", "OutSize"}},
    {"resnet_basic_block",
     {"X",
      "Filter1",
      "Scale1",
      "Bias1",
      "Mean1",
      "Var1",
      "Filter2",
      "Scale2",
      "Bias2",
      "Mean2",
      "Var2",
      "Filter3",
      "Scale3",
      "Bias3",
      "Mean3",
      "Var3"}},
    {"graph_send_recv", {"X", "Src_index", "Dst_index", "Out_size"}},
    {"graph_send_ue_recv", {"X", "Y", "Src_index", "Dst_index", "Out_size"}},
};

std::map<std::string, std::set<std::string>> op_outs_map = {
    {"rank_attention", {"InputHelp", "Out", "InsRank"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"batch_norm",
     {"Y",
      "MeanOut",
      "VarianceOut",
      "SavedMean",
      "SavedVariance",
      "ReserveSpace"}},
    {"lstsq", {"Solution", "Residuals", "Rank", "SingularValues"}},
    {"fused_attention", {"LnMean",         "LnVariance",
                         "LnOut",          "QKVOut",
                         "QKVBiasOut",     "TransposeOut2",
                         "QKOut",          "QKTVOut",
                         "SoftmaxOut",     "AttnDropoutMaskOut",
                         "AttnDropoutOut", "SrcMaskOut",
                         "FMHAOut",        "OutLinearOut",
                         "DropoutMaskOut", "Ln2Mean",
                         "Ln2Variance",    "BiasDropoutResidualOut",
                         "CacheKVOut",     "Y"}},
    {"fused_bias_dropout_residual_layer_norm",
     {"BiasDropoutResidualOut", "DropoutMaskOut", "LnMean", "LnVariance", "Y"}},
    {"fused_gate_attention",
     {"QueryTransposeOut",
      "KeyTransposeOut",
      "ValueTransposeOut",
      "QKVTransposeOut",
      "SoftmaxOut",
      "FMHAOut",
      "GateOut",
      "Out"}},
    {"sync_batch_norm",
     {"Y",
      "MeanOut",
      "VarianceOut",
      "SavedMean",
      "SavedVariance",
      "ReserveSpace"}},
    {"adadelta",
     {"ParamOut",
      "AvgSquaredGradOut",
      "AvgSquaredUpdateOut",
      "MasterParamOut"}},
    {"unique", {"Out", "Index", "Indices", "Counts"}},
    {"unique_consecutive", {"Out", "Index", "Counts"}},
    {"generate_proposals", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
    {"collect_fpn_proposals", {"FpnRois", "RoisNum"}},
    {"matrix_nms", {"Out", "Index", "RoisNum"}},
    {"distribute_fpn_proposals",
     {"MultiFpnRois", "RestoreIndex", "MultiLevelRoIsNum"}},
    {"moving_average_abs_max_scale",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"rmsprop",
     {"ParamOut",
      "MomentOut",
      "MeanSquareOut",
      "MeanGradOut",
      "MasterParamOut"}},
    {"multiclass_nms3", {"Out", "NmsRoisNum"}},
    {"generate_proposals_v2", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
    {"momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"merged_momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"sparse_momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"rnn", {"DropoutState", "Reserve", "Out", "State"}},
    {"run_program", {"DOut", "CUDAGraph"}},
    {"adam",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Moment2MaxOut",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"merged_adam",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Moment2MaxOut",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"fused_adam",
     {"ParamsOut",
      "Moments1Out",
      "Moments2Out",
      "Moments2MaxOut",
      "Beta1PowsOut",
      "Beta2PowsOut",
      "MasterParamsOut"}},
    {"adamw",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Moment2MaxOut",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"adamax",
     {"ParamOut", "MomentOut", "InfNormOut", "Beta1Pow", "MasterParamOut"}},
    {"sgd", {"ParamOut", "MasterParamOut"}},
    {"adagrad", {"ParamOut", "MomentOut", "MasterParamOut"}},
    {"lamb",
     {"ParamOut",
      "Moment1Out",
      "Moment2Out",
      "Beta1PowOut",
      "Beta2PowOut",
      "MasterParamOut"}},
    {"fused_multi_transformer_int8", {"CacheKVOut", "Out"}},
    {"resnet_basic_block",
     {"Y",         "Conv1",     "SavedMean1", "SavedInvstd1", "Mean1Out",
      "Var1Out",   "Conv2",     "SavedMean2", "SavedInvstd2", "Mean2Out",
      "Var2Out",   "Conv3",     "SavedMean3", "SavedInvstd3", "Mean3Out",
      "Var3Out",   "MaxInput1", "MaxFilter1", "MaxInput2",    "MaxFilter2",
      "MaxInput3", "MaxFilter3"}},
};

std::set<std::string> special_inplace_op_set = {
    "sum",     // `sum` op has duplicate input
    "assign",  // output of `assign` op is in `op_passing_outs_map`
};

std::set<std::string> special_no_need_buffer_op_set = {
    "sequence_conv",
};

int run_generator(int argc, char* argv[]) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  // We need a fake device to trigger the registration of the common kernel and
  // generate api
  paddle::operators::RegisterCustomDeviceCommonKernel("fake_device");
#endif

  std::string eager_root = argv[1];
  int split_count = atoi(argv[2]);

  paddle::framework::PrepareAttrMapForOps();

  paddle::framework::DygraphCodeGeneration(eager_root, split_count);

  return 0;
}
